# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch
import torch.nn.functional as F

# Import flash/sage attn with automatic fallback from compatibility layer
from ...optimization.compatibility import (
    call_flash_attn_2_varlen, call_flash_attn_3_varlen,
    call_sage_attn_2_varlen, call_sage_attn_3_varlen
)

from torch import nn


def pytorch_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False):
    """
    A PyTorch-based implementation of variable-length attention to replace flash_attn_varlen_func.
    It processes each sequence in the batch individually.
    
    NOTE: max_seqlen_q and max_seqlen_k are accepted for API compatibility but not used.
    PyTorch's scaled_dot_product_attention automatically handles variable sequence lengths.
    
    COMPILE OPTIMIZATION: Uses torch.tensor_split to avoid .item() graph breaks
    """
    # Split q, k, v using cumulative sequence lengths
    # NOTE: torch.tensor_split requires int64 dtype and CPU device (PyTorch requirements)
    q_splits = list(torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0))
    k_splits = list(torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0))
    v_splits = list(torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0))

    # Process each sequence
    output_splits = []
    for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
        # Reshape for torch's scaled_dot_product_attention which expects (batch, heads, seq, dim).
        # Here, we treat each sequence as a batch of 1.
        q_i = q_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_q, head_dim)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_k, head_dim)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_k, head_dim)

        # Use PyTorch's built-in scaled dot-product attention.
        output_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, 
            dropout_p=dropout_p if not deterministic else 0.0,
            is_causal=causal
        )

        # Reshape the output back to the original format (seq_len, heads, head_dim)
        output_i = output_i.squeeze(0).permute(1, 0, 2)
        output_splits.append(output_i)
    
    # Concatenate all outputs
    return torch.cat(output_splits, dim=0)


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    """
    Variable-length attention with configurable backend.
    
    Supported backends:
    - sdpa: PyTorch SDPA (fully compilable, always available)
    - flash_attn_2: Flash Attention 2 (Ampere+)
    - flash_attn_3: Flash Attention 3 (Hopper+)
    - sageattn_2: SageAttention 2
    - sageattn_3: SageAttention 3 (Blackwell/RTX 50xx)
    
    All non-SDPA backends use @torch._dynamo.disable wrapper (C++ extensions).
    """

    def __init__(self, attention_mode: str = 'sdpa', compute_dtype: torch.dtype = None):
        """
        Initialize with specified attention backend.
        
        Args:
            attention_mode: 'sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3'
            compute_dtype: Compute dtype for attention (set by pipeline, defaults to None for auto-detection)
        """
        super().__init__()
        self.attention_mode = attention_mode
        self.compute_dtype = compute_dtype

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        
        # Convert to pipeline compute_dtype if configured (handles FP8 → fp16/bf16)
        if self.compute_dtype is not None and q.dtype != self.compute_dtype:
            q = q.to(self.compute_dtype)
            k = k.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        
        if self.attention_mode == 'flash_attn_3':
            return call_flash_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'flash_attn_2':
            return call_flash_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_3':
            return call_sage_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_2':
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        else:
            # PyTorch SDPA
            return pytorch_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )