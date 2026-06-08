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
from collections import defaultdict

# Import flash/sage attn with automatic fallback from compatibility layer
from ...optimization.compatibility import (
    call_flash_attn_2_varlen, call_flash_attn_3_varlen,
    call_sage_attn_2_varlen, call_sage_attn_3_varlen,
    call_omni_xpu_sdp, OMNI_XPU_AVAILABLE
)

from torch import nn


_OMNI_XPU_LOG_COUNTS = defaultdict(int)


def _log_omni_xpu(debug, message: str, level: str = "INFO", category: str = "setup", limit: int = 1) -> None:
    key = (level, message)
    _OMNI_XPU_LOG_COUNTS[key] += 1
    if _OMNI_XPU_LOG_COUNTS[key] > limit:
        return
    if debug is not None:
        debug.log(message, level=level, category=category, force=True)
    else:
        print(f"[SeedVR2] {message}")


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


def omni_xpu_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False, debug=None):
    """
    Intel Omni XPU SDP adapter for SeedVR2 varlen attention.

    omni_xpu_kernel.sdp expects (batch, seq, heads, head_dim), while SeedVR2
    varlen attention uses (total_seq, heads, head_dim). Process one packed
    sequence at a time and fall back to PyTorch SDPA for unsupported cases.
    """
    fallback = lambda: pytorch_varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    fallback_reason = None
    if not OMNI_XPU_AVAILABLE:
        fallback_reason = "omni_xpu_kernel SDP is not available"
    elif q.device.type != "xpu" or k.device.type != "xpu" or v.device.type != "xpu":
        fallback_reason = f"device q/k/v={q.device.type}/{k.device.type}/{v.device.type}"
    elif q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        fallback_reason = f"dtype q/k/v={q.dtype}/{k.dtype}/{v.dtype}"
    elif q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        fallback_reason = f"rank q/k/v={q.ndim}/{k.ndim}/{v.ndim}"
    elif q.shape[1:] != k.shape[1:] or k.shape[1:] != v.shape[1:]:
        fallback_reason = f"shape q/k/v={tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}"
    elif q.shape[-1] not in (64, 128):
        fallback_reason = f"head_dim={q.shape[-1]}"
    elif dropout_p not in (None, 0, 0.0):
        fallback_reason = f"dropout_p={dropout_p}"
    elif softmax_scale is not None:
        fallback_reason = f"softmax_scale={softmax_scale}"
    elif causal:
        fallback_reason = "causal=True"

    if fallback_reason is not None:
        _log_omni_xpu(
            debug,
            f"OmniXPU attention fallback to SDPA: {fallback_reason}",
            level="WARNING",
            category="setup",
            limit=3,
        )
        return fallback()

    q_splits = list(torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0))
    k_splits = list(torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0))
    v_splits = list(torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0))

    output_splits = []
    try:
        for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
            if q_i.shape[0] != k_i.shape[0] or k_i.shape[0] != v_i.shape[0]:
                _log_omni_xpu(
                    debug,
                    f"OmniXPU attention fallback to SDPA: non-self attention lengths q/k/v={q_i.shape[0]}/{k_i.shape[0]}/{v_i.shape[0]}",
                    level="WARNING",
                    category="setup",
                    limit=3,
                )
                return fallback()

            _log_omni_xpu(
                debug,
                f"Using OmniXPU attention: seq={q_i.shape[0]}, heads={q_i.shape[1]}, head_dim={q_i.shape[2]}, dtype={q_i.dtype}",
                category="success",
                limit=1,
            )
            output_i = call_omni_xpu_sdp(
                q_i.unsqueeze(0).contiguous(),
                k_i.unsqueeze(0).contiguous(),
                v_i.unsqueeze(0).contiguous(),
            )
            if isinstance(output_i, tuple):
                output_i = output_i[0]
            output_splits.append(output_i.squeeze(0))
    except Exception as e:
        _log_omni_xpu(
            debug,
            f"OmniXPU attention fallback to SDPA after runtime error: {str(e).splitlines()[0][:160]}",
            level="WARNING",
            category="setup",
            limit=3,
        )
        return fallback()

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
    - omni_xpu: Intel Omni XPU SDP (Intel XPU)
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
            attention_mode: 'sdpa', 'omni_xpu', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3'
            compute_dtype: Compute dtype for attention (set by pipeline, defaults to None for auto-detection)
        """
        super().__init__()
        self.attention_mode = attention_mode
        self.compute_dtype = compute_dtype
        self.debug = None

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
        
        if self.attention_mode == 'omni_xpu':
            return omni_xpu_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, debug=getattr(self, "debug", None), **kwargs
            )
        elif self.attention_mode == 'flash_attn_3':
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
