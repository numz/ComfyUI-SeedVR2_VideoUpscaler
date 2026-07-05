"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import triton
import triton.language as tl
from torch import Tensor


def precision_metric(quant_o, fa2_o, verbose=True, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}

def hyperparameter_check(hyper, H, device):
    if type(hyper) == float or type(hyper) == int:
        hyper = torch.full((H,), float(hyper), device=device)
    elif isinstance(hyper, Tensor):
        assert len(hyper.shape) <= 1, "Hyperparameter tensor must be 1D"
        if len(hyper.shape) == 0:
            hyper = torch.full((H,), hyper.item(), device=device)
        assert hyper.numel() == H, f"Hyperparameter tensor must have {H} elements, but has {hyper.numel()}"
        hyper = hyper.to(device)
    else:
        print(hyper)
        raise ValueError("Hyperparameter must be a float or a tensor")
    return hyper



@triton.jit
def triton_block_map_to_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, num_block_k):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    valid_block_num = 0

    map_ptr = map_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    lut_ptr = lut_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    valid_block_num_ptr = valid_block_num_ptr + b * H * Q + h * Q + q
    
    valid_block_num = 0
    prev_block = 0

    for i in range(num_block_k):
        cur_block = tl.load(map_ptr + i)
        if cur_block:
            tl.store(lut_ptr + valid_block_num, i - prev_block)
            valid_block_num += 1
            prev_block = i

    tl.store(valid_block_num_ptr, valid_block_num)

def block_map_lut_triton(block_map):
    assert block_map.dim() == 4
    assert block_map.is_contiguous()

    B, H, Q, K = block_map.shape
    lut = torch.zeros((B, H, Q, K), dtype=torch.int32, device=block_map.device)
    valid_block_num = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)

    grid = (B, H, Q)
    triton_block_map_to_lut_kernel[grid](block_map, lut, valid_block_num, K)

    return lut, valid_block_num

@triton.jit
def qk_quantize(
    # Pointers
    x_ptr,
    xm_ptr,
    x_quant_ptr,
    scale_ptr,
    # Constexpr dimensions
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    fuse_mean: tl.constexpr
):
    """
    Triton kernel to perform per-block quantization of a tensor X to int8.
    It loads a block of X, optionally subtracts a mean vector, then calculates
    a scaling factor for the block and quantizes the data to int8.

    Grid: (B, H, NB)
        B: Batch size
        H: Number of heads
        NB: Number of blocks in the N dimension (N // BS)
    """
    # 1. Get program IDs to identify the current block
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    # 2. Calculate pointers for the input block X
    block_offset = b * H * N * D + h * N * D + nb * BS * D
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    
    # Create a mask to handle the last block if N is not a multiple of BS
    xmask = (nb * BS + tl.arange(0, BS)[:, None]) < N
    
    # Load the input block
    x = tl.load(x_ptrs, mask=xmask, other=0.0)

    # 3. (Optional) Subtract the mean if fuse_mean is enabled
    if fuse_mean:
        xm_ptrs = xm_ptr + b * H * D + h * D + tl.arange(0, D)
        x_mean = tl.load(xm_ptrs)
        x -= x_mean
        # Re-apply mask to zero out padded values after subtraction
        x = tl.where(xmask, x, 0.0)

    # 4. Perform quantization
    # Convert to float32 for stable calculations
    x_fp32 = x.to(tl.float32)

    # Calculate the scale: max(abs(x)) / 127.0
    # The scale is per-block
    scale = tl.max(tl.abs(x_fp32)) / 127.0
    # Add a small epsilon to avoid division by zero
    scale += 1e-7

    # Quantize to int8: (x / scale) and round to nearest integer
    x_int8 = x_fp32 / scale
    # Round to nearest: add 0.5 for positive, -0.5 for negative
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)

    # 5. Calculate output pointers and store the results
    # Pointers for the quantized output tensor
    x_quant_ptrs = x_quant_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    # Pointer for the scale value of this block
    scale_ptrs = scale_ptr + b * H * NB + h * NB + nb

    # Store the quantized int8 values
    tl.store(x_quant_ptrs, x_int8, mask=xmask)
    # Store the scale value
    tl.store(scale_ptrs, scale)

@triton.jit
def triton_bmm_pool_sim_simmean_fuse_quant(
    x_ptr,
    xm_ptr,
    pool_ptr,
    sim_ptr,
    x_quant_ptr,
    scale_ptr,
    simthreshd1,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    fuse_mean: tl.constexpr
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    if fuse_mean:
        xm_ptrs = xm_ptr + b * H * D + h * D + tl.arange(0, D)
        x_mean = tl.load(xm_ptrs)
        x -= x_mean
        x = tl.where(xmask, x, 0)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)

    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)

    scale = tl.max(tl.abs(x_fp32)) / 127.
    scale += 0.0000001
    x_int8 = x_fp32 / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    x_quant_ptrs = x_quant_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    scale_ptrs = scale_ptr + b * H * NB + h * NB + nb
    tl.store(x_quant_ptrs, x_int8, mask = xmask)
    tl.store(scale_ptrs, scale)

@triton.jit
def triton_bmm_pool_sim_simmean(x_ptr, pool_ptr, sim_ptr, simthreshd1, N: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)
    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)
    
    
def get_pool_sim_triton_simmean(x, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    # Launch kernel
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks
 
#todo(xingyang): wrapper for tensor quantization
def get_quant(x, x_mean, block_size):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size
    x_quant = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    x_scale = torch.empty((B, H, nblock), device=x.device, dtype=torch.float32)
    grid = (B, H, nblock)
    qk_quantize[grid](x, x_mean, x_quant, x_scale, N=N, D=D, BS=block_size, fuse_mean=(True if x_mean is not None else False))
    return x_quant, x_scale

def get_vanilla_qk_quant(q, k, km=None, BLKQ=128, BLKK=64):
    q_int8, q_scale = get_quant(q, None, BLKQ)
    k_int8, k_scale = get_quant(k, km, BLKK)
    return q_int8, q_scale, k_int8, k_scale

def get_pool_sim_triton_simmean_fuse_quant(x, x_mean, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    x_quant = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    x_scale = torch.empty((B, H, nblock), device=x.device, dtype=torch.float32)
    grid = (B, H, nblock)
    triton_bmm_pool_sim_simmean_fuse_quant[grid](x, x_mean, pool, sim_blocks, x_quant, x_scale, simthreshd1, N=N, D=D, BS=block_size, fuse_mean=(True if x_mean is not None else False))
    return pool, sim_blocks, x_quant, x_scale

@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)
    

def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map

@triton.jit
def triton_fill_causal_mask(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)

def fill_causal_mask_triton(mask, BqdivBk:float):
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


def get_block_map_meansim(q, k, is_causal=False, BLKQ=128, BLKK=64, simthreshd1=0.1, cdfthreshd=0.9, topk=None, is_sparse=True, return_lut=False, attention_sink=False):
    assert (cdfthreshd is None and topk is not None) \
        or (cdfthreshd is not None and topk is None), "Only one of cdfthreshd and topk can be set."

    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    if cdfthreshd is not None:
        cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    if topk is not None:
        topk = hyperparameter_check(topk, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    if cdfthreshd is not None:
        cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
        cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    else:
        num_to_select = (topk * K).to(torch.int64).view(1, H, 1).expand(B, -1, Q).contiguous()

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    if not return_lut:
        return final_map
    else:
        lut, valid_block_num = block_map_lut_triton(final_map)
        return lut, valid_block_num

def get_block_map_meansim_fuse_quant(q, k, km=None, is_causal=False, BLKQ=128, BLKK=64, simthreshd1=0.1, cdfthreshd=0.9, topk=None, is_sparse=True, return_lut=False, attention_sink=False):
    assert (cdfthreshd is None and topk is not None) \
        or (cdfthreshd is not None and topk is None), "Only one of cdfthreshd and topk can be set."
    
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    if cdfthreshd is not None:
        cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    if topk is not None:
        topk = hyperparameter_check(topk, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks, q_int8, q_scale = get_pool_sim_triton_simmean_fuse_quant(q, None, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks, k_int8, k_scale = get_pool_sim_triton_simmean_fuse_quant(k, km, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    if cdfthreshd is not None:
        cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
        cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    else:
        num_to_select = (topk * K).to(torch.int64).view(1, H, 1).expand(B, -1, Q).contiguous()
    
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    if not return_lut:
        return final_map, q_int8, q_scale, k_int8, k_scale
    else:
        lut, valid_block_num = block_map_lut_triton(final_map)
        return lut, valid_block_num, q_int8, q_scale, k_int8, k_scale


def block_map_to_mask(block_map, BLKQ=128, BLKK=64):
    B, H, x, y = block_map.shape

    expanded_mask = torch.zeros((B, H, x * BLKQ, y * BLKK), dtype=torch.bool, device=block_map.device)
    for i in range(x):
        for j in range(y):
            expanded_mask[..., i * BLKQ: (i + 1) * BLKQ, j * BLKK: (j + 1) * BLKK] = block_map[..., i:i+1, j:j+1]

    return expanded_mask

def block_map_lut(block_map):
    valid_entry_num = block_map.to(torch.int32).sum(dim=-1)

    B, H, x, y = block_map.shape

    one_matrix = torch.ones((B, H, x, y), dtype=torch.int32, device=block_map.device)
    cum_matrix = torch.cumsum(one_matrix, dim=-1)
    masked_cum_matrix = cum_matrix * block_map.to(torch.int32)
    filled_matrix = masked_cum_matrix.clone()
    filled_matrix[~block_map] = 10000000
    lut = torch.sort(filled_matrix, dim=-1)[0] - 1 # make index start from 0
    lut[:, :, :, 1:] = lut[:, :, :, 1:] - lut[:, :, :, :-1]

    return lut.to(torch.int32), valid_entry_num.to(torch.int32)