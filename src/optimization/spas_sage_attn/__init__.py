"""
Local vendored SpargeAttn/Sage2 implementation for ComfyUI-SeedVR2.5

This is a local copy of the SpargeAttn library (https://github.com/thu-ml/SpargeAttn)
modified for local JIT compilation without requiring global installation.

The implementation uses Triton kernels that compile just-in-time (JIT) on first use,
specifically optimized for NVIDIA Blackwell (RTX 50xx) GPUs with CUDA 12.8+/13.x.

Original Copyright (c) 2025 by SpargeAttn team.
Licensed under the Apache License, Version 2.0
"""

from .core import (
    spas_sage2_attn_meansim_topk_cuda,
    block_sparse_sage2_attn_cuda,
    spas_sage_attn_meansim_topk_cuda,
    SPARGE_LOCAL_AVAILABLE,
    SPARGE_LOCAL_VERSION,
    TRITON_AVAILABLE,
    TRITON_IMPORT_ERROR,
    get_blackwell_config,
)

__all__ = [
    'spas_sage2_attn_meansim_topk_cuda',
    'block_sparse_sage2_attn_cuda', 
    'spas_sage_attn_meansim_topk_cuda',
    'SPARGE_LOCAL_AVAILABLE',
    'SPARGE_LOCAL_VERSION',
    'TRITON_AVAILABLE',
    'TRITON_IMPORT_ERROR',
    'get_blackwell_config',
]

__version__ = "0.1.0-local"
