"""
Optimization utilities for SeedVR2 Video Upscaler.

Exports:
    IS_ROCM: Boolean indicating if running on AMD ROCm/HIP backend
    portable_repeat_interleave: Cross-platform repeat_interleave that works on ROCm
"""

from .compatibility import IS_ROCM, portable_repeat_interleave

__all__ = [
    "IS_ROCM",
    "portable_repeat_interleave",
]
