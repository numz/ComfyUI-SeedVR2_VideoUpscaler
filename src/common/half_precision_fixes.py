"""
Half Precision Compatibility Fixes for SeedVR2

Provides safe wrappers around PyTorch operations that may fail with reduced
precision dtypes (float16, bfloat16, float8). These utilities automatically
detect precision issues and apply temporary conversions when needed, then
restore original dtypes.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def safe_pad_operation(
    x: torch.Tensor,
    padding: Union[Tuple[int, ...], int],
    mode: str = 'constant',
    value: float = 0.0
) -> torch.Tensor:
    """
    Safe padding operation with automatic float16 compatibility handling.
    
    Certain padding modes ('replicate', 'reflect', 'circular') are not implemented
    for float16 on some backends. This function automatically detects failures and
    applies a temporary float32 conversion, then restores the original dtype.
    
    Args:
        x: Input tensor of any dtype
        padding: Padding specification (left, right, top, bottom, front, back)
        mode: Padding mode - 'constant', 'replicate', 'reflect', or 'circular'
        value: Fill value for 'constant' mode (default: 0.0)
    
    Returns:
        Padded tensor in original dtype
    """
    # Modes that may require float16 compatibility fixes
    problematic_modes = ['replicate', 'reflect', 'circular']
    
    if mode in problematic_modes:
        try:
            return F.pad(x, padding, mode=mode, value=value)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                original_dtype = x.dtype
                result = F.pad(x.float(), padding, mode=mode, value=value)
                return result.to(original_dtype)
            else:
                raise
    else:
        # 'constant' and other compatible modes work natively
        return F.pad(x, padding, mode=mode, value=value)


def safe_interpolate_operation(
    x: torch.Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None
) -> torch.Tensor:
    """
    Safe interpolation operation with automatic float16 compatibility handling.
    
    Interpolation modes like 'bilinear', 'bicubic', and 'trilinear' may not be
    implemented for float16 on some backends. This function automatically detects
    failures and applies a temporary float32 conversion, then restores the original dtype.
    
    Args:
        x: Input tensor of any dtype
        size: Target output size (height, width) or (depth, height, width)
        scale_factor: Multiplier for spatial size (alternative to size)
        mode: Interpolation mode - 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        align_corners: If True, align corner pixels of input and output
        recompute_scale_factor: Recompute scale_factor for use in interpolation
    
    Returns:
        Interpolated tensor in original dtype
    """
    # Modes that may require float16 compatibility fixes
    problematic_modes = ['bilinear', 'bicubic', 'trilinear']
    
    if mode in problematic_modes:
        try:
            return F.interpolate(
                x,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor
            )
        except RuntimeError as e:
            # Check for float16 incompatibility errors
            if ("not implemented for 'Half'" in str(e) or
                "compute_indices_weights" in str(e)):
                original_dtype = x.dtype
                result = F.interpolate(
                    x.float(),
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor
                )
                return result.to(original_dtype)
            else:
                raise
    else:
        # 'nearest' and other compatible modes work natively
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )


def ensure_float32_precision(
    tensor: torch.Tensor,
    force_float32: bool = True
) -> Tuple[torch.Tensor, torch.dtype]:
    """
    Ensure tensor is in float32 for precision-sensitive operations.
    
    Many numerical operations require full precision to avoid accumulated errors:
    - Color space conversions (RGB↔LAB, RGB↔HSV) with matrix multiplications
    - Statistical operations (mean, variance, standard deviation)
    - Edge detection using derivative filters (Sobel, Canny)
    - Histogram matching and CDF computations
    - Guided filtering with covariance calculations
    
    This function upgrades reduced precision dtypes (float16, bfloat16, float8)
    to float32, while preserving the original dtype for restoration after computation.
    
    Args:
        tensor: Input tensor of any dtype
        force_float32: If True, convert reduced precision dtypes to float32.
                      If False, return tensor unchanged (useful for disabling conversion)
    
    Returns:
        Tuple of (converted_tensor, original_dtype) for easy restoration:
        - converted_tensor: Tensor in float32 (or original dtype if already full precision)
        - original_dtype: Original tensor dtype for restoration via .to(original_dtype)
    """
    original_dtype = tensor.dtype
    
    # Skip conversion if disabled
    if not force_float32:
        return tensor, original_dtype
    
    # Convert reduced precision dtypes to float32
    if original_dtype not in (torch.float32, torch.float64):
        return tensor.float(), original_dtype
    
    # Already full precision - return as-is (no copy)
    return tensor, original_dtype