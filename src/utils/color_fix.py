"""
Color Correction Module for SeedVR2

Provides perceptually-accurate color correction methods to match upscaled video
frames to their original color characteristics. All methods preserve spatial details
while transferring color distributions.

Available Methods:
- lab: Full perceptual color matching with optional detail preservation (recommended)
- wavelet: Frequency-based color transfer preserving high-frequency details
- wavelet_adaptive: Wavelet with targeted saturation correction
- hsv: Hue-conditional saturation histogram matching
- adain: Adaptive instance normalization style transfer
"""

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from typing import Optional
from torchvision.transforms import ToTensor, ToPILImage
from ..common.half_precision_fixes import safe_pad_operation, safe_interpolate_operation, ensure_float32_precision


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """
    Apply AdaIN color correction to PIL images.
    
    Args:
        target: PIL Image with desired details
        source: PIL Image with desired colors
        
    Returns:
        PIL Image with corrected colors
    """
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)
    
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    
    return result_image


def wavelet_color_fix(target: Image.Image, source: Image.Image, debug: Optional['Debug'] = None) -> Image.Image:
    """
    Apply wavelet-based color correction to PIL images.
    
    Args:
        target: PIL Image with desired details
        source: PIL Image with desired colors
        debug: Debug instance for logging
        
    Returns:
        PIL Image with corrected colors
    """
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor, debug)
    
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    
    return result_image


def calc_mean_std(feat: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """
    Calculate channel-wise mean and standard deviation.
    
    Args:
        feat: 4D tensor [B, C, H, W]
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of (mean, std) tensors [B, C, 1, 1]
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.
    
    Transfers the color distribution (mean and variance) from style to content
    while preserving content structure.
    
    Args:
        content_feat: Target tensor [B, C, H, W] with desired structure
        style_feat: Source tensor [B, C, H, W] with desired color statistics
        
    Returns:
        Normalized tensor with style statistics and content structure
    """
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    
    # Normalize content to zero mean, unit variance (avoid expand by using broadcasting)
    normalized_feat = (content_feat - content_mean) / content_std
    del content_mean, content_std
    
    # Apply style statistics (broadcasting avoids expand)
    result = normalized_feat * style_std + style_mean
    del normalized_feat, style_mean, style_std
    
    return result


def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    """
    Apply Gaussian-like blur using dilated convolution for wavelet decomposition.
    
    Automatically limits radius to prevent numerical instability at high resolutions.
    Supports arbitrary number of channels.
    
    Args:
        image: Input tensor [B, C, H, W]
        radius: Dilation radius for blur kernel
        
    Returns:
        Blurred tensor [B, C, H, W]
    """
    # Prevent excessive dilation that causes OOM/numerical issues
    # Conservative limit: 1/8 of smallest spatial dimension
    max_safe_radius = max(1, min(image.shape[-2:]) // 8)
    if radius > max_safe_radius:
        radius = max_safe_radius
    
    num_channels = image.shape[1]
    
    # 3x3 Gaussian-approximation kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(num_channels, 1, 1, 1)
    
    # Apply padding and grouped convolution
    image = safe_pad_operation(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=num_channels, dilation=radius)
    
    return output


def wavelet_decomposition(image: Tensor, levels: int = 5) -> tuple[Tensor, Tensor]:
    """
    Multi-scale wavelet decomposition to separate frequency components.
    
    Decomposes image into high-frequency (details/edges) and low-frequency
    (color/illumination) components using iterative Gaussian pyramid.
    
    Args:
        image: Input tensor [B, C, H, W]
        levels: Number of decomposition levels (default: 5)
        
    Returns:
        Tuple of (high_freq, low_freq) tensors
        - high_freq: Detail information [B, C, H, W]
        - low_freq: Color/illumination information [B, C, H, W]
    """
    high_freq = torch.zeros_like(image)
    
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq.add_(image).sub_(low_freq)
        image = low_freq
    
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor, debug: Optional['Debug'] = None) -> Tensor:
    """
    Apply wavelet-based color transfer from style to content.
    
    Preserves high-frequency details from content while adopting low-frequency 
    color information from style using multi-resolution wavelet decomposition.
    
    Algorithm:
    1. Decompose both images into high (detail) and low (color) frequency components
    2. Combine content's high frequencies with style's low frequencies
    3. Reconstruct the image preserving details with transferred colors
    
    Args:
        content_feat: Target tensor with desired details [B, C, H, W] in [-1,1]
        style_feat: Source tensor with desired colors [B, C, H, W] in [-1,1]
        debug: Debug instance for logging
        
    Returns:
        Tensor: Reconstructed tensor with content details and style colors in [-1,1]
    """
    # Handle dimension mismatch if needed
    if content_feat.shape != style_feat.shape:
        debug.log(f"Dimension mismatch: content {content_feat.shape} vs style {style_feat.shape}", 
                  level="WARNING", category="precision", force=True)
        
        # Resize style to match content spatial dimensions
        if len(content_feat.shape) >= 3:
            # safe_interpolate_operation handles FP16 conversion automatically
            style_feat = safe_interpolate_operation(
                style_feat, 
                size=content_feat.shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            debug.log(f"Style resized to: {style_feat.shape}", category="precision", force=True)
    
    # Decompose both features into frequency components
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq  # Free memory immediately
    
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)  
    del style_high_freq  # Free memory immediately
    
    # Safety check (should not happen after resize)
    if content_high_freq.shape != style_low_freq.shape:
        debug.log(f"Final dimension adjustment needed", level="WARNING", category="precision", force=True)
        style_low_freq = safe_interpolate_operation(
            style_low_freq,
            size=content_high_freq.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Reconstruct: content details + style color (in-place on content_high_freq)
    content_high_freq.add_(style_low_freq)
    
    # Safety clamp for normalized SDR range (in-place)
    # This prevents numerical errors from propagating
    # Note: For HDR support, this would need to be removed
    return content_high_freq.clamp_(-1.0, 1.0)


def lab_color_transfer(
    content_feat: Tensor,
    style_feat: Tensor,
    debug,
    luminance_weight: float = 0.8
) -> Tensor:
    """
    Perceptually-accurate color transfer using CIELAB color space.
    
    Combines wavelet reconstruction (for spatial continuity) with LAB histogram
    matching (for precise color matching). This eliminates tile artifacts while
    providing superior color accuracy.
    
    Algorithm:
    1. Apply wavelet reconstruction to get artifact-free base
    2. Convert both wavelet result and style to LAB color space
    3. Apply histogram matching to LAB channels
    4. Convert back to RGB
    
    Args:
        content_feat: Target tensor [B, C, H, W] in [-1, 1] with upscaled details
        style_feat: Source tensor [B, C, H, W] in [-1, 1] with original colors
        debug: Debug instance for logging
        luminance_weight: How much content luminance to preserve (0.0-1.0)
                         0.0 = full color match, 1.0 = preserve all detail
                         Default: 0.8 (slight color adjustment, strong detail preservation)
        
    Returns:
        Color-corrected tensor [B, C, H, W] in [-1, 1]
    """
    # Step 1: Apply wavelet to get artifact-free base with correct spatial structure
    content_feat = wavelet_reconstruction(content_feat, style_feat, debug=None)
    
    # Handle spatial dimension mismatch (should already match after wavelet)
    if content_feat.shape != style_feat.shape:
        debug.log(
            f"LAB: Resizing style {style_feat.shape} to match content {content_feat.shape}",
            level="WARNING", category="precision", force=True
        )
        style_feat = safe_interpolate_operation(
            style_feat,
            size=content_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Store device and convert to float32
    device = content_feat.device
    
    # Convert to float32 for accurate color space conversion
    content_feat, original_dtype = ensure_float32_precision(content_feat)
    style_feat, _ = ensure_float32_precision(style_feat)
    
    # Precompute color space conversion matrices
    rgb_to_xyz_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=torch.float32, device=device)
    
    xyz_to_rgb_matrix = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=torch.float32, device=device)
    
    # LAB conversion constants
    epsilon = 6.0 / 29.0
    kappa = (29.0 / 3.0) ** 3
    
    # Convert from [-1, 1] to [0, 1] range (in-place)
    content_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    style_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    
    # Convert to LAB color space
    content_lab = _rgb_to_lab_batch(content_feat, device, rgb_to_xyz_matrix, epsilon, kappa)
    del content_feat
    
    style_lab = _rgb_to_lab_batch(style_feat, device, rgb_to_xyz_matrix, epsilon, kappa)
    del style_feat, rgb_to_xyz_matrix
    
    # Match chrominance channels (a*, b*) for accurate color transfer
    matched_a = _histogram_matching_channel(content_lab[:, 1], style_lab[:, 1], device)
    matched_b = _histogram_matching_channel(content_lab[:, 2], style_lab[:, 2], device)
    
    # Handle luminance with weighted blending
    if luminance_weight < 1.0:
        # Partially match luminance for better overall color accuracy
        matched_L = _histogram_matching_channel(content_lab[:, 0], style_lab[:, 0], device)
        # Blend: preserve some content L* for detail, adopt some style L* for color
        result_L = content_lab[:, 0].mul(luminance_weight).add_(matched_L.mul(1.0 - luminance_weight))
        del matched_L
    else:
        # Fully preserve content luminance
        result_L = content_lab[:, 0]
    
    del content_lab, style_lab
    
    # Reconstruct LAB with corrected channels
    result_lab = torch.stack([result_L, matched_a, matched_b], dim=1)
    del result_L, matched_a, matched_b
    
    # Convert back to RGB
    result_rgb = _lab_to_rgb_batch(result_lab, device, xyz_to_rgb_matrix, epsilon, kappa)
    del result_lab, xyz_to_rgb_matrix
    
    # Convert back to [-1, 1] range (in-place)
    result = result_rgb.mul_(2.0).sub_(1.0)
    del result_rgb
    
    # Restore original dtype
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    
    debug.log(f"LAB color transfer completed (luminance_weight={luminance_weight})", category="video", indent_level=1)
    
    return result


def _rgb_to_lab_batch(rgb: Tensor, device: torch.device, matrix: Tensor, epsilon: float, kappa: float) -> Tensor:
    """Convert batch of RGB images to CIELAB color space using D65 illuminant."""
    # Apply sRGB gamma correction (linearize)
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92
    )
    del mask
    
    # Matrix multiplication: RGB -> XYZ
    B, C, H, W = rgb_linear.shape
    rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
    del rgb_linear
    
    # Ensure dtype consistency for matrix multiplication
    rgb_flat = rgb_flat.to(dtype=matrix.dtype)
    xyz_flat = torch.matmul(rgb_flat, matrix.T)
    del rgb_flat
    
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    del xyz_flat
    
    # Normalize by D65 white point (in-place)
    xyz[:, 0].div_(0.95047)  # X
    # xyz[:, 1] /= 1.00000  # Y (no-op, skip)
    xyz[:, 2].div_(1.08883)  # Z
    
    # XYZ to LAB transformation
    epsilon_cubed = epsilon ** 3
    mask = xyz > epsilon_cubed
    f_xyz = torch.where(
        mask,
        torch.pow(xyz, 1.0 / 3.0),
        xyz.mul(kappa).add_(16.0).div_(116.0)
    )
    del xyz, mask
    
    # Extract channels and compute LAB
    L = f_xyz[:, 1].mul(116.0).sub_(16.0)  # Lightness [0, 100]
    a = (f_xyz[:, 0] - f_xyz[:, 1]).mul_(500.0)  # Green-Red [-128, 127]
    b = (f_xyz[:, 1] - f_xyz[:, 2]).mul_(200.0)  # Blue-Yellow [-128, 127]
    del f_xyz
    
    return torch.stack([L, a, b], dim=1)


def _lab_to_rgb_batch(lab: Tensor, device: torch.device, matrix_inv: Tensor, epsilon: float, kappa: float) -> Tensor:
    """Convert batch of CIELAB images to RGB color space."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    
    # LAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = a.div(500.0).add_(fy)
    fz = fy - b / 200.0
    del L, a, b
    
    # XYZ transformation
    x = torch.where(
        fx > epsilon,
        torch.pow(fx, 3.0),
        fx.mul(116.0).sub_(16.0).div_(kappa)
    )
    y = torch.where(
        fy > epsilon,
        torch.pow(fy, 3.0),
        fy.mul(116.0).sub_(16.0).div_(kappa)
    )
    z = torch.where(
        fz > epsilon,
        torch.pow(fz, 3.0),
        fz.mul(116.0).sub_(16.0).div_(kappa)
    )
    del fx, fy, fz
    
    # Apply D65 white point (in-place)
    x.mul_(0.95047)
    # y *= 1.00000  # (no-op, skip)
    z.mul_(1.08883)
    
    xyz = torch.stack([x, y, z], dim=1)
    del x, y, z
    
    # Matrix multiplication: XYZ -> RGB
    B, C, H, W = xyz.shape
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    del xyz
    
    # Ensure dtype consistency for matrix multiplication
    xyz_flat = xyz_flat.to(dtype=matrix_inv.dtype)
    rgb_linear_flat = torch.matmul(xyz_flat, matrix_inv.T)
    del xyz_flat
    
    rgb_linear = rgb_linear_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    del rgb_linear_flat
    
    # Apply inverse gamma correction (delinearize)
    mask = rgb_linear > 0.0031308
    rgb = torch.where(
        mask,
        torch.pow(torch.clamp(rgb_linear, min=0.0), 1.0 / 2.4).mul_(1.055).sub_(0.055),
        rgb_linear * 12.92
    )
    del mask, rgb_linear
    
    return torch.clamp(rgb, 0.0, 1.0)


def _histogram_matching_channel(source: Tensor, reference: Tensor, device: torch.device) -> Tensor:
    """
    Match histogram of source channel to reference using CDF mapping.
    
    Args:
        source: Source channel tensor [B, H, W]
        reference: Reference channel tensor [B, H, W]
        
    Returns:
        Matched channel tensor [B, H, W]
    """
    original_shape = source.shape
    
    # Flatten
    source_flat = source.flatten()
    reference_flat = reference.flatten()
    
    # Sort both arrays
    source_sorted, source_indices = torch.sort(source_flat)
    reference_sorted, _ = torch.sort(reference_flat)
    del reference_flat
    
    # Quantile mapping
    n_source = len(source_sorted)
    n_reference = len(reference_sorted)
    
    if n_source == n_reference:
        matched_sorted = reference_sorted
    else:
        # Interpolate reference to match source quantiles
        source_quantiles = torch.linspace(0, 1, n_source, device=device)
        ref_indices = (source_quantiles * (n_reference - 1)).long()
        ref_indices.clamp_(0, n_reference - 1)
        matched_sorted = reference_sorted[ref_indices]
        del source_quantiles, ref_indices, reference_sorted
    
    del source_sorted, source_flat
    
    # Reconstruct using argsort (portable across CUDA/ROCm/MPS)
    inverse_indices = torch.argsort(source_indices)
    del source_indices
    matched_flat = matched_sorted[inverse_indices]
    del matched_sorted, inverse_indices
    
    return matched_flat.reshape(original_shape)


def hsv_saturation_histogram_match(content_feat: Tensor, style_feat: Tensor, debug: Optional['Debug'] = None) -> Tensor:
    """
    Hue-conditional saturation histogram matching in HSV color space.
    
    Matches saturation distribution from style to content separately for each hue bin
    to handle color-specific oversaturation (e.g., overly saturated reds).
    
    Based on: Neumann et al. (2005) "Color Style Transfer using Hue, Lightness 
    and Saturation Histogram Matching"
    
    Algorithm:
    1. Convert both images to HSV color space
    2. Divide hue circle into 12 bins (30° each)
    3. For each hue bin, match saturation histograms independently
    4. Reconstruct HSV with matched saturation
    5. Convert back to RGB
    
    Args:
        content_feat: Target tensor [B, C, H, W] in [-1, 1] with upscaled details
        style_feat: Source tensor [B, C, H, W] in [-1, 1] with original saturation
        debug: Debug instance for logging
        
    Returns:
        Saturation-corrected tensor [B, C, H, W] in [-1, 1]
    """
    # Handle spatial dimension mismatch
    if content_feat.shape != style_feat.shape:
        debug.log(
            f"HSV: Resizing style {style_feat.shape} to match content {content_feat.shape}",
            level="WARNING", category="precision", force=True
        )
        style_feat = safe_interpolate_operation(
            style_feat,
            size=content_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Convert to float32 for processing
    content_feat, original_dtype = ensure_float32_precision(content_feat)
    style_feat, _ = ensure_float32_precision(style_feat)
    
    # Convert from [-1, 1] to [0, 1] range (in-place)
    content_rgb = content_feat.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
    style_rgb = style_feat.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
    del content_feat, style_feat
    
    # Convert to HSV
    content_hsv = _rgb_to_hsv_batch(content_rgb)
    del content_rgb
    
    style_hsv = _rgb_to_hsv_batch(style_rgb)
    del style_rgb
    
    # Extract channels
    content_h = content_hsv[:, 0]
    content_s = content_hsv[:, 1]
    content_v = content_hsv[:, 2]
    del content_hsv
    
    style_h = style_hsv[:, 0]
    style_s = style_hsv[:, 1]
    del style_hsv
    
    # Match saturation per hue bin
    matched_s = _hue_conditional_saturation_match(content_h, content_s, style_h, style_s)
    del style_h, style_s
    
    # Reconstruct HSV: preserve H and V from content, use matched S
    result_hsv = torch.stack([content_h, matched_s, content_v], dim=1)
    del content_h, matched_s, content_v
    
    # Convert back to RGB
    result_rgb = _hsv_to_rgb_batch(result_hsv)
    del result_hsv
    result_rgb.clamp_(0.0, 1.0)
    
    # Convert back to [-1, 1] range (in-place)
    result = result_rgb.mul_(2.0).sub_(1.0)
    del result_rgb
    
    # Restore original dtype
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    
    debug.log("HSV hue-conditional saturation matching completed", category="video", indent_level=1)
    
    return result


def _rgb_to_hsv_batch(rgb: Tensor) -> Tensor:
    """Convert batch of RGB images to HSV color space. All channels in [0, 1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    maxc = torch.max(rgb, dim=1)[0]
    minc = torch.min(rgb, dim=1)[0]
    rangec = maxc - minc
    
    # Avoid division by zero
    rangec_nz = torch.where(rangec > 1e-10, rangec, torch.ones_like(rangec))
    
    # Hue calculation (in [0, 1])
    h = torch.zeros_like(maxc)
    
    mask_r = (maxc == r) & (rangec > 1e-10)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / rangec_nz[mask_r]) % 6.0
    del mask_r
    
    mask_g = (maxc == g) & (rangec > 1e-10)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / rangec_nz[mask_g]) + 2.0
    del mask_g
    
    mask_b = (maxc == b) & (rangec > 1e-10)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / rangec_nz[mask_b]) + 4.0
    del mask_b
    
    h.div_(6.0)  # Normalize to [0, 1] (in-place)
    
    # Saturation calculation
    s = torch.where(maxc > 1e-10, rangec / torch.clamp(maxc, min=1e-10), torch.zeros_like(maxc))
    del rangec, rangec_nz
    
    # Value calculation
    v = maxc
    
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb_batch(hsv: Tensor) -> Tensor:
    """Convert batch of HSV images to RGB color space."""
    h = hsv[:, 0].mul(6.0)  # Convert to [0, 6]
    s = hsv[:, 1]
    v = hsv[:, 2]
    
    i = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    # Initialize RGB channels
    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)
    
    # Apply hue sector transformations
    mask0 = (i == 0)
    r[mask0], g[mask0], b[mask0] = v[mask0], t[mask0], p[mask0]
    del mask0
    
    mask1 = (i == 1)
    r[mask1], g[mask1], b[mask1] = q[mask1], v[mask1], p[mask1]
    del mask1
    
    mask2 = (i == 2)
    r[mask2], g[mask2], b[mask2] = p[mask2], v[mask2], t[mask2]
    del mask2
    
    mask3 = (i == 3)
    r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v[mask3]
    del mask3
    
    mask4 = (i == 4)
    r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], v[mask4]
    del mask4
    
    mask5 = (i == 5)
    r[mask5], g[mask5], b[mask5] = v[mask5], p[mask5], q[mask5]
    del mask5, i, f, p, q, t, h, s, v
    
    return torch.stack([r, g, b], dim=1)


def _hue_conditional_saturation_match(
    content_h: Tensor,
    content_s: Tensor,
    style_h: Tensor,
    style_s: Tensor
) -> Tensor:
    """
    Match saturation histogram conditionally per hue bin.
    
    Divides hue circle into 12 bins (30° each) and matches saturation
    separately for each bin to handle color-specific oversaturation.
    """
    num_bins = 12
    bin_width = 1.0 / num_bins
    min_pixels = 100  # Minimum pixels for reliable histogram matching
    
    matched_s = content_s.clone()
    device = content_s.device
    
    for bin_idx in range(num_bins):
        bin_start = bin_idx * bin_width
        bin_end = (bin_idx + 1) * bin_width
        
        # Handle hue wrap-around for red (0°/360°)
        if bin_idx == 0:
            content_mask = ((content_h >= 0) & (content_h < bin_end)) | (content_h >= (1.0 - bin_width))
            style_mask = ((style_h >= 0) & (style_h < bin_end)) | (style_h >= (1.0 - bin_width))
        else:
            content_mask = (content_h >= bin_start) & (content_h < bin_end)
            style_mask = (style_h >= bin_start) & (style_h < bin_end)
        
        # Extract saturation values for this hue bin
        content_s_bin = content_s[content_mask]
        style_s_bin = style_s[style_mask]
        
        # Only match if both bins have sufficient pixels
        if len(content_s_bin) > min_pixels and len(style_s_bin) > min_pixels:
            matched_s_bin = _histogram_match_1d(content_s_bin, style_s_bin, device)
            matched_s[content_mask] = matched_s_bin
            del matched_s_bin
        
        del content_mask, style_mask, content_s_bin, style_s_bin
    
    return matched_s


def _histogram_match_1d(source: Tensor, reference: Tensor, device: torch.device) -> Tensor:
    """Match 1D histogram using CDF mapping."""
    source_sorted, source_indices = torch.sort(source)
    reference_sorted, _ = torch.sort(reference)
    
    n_source = len(source_sorted)
    n_reference = len(reference_sorted)
    
    if n_source == n_reference:
        matched_sorted = reference_sorted
    else:
        source_quantiles = torch.linspace(0, 1, n_source, device=device)
        ref_indices = (source_quantiles * (n_reference - 1)).long()
        ref_indices.clamp_(0, n_reference - 1)
        matched_sorted = reference_sorted[ref_indices]
        del source_quantiles, ref_indices, reference_sorted
    
    del source_sorted
    
    # Reconstruct using argsort (portable across CUDA/ROCm/MPS)
    inverse_indices = torch.argsort(source_indices)
    del source_indices
    matched = matched_sorted[inverse_indices]
    del matched_sorted, inverse_indices
    
    return matched


def wavelet_adaptive_color_correction(content_feat: Tensor, style_feat: Tensor, debug: Optional['Debug'] = None) -> Tensor:
    """
    Adaptive hybrid color correction combining wavelet and HSV methods.
    
    Uses wavelet as the base correction for natural colors, then selectively
    applies HSV saturation correction only to oversaturated regions.
    
    Algorithm:
    1. Apply wavelet reconstruction (natural color base)
    2. Apply HSV saturation matching (targeted correction)
    3. Detect oversaturated pixels by comparing saturation levels
    4. Blend HSV correction only into oversaturated areas via sigmoid
    5. Keep wavelet colors everywhere else
    
    Args:
        content_feat: Target tensor [B, C, H, W] in [-1, 1] with upscaled details
        style_feat: Source tensor [B, C, H, W] in [-1, 1] with original colors
        debug: Debug instance for logging
        
    Returns:
        Adaptively corrected tensor [B, C, H, W] in [-1, 1]
    """
    # Handle spatial dimension mismatch
    if content_feat.shape != style_feat.shape:
        debug.log(
            f"Wavelet Adaptive: Resizing style {style_feat.shape} to match content {content_feat.shape}",
            level="WARNING", category="precision", force=True
        )
        style_feat = safe_interpolate_operation(
            style_feat,
            size=content_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Convert to float32 for processing
    content_feat, original_dtype = ensure_float32_precision(content_feat)
    style_feat, _ = ensure_float32_precision(style_feat)
    
    # Step 1: Apply wavelet (base correction)
    wavelet_result = wavelet_reconstruction(content_feat, style_feat, debug)
    
    # Step 2: Apply HSV saturation matching (targeted correction)
    hsv_result = hsv_saturation_histogram_match(content_feat, style_feat, debug)
    
    # Step 3: Compute saturation maps to detect oversaturation
    content_sat = _get_saturation_map(content_feat)
    style_sat = _get_saturation_map(style_feat)
    wavelet_sat = _get_saturation_map(wavelet_result)
    
    # Step 4: Create adaptive blend mask based on saturation difference
    sat_difference = content_sat - style_sat
    del content_sat
    
    # Parameters for blending
    oversaturation_threshold = 0.15  # Saturation difference threshold
    blend_sharpness = 5.0  # Sigmoid sharpness for smooth transitions
    
    # Sigmoid blend: 0 = use wavelet, 1 = use HSV correction
    blend_weight = torch.sigmoid(blend_sharpness * (sat_difference - oversaturation_threshold))
    del sat_difference
    
    # Only correct if wavelet itself is still oversaturated
    wavelet_oversaturated = (wavelet_sat - style_sat) > (oversaturation_threshold * 0.5)
    del wavelet_sat, style_sat
    
    blend_weight.mul_(wavelet_oversaturated.float())
    del wavelet_oversaturated
    blend_weight.clamp_(0.0, 1.0)
    
    # Step 5: Adaptive blending (in-place where possible)
    result = wavelet_result.mul(1.0 - blend_weight).add_(hsv_result.mul(blend_weight))
    
    # Log statistics
    correction_pct = (blend_weight > 0.01).float().mean().item() * 100
    del blend_weight, wavelet_result, hsv_result
    
    # Restore original dtype
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    
    debug.log(f"Wavelet Adaptive: {correction_pct:.1f}% pixels use HSV correction", category="video", indent_level=1)
    
    return result


def _get_saturation_map(tensor: Tensor) -> Tensor:
    """Extract saturation channel from RGB tensor in [-1, 1]."""
    rgb = tensor.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
    
    maxc = torch.max(rgb, dim=1, keepdim=True)[0]
    minc = torch.min(rgb, dim=1, keepdim=True)[0]
    del rgb
    
    saturation = torch.where(
        maxc > 1e-10,
        (maxc - minc) / torch.clamp(maxc, min=1e-10),
        torch.zeros_like(maxc)
    )
    del maxc, minc
    
    return saturation