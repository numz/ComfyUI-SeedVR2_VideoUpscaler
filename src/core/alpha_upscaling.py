"""
Alpha Channel Upscaling Module

Provides edge-guided Alpha upscaling that matches RGB structural changes
without diffusion artifacts. Uses joint bilateral filtering and edge detection
to maintain alignment between upscaled RGB and Alpha channels.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Any, List
from ..common.half_precision_fixes import ensure_float32_precision
from ..optimization.memory_manager import manage_tensor


def process_alpha_for_batch(
    rgb_samples: List[torch.Tensor],
    alpha_original: torch.Tensor,
    rgb_original: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
    debug: Optional['Debug'] = None
) -> List[torch.Tensor]:
    """
    Process Alpha channel for an entire batch with temporal consistency.
    Called during postprocess phase when VRAM is available.
    
    Args:
        rgb_samples: List of decoded RGB samples from VAE
        alpha_original: Original Alpha channel (C, T, H, W)
        rgb_original: Original RGB for guidance (C, T, H, W)
        device: Target device for processing (typically CUDA)
        compute_dtype: Pipeline compute dtype (e.g., bfloat16) for final output.
        debug: Debug instance for logging
        
    Returns:
        List of RGBA samples with Alpha merged, in compute_dtype
    """
    # Move alpha and RGB guidance tensors to processing device (GPU)
    alpha_original = manage_tensor(
        tensor=alpha_original,
        target_device=device,
        tensor_name="alpha_original",
        debug=debug,
        reason="Alpha processing",
        indent_level=1
    )
    rgb_original = manage_tensor(
        tensor=rgb_original,
        target_device=device,
        tensor_name="rgb_original",
        debug=debug,
        reason="Alpha processing",
        indent_level=1
    )
    
    # Process each RGB sample individually to handle variable frame counts
    processed_samples = []
    
    for rgb_sample in rgb_samples:
        # Move RGB sample to processing device before use
        rgb_sample = manage_tensor(
            tensor=rgb_sample,
            target_device=device,
            tensor_name="rgb_sample",
            debug=debug,
            reason="Alpha processing",
            indent_level=1
        )
        # Handle dimension variations between single frames and video sequences
        if rgb_sample.ndim == 3:
            # Single frame: (C, H, W) -> need to add T dimension
            rgb_sample_4d = rgb_sample.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            alpha_4d = alpha_original.unsqueeze(1) if alpha_original.ndim == 3 else alpha_original
            rgb_guide_4d = rgb_original.unsqueeze(1) if rgb_original.ndim == 3 else rgb_original
            was_single_frame = True
        else:
            # Video: (T, C, H, W) -> already correct
            rgb_sample_4d = rgb_sample
            alpha_4d = alpha_original
            rgb_guide_4d = rgb_original
            was_single_frame = False
        
        # Apply edge-guided upscaling to match RGB resolution while preserving alpha structure
        alpha_upscaled = edge_guided_alpha_upscale(
            input_alpha=alpha_4d[:, :1, :, :] if alpha_4d.shape[1] > 1 else alpha_4d,  # Ensure (T, 1, H, W)
            input_rgb=rgb_guide_4d[:, :3, :, :],  # Ensure (T, 3, H, W)
            upscaled_rgb=rgb_sample_4d[:, :3, :, :],  # Ensure (T, 3, H, W)
            method='guided',
            debug=debug
        )
        
        # Convert Alpha from float32 to compute dtype
        if alpha_upscaled.dtype != compute_dtype:
            alpha_upscaled = manage_tensor(
                tensor=alpha_upscaled,
                target_device=alpha_upscaled.device,
                tensor_name="alpha_upscaled",
                dtype=compute_dtype,
                debug=debug,
                reason="dtype alignment for RGBA concatenation",
                indent_level=1
            )
        
        # Concatenate RGB and upscaled alpha to create RGBA output (T, 4, H, W)
        rgba_sample = torch.cat([rgb_sample_4d[:, :3, :, :], alpha_upscaled], dim=1)
        
        # Restore original format
        if was_single_frame:
            rgba_sample = rgba_sample.squeeze(0)  # (1, 4, H, W) -> (4, H, W)
        
        processed_samples.append(rgba_sample)
        
        # Release memory immediately after processing each sample
        del alpha_upscaled
    
    # Clean up
    del alpha_original, rgb_original
    
    return processed_samples


def detect_edges_batch(
    images: torch.Tensor,
    method: str = 'sobel',
    debug: Optional['Debug'] = None
    ) -> torch.Tensor:
    """
    Detect edges in a batch of images using Sobel or Canny.
    
    Args:
        images: Tensor of shape (T, C, H, W) in range [-1, 1] or [0, 1]
        method: 'sobel' or 'canny'
        debug: Optional debug instance for logging
        
    Returns:
        Edge map tensor of shape (T, 1, H, W) in range [0, 1]
    """
    # Convert to float32 for OpenCV processing (will be converted to numpy anyway)
    images, images_dtype = ensure_float32_precision(images, force_float32=True)
    
    images_np = images.cpu().numpy()
    T, C, H, W = images_np.shape
    
    # Denormalize if needed
    if images_np.min() < 0:
        images_np = (images_np + 1) / 2
    
    # Convert to 0-255 uint8
    images_np = (images_np * 255).clip(0, 255).astype(np.uint8)
    
    edges = []
    for t in range(T):
        frame = images_np[t].transpose(1, 2, 0)
        
        if C == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame[..., 0]
        
        if method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge = np.sqrt(sobelx**2 + sobely**2)
            edge = (edge / edge.max() * 255).astype(np.uint8)
        else:
            edge = cv2.Canny(gray, 50, 150)
        
        edges.append(edge)
    
    edges = np.stack(edges)
    edges = torch.from_numpy(edges).float() / 255.0
    edges = edges.unsqueeze(1)
    
    # Move back to images device after CPU numpy processing
    edges = manage_tensor(
        tensor=edges,
        target_device=images.device,
        tensor_name="edge_map",
        dtype=images_dtype,
        debug=debug,
        reason="edge detection result",
        indent_level=1
    )
    
    return edges


def guided_filter_pytorch(guide: torch.Tensor, src: torch.Tensor, 
                          radius: int = 8, eps: float = 0.01) -> torch.Tensor:
    """
    Apply guided filter for edge-preserving smoothing.
    
    Args:
        guide: Guidance image (T, C, H, W)
        src: Input to filter (T, 1, H, W)
        radius: Filter radius
        eps: Regularization parameter
        
    Returns:
        Filtered output (T, 1, H, W)
    """
    # Convert to float32 for numerical stability in statistical operations
    guide, guide_dtype = ensure_float32_precision(guide, force_float32=True)
    src, _ = ensure_float32_precision(src, force_float32=True)
    
    T, C, H, W = guide.shape
    
    # Convert to grayscale if RGB
    if C == 3:
        guide_gray = guide.mean(dim=1, keepdim=True)
    else:
        guide_gray = guide
    
    # Apply guided filter
    output = _apply_guided_filter(guide_gray, src, radius, eps)
    
    # Restore original dtype after float32 processing
    if output.dtype != guide_dtype:
        output = manage_tensor(
            tensor=output,
            target_device=output.device,
            tensor_name="filtered_output",
            dtype=guide_dtype,
            debug=debug,
            reason="dtype restoration"
        )
    
    return output


def _apply_guided_filter(guide_gray: torch.Tensor, src: torch.Tensor,
                        radius: int, eps: float) -> torch.Tensor:
    """
    Apply guided filter algorithm.
    
    Args:
        guide_gray: Grayscale guidance image (T, 1, H, W)
        src: Source image to filter (T, 1, H, W)
        radius: Filter radius in pixels
        eps: Regularization parameter for edge preservation
        
    Returns:
        Filtered output tensor (T, 1, H, W)
        
    Note:
        Uses box filtering for efficient implementation
    """
    
    def box_filter(x, r):
        return F.avg_pool2d(x, kernel_size=2*r+1, stride=1, padding=r)
    
    # Compute means
    mean_guide = box_filter(guide_gray, radius)
    mean_src = box_filter(src, radius)
    
    # Compute correlations
    corr_guide = box_filter(guide_gray * guide_gray, radius)
    corr_guide_src = box_filter(guide_gray * src, radius)
    
    # Compute variance and covariance
    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src
    
    del corr_guide, corr_guide_src
    
    # Linear coefficients
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    
    del cov_guide_src, var_guide, mean_guide, mean_src
    
    # Average coefficients
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    del a, b
    
    # Apply filter
    output = mean_a * guide_gray + mean_b
    
    del mean_a, mean_b
    
    return output


def edge_guided_alpha_upscale(
    input_alpha: torch.Tensor,
    input_rgb: torch.Tensor,
    upscaled_rgb: torch.Tensor,
    method: str = 'guided',
    debug: Optional['Debug'] = None
) -> torch.Tensor:
    """
    Upscale Alpha channel using RGB edge structure as guidance.
    Creates tight, smooth transitions aligned with RGB edges.
    
    Args:
        input_alpha: Original Alpha (T, 1, H_in, W_in) in [0, 1]
        input_rgb: Original RGB (T, 3, H_in, W_in) in [-1, 1] or [0, 1]
        upscaled_rgb: Upscaled RGB (T, 3, H_out, W_out) in [-1, 1] or [0, 1]
        method: 'guided' or 'bilateral'
        debug: Debug instance
        
    Returns:
        Upscaled Alpha (T, 1, H_out, W_out) in [0, 1]
    """
    T, _, H_out, W_out = upscaled_rgb.shape
    device = input_alpha.device
    dtype = input_alpha.dtype
    
    # Convert to float32 for numerical stability in edge detection and filtering
    input_alpha, alpha_dtype = ensure_float32_precision(input_alpha, force_float32=True)
    upscaled_rgb, _ = ensure_float32_precision(upscaled_rgb, force_float32=True)
    input_rgb, _ = ensure_float32_precision(input_rgb, force_float32=True)
    
    # Analyze alpha distribution to detect binary masks vs gradient alphas
    alpha_flat = input_alpha.flatten()
    near_zero = (alpha_flat < 0.1).sum().float()
    near_one = (alpha_flat > 0.9).sum().float()
    binary_ratio = (near_zero + near_one) / alpha_flat.numel()
    is_binary_mask = binary_ratio > 0.95
    
    if debug:
        debug.log(f"Alpha type: {'binary mask' if is_binary_mask else 'gradient alpha'}", category="alpha", indent_level=1)
        debug.log(f"Binary ratio: {binary_ratio:.2%}", category="alpha", indent_level=1)
        debug.log("Creating tight edge-aligned transitions", category="alpha", indent_level=1)
    
    # Normalize RGB from [-1, 1] to [0, 1] for edge detection
    rgb_normalized = upscaled_rgb.clone()
    if rgb_normalized.min() < 0:
        rgb_normalized = (rgb_normalized + 1) / 2
    
    # Detect edges in upscaled RGB
    rgb_edges = detect_edges_batch(images=rgb_normalized, method='sobel', debug=debug)
    
    # Step 1: Initial bicubic upscale provides smooth base before edge refinement
    # MPS on PyTorch < 2.8 doesn't support bicubic+antialias - use CPU fallback
    try:
        alpha_upscaled = F.interpolate(
            input_alpha,
            size=(H_out, W_out),
            mode='bicubic',
            align_corners=False,
            antialias=True
        ).clamp(0, 1)
    except NotImplementedError:
        alpha_upscaled = F.interpolate(
            input_alpha.cpu(),
            size=(H_out, W_out),
            mode='bicubic',
            align_corners=False,
            antialias=True
        ).to(device).clamp(0, 1)
    
    if is_binary_mask:
        if debug:
            debug.log("Applying tight edge-aware refinement", category="alpha", indent_level=1)
        
        # Step 2: Single-pass guided filter with small radius for tight edges
        alpha_refined = guided_filter_pytorch(
            guide=rgb_normalized,
            src=alpha_upscaled,
            radius=2,  # Reduced from 3 for tighter edges
            eps=0.002
        )
        
        # Step 3: Create tight transition zone using 3x3 max pooling on edge map
        edge_map = rgb_edges
        transition_zone = F.max_pool2d(edge_map, kernel_size=3, stride=1, padding=1)
        
        # Step 4: Identify solid regions (far from edges) vs transition regions (near edges)
        solid_threshold = 0.05  # Reduced from 0.08
        is_solid = transition_zone < solid_threshold
        
        # Convert solid regions to binary (0 or 1) based on 0.5 threshold
        alpha_binary = (alpha_refined > 0.5).float()
        
        # Step 5: Apply strong contrast enhancement in narrow transition zones to complete gradients
        in_edge_region = (transition_zone >= solid_threshold) & (transition_zone < 0.25)  # Narrow band
        
        # Apply strong sigmoid (strength=12.0) to push alpha values toward binary endpoints
        contrast_enhanced = torch.sigmoid((alpha_refined - 0.5) * 12.0)  # Increased from 7.0
        
        # Blend original and enhanced alpha based on edge strength for smooth transitions
        edge_strength = torch.clamp(edge_map / 0.25, 0, 1)
        alpha_in_edges = alpha_refined * (1 - edge_strength) + contrast_enhanced * edge_strength
        
        # Step 6: Combine solid binary regions with enhanced transition regions
        alpha_combined = torch.where(is_solid, alpha_binary, alpha_in_edges)
        
        # Step 7: Snap remaining mid-gray values to binary outside tightest edge zone
        very_solid = transition_zone < 0.03  # Even tighter threshold
        final_binary = (alpha_combined > 0.5).float()
        alpha_final = torch.where(very_solid, final_binary, alpha_combined)
        
        # Step 8: Final cleanup - eliminate mid-gray artifacts, preserving only smooth edges in tight regions
        tightest_edges = edge_map > 0.15
        mid_gray = (alpha_final > 0.3) & (alpha_final < 0.7)
        should_be_binary = mid_gray & ~tightest_edges
        
        alpha_final = torch.where(
            should_be_binary,
            (alpha_final > 0.5).float(),
            alpha_final
        )
        
    else:
        # For gradient alphas: single guided filter pass preserves smooth transitions
        if debug:
            debug.log("Applying guided filter for gradient alpha", category="alpha", indent_level=1)
        
        alpha_final = alpha_upscaled.clone()
        
        # Single pass with optimized parameters for gradients
        alpha_final = guided_filter_pytorch(
            guide=rgb_normalized,
            src=alpha_final,
            radius=3,
            eps=0.002
        )
    
   # Clamp output to valid alpha range [0, 1]
    alpha_final = alpha_final.clamp(0, 1)
    
    if debug:
        final_values = alpha_final.flatten()
        exact_zeros = (final_values < 0.02).sum().item()
        exact_ones = (final_values > 0.98).sum().item()
        smooth_edges = ((final_values >= 0.02) & (final_values <= 0.98)).sum().item()
        mid_grays = ((final_values >= 0.35) & (final_values <= 0.65)).sum().item()
        total = final_values.numel()
        
        debug.log(f"Pure 0s: {exact_zeros/total*100:.1f}% | Pure 1s: {exact_ones/total*100:.1f}% | Smooth edges: {smooth_edges/total*100:.1f}% | Mid-grays: {mid_grays/total*100:.1f}%", category="alpha", indent_level=1)
    
    return alpha_final