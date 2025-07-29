import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from src.common.half_precision_fixes import safe_pad_operation, safe_interpolate_operation
from torchvision.transforms import ToTensor, ToPILImage

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # Limit radius to prevent numerical instability at high resolutions
    max_radius = min(radius, min(image.shape[-2:]) // 4)
    if max_radius != radius:
        print(f"⚠️ Limiting wavelet blur radius from {radius} to {max_radius} for stability")
        radius = max_radius
    
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    
    # Use safer padding for high resolution
    try:
        image = safe_pad_operation(image, (radius, radius, radius, radius), mode='replicate')
        # apply convolution
        output = F.conv2d(image, kernel, groups=3, dilation=radius)
    except RuntimeError as e:
        print(f"⚠️ Wavelet blur failed with radius {radius}, using simpler blur: {e}")
        # Fallback to simple averaging
        output = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
    
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    # Ensure input is in a stable dtype for high resolution processing
    original_dtype = image.dtype
    if original_dtype == torch.float16:
        image = image.float()  # Use FP32 for numerical stability
    
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        
        # Add numerical stability check
        diff = image - low_freq
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            print(f"⚠️ NaN/Inf in wavelet level {i}, skipping this level")
            break
            
        high_freq += diff
        image = low_freq
    
    # Convert back to original dtype
    if original_dtype == torch.float16:
        high_freq = high_freq.half()
        image = image.half()

    return high_freq, image



def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # Store original dtype for final output
    original_dtype = content_feat.dtype
    
    # Use FP32 for numerical stability in high resolution processing
    if content_feat.dtype == torch.float16:
        content_feat = content_feat.float()
    if style_feat.dtype == torch.float16:
        style_feat = style_feat.float()
    
    # Vérifier et ajuster les dimensions si nécessaire
    if content_feat.shape != style_feat.shape:
        print(f"⚠️ Dimension mismatch détectée: content {content_feat.shape} vs style {style_feat.shape}")
        
        # Redimensionner style_feat pour correspondre à content_feat
        target_shape = content_feat.shape
        if len(target_shape) >= 3:  # Au moins 3 dimensions
            # Utiliser interpolation pour ajuster les dimensions spatiales
            style_feat = safe_interpolate_operation(
                style_feat, 
                size=target_shape[-2:],  # Dernières 2 dimensions (H, W)
                mode='bilinear', 
                align_corners=False
            )
            print(f"✅ Style redimensionné vers: {style_feat.shape}")
    
    # Add input validation
    if torch.isnan(content_feat).any() or torch.isinf(content_feat).any():
        print(f"⚠️ NaN/Inf in content_feat input, cleaning...")
        content_feat = torch.nan_to_num(content_feat, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if torch.isnan(style_feat).any() or torch.isinf(style_feat).any():
        print(f"⚠️ NaN/Inf in style_feat input, cleaning...")
        style_feat = torch.nan_to_num(style_feat, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    
    # Vérification finale avant addition
    if content_high_freq.shape != style_low_freq.shape:
        print(f"⚠️ Ajustement final nécessaire: {content_high_freq.shape} vs {style_low_freq.shape}")
        style_low_freq = safe_interpolate_operation(
            style_low_freq,
            size=content_high_freq.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # reconstruct the content feature with the style's high frequency
    result = content_high_freq + style_low_freq
    
    # Final validation and cleanup
    if torch.isnan(result).any() or torch.isinf(result).any():
        print(f"⚠️ NaN/Inf in wavelet reconstruction result, applying selective fix")
        # Selective fix: only replace problematic pixels, preserve good ones
        finite_mask = torch.isfinite(result)
        if finite_mask.any():
            valid_values = result[finite_mask]
            replacement_value = torch.median(valid_values)
            result = torch.where(finite_mask, result, replacement_value)
            print(f"   Replaced {(~finite_mask).sum().item()} pixels with median {replacement_value:.4f}")
        else:
            # Complete failure - use original content
            print(f"   Complete reconstruction failure, using original content")
            result = content_feat
    
    # Convert back to original dtype
    if original_dtype == torch.float16:
        result = result.half()
    
    return result
