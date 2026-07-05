"""
Wan2.2 VAE Implementation for ComfyUI
Includes patchify/unpatchify operations, spatial downsampling/upsampling,
and the main Wan2_2_VAE wrapper class with proper normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert image to patches.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        patch_size: Size of patches (assumed square)
    
    Returns:
        Patched tensor of shape (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})"
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Reshape: (B, C, H, W) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = x.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    # Permute: (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    #       -> (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    x = x.permute(0, 1, 2, 4, 3, 5)
    
    return x


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert patches back to image.
    
    Args:
        x: Patched tensor of shape (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        patch_size: Size of patches
    
    Returns:
        Image tensor of shape (B, C, H, W)
    """
    B, C, num_patches_h, num_patches_w, _, _ = x.shape
    
    # Permute: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    #       -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = x.permute(0, 1, 2, 4, 3, 5)
    # Reshape: (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    #       -> (B, C, H, W)
    x = x.reshape(B, C, num_patches_h * patch_size, num_patches_w * patch_size)
    
    return x


class AvgDown3D(nn.Module):
    """
    3D Average Downsampling module.
    Performs average pooling in spatial and channel dimensions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Learnable projection for channel dimension reduction/mapping
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W) or (B, C, H, W)
        
        Returns:
            Downsampled tensor
        """
        if x.dim() == 4:
            # 2D case: (B, C, H, W)
            B, C, H, W = x.shape
            
            # Apply average pooling
            x = F.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            
            # Project channels if needed
            if self.proj is not None:
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
                x = self.proj(x)
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            return x
        
        elif x.dim() == 5:
            # 3D case: (B, C, D, H, W)
            B, C, D, H, W = x.shape
            
            # Apply average pooling
            x = F.avg_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            
            # Project channels if needed
            if self.proj is not None:
                B, C, D, H, W = x.shape
                x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
                x = self.proj(x)
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
            
            return x
        
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D")


class DupUp3D(nn.Module):
    """
    3D Duplication Upsampling module.
    Performs nearest-neighbor upsampling (duplication of values).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        
        # Learnable projection for channel dimension mapping
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W) or (B, C, H, W)
        
        Returns:
            Upsampled tensor
        """
        if x.dim() == 4:
            # 2D case: (B, C, H, W)
            # Use nearest neighbor interpolation
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='nearest'
            )
            
            # Project channels if needed
            if self.proj is not None:
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
                x = self.proj(x)
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            return x
        
        elif x.dim() == 5:
            # 3D case: (B, C, D, H, W)
            # Use nearest neighbor interpolation
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='nearest'
            )
            
            # Project channels if needed
            if self.proj is not None:
                B, C, D, H, W = x.shape
                x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
                x = self.proj(x)
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
            
            return x
        
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D")


class Wan2_2_VAE(nn.Module):
    """
    Wan2.2 VAE wrapper class with proper normalization parameters.
    
    This module wraps a pre-trained VAE model and provides:
    - Normalization/denormalization utilities
    - Encoding and decoding interfaces
    - Proper parameter initialization
    
    Attributes:
        scale_factor: Scaling factor for latent space
        shift_factor: Shifting factor for normalization
        mean: Mean values for normalization (per-channel)
        std: Standard deviation values for normalization (per-channel)
    """
    
    def __init__(
        self,
        vae_model: Optional[nn.Module] = None,
        latent_channels: int = 4,
        scale_factor: float = 0.18215,
        shift_factor: float = 0.0,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        use_quant: bool = True,
        quant_conv_in: int = 8,
        quant_conv_out: int = 8
    ):
        """
        Initialize Wan2.2 VAE wrapper.
        
        Args:
            vae_model: Pre-trained VAE model (optional)
            latent_channels: Number of channels in latent space
            scale_factor: Scaling factor for latent embeddings (default: 0.18215)
            shift_factor: Shifting factor for normalization (default: 0.0)
            mean: Mean normalization values (default: zeros)
            std: Standard deviation normalization values (default: ones)
            use_quant: Whether to use quantization convolutions
            quant_conv_in: Input channels for quantization conv
            quant_conv_out: Output channels for quantization conv
        """
        super().__init__()
        
        self.vae_model = vae_model
        self.latent_channels = latent_channels
        self.scale_factor = float(scale_factor)
        self.shift_factor = float(shift_factor)
        self.use_quant = use_quant
        
        # Register normalization parameters
        if mean is None:
            mean = torch.zeros(latent_channels)
        if std is None:
            std = torch.ones(latent_channels)
        
        # Ensure proper shapes
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean, dtype=torch.float32)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std, dtype=torch.float32)
        
        # Reshape to (1, C, 1, 1) for broadcasting
        self.register_buffer('mean', mean.view(1, -1, 1, 1))
        self.register_buffer('std', std.view(1, -1, 1, 1))
        
        # Quantization convolutions (if enabled)
        if self.use_quant:
            self.quant_conv = nn.Conv2d(quant_conv_in, quant_conv_out, 1)
            self.post_quant_conv = nn.Conv2d(quant_conv_out, quant_conv_in, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize module weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode an image to latent space.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            return_dict: Whether to return as dictionary
        
        Returns:
            Dictionary containing 'latent' and optional 'distribution' info
        """
        if self.vae_model is None:
            raise RuntimeError("VAE model not initialized")
        
        # Pass through VAE encoder
        if hasattr(self.vae_model, 'encode'):
            posterior = self.vae_model.encode(x)
            latent = posterior.sample() if hasattr(posterior, 'sample') else posterior
        else:
            latent = self.vae_model(x)
        
        # Apply quantization if enabled
        if self.use_quant and hasattr(self, 'quant_conv'):
            latent = self.quant_conv(latent)
        
        # Normalize
        latent = self.normalize(latent)
        
        if return_dict:
            return {'latent': latent}
        return latent
    
    def decode(
        self,
        latent: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to image space.
        
        Args:
            latent: Latent tensor of shape (B, C, H_latent, W_latent)
            return_dict: Whether to return as dictionary
        
        Returns:
            Dictionary containing 'sample' key with decoded image
        """
        if self.vae_model is None:
            raise RuntimeError("VAE model not initialized")
        
        # Denormalize
        latent = self.denormalize(latent)
        
        # Apply post-quantization conv if enabled
        if self.use_quant and hasattr(self, 'post_quant_conv'):
            latent = self.post_quant_conv(latent)
        
        # Pass through VAE decoder
        if hasattr(self.vae_model, 'decode'):
            sample = self.vae_model.decode(latent)
        else:
            sample = self.vae_model(latent)
        
        if return_dict:
            return {'sample': sample}
        return sample
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize latent representation.
        
        Args:
            x: Input tensor
        
        Returns:
            Normalized tensor
        """
        x = (x - self.shift_factor) * self.scale_factor
        return x
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize latent representation.
        
        Args:
            x: Normalized tensor
        
        Returns:
            Denormalized tensor
        """
        x = (x / self.scale_factor) + self.shift_factor
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VAE (encode + decode).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Reconstructed tensor of same shape
        """
        encoded = self.encode(x, return_dict=True)
        decoded = self.decode(encoded['latent'], return_dict=True)
        return decoded['sample']
    
    def set_normalization(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None
    ):
        """
        Set normalization parameters.
        
        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        if mean is not None:
            if isinstance(mean, (list, tuple)):
                mean = torch.tensor(mean, dtype=torch.float32)
            self.register_buffer('mean', mean.view(1, -1, 1, 1))
        
        if std is not None:
            if isinstance(std, (list, tuple)):
                std = torch.tensor(std, dtype=torch.float32)
            self.register_buffer('std', std.view(1, -1, 1, 1))
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'latent_channels': self.latent_channels,
            'scale_factor': self.scale_factor,
            'shift_factor': self.shift_factor,
            'use_quant': self.use_quant,
            'mean': self.mean.squeeze().tolist() if self.mean is not None else None,
            'std': self.std.squeeze().tolist() if self.std is not None else None,
        }
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        vae_model: Optional[nn.Module] = None
    ) -> 'Wan2_2_VAE':
        """
        Create VAE instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            vae_model: Pre-trained VAE model
        
        Returns:
            Wan2_2_VAE instance
        """
        return cls(
            vae_model=vae_model,
            latent_channels=config.get('latent_channels', 4),
            scale_factor=config.get('scale_factor', 0.18215),
            shift_factor=config.get('shift_factor', 0.0),
            mean=config.get('mean'),
            std=config.get('std'),
            use_quant=config.get('use_quant', True),
        )


# Utility functions for common operations

def create_wan2_2_vae(
    latent_channels: int = 4,
    scale_factor: float = 0.18215,
    device: Optional[torch.device] = None
) -> Wan2_2_VAE:
    """
    Create a Wan2.2 VAE instance with default configuration.
    
    Args:
        latent_channels: Number of latent channels
        scale_factor: Scaling factor for normalization
        device: Device to create tensors on
    
    Returns:
        Configured Wan2_2_VAE instance
    """
    vae = Wan2_2_VAE(
        latent_channels=latent_channels,
        scale_factor=scale_factor
    )
    
    if device is not None:
        vae = vae.to(device)
    
    return vae


def calculate_spatial_dimensions(
    input_size: int,
    scale_factor: int = 8
) -> int:
    """
    Calculate latent spatial dimensions.
    
    Args:
        input_size: Input spatial dimension
        scale_factor: Downsampling scale factor
    
    Returns:
        Latent spatial dimension
    """
    return input_size // scale_factor


if __name__ == "__main__":
    # Example usage
    print("Wan2.2 VAE Module loaded successfully")
    
    # Test patchify/unpatchify
    x = torch.randn(2, 3, 512, 512)
    patches = patchify(x, patch_size=16)
    x_recon = unpatchify(patches, patch_size=16)
    print(f"Patchify test - Input shape: {x.shape}, Output shape: {x_recon.shape}")
    assert torch.allclose(x, x_recon), "Patchify/unpatchify mismatch"
    
    # Test AvgDown3D
    down = AvgDown3D(3, 3, kernel_size=2, stride=2)
    x_down = down(x)
    print(f"AvgDown3D test - Input: {x.shape}, Output: {x_down.shape}")
    
    # Test DupUp3D
    up = DupUp3D(3, 3, scale_factor=2)
    x_up = up(x_down)
    print(f"DupUp3D test - Input: {x_down.shape}, Output: {x_up.shape}")
    
    # Test Wan2_2_VAE
    vae = create_wan2_2_vae(latent_channels=4)
    print(f"Wan2_2_VAE config: {vae.get_config()}")
    print("All tests passed!")
