"""
Wan2.1 VAE Implementation for ComfyUI-SeedVR2.5
Includes encoder/decoder blocks and Wan2_1_VAE wrapper class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_dropout: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=True)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=True),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )
        else:
            self.skip = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.skip is not None:
            identity = self.skip(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Reshape and normalize
        x_norm = self.norm(x)
        x_flat = x_norm.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Project to Q, K, V
        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(batch_size, height * width, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, height * width, channels)
        out = self.proj(out)
        
        # Reshape back and add residual
        out = out.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        out = out + x
        
        return out


class EncoderBlock(nn.Module):
    """Encoder block with convolution, residuals, and optional attention"""
    
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2,
                 stride: int = 1, use_attention: bool = False, attention_heads: int = 4):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                 stride=stride, padding=1, bias=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, out_channels, use_dropout=False)
            for _ in range(num_res_blocks)
        ])
        
        # Attention
        self.attention = None
        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads=attention_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, convolution, residuals, and optional attention"""
    
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2,
                 scale_factor: float = 2.0, use_attention: bool = False, 
                 attention_heads: int = 4):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                 padding=1, bias=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, out_channels, use_dropout=False)
            for _ in range(num_res_blocks)
        ])
        
        # Attention
        self.attention = None
        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads=attention_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample
        if self.scale_factor > 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        
        x = self.conv_in(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class Wan2_1_Encoder(nn.Module):
    """Complete Wan2.1 VAE Encoder"""
    
    def __init__(self, in_channels: int = 3, z_channels: int = 4, 
                 base_channels: int = 64, channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2, attention_at_res: int = 2):
        super().__init__()
        
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                                 stride=1, padding=1, bias=True)
        
        # Encoder blocks with downsampling
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            use_attn = (mult >= attention_at_res)
            
            block = EncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                num_res_blocks=num_res_blocks,
                stride=2,
                use_attention=use_attn,
                attention_heads=min(4, out_ch // 64)
            )
            self.down_blocks.append(block)
            in_ch = out_ch
        
        # Middle blocks
        self.middle_res_blocks = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, use_dropout=False)
            for _ in range(num_res_blocks)
        ])
        self.middle_attention = AttentionBlock(in_ch, num_heads=min(4, in_ch // 64))
        
        # Output projection to latent space
        self.norm_out = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, kernel_size=3, 
                                  stride=1, padding=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution
        Returns: mean and logvar for reparameterization
        """
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling blocks
        for block in self.down_blocks:
            h = block(h)
        
        # Middle blocks
        for res_block in self.middle_res_blocks:
            h = res_block(h)
        h = self.middle_attention(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z


class Wan2_1_Decoder(nn.Module):
    """Complete Wan2.1 VAE Decoder"""
    
    def __init__(self, z_channels: int = 4, out_channels: int = 3, 
                 base_channels: int = 64, channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2, attention_at_res: int = 2):
        super().__init__()
        
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        
        # Input projection from latent space
        num_down = len(channel_multipliers)
        self.z_to_h = nn.Conv2d(z_channels, base_channels * channel_multipliers[-1],
                               kernel_size=1, stride=1, bias=True)
        
        # Middle blocks
        in_ch = base_channels * channel_multipliers[-1]
        self.middle_res_blocks = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, use_dropout=False)
            for _ in range(num_res_blocks)
        ])
        self.middle_attention = AttentionBlock(in_ch, num_heads=min(4, in_ch // 64))
        
        # Decoder blocks with upsampling
        self.up_blocks = nn.ModuleList()
        mults = list(reversed(channel_multipliers))
        
        for i, mult in enumerate(mults):
            out_ch = base_channels * mult
            use_attn = (mult >= attention_at_res)
            
            block = DecoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                num_res_blocks=num_res_blocks,
                scale_factor=2.0,
                use_attention=use_attn,
                attention_heads=min(4, out_ch // 64)
            )
            self.up_blocks.append(block)
            in_ch = out_ch
        
        # Output convolution
        self.norm_out = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, 
                                  stride=1, padding=1, bias=True)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to image"""
        # Input projection
        h = self.z_to_h(z)
        
        # Middle blocks
        for res_block in self.middle_res_blocks:
            h = res_block(h)
        h = self.middle_attention(h)
        
        # Upsampling blocks
        for block in self.up_blocks:
            h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        h = torch.tanh(h)  # Ensure output is in [-1, 1]
        
        return h


class Wan2_1_VAE(nn.Module):
    """
    Complete Wan2.1 VAE wrapper class combining encoder and decoder
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 z_channels: int = 4, base_channels: int = 64,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2, attention_at_res: int = 2,
                 use_ema: bool = False, ema_decay: float = 0.99):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.base_channels = base_channels
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Encoder and Decoder
        self.encoder = Wan2_1_Encoder(
            in_channels=in_channels,
            z_channels=z_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_at_res=attention_at_res
        )
        
        self.decoder = Wan2_1_Decoder(
            z_channels=z_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_at_res=attention_at_res
        )
        
        # EMA tracking if enabled
        if use_ema:
            self.register_buffer('ema_step', torch.tensor(0, dtype=torch.long))
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Tuple of (mean, logvar) for latent distribution
        """
        mean, logvar = self.encoder(x)
        return mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to image
        Args:
            z: Latent code tensor (B, C, H, W)
        Returns:
            Reconstructed image tensor
        """
        x_recon = self.decoder(z)
        return x_recon
    
    def sample(self, num_samples: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from standard normal distribution and decode
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        Returns:
            Generated image tensor
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard normal
        # Assuming 4x4 latent space for typical downsampling
        z = torch.randn(num_samples, self.z_channels, 4, 4, device=device)
        x_samples = self.decode(z)
        
        return x_samples
    
    def forward(self, x: torch.Tensor, return_loss: bool = False) -> torch.Tensor:
        """
        Full VAE forward pass: encode -> reparameterize -> decode
        Args:
            x: Input image tensor (B, C, H, W)
            return_loss: If True, returns (reconstruction, kl_loss)
        Returns:
            Reconstructed image or (reconstruction, kl_loss) if return_loss=True
        """
        # Encode
        mean, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.encoder.reparameterize(mean, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        if return_loss:
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_loss.mean()
            return x_recon, kl_loss
        
        return x_recon
    
    def update_ema(self) -> None:
        """Update EMA weights if enabled"""
        if not self.use_ema:
            return
        
        self.ema_step += 1
        current_decay = min(self.ema_decay, 1.0 - 1.0 / (self.ema_step.item() + 1))
        
        # Update EMA parameters (would be implemented with separate EMA model in practice)
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'z_channels': self.z_channels,
            'base_channels': self.base_channels,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
        }
    
    @staticmethod
    def from_pretrained(pretrained_path: str, device: Optional[torch.device] = None) -> 'Wan2_1_VAE':
        """
        Load pretrained Wan2.1 VAE from checkpoint
        Args:
            pretrained_path: Path to checkpoint file
            device: Device to load model on
        Returns:
            Loaded Wan2_1_VAE model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Extract config if available
        config = checkpoint.get('config', {})
        model = Wan2_1_VAE(**config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model
    
    def save_checkpoint(self, save_path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: int = 0, step: int = 0) -> None:
        """
        Save model checkpoint
        Args:
            save_path: Path to save checkpoint to
            optimizer: Optional optimizer state to save
            epoch: Current epoch
            step: Current step
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'epoch': epoch,
            'step': step,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, save_path)


# Convenience function for creating standard Wan2.1 VAE
def create_wan2_1_vae(z_channels: int = 4, pretrained: Optional[str] = None,
                      device: Optional[torch.device] = None) -> Wan2_1_VAE:
    """
    Create a Wan2.1 VAE model with standard configuration
    Args:
        z_channels: Latent space dimensions
        pretrained: Path to pretrained weights (optional)
        device: Device to create model on
    Returns:
        Wan2_1_VAE model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Wan2_1_VAE(
        in_channels=3,
        out_channels=3,
        z_channels=z_channels,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_at_res=2,
        use_ema=False
    ).to(device)
    
    if pretrained is not None:
        model = Wan2_1_VAE.from_pretrained(pretrained, device=device)
    
    return model
