"""
GGUF Quantization Operations Module

This module provides runtime quantization support for GGUF models,
handling on-the-fly dequantization during forward passes while
keeping tensors in quantized format in VRAM.
"""

import torch
import torch.nn as nn
import gguf
import warnings
from typing import Optional, Tuple


class GGUFQuantizedLinear(nn.Module):
    """
    Quantized Linear layer that dequantizes weights on-the-fly
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.quantized_weight = None
        self.weight_qtype = None
        self.weight_shape = None
        
    def load_quantized_weight(self, weight_tensor, bias_tensor=None):
        """Load quantized weight tensor"""
        if hasattr(weight_tensor, 'tensor_type') and hasattr(weight_tensor, 'tensor_shape'):
            # This is a quantized tensor
            self.quantized_weight = weight_tensor
            self.weight_qtype = weight_tensor.tensor_type
            self.weight_shape = weight_tensor.tensor_shape
            print(f"🔍 GGUFQuantizedLinear loaded weight: shape={weight_tensor.tensor_shape}, type={weight_tensor.tensor_type}")
        else:
            # This is a regular tensor
            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
            print(f"🔍 GGUFQuantizedLinear loaded regular weight: shape={weight_tensor.shape}")
            
        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
            
    def dequantize_weight(self, device=None, dtype=torch.float16):
        """Dequantize weight tensor on-the-fly"""
        if self.quantized_weight is None:
            return self.weight
            
        # Check if it's our custom GGUFTensor or parameter with gguf_dequantize
        if hasattr(self.quantized_weight, 'gguf_dequantize'):
            return self.quantized_weight.gguf_dequantize(device, dtype)
        elif hasattr(self.quantized_weight, 'dequantize'):
            return self.quantized_weight.dequantize(device, dtype)
            
        if self.weight_qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Already unquantized
            return self.quantized_weight.to(device, dtype)
        
        try:
            # Dequantize using gguf
            numpy_data = self.quantized_weight.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.weight_qtype)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            return result.reshape(self.weight_shape)
        except Exception as e:
            print(f"Warning: Could not dequantize weight: {e}")
            return self.quantized_weight.to(device, dtype)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        dtype = input.dtype
        
        # Dequantize weight on-the-fly
        weight = self.dequantize_weight(device, dtype)
        
        # PyTorch linear expects weight to be [out_features, in_features]
        # but GGUF provides [in_features, out_features], so we need to transpose
        if len(weight.shape) == 2:
            original_shape = weight.shape
            weight = weight.transpose(0, 1)
            
            # Debug output for shape mismatches
            expected_out_features = self.out_features
            expected_in_features = self.in_features
            if weight.shape != (expected_out_features, expected_in_features):
                print(f"⚠️ GGUFQuantizedLinear shape mismatch:")
                print(f"   Input: {input.shape}")
                print(f"   Expected weight: [{expected_out_features}, {expected_in_features}]")
                print(f"   Actual weight: {weight.shape} (transposed from {original_shape})")
                print(f"   Layer expects: {expected_in_features} -> {expected_out_features}")
        
        # Standard linear operation
        return torch.nn.functional.linear(input, weight, self.bias)


class GGUFQuantizedConv2d(nn.Module):
    """
    Quantized Conv2d layer that dequantizes weights on-the-fly
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = None
        self.bias = None
        self.quantized_weight = None
        self.weight_qtype = None
        self.weight_shape = None
        
    def load_quantized_weight(self, weight_tensor, bias_tensor=None):
        """Load quantized weight tensor"""
        if hasattr(weight_tensor, 'tensor_type') and hasattr(weight_tensor, 'tensor_shape'):
            # This is a quantized tensor
            self.quantized_weight = weight_tensor
            self.weight_qtype = weight_tensor.tensor_type
            self.weight_shape = weight_tensor.tensor_shape
        else:
            # This is a regular tensor
            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
            
        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
            
    def dequantize_weight(self, device=None, dtype=torch.float16):
        """Dequantize weight tensor on-the-fly"""
        if self.quantized_weight is None:
            return self.weight
            
        # Check if it's our custom GGUFTensor or parameter with gguf_dequantize
        if hasattr(self.quantized_weight, 'gguf_dequantize'):
            return self.quantized_weight.gguf_dequantize(device, dtype)
        elif hasattr(self.quantized_weight, 'dequantize'):
            return self.quantized_weight.dequantize(device, dtype)
            
        if self.weight_qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Already unquantized
            return self.quantized_weight.to(device, dtype)
        
        try:
            # Dequantize using gguf
            numpy_data = self.quantized_weight.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.weight_qtype)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            return result.reshape(self.weight_shape)
        except Exception as e:
            print(f"Warning: Could not dequantize weight: {e}")
            return self.quantized_weight.to(device, dtype)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        dtype = input.dtype
        
        # Dequantize weight on-the-fly
        weight = self.dequantize_weight(device, dtype)
        
        # Standard conv2d operation
        return torch.nn.functional.conv2d(input, weight, self.bias, self.stride, 
                                         self.padding, self.dilation, self.groups)


def is_quantized_tensor(tensor):
    """Check if a tensor is quantized"""
    # Check if it's our GGUFTensor class
    if hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape'):
        return True
    # Check if it's marked as quantized
    if hasattr(tensor, '_gguf_quantized'):
        return True
    # Check if it's a GGUFTensor by class name
    if 'GGUFTensor' in str(type(tensor)):
        return True
    return False


def replace_linear_with_quantized(module, prefix=""):
    """
    Replace Linear layers with quantized versions if they have quantized weights
    """
    replacements_made = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if weight is quantized
            if is_quantized_tensor(child.weight):
                # Create quantized linear layer
                quantized_linear = GGUFQuantizedLinear(
                    child.in_features, 
                    child.out_features,
                    bias=child.bias is not None
                )
                quantized_linear.load_quantized_weight(child.weight, child.bias)
                setattr(module, name, quantized_linear)
                print(f"✅ Replaced {prefix}.{name} with quantized linear layer ({child.in_features} -> {child.out_features})")
                replacements_made += 1
        elif isinstance(child, nn.Conv2d):
            # Check if weight is quantized
            if is_quantized_tensor(child.weight):
                # Create quantized conv2d layer
                quantized_conv = GGUFQuantizedConv2d(
                    child.in_channels,
                    child.out_channels, 
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None
                )
                quantized_conv.load_quantized_weight(child.weight, child.bias)
                setattr(module, name, quantized_conv)
                print(f"✅ Replaced {prefix}.{name} with quantized conv2d layer")
                replacements_made += 1
        else:
            # Recursively replace in child modules
            replacements_made += replace_linear_with_quantized(child, f"{prefix}.{name}" if prefix else name)
    
    return replacements_made


def apply_quantized_ops(model):
    """
    Apply quantized operations to a model loaded with GGUF tensors
    """
    print("🔄 Applying quantized operations to model...")
    
    # Count quantized parameters before replacement
    total_params = 0
    quantized_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if is_quantized_tensor(param):
            quantized_params += 1
    
    print(f"📊 Before replacement: {quantized_params}/{total_params} quantized parameters")
    
    replacements_made = replace_linear_with_quantized(model)
    print(f"✅ Quantized operations applied successfully! Made {replacements_made} replacements")
    return model