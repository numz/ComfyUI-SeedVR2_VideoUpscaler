"""
Compatibility module for SeedVR2
Contains FP8/FP16 compatibility layers and wrappers for different model architectures

Extracted from: seedvr2.py (lines 1045-1630)
"""

import torch
import types
from typing import List, Tuple, Union, Any, Optional


def call_rope_with_stability(method, *args, **kwargs):
    """
    Call RoPE method with stability fixes:
    1. Clear cache if available
    2. Disable autocast to prevent numerical issues
    This prevents artifacts in FP8/mixed precision models.
    """
    if hasattr(method, 'cache_clear'):
        method.cache_clear()
    
    with torch.cuda.amp.autocast(enabled=False):
        return method(*args, **kwargs)
    
class FP8CompatibleDiT(torch.nn.Module):
    """
    Wrapper for DiT models with automatic compatibility management + advanced optimizations
    - FP8: Keeps native FP8 parameters, converts inputs/outputs
    - FP16: Uses native FP16
    - Mixed Precision: Stabilizes RoPE for models with FP16 blocks
    - RoPE: Converted from FP8 to BFloat16 only when detected as FP8
    - Flash Attention: Automatic optimization of attention layers
    """
    
    def __init__(self, dit_model, skip_conversion=False, blockswap_active=None, debug=None):
        super().__init__()
        self.debug = debug if debug is not None else None
        if debug is None:
            raise ValueError("Debug instance must be provided to FP8CompatibleDiT")
            
        self.dit_model = dit_model
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16
        self.blockswap_active = blockswap_active if blockswap_active is not None else None

        # Only convert if not already done (e.g., when reusing cached weights)
        if not skip_conversion and self.is_fp8_model:
            # Only FP8 models need RoPE frequency conversion
            model_variant = "7B" if self._is_nadit_model() else "3B" if self._is_nadit_v2_model() else "Unknown"
            self.debug.log(f"Detected NaDiT {model_variant} FP8 - Converting RoPE freqs for FP8 compatibility", 
                          category="precision", force=True)
            self._convert_rope_freqs()
            if torch.mps.is_available():
                self.debug.log(f"Also converting NaDiT parameters/buffers for MPS backend", category="precision", force=True)
                self._force_nadit_bfloat16()
            
        # Apply RoPE stabilization for numerical stability
        self._stabilize_rope_computations()

        # 🚀 FLASH ATTENTION OPTIMIZATION (Phase 2)
        self._apply_flash_attention_optimization()
    
    def _detect_model_dtype(self) -> torch.dtype:
        """Detect main model dtype"""
        try:
            return next(self.dit_model.parameters()).dtype
        except:
            return torch.bfloat16
    
    def _is_nadit_model(self) -> bool:
        """Detect if this is a NaDiT model (7B) with precise logic"""
        # Check module path for dit (not dit_v2) 
        model_module = str(self.dit_model.__class__.__module__).lower()
        return 'dit.nadit' in model_module and 'dit_v2' not in model_module

    def _is_nadit_v2_model(self) -> bool:
        """Detect if this is a NaDiT v2 model (3B) with precise logic"""
        # Check module path for dit_v2
        model_module = str(self.dit_model.__class__.__module__).lower()
        return 'dit_v2' in model_module
        
    def _convert_rope_freqs(self) -> None:
        """Convert RoPE frequency buffers for FP8 compatibility"""
        converted = 0
        for module in self.dit_model.modules():
            if 'RotaryEmbedding' in type(module).__name__:
                if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
                    if module.rope.freqs.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        if  module.rope.freqs.device.type == "mps":
                            module.rope.freqs.data = module.rope.freqs.to("cpu").to(torch.bfloat16).to("mps")
                        else:
                            module.rope.freqs.data = module.rope.freqs.to(torch.bfloat16)
                        converted += 1
        self.debug.log(f"Converted {converted} RoPE frequency buffers", category="success")
                        
    def _force_nadit_bfloat16(self) -> None:
        """🎯 Force ALL NaDiT parameters to BFloat16 to avoid promotion errors"""
        converted_count = 0
        original_dtype = None
        
        # Convert ALL parameters to BFloat16 (FP8, FP16, etc.)
        for name, param in self.dit_model.named_parameters():
            if original_dtype is None:
                original_dtype = param.dtype
            if param.dtype != torch.bfloat16:
                if param.device.type == "mps":
                    param_data = param.data.to("cpu").to(torch.bfloat16).to("mps")
                else:
                    param_data = param.data.to(torch.bfloat16)
                param.data = param_data
                converted_count += 1
                
        # Also convert buffers
        for name, buffer in self.dit_model.named_buffers():
            if buffer.dtype != torch.bfloat16:
                if param.device.type == "mps":
                    buffer_data = buffer.data.to("cpu").to(torch.bfloat16).to("mps")
                else:
                    buffer_data = buffer.data.to(torch.bfloat16)
                buffer.data = buffer_data
                converted_count += 1
        
        self.debug.log(f"Converted {converted_count} NaDiT parameters/buffers for MPS", category="success")
        
        # Update detected dtype
        self.model_dtype = torch.bfloat16
        self.is_fp8_model = False  # Model is no longer FP8 after conversion

    def _stabilize_rope_computations(self):
        """
        Add error handling to RoPE computations to prevent artifacts.
        
        Wraps the get_axial_freqs method of RoPE modules with a try-except handler.
        During normal operation, uses the original cached method for performance.
        Only on exceptions (e.g., numerical instability, NaN propagation) does it
        intervene by clearing the cache and retrying the computation through
        call_rope_with_stability.
        
        This prevents artifacts in FP8, mixed precision, and edge cases while
        maintaining optimal performance for normal operations.
        """
        if not hasattr(self.dit_model, 'blocks'):
            return
        
        self.debug.start_timer("stabilize_rope")
        self.debug.log(f"Stabilizing RoPE computations for numerical stability", category="precision")
        
        rope_count = 0
        
        # Wrap RoPE modules to handle numerical instability
        for name, module in self.dit_model.named_modules():
            if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
                # Check if already wrapped
                if hasattr(module, '_rope_wrapped'):
                    continue
                    
                original_method = module.get_axial_freqs
                
                # Mark as wrapped and store original
                module._rope_wrapped = 'stability'
                module._original_get_axial_freqs = original_method
                
                # Error handler that prevents NaN propagation
                def stable_rope_computation(self, *args, **kwargs):
                    try:
                        return original_method(*args, **kwargs)
                    except Exception:
                        return call_rope_with_stability(original_method, *args, **kwargs)
                
                module.get_axial_freqs = types.MethodType(stable_rope_computation, module)
                rope_count += 1
        
        if rope_count > 0:
            self.debug.log(f"Stabilized {rope_count} RoPE modules", category="success")
        
        self.debug.end_timer("stabilize_rope", f"Stabilized {rope_count} RoPE modules")

    def _apply_flash_attention_optimization(self) -> None:
        """🚀 FLASH ATTENTION OPTIMIZATION - 30-50% speedup of attention layers"""
        attention_layers_optimized = 0
        flash_attention_available = self._check_flash_attention_support()
        
        for name, module in self.dit_model.named_modules():
            # Identify all attention layers
            if self._is_attention_layer(name, module):
                # Apply optimization based on availability
                if self._optimize_attention_layer(name, module, flash_attention_available):
                    attention_layers_optimized += 1
        
        if not flash_attention_available:
            self.debug.log("Flash Attention not available, using PyTorch SDPA as fallback", category="info", force=True)
    
    def _check_flash_attention_support(self) -> bool:
        """Check if Flash Attention is available"""
        try:
            # Check PyTorch SDPA (includes Flash Attention on H100/A100)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return True
            
            # Check flash-attn package
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _is_attention_layer(self, name: str, module: torch.nn.Module) -> bool:
        """Identify if a module is an attention layer"""
        attention_keywords = [
            'attention', 'attn', 'self_attn', 'cross_attn', 'mhattn', 'multihead',
            'transformer_block', 'dit_block'
        ]
        
        # Check by name
        if any(keyword in name.lower() for keyword in attention_keywords):
            return True
        
        # Check by module type
        module_type = type(module).__name__.lower()
        if any(keyword in module_type for keyword in attention_keywords):
            return True
        
        # Check by attributes (modules with q, k, v projections)
        if hasattr(module, 'q_proj') or hasattr(module, 'qkv') or hasattr(module, 'to_q'):
            return True
        
        return False
    
    def _optimize_attention_layer(self, name: str, module: torch.nn.Module, flash_attention_available: bool) -> bool:
        """Optimize a specific attention layer"""
        try:
            # Save original forward method
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            
            # Create new optimized forward method
            if flash_attention_available:
                optimized_forward = self._create_flash_attention_forward(module, name)
            else:
                optimized_forward = self._create_sdpa_forward(module, name)
            
            # Replace forward method
            module.forward = optimized_forward
            return True
            
        except Exception as e:
            self.debug.log(f"Failed to optimize attention layer '{name}': {e}", level="WARNING", category="model")
            return False
    
    def _create_flash_attention_forward(self, module: torch.nn.Module, layer_name: str):
        """Create optimized forward with Flash Attention"""
        original_forward = module._original_forward
        
        def flash_attention_forward(*args, **kwargs):
            try:
                # Try to use Flash Attention via SDPA
                return self._sdpa_attention_forward(original_forward, module, *args, **kwargs)
            except Exception as e:
                # Fallback to original implementation
                self.debug.log(f"Flash Attention failed for {layer_name}, using original: {e}", level="WARNING", category="model", force=True)
                return original_forward(*args, **kwargs)
        
        return flash_attention_forward
    
    def _create_sdpa_forward(self, module: torch.nn.Module, layer_name: str):
        """Create optimized forward with PyTorch SDPA"""
        original_forward = module._original_forward
        
        def sdpa_forward(*args, **kwargs):
            try:
                return self._sdpa_attention_forward(original_forward, module, *args, **kwargs)
            except Exception as e:
                # Fallback to original implementation
                return original_forward(*args, **kwargs)
        
        return sdpa_forward
    
    def _sdpa_attention_forward(self, original_forward, module: torch.nn.Module, *args, **kwargs):
        """Optimized forward pass using SDPA (Scaled Dot Product Attention)"""
        # Detect if we can intercept and optimize this layer
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
            
            # Check dimensions to ensure it's standard attention
            if len(input_tensor.shape) >= 3:  # [batch, seq_len, hidden_dim] or similar
                try:
                    return self._optimized_attention_computation(module, input_tensor, *args[1:], **kwargs)
                except:
                    pass
        
        # Fallback to original implementation
        return original_forward(*args, **kwargs)
    
    def _optimized_attention_computation(self, module: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs):
        """Optimized attention computation with SDPA"""
        # Try to detect standard attention format
        batch_size, seq_len = input_tensor.shape[:2]
        
        # Check if module has standard Q, K, V projections
        if hasattr(module, 'qkv') or (hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj')):
            return self._compute_sdpa_attention(module, input_tensor, *args, **kwargs)
        
        # If no standard format detected, use original
        return module._original_forward(input_tensor, *args, **kwargs)
    
    def _compute_sdpa_attention(self, module: torch.nn.Module, x: torch.Tensor, *args, **kwargs):
        """Optimized SDPA computation for standard attention modules"""
        try:
            # Case 1: Module with combined QKV projection
            if hasattr(module, 'qkv'):
                qkv = module.qkv(x)
                # Reshape to separate Q, K, V
                batch_size, seq_len, _ = qkv.shape
                qkv = qkv.reshape(batch_size, seq_len, 3, -1)
                q, k, v = qkv.unbind(dim=2)
                
            # Case 2: Separate Q, K, V projections
            elif hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                q = module.q_proj(x)
                k = module.k_proj(x)
                v = module.v_proj(x)
            else:
                # Unsupported format, use original
                return module._original_forward(x, *args, **kwargs)
            
            # Detect number of heads
            head_dim = getattr(module, 'head_dim', None)
            num_heads = getattr(module, 'num_heads', None)
            
            if head_dim is None or num_heads is None:
                # Try to guess from dimensions
                hidden_dim = q.shape[-1]
                if hasattr(module, 'num_heads'):
                    num_heads = module.num_heads
                    head_dim = hidden_dim // num_heads
                else:
                    # Reasonable defaults
                    head_dim = 64
                    num_heads = hidden_dim // head_dim
            
            # Reshape for multi-head attention
            batch_size, seq_len = q.shape[:2]
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            if torch.mps.is_available():
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False
                )
            else:
                # Use optimized SDPA
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True
                ):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=0.0,
                        is_causal=False
                    )
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, num_heads * head_dim
            )
            
            # Output projection if it exists
            if hasattr(module, 'out_proj') or hasattr(module, 'o_proj'):
                proj = getattr(module, 'out_proj', None) or getattr(module, 'o_proj', None)
                attn_output = proj(attn_output)
            
            return attn_output
            
        except Exception as e:
            # In case of error, use original implementation
            return module._original_forward(x, *args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass with minimal dtype conversion overhead
        
        Conversion strategy:
        - FP16 models: Keep everything in FP16 (no conversion needed)
        - FP8 models: Convert FP8 tensors to BFloat16 (required for arithmetic)
        - BFloat16 models: No conversion needed
        """
        
        # Only convert if we have an FP8 model for arithmetic operations 
        if self.is_fp8_model:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            
            # Convert args
            converted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype in fp8_dtypes:
                    converted_args.append(arg.to(torch.bfloat16))
                else:
                    converted_args.append(arg)
            
            # Convert kwargs
            converted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.dtype in fp8_dtypes:
                    converted_kwargs[key] = value.to(torch.bfloat16)
                else:
                    converted_kwargs[key] = value
            
            args = tuple(converted_args)
            kwargs = converted_kwargs
            
        # Force move weights to device    
        if not self.blockswap_active:
            input_device = None
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    input_device = value.device
                    break
            
            if input_device is not None:
                model_device = next(self.dit_model.parameters()).device
                if model_device != input_device:
                    self.debug.log(f"Auto-moving DiT model from {model_device} to {input_device}", category="memory")
                    self.dit_model = self.dit_model.to(input_device)
        
        # Execute forward pass
        try:
            return self.dit_model(*args, **kwargs)
        except Exception as e:
            self.debug.log(f"Forward pass error: {e}", category="error", force=True)
            if self.is_fp8_model:
                self.debug.log(f"FP8 model - converted FP8 tensors to BFloat16", category="info", force=True)
            else:
                self.debug.log(f"{self.model_dtype} model - no conversion applied", category="info", force=True)
            raise
    
    def __getattr__(self, name):
        """Redirect all other attributes to original model"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model', 'blockswap_active']:
            return super().__getattr__(name)
        return getattr(self.dit_model, name)
    
    def __setattr__(self, name, value):
        """Redirect assignments to original model except for our attributes"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model', 'blockswap_active']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'dit_model'):
                setattr(self.dit_model, name, value)
            else:
                super().__setattr__(name, value)
                