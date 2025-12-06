"""
Compatibility module for SeedVR2
Contains FP8/FP16 compatibility layers and wrappers for different model architectures

Extracted from: seedvr2.py (lines 1045-1630)
"""

# Triton compatibility shim for bitsandbytes 0.45+ with triton 3.0+
# Must be called before any diffusers import
import sys

def ensure_triton_compat():
    """Create minimal triton.ops stubs only if missing, to allow bitsandbytes import."""
    if 'triton.ops.matmul_perf_model' in sys.modules:
        return
    
    try:
        from triton.ops.matmul_perf_model import early_config_prune  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    
    import types
    
    if 'triton.ops' not in sys.modules:
        sys.modules['triton.ops'] = types.ModuleType('triton.ops')
    
    matmul_perf = types.ModuleType('triton.ops.matmul_perf_model')
    matmul_perf.early_config_prune = lambda configs, *a, **kw: configs
    matmul_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    
    sys.modules['triton.ops'].matmul_perf_model = matmul_perf
    sys.modules['triton.ops.matmul_perf_model'] = matmul_perf

# Run immediately on import
ensure_triton_compat()


import torch
import types
import os


# Flash Attention & Triton Compatibility Layer
# 1. Flash Attention - speedup for attention operations
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func = None
    FLASH_ATTN_AVAILABLE = False

def validate_flash_attention_availability(requested_mode: str, debug=None) -> str:
    """
    Validate Flash Attention availability and warn if fallback needed.
    
    Args:
        requested_mode: Either 'flash_attn' or 'sdpa'
        debug: Optional debug instance for logging
        
    Returns:
        Validated mode ('flash_attn' or 'sdpa')
    """
    if requested_mode == 'flash_attn' and not FLASH_ATTN_AVAILABLE:
        error_msg = (
            f"Cannot use 'flash_attn' attention mode: Flash Attention is not installed.\n"
            f"\n"
            f"Flash Attention provides speedup on some hardware through optimized CUDA kernels.\n"
            f"Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install Flash Attention: pip install flash-attn\n"
            f"  2. OR change attention_mode to 'sdpa' (default, always available)\n"
            f"\n"
            f"For more info: https://github.com/Dao-AILab/flash-attention"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        
        return 'sdpa'
    
    return requested_mode


# 2. Triton - Required for torch.compile with inductor backend
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# 3. GGUF - Required for quantized model loading
try:
    import gguf
    from gguf import GGMLQuantizationType
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None
    GGMLQuantizationType = None


def validate_gguf_availability(operation: str = "load GGUF model", debug=None) -> None:
    """
    Validate GGUF availability and raise error if not installed.
    
    Args:
        operation: Description of the operation requiring GGUF
        debug: Optional debug instance for logging
        
    Raises:
        RuntimeError: If GGUF is not available
    """
    if not GGUF_AVAILABLE:
        error_msg = (
            f"Cannot {operation}: GGUF library is not installed.\n"
            f"\n"
            f"GGUF provides quantized model support for memory-efficient loading.\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install GGUF: pip install gguf\n"
            f"  2. OR use a non-quantized model format (.safetensors)\n"
            f"\n"
            f"For more info: https://github.com/ggerganov/ggml"
        )
        if debug:
            debug.log(error_msg, level="ERROR", category="setup", force=True)
        raise RuntimeError(f"GGUF library required to {operation}")


# 4. NVIDIA Conv3d Memory Bug - Workaround for PyTorch 2.9-2.10 + cuDNN >= 91002
def _check_conv3d_memory_bug():
    """
    Check if Conv3d memory bug workaround needed.
    Bug: PyTorch 2.9-2.10 with cuDNN >= 91002 uses 3x memory for Conv3d 
    with fp16/bfloat16 due to buggy dispatch layer.
    """
    try:
        # Exclude AMD ROCm/HIP builds (they use MIOpen, not cuDNN)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return False
        
        # Must have CUDA available
        if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
            return False
        
        # Must have cuDNN actually available (not just the attribute)
        if not (hasattr(torch.backends.cudnn, 'is_available') and 
                torch.backends.cudnn.is_available()):
            return False
        
        # Check device capability (NVIDIA GPUs)
        if torch.cuda.get_device_capability()[0] < 3:
            return False
        
        # Parse torch version
        version_str = torch.__version__.split('+')[0]
        parts = version_str.split('.')
        torch_version = tuple(int(p) for p in parts[:2])
        
        if not ((2, 9) <= torch_version <= (2, 10)):
            return False
        
        if not hasattr(torch.backends.cudnn, 'version'):
            return False
            
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None or cudnn_version < 91002:
            return False
        
        return True
    except:
        return False

NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = _check_conv3d_memory_bug()


# Log all optimization status once globally (cross-process) using environment variable
if not os.environ.get("SEEDVR2_OPTIMIZATIONS_LOGGED"):
    os.environ["SEEDVR2_OPTIMIZATIONS_LOGGED"] = "1"
    
    # Flash Attention & Triton status
    has_both = FLASH_ATTN_AVAILABLE and TRITON_AVAILABLE
    has_neither = not FLASH_ATTN_AVAILABLE and not TRITON_AVAILABLE
    
    if has_both:
        print("⚡ SeedVR2 optimizations check: Flash Attention ✅ | Triton ✅")
    elif has_neither:
        print("⚠️  SeedVR2 optimizations check: Flash Attention ❌ | Triton ❌")
        print("💡 For best performance: pip install flash-attn triton")
    elif FLASH_ATTN_AVAILABLE:
        print("⚡ SeedVR2 optimizations check: Flash Attention ✅ | Triton ❌")
        print("💡 Install Triton for torch.compile: pip install triton")
    else:  # TRITON_AVAILABLE only
        print("⚠️  SeedVR2 optimizations check: Flash Attention ❌ | Triton ✅")
        print("💡 Install Flash Attention for faster inference: pip install flash-attn")
    
    # Conv3d workaround status (if applicable)
    if NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND:
        torch_ver = torch.__version__.split('+')[0]
        cudnn_ver = torch.backends.cudnn.version()
        print(f"🔧 Conv3d workaround active: PyTorch {torch_ver}, cuDNN {cudnn_ver} (fixing VAE 3x memory bug)")


# Bfloat16 CUBLAS support
def _probe_bfloat16_support() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda:0')
        _ = torch.matmul(a, a)
        del a
        return True
    except RuntimeError as e:
        if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
            return False
        raise

BFLOAT16_SUPPORTED = _probe_bfloat16_support()
COMPUTE_DTYPE = torch.bfloat16 if BFLOAT16_SUPPORTED else torch.float16


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
    
    Precision Handling:
    - FP8: Keeps native FP8 parameters (memory efficient), converts inputs/outputs to compute_dtype for arithmetic
    - FP16: Uses native FP16 precision throughout
    - BFloat16: Uses native BFloat16 precision throughout
    - Float32: Uses full precision for maximum quality
    - RoPE: Converted from FP8 to compute_dtype for numerical consistency
    
    Optimizations:
    - Flash Attention: Automatic optimization of attention layers
    - RoPE Stabilization: Error handling for numerical stability in mixed precision
    - MPS Compatibility: Unified dtype conversion for Apple Silicon backends
    """
    
    def __init__(self, dit_model, debug: 'Debug', compute_dtype: torch.dtype = torch.bfloat16, skip_conversion: bool = False):
        super().__init__()
        self.dit_model = dit_model
        self.debug = debug
        self.compute_dtype = compute_dtype
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16

        # Only convert if not already done (e.g., when reusing cached weights)
        if not skip_conversion and self.is_fp8_model:
            # FP8 models need RoPE frequency conversion to compute dtype
            model_variant = self._get_model_variant()
            self.debug.log(f"Detected NaDiT {model_variant} FP8 - Converting RoPE freqs for FP8 compatibility", 
                        category="precision")
            self.debug.start_timer("_convert_rope_freqs")
            self._convert_rope_freqs(target_dtype=self.compute_dtype)
            self.debug.end_timer("_convert_rope_freqs", "RoPE freqs conversion")
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.debug.log(f"Also converting NaDiT parameters/buffers for MPS backend", category="setup", force=True)
                self.debug.start_timer("_force_nadit_precision")
                self._force_nadit_precision(target_dtype=self.compute_dtype)
                self.debug.end_timer("_force_nadit_precision", "NaDiT parameters/buffers conversion")
            
        # Apply RoPE stabilization for numerical stability
        self.debug.log(f"Stabilizing RoPE computations for numerical stability", category="setup")
        self.debug.start_timer("_stabilize_rope_computations")
        self._stabilize_rope_computations()
        self.debug.end_timer("_stabilize_rope_computations", "RoPE stabilization")

        # 🚀 FLASH ATTENTION OPTIMIZATION (Phase 2)
        self.debug.start_timer("_apply_flash_attention_optimization")
        self._apply_flash_attention_optimization()
        self.debug.end_timer("_apply_flash_attention_optimization", "Flash Attention application")
    
    def _detect_model_dtype(self) -> torch.dtype:
        """Detect main model dtype"""
        try:
            return next(self.dit_model.parameters()).dtype
        except:
            return torch.bfloat16
    
    def _get_model_variant(self) -> str:
        """Detect model variant from module path"""
        model_module = str(self.dit_model.__class__.__module__).lower()
        if 'dit_7b' in model_module:
            return "7B"
        elif 'dit_3b' in model_module:
            return "3B"
        else:
            return "Unknown"
        
    def _convert_rope_freqs(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Convert RoPE frequency buffers from FP8 to target dtype for compatibility.
        
        Args:
            target_dtype: Target dtype for RoPE freqs (default: bfloat16 for stability)
        """
        converted = 0
        for module in self.dit_model.modules():
            if 'RotaryEmbedding' in type(module).__name__:
                if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
                    if module.rope.freqs.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        if module.rope.freqs.device.type == "mps":
                            module.rope.freqs.data = module.rope.freqs.to("cpu").to(target_dtype).to("mps")
                        else:
                            module.rope.freqs.data = module.rope.freqs.to(target_dtype)
                        converted += 1
        self.debug.log(f"Converted {converted} RoPE frequency buffers from FP8 to {target_dtype} for compatibility", category="success")
                        
    def _force_nadit_precision(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Force ALL NaDiT parameters to target dtype to avoid promotion errors (MPS requirement).
        
        Args:
            target_dtype: Target dtype for all parameters (default: bfloat16 for MPS compatibility)
        """
        converted_count = 0
        original_dtype = None
        
        # Convert ALL parameters to target dtype
        for name, param in self.dit_model.named_parameters():
            if original_dtype is None:
                original_dtype = param.dtype
            if param.dtype != target_dtype:
                if param.device.type == "mps":
                    temp_cpu = param.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    param.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    param.data = param.data.to(target_dtype)
                converted_count += 1
                
        # Also convert buffers
        for name, buffer in self.dit_model.named_buffers():
            if buffer.dtype != target_dtype:
                if buffer.device.type == "mps":
                    temp_cpu = buffer.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    buffer.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    buffer.data = buffer.data.to(target_dtype)
                converted_count += 1
        
        self.debug.log(f"Converted {converted_count} NaDiT parameters/buffers to {target_dtype} for MPS", category="success")
        
        # Update detected dtype
        self.model_dtype = target_dtype
        self.is_fp8_model = (target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2))

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
        # Check PyTorch SDPA (includes Flash Attention on H100/A100)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return True
        
        # Check flash-attn package (uses module-level check from top of file)
        return FLASH_ATTN_AVAILABLE
    
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
            self.debug.log(f"Failed to optimize attention layer '{name}': {e}", level="WARNING", category="dit", force=True)
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
                self.debug.log(f"Flash Attention failed for {layer_name}, using original: {e}", level="WARNING", category="dit", force=True)
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
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False
                )
            else:
                # Use optimized SDPA - PyTorch 2.3+ API with CUDNN support, fallback for older versions
                if hasattr(torch.nn.attention, 'sdpa_kernel'):
                    ctx = torch.nn.attention.sdpa_kernel([
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                        torch.nn.attention.SDPBackend.MATH])
                else:
                    ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
                
                with ctx:
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
        """
        Forward pass with minimal dtype conversion overhead
        
        Conversion strategy:
            - FP16/BFloat16/Float32 models: Use native precision (no conversion needed)
            - FP8 models: Convert FP8 tensors to compute_dtype for arithmetic operations
            (FP8 parameters stay in FP8 for memory efficiency, only converted for computation)
        """
        
        # Only convert if we have an FP8 model for arithmetic operations 
        if self.is_fp8_model:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            target_dtype = self.compute_dtype
            
            # Convert args
            converted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype in fp8_dtypes:
                    converted_args.append(arg.to(target_dtype))
                else:
                    converted_args.append(arg)
            
            # Convert kwargs
            converted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.dtype in fp8_dtypes:
                    converted_kwargs[key] = value.to(target_dtype)
                else:
                    converted_kwargs[key] = value
            
            args = tuple(converted_args)
            kwargs = converted_kwargs
        
        # Execute forward pass
        try:
            return self.dit_model(*args, **kwargs)
        except Exception as e:
            self.debug.log(f"Forward pass error: {e}", level="ERROR", category="generation", force=True)
            if self.is_fp8_model:
                self.debug.log(f"FP8 model - converted FP8 tensors to {self.compute_dtype}", category="info", force=True)
            else:
                self.debug.log(f"{self.model_dtype} model - no conversion applied", category="info", force=True)
            raise
    
    def __getattr__(self, name):
        """Redirect all other attributes to original model"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            return super().__getattr__(name)
        return getattr(self.dit_model, name)
    
    def __setattr__(self, name, value):
        """Redirect assignments to original model except for our attributes"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'dit_model'):
                setattr(self.dit_model, name, value)
            else:
                super().__setattr__(name, value)
                