"""
Model Management Module for SeedVR2

This module handles all model-related operations including:
- Model configuration and path resolution
- Model loading with format detection (SafeTensors, PyTorch)
- DiT and VAE model setup and inference configuration
- State dict management with native FP8 support
- Universal compatibility wrappers

Key Features:
- Dynamic import path resolution for different ComfyUI environments
- Native FP8 model support with optimal performance
- Automatic compatibility mode for model architectures
- Memory-efficient model loading and configuration
"""

import os
import time
import torch
from omegaconf import DictConfig, OmegaConf

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️ SafeTensors not available, recommended install: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

# Import GGUF with fallback
try:
    import gguf
    import warnings
    GGUF_AVAILABLE = True
except ImportError:
    print("⚠️ GGUF not available, recommended install: pip install gguf")
    GGUF_AVAILABLE = False

from src.optimization.memory_manager import get_basic_vram_info, clear_vram_cache
from src.optimization.compatibility import FP8CompatibleDiT
from src.optimization.memory_manager import preinitialize_rope_cache, clear_rope_lru_caches
from src.common.config import load_config, create_object
from src.core.infer import VideoDiffusionInfer
from src.optimization.blockswap import apply_block_swap_to_dit

# Import GGUF ops for quantized model support
try:
    from src.optimization.gguf_ops import apply_quantized_ops
    GGUF_OPS_AVAILABLE = True
except ImportError:
    GGUF_OPS_AVAILABLE = False

# Get script directory for config paths
script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global cache for state dicts when cache_model is enabled
_state_dict_cache = {}


def clear_state_dict_cache():
    """Clear the global state dict cache to free RAM"""
    global _state_dict_cache
    if _state_dict_cache:
        print(f"🧹 Clearing state dict cache ({len(_state_dict_cache)} entries)")
        # Properly cleanup tensors
        for cache_key, cached_state in _state_dict_cache.items():
            for key, tensor in cached_state.items():
                if hasattr(tensor, 'cpu'):
                    del tensor
        _state_dict_cache.clear()


def get_cache_info():
    """Get information about the current state dict cache"""
    if not _state_dict_cache:
        return "State dict cache is empty"
    
    info = []
    for cache_key, cached_state in _state_dict_cache.items():
        # Extract model name from cache key (first part before first underscore)
        model_name = cache_key.split('_')[0].split('/')[-1] if cache_key else "unknown"
        tensor_count = len(cached_state)
        info.append(f"  - {model_name}: {tensor_count} tensors")
    
    return f"State dict cache contains {len(_state_dict_cache)} entries:\n" + "\n".join(info)


def get_orig_shape(reader, tensor_name):
    """Get original tensor shape from GGUF metadata"""
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        return tensor

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.requires_grad_(False)  # Ensure no gradients
        return new

    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
    
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
    
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
        
        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            from src.optimization.gguf_dequant import dequantize_tensor as fast_dequantize
            result = fast_dequantize(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
            
            return final_result
        except Exception as e:
            print(f"❌ Fast dequantization failed: {e}")
            print(f"🔄 Falling back to numpy dequantization")
        
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            
            # Ensure we return a regular tensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
            
            return final_result
        except Exception as e:
            print(f"❌ Numpy fallback also failed: {e}")
            print(f"   Tensor type: {self.tensor_type}")
            print(f"   Shape: {self.shape}")
            print(f"   Target shape: {self.tensor_shape}")
            import traceback
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized, delegate to parent
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Unquantized, delegate to parent
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if this is a linear function call
        if func == torch.nn.functional.linear:
            # Dequantize the weight (self) before calling linear
            if len(args) >= 2 and args[1] is self:  # weight is the second argument
                try:
                    dequantized_weight = self.dequantize(device=args[0].device, dtype=args[0].dtype)
                    # Replace the weight with dequantized version
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    print(f"❌ Error in linear dequantization: {e}")
                    print(f"  Function: {func}")
                    print(f"  Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}")
                    raise
        
        # For tensor operations that might need dequantization, dequantize first
        if func in [torch.matmul, torch.mm, torch.bmm, torch.addmm]:
            try:
                dequantized_self = self.dequantize()
                # Replace self with dequantized version in args
                new_args = tuple(dequantized_self if arg is self else arg for arg in args)
                return func(*new_args, **kwargs)
            except Exception as e:
                print(f"❌ Error in {func.__name__} dequantization: {e}")
                raise
        
        # For other operations, delegate to parent without dequantization
        return super().__torch_function__(func, types, args, kwargs)


def load_gguf_state_dict(path, handle_prefix="model.diffusion_model.", device="cuda"):
    """
    Load GGUF state dict keeping tensors quantized for VRAM efficiency
    """
    if not GGUF_AVAILABLE:
        raise ImportError("GGUF required to load this model. Install with: pip install gguf")
    
    reader = gguf.GGUFReader(path)
    
    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # Load tensors while preserving quantization
    state_dict = {}
    print(f"🔄 Loading {len(tensors)} tensors directly to {device}...")
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Load tensor data with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            # Create tensor on CPU but immediately move to GPU to minimize RAM usage
            torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=True)
        
        # Get original shape from metadata or infer from tensor shape
        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting and memory management
        if i % 100 == 0:
            print(f"   Loaded {i+1}/{len(tensors)} tensors...")
            # Force garbage collection to minimize RAM accumulation
            import gc
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()
    
    print(f"✅ Successfully loaded {len(state_dict)} tensors to {device}")
    return state_dict


def configure_runner(model, base_cache_dir, preserve_vram=False, debug=False, block_swap_config=None, cached_runner=None):
    """
    Configure and create a VideoDiffusionInfer runner for the specified model
    
    Args:
        model (str): Model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir (str): Base directory containing model files
        preserve_vram (bool): Whether to preserve VRAM
        debug (bool): Enable debug logging
        block_swap_config (dict): Optional BlockSwap configuration
        cached_runner: Optional cached runner to reuse entirely (not just DiT)
        
    Returns:
        VideoDiffusionInfer: Configured runner instance ready for inference
        
    Features:
        - Dynamic config loading based on model type (3B vs 7B)
        - Automatic import path resolution for different environments
        - VAE configuration with proper parameter handling
        - Memory optimization and RoPE cache pre-initialization
    """

    # Check if we can fully reuse the cached runner
    # Note: This checks for runner-level caching. Additionally, state dicts are now
    # cached in RAM separately via _state_dict_cache when cache_model=True
    if cached_runner and block_swap_config and block_swap_config.get("cache_model", False):        
        # Clear RoPE caches before reuse
        if hasattr(cached_runner, 'dit'):
            dit_model = cached_runner.dit
            if hasattr(dit_model, 'dit_model'):
                dit_model = dit_model.dit_model
            clear_rope_lru_caches(dit_model)
        
        print(f"♻️ Reusing cached runner for {model}")
        
        # Check if blockswap needs to be applied
        blockswap_needed = block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
        
        if blockswap_needed:
            # Check if we have cached configuration
            has_cached_config = hasattr(cached_runner, "_cached_blockswap_config")
            
            if has_cached_config:
                # Compare configurations
                cached_config = cached_runner._cached_blockswap_config
                config_matches = (
                    cached_config.get("blocks_to_swap") == block_swap_config.get("blocks_to_swap") and
                    cached_config.get("offload_io_components") == block_swap_config.get("offload_io_components", False) and
                    cached_config.get("use_non_blocking") == block_swap_config.get("use_non_blocking", True)
                )
                
                if config_matches:
                    # Configuration matches - fast re-application
                    print("✅ BlockSwap config matches, performing fast re-application")
                    
                    # Mark as active before applying
                    cached_runner._blockswap_active = True
                    
                    # Apply BlockSwap (will be fast since model structure is intact)
                    apply_block_swap_to_dit(cached_runner, block_swap_config)
                else:
                    # Configuration changed - apply new config
                    print("🔄 BlockSwap configuration changed, applying new config")
                    apply_block_swap_to_dit(cached_runner, block_swap_config)
            else:
                # No cached config - apply fresh
                print("🔄 Applying BlockSwap to cached runner")
                apply_block_swap_to_dit(cached_runner, block_swap_config)
            
            return cached_runner
        else:
            # No BlockSwap needed
            return cached_runner
    
    # If we reach here, create a new runner
    t = time.time()
    vram_info = get_basic_vram_info()
    if debug:
        print(f"🔄 RUNNER : VRAM INFO: {vram_info}")
    # Select config based on model type


    
    if "7b" in model:
        config_path = os.path.join(script_directory, './configs_7b', 'main.yaml')
        if "fp8" in model:
            model_weight = "7b_fp8"
        elif model.endswith('.gguf'):
            model_weight = "7b_gguf"
        else:
            model_weight = "7b_fp16"
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
        if "fp8" in model:
            model_weight = "3b_fp8"
        elif model.endswith('.gguf'):
            model_weight = "3b_gguf"
        else:
            model_weight = "3b_fp16"
    
    config = load_config(config_path)
    if debug:
        print(f"🔄 RUNNER : CONFIG LOAD TIME: {time.time() - t} seconds")
    # DiT model configuration is now handled directly in the YAML config files
    # No need for dynamic path resolution here anymore!

    # Load and configure VAE with additional parameters
    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    t = time.time()
    vae_config = OmegaConf.load(vae_config_path)
    if debug:
        print(f"🔄 RUNNER : VAE CONFIG LOAD TIME: {time.time() - t} seconds")
    
    t = time.time()
    # Configure VAE parameters
    spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    if debug:
        print(f"🔄 RUNNER : VAE CONFIG SET TIME: {time.time() - t} seconds")
    
    # Merge additional VAE config with main config (preserving __object__ from main config)
    t = time.time()
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    if debug:
        print(f"🔄 RUNNER : VAE CONFIG MERGE TIME: {time.time() - t} seconds")
    
    t = time.time()
    # Create runner
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    # Store model name for cache validation
    runner._model_name = model
    # Store GGUF flag for dtype handling
    runner._is_gguf_model = model.endswith('.gguf')
    if debug:
        print(f"🔄 RUNNER : RUNNER VIDEO DIFFUSION INFER TIME: {time.time() - t} seconds")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure models
    checkpoint_path = os.path.join(base_cache_dir, f'./{model}')
    t = time.time()
    runner = configure_dit_model_inference(runner, device, checkpoint_path, config, preserve_vram, model_weight, vram_info, debug, block_swap_config)
    if debug:
        print(f"🔄 RUNNER : DIT MODEL INFERENCE TIME: {time.time() - t} seconds")
    
    t = time.time()
    checkpoint_path = os.path.join(base_cache_dir, f'./{config.vae.checkpoint}')
    runner = configure_vae_model_inference(runner, device, checkpoint_path, config, preserve_vram, model_weight, vram_info, debug, block_swap_config)
    if debug:
        print(f"🔄 RUNNER : VAE MODEL INFERENCE TIME: {time.time() - t} seconds")
    
    t = time.time()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    if debug:
        print(f"🔄 RUNNER : VAE MEMORY LIMIT TIME: {time.time() - t} seconds")
    
    # Check if BlockSwap is active
    blockswap_active = (
        block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    )
    
    # Pre-initialize RoPE cache for optimal performance if BlockSwap is NOT active
    if not blockswap_active:
        t = time.time()
        preinitialize_rope_cache(runner)
        if debug:
            print(f"🔄 RUNNER : ROPE CACHE PREINITIALIZE TIME: {time.time() - t} seconds")
    else:
        if debug:
            print(f"🔄 RUNNER : Skipping RoPE pre-init due to BlockSwap")
    
    # Apply BlockSwap if configured
    if blockswap_active:
        apply_block_swap_to_dit(runner, block_swap_config)
    
    # Log cache status
    cache_model = block_swap_config and block_swap_config.get("cache_model", False)
    if cache_model:
        print(f"💾 Model caching enabled - state dicts will persist in RAM between runs")
        if debug:
            print(f"💾 Current cache status: {get_cache_info()}")
    
    #clear_vram_cache()
    return runner


def load_quantized_state_dict(checkpoint_path, device="cpu", keep_native_fp8=True, cache_enabled=False):
    """
    Load state dict from SafeTensors, PyTorch, or GGUF with optimal format support
    
    Args:
        checkpoint_path (str): Path to model checkpoint (.safetensors, .pth, or .gguf)
        device (str): Target device for loading
        keep_native_fp8 (bool): Whether to preserve native FP8 format for performance
        cache_enabled (bool): Whether to use RAM caching for state dict
        
    Returns:
        dict: State dictionary with optimal dtype handling
        
    Features:
        - Automatic format detection (SafeTensors vs PyTorch vs GGUF)
        - Native FP8 preservation for 2x speedup and 50% VRAM reduction
        - GGUF quantization support without type conversion
        - Intelligent dtype conversion when needed for compatibility
        - Memory-mapped loading for large models
        - RAM caching for faster subsequent loads when cache_enabled=True
    """
    # Check cache first if caching is enabled
    if cache_enabled:
        cache_key = f"{checkpoint_path}_{device}_{keep_native_fp8}"
        print(f"🔍 Cache check - cache_enabled: {cache_enabled}, cache_key: {os.path.basename(checkpoint_path)}")
        if cache_key in _state_dict_cache:
            print(f"⚡ CACHE HIT: Loading {os.path.basename(checkpoint_path)} from RAM cache (skipping disk I/O)")
            cached_state = _state_dict_cache[cache_key]
            
            # Create a copy of cached state dict with proper device placement
            state = {}
            for key, tensor in cached_state.items():
                if hasattr(tensor, 'to'):
                    # Move tensor to target device
                    state[key] = tensor.to(device)
                else:
                    # For non-tensor objects, just copy the reference
                    state[key] = tensor
            return state
        else:
            print(f"🔍 Cache miss - will load from disk and cache: {os.path.basename(checkpoint_path)}")
    else:
        print(f"🔍 Cache disabled - loading from disk: {os.path.basename(checkpoint_path)}")
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors required to load this model. Install with: pip install safetensors")
        state = load_safetensors_file(checkpoint_path, device=device)
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    elif checkpoint_path.endswith('.gguf'):
        if not GGUF_AVAILABLE:
            raise ImportError("GGUF required to load this model. Install with: pip install gguf")
        # Load GGUF model using our custom loader
        # GGUF models maintain their quantization level and should not be type converted
        state = load_gguf_state_dict(checkpoint_path, handle_prefix="model.diffusion_model.", device=device)
        print(f"🚀 Loaded GGUF model: {checkpoint_path}")
        return state
    else:
        raise ValueError(f"Unsupported format. Expected .safetensors, .pth, or .gguf, got: {checkpoint_path}")
    
    # FP8 optimization: Keep native format for maximum performance
    fp8_detected = False
    fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2) if hasattr(torch, 'float8_e4m3fn') else ()
    
    if fp8_types:
        # Check if model contains FP8 tensors
        for key, tensor in state.items():
            if hasattr(tensor, 'dtype') and tensor.dtype in fp8_types:
                fp8_detected = True
                break
    
    if fp8_detected:
        if keep_native_fp8:
            # Keep native FP8 format for optimal performance
            # Benefits: ~50% less VRAM, ~2x faster inference
            return state
        else:
            # Convert FP8 → BFloat16 for compatibility
            converted_state = {}
            converted_count = 0
            
            for key, tensor in state.items():
                if hasattr(tensor, 'dtype') and tensor.dtype in fp8_types:
                    converted_state[key] = tensor.to(torch.bfloat16)
                    converted_count += 1
                else:
                    converted_state[key] = tensor
            
            return converted_state
    
    # Cache the state dict if caching is enabled
    if cache_enabled:
        cache_key = f"{checkpoint_path}_{device}_{keep_native_fp8}"
        if cache_key not in _state_dict_cache:
            print(f"💾 CACHING: Storing {os.path.basename(checkpoint_path)} state dict in RAM for future runs")
            # Store a copy of the state dict in CPU memory for caching
            cached_state = {}
            for key, tensor in state.items():
                if hasattr(tensor, 'cpu'):
                    # Move to CPU for storage, preserve all attributes
                    cpu_tensor = tensor.cpu()
                    # Preserve special attributes for GGUF tensors
                    if hasattr(tensor, 'tensor_type'):
                        cpu_tensor.tensor_type = tensor.tensor_type
                    if hasattr(tensor, 'tensor_shape'):
                        cpu_tensor.tensor_shape = tensor.tensor_shape
                    cached_state[key] = cpu_tensor
                else:
                    # For non-tensor objects, just copy the reference
                    cached_state[key] = tensor
            _state_dict_cache[cache_key] = cached_state
    
    return state



def configure_dit_model_inference(runner, device, checkpoint, config, preserve_vram=False, model_weight=None, vram_info=None, debug=False, block_swap_config=None):
    """
    Configure DiT model for inference without distributed decorators
    
    Args:
        runner: VideoDiffusionInfer instance
        device (str): Target device
        checkpoint (str): Path to model checkpoint
        config: Model configuration
        block_swap_config (dict): Optional BlockSwap configuration
        
    Features:
        - Automatic format detection and optimal loading
        - Native FP8 support with universal compatibility wrapper
        - Gradient checkpointing configuration
        - Intelligent dtype handling for all model architectures
        - BlockSwap support for low VRAM systems
    """
    
    # Create dit model
    t = time.time()

    # Check if BlockSwap is active
    blockswap_active = (
        block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    )

    # Check if cache_model is active
    cache_model = (
        block_swap_config and block_swap_config.get("cache_model", False)
    )
    
    if debug:
        print(f"🔄 CONFIG DIT : cache_model setting: {cache_model}")
        print(f"🔄 CONFIG DIT : block_swap_config: {block_swap_config}")
        print(f"🔄 CONFIG DIT : Current cache entries: {len(_state_dict_cache)}")
        if _state_dict_cache:
            for key in _state_dict_cache.keys():
                model_name = key.split('_')[0].split('/')[-1]
                print(f"🔄 CONFIG DIT :   - Cached: {model_name}")
                break  # Just show one example

    # For GGUF models, ALWAYS create on CPU since we'll replace all parameters anyway
    # This prevents the 30GB+ VRAM spike from uninitialized parameters
    is_gguf = checkpoint.endswith('.gguf')
    loading_device = "cpu" if (preserve_vram or blockswap_active or is_gguf) else device
    
    if (blockswap_active or is_gguf) and debug:
        reason = "BlockSwap active" if blockswap_active else "GGUF model"
        print(f"🔄 CONFIG DIT : {reason} - creating model on CPU")

    with torch.device(loading_device):
        runner.dit = create_object(config.dit.model)
    # Passer les opérations au modèle

    if debug:
        print(f"🔄 CONFIG DIT : MODEL CREATE TIME: {time.time() - t} seconds device: {device}")
    t = time.time()
    runner.dit.set_gradient_checkpointing(config.dit.gradient_checkpoint)

    # Detect and log model format
    print(f"🚀 Loading model_weight: {model_weight}")

    t = time.time()
    # For GGUF models, load directly to GPU to avoid RAM usage
    if checkpoint.endswith('.gguf'):
        state_loading_device = device  # Load directly to GPU
    else:
        state_loading_device = "cpu" if "7b" in model_weight and vram_info['total_gb'] < 25 else device
    
    # For GGUF models, keep_native_fp8 should be False since they have their own quantization
    keep_native_fp8 = not checkpoint.endswith('.gguf')
    state = load_quantized_state_dict(checkpoint, state_loading_device, keep_native_fp8=keep_native_fp8, cache_enabled=cache_model)

    if debug:
        print(f"🔄 CONFIG DIT : DiT load state dict time: {time.time() - t} seconds")
    t = time.time()
    # Handle GGUF models with custom loading approach
    if checkpoint.endswith('.gguf'):
        print("🔄 Loading GGUF model - keeping tensors quantized...")
        
        # Check for architecture mismatch first
        model_state = runner.dit.state_dict()
        
        # Check a few key parameters to detect architecture mismatch
        key_params_to_check = [
            "blocks.0.attn.proj_qkv.vid.weight",
            "blocks.0.attn.proj_qkv.txt.weight", 
            "blocks.0.mlp.vid.proj_in.weight"
        ]
        
        architecture_mismatch = False
        for key in key_params_to_check:
            if key in state and key in model_state:
                model_shape = model_state[key].shape
                gguf_param = state[key]
                
                # Use tensor_shape if available, otherwise use shape
                gguf_shape = gguf_param.tensor_shape if hasattr(gguf_param, 'tensor_shape') else gguf_param.shape
                
                if model_shape != gguf_shape:
                    print(f"❌ Shape mismatch detected!")
                    print(f"   Parameter: {key}")
                    print(f"   Model expects: {model_shape}")
                    print(f"   GGUF provides: {gguf_shape}")
                    
                    # Check if this is a quantization issue rather than architecture mismatch
                    if hasattr(gguf_param, 'tensor_type') and gguf_param.tensor_type not in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                        print(f"   This is a quantized tensor - the shape difference might be due to quantization")
                        print(f"   Raw tensor shape: {gguf_param.shape}")
                        print(f"   Logical tensor shape: {gguf_shape}")
                        
                        # For quantized tensors, we should use the logical shape, not the raw shape
                        if hasattr(gguf_param, 'tensor_shape') and gguf_param.tensor_shape == model_shape:
                            print(f"   ✅ Logical shapes match, continuing with quantized tensor")
                            continue
                    
                    architecture_mismatch = True
                    break
        
        print(f"🔍 Architecture check complete. Mismatch: {architecture_mismatch}")
        
        if architecture_mismatch:
            error_msg = (
                f"GGUF model architecture mismatch: This GGUF model is incompatible with the current 7B architecture.\n\n"
                f"Possible solutions:\n"
                f"1. Use a GGUF model that matches the 7B architecture (3072 dimensions)\n"
                f"2. The current GGUF model appears to be from a 3B model (1728 dimensions)\n"
                f"3. Try using a regular FP16 model instead: 'seedvr2_ema_7b_fp16.safetensors'\n"
                f"4. Check if you have a compatible GGUF model for the 7B architecture\n\n"
                f"The model configuration is expecting 7B dimensions but the GGUF provides different dimensions."
            )
            raise ValueError(error_msg)
        
        # If we get here, shapes should match - load without converting
        loaded_params = set()
        quantized_params = 0
        
        for name, param in state.items():
            if name in model_state:
                model_param = model_state[name]
                
                # Verify shape match
                param_shape = param.tensor_shape if hasattr(param, 'tensor_shape') else param.shape
                if param_shape == model_param.shape:
                    # Debug output for parameter loading
                    if hasattr(param, 'tensor_type') and debug:
                        print(f"📝 Loading quantized parameter: {name}")
                        print(f"   Shape: {param_shape}")
                        print(f"   Type: {param.tensor_type}")
                    
                    # Replace the parameter with quantized version - NO CONVERSION
                    with torch.no_grad():
                        if hasattr(param, 'tensor_type'):
                            # Keep quantized tensor as-is - navigate to the actual parameter
                            module = runner.dit
                            param_path = name.split('.')
                            
                            # Navigate to the parent module
                            for attr in param_path[:-1]:
                                module = getattr(module, attr)
                            
                            # Replace the parameter directly - preserve GGUFTensor attributes
                            param_name = param_path[-1]
                            
                            # Create a parameter that preserves GGUFTensor attributes
                            if hasattr(param, 'tensor_type') and hasattr(param, 'tensor_shape'):
                                # Create a new parameter but preserve the GGUFTensor attributes
                                # Use the tensor directly, not .data, to avoid triggering dequantization
                                new_param = torch.nn.Parameter(param, requires_grad=False)
                                new_param.tensor_type = param.tensor_type
                                new_param.tensor_shape = param.tensor_shape
                                
                                # Add custom dequantize method to the parameter
                                # We need to capture the original tensor and its methods
                                original_tensor = param
                                def gguf_dequantize(device=None, dtype=torch.float16):
                                    # Use the original GGUFTensor's dequantize method
                                    if hasattr(original_tensor, 'dequantize'):
                                        return original_tensor.dequantize(device, dtype)
                                    else:
                                        # Fallback: manually dequantize using gguf
                                        try:
                                            import gguf
                                            numpy_data = original_tensor.cpu().numpy()
                                            dequantized = gguf.quants.dequantize(numpy_data, original_tensor.tensor_type)
                                            result = torch.from_numpy(dequantized).to(device, dtype)
                                            result.requires_grad_(False)
                                            result = result.reshape(original_tensor.tensor_shape)
                                            return result
                                        except Exception as e:
                                            print(f"Warning: Could not dequantize tensor: {e}")
                                            return original_tensor.to(device, dtype)
                                new_param.gguf_dequantize = gguf_dequantize
                                
                                setattr(module, param_name, new_param)
                            else:
                                setattr(module, param_name, torch.nn.Parameter(param, requires_grad=False))
                            
                            quantized_params += 1
                        else:
                            # Regular tensor copy
                            model_param.copy_(param)
                    loaded_params.add(name)
                else:
                    print(f"❌ Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}")
                    raise ValueError(f"Shape mismatch for parameter {name}")
        
        print(f"✅ GGUF loading complete: {len(loaded_params)} parameters loaded")
        print(f"📊 Quantized parameters: {quantized_params}")
        print(f"🔥 VRAM savings: Tensors kept in quantized format")
        
        # Debug: check for shape mismatches
        unmatched_params = []
        for name, param in state.items():
            if name not in model_state:
                unmatched_params.append(name)
        if unmatched_params:
            print(f"⚠️ Warning: {len(unmatched_params)} parameters from GGUF not found in model")
            print(f"   First few unmatched: {unmatched_params[:5]}")
        
        missing_params = []
        for name in model_state:
            if name not in loaded_params:
                missing_params.append(name)
        if missing_params:
            print(f"⚠️ Warning: {len(missing_params)} model parameters not loaded from GGUF")
            print(f"   First few missing: {missing_params[:5]}")
        success_rate = 1.0  # If we get here, loading was successful

        # Move model to GPU after GGUF loading (if not preserving VRAM and no BlockSwap)
        if not preserve_vram and not blockswap_active:
            print("🔄 Moving GGUF model from CPU to GPU...")
            runner.dit = runner.dit.to(device)

        
    else:
        # Standard loading for non-GGUF models
        missing_keys, unexpected_keys = runner.dit.load_state_dict(state, strict=True, assign=True)
        success_rate = 1.0  # Assume success for non-GGUF
        
        if missing_keys or unexpected_keys:
            print(f"⚠️ Model loading issues:")
            if missing_keys:
                print(f"  Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys[:5]}...")

    if 'state' in locals():
        del state
            
    if debug:
        print(f"🔄 CONFIG DIT : DiT load time: {time.time() - t} seconds")
    #state.to("cpu")
    #runner.dit = runner.dit.to(device)

    # Apply quantized operations for GGUF models
    if checkpoint.endswith('.gguf') and GGUF_OPS_AVAILABLE:
        t = time.time()
        print("🔧 Skipping quantized operations replacement - using direct parameter approach")
        # runner.dit = apply_quantized_ops(runner.dit)
        # print("🔧 Applied quantized operations for GGUF model")
        if debug:
            print(f"🔄 CONFIG DIT : GGUF quantized ops time: {time.time() - t} seconds")
    elif checkpoint.endswith('.gguf'):
        print("⚠️ GGUF ops not available - model will run with standard operations")
        print("   Install missing dependencies for optimal GGUF performance")

    # Apply universal compatibility wrapper to ALL models
    # This ensures RoPE compatibility and optimal performance across all architectures
    t = time.time()
    # Check if already wrapped to avoid double wrapping
    if not isinstance(runner.dit, FP8CompatibleDiT):
        runner.dit = FP8CompatibleDiT(runner.dit, skip_conversion=False)
    if debug:
        print(f"🔄 CONFIG DIT : FP8CompatibleDiT time: {time.time() - t} seconds")

    # Move DiT to CPU to prevent VRAM leaks (especially for 3B model with complex RoPE)
    if preserve_vram and not blockswap_active:
        if debug:
            print(f"🔄 CONFIG DIT : dit to cpu cause preserve_vram: {preserve_vram}")
        runner.dit = runner.dit.to("cpu")
        if "7b" in model_weight:
            clear_vram_cache()
    else:
        if state_loading_device == "cpu" and not blockswap_active:
            runner.dit.to(device)

    # Log BlockSwap status if active
    if blockswap_active and debug:
        print(f"🔄 CONFIG DIT : BlockSwap active ({block_swap_config.get('blocks_to_swap', 0)} blocks) - placement handled by BlockSwap")

    return runner


def configure_vae_model_inference(runner, device, checkpoint_path, config, preserve_vram=False, model_weight=None, vram_info=None, debug=False, block_swap_config=None):
    """
    Configure VAE model for inference without distributed decorators
    
    Args:
        runner: VideoDiffusionInfer instance  
        config: Model configuration
        device (str): Target device
        block_swap_config (dict): Optional BlockSwap configuration
        
    Features:
        - Dynamic path resolution for VAE checkpoints
        - SafeTensors and PyTorch format support
        - FP8 and FP16 VAE handling
        - Causal slicing configuration
    """
    
    # Create vae model
    
    dtype = getattr(torch, config.vae.dtype)
    t = time.time()
    loading_device = "cpu" if preserve_vram else device
    
    with torch.device(device):
        runner.vae = create_object(config.vae.model)
    if debug:
        print(f"🔄 CONFIG VAE : MODEL CREATE TIME: {time.time() - t} seconds device: {device} dtype: {dtype}")
    t = time.time()
    runner.vae.requires_grad_(False).eval()
    if debug:
        print(f"🔄 CONFIG VAE : MODEL REQUIRES GRAD TIME: {time.time() - t} seconds device: {device} dtype: {dtype}")
    t = time.time()
    
    #runner.vae.to(device=loading_device, dtype=dtype)
    #if debug:
    #    print(f"🔄 CONFIG VAE : TO CPU TIME: {time.time() - t} seconds device: {device} dtype: {dtype}")
    # Resolve VAE checkpoint path dynamically
    '''
    checkpoint_path = config.vae.checkpoint
    
    possible_paths = [
        checkpoint_path,  # Original path
        os.path.join("ComfyUI", checkpoint_path),  # With ComfyUI prefix
        os.path.join(script_directory, checkpoint_path),  # Relative to script directory
        os.path.join(script_directory, "..", "..", checkpoint_path),  # From ComfyUI root
    ]
    t = time.time()
    vae_checkpoint_path = None
    for path in possible_paths:
        if os.path.exists(path):
            vae_checkpoint_path = path
            if debug:
                print(f"🔄 CONFIG VAE : Found VAE checkpoint at: {vae_checkpoint_path}")
            break
    if debug:
        print(f"🔄 CONFIG VAE : VAE CHECKPOINT PATH TIME: {time.time() - t} seconds")
    if vae_checkpoint_path is None:
        raise FileNotFoundError(f"VAE checkpoint not found. Tried paths: {possible_paths}")
    '''
    # Load VAE with format detection
    t = time.time()
    state_loading_device = "cpu" if "7b" in model_weight and vram_info['total_gb'] < 25 else device
    print(f"🚀 Loading VAE: {checkpoint_path}")
    # Check if VAE caching is enabled (use same cache_model setting from block_swap_config)
    vae_cache_enabled = block_swap_config and block_swap_config.get("cache_model", False)
    
    # Use optimized loading for all formats
    if "fp8_e4m3fn" in checkpoint_path:
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=True, cache_enabled=vae_cache_enabled)
    elif checkpoint_path.endswith('.gguf'):
        # For GGUF models, disable native FP8 since they have their own quantization
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=False, cache_enabled=vae_cache_enabled)
    else:
        # For FP16 SafeTensors, disable native FP8
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=False, cache_enabled=vae_cache_enabled)

    if debug:
        print(f"🔄 CONFIG VAE : VAE LOAD TIME: {time.time() - t} seconds")
    t = time.time()
    runner.vae.load_state_dict(state)
    
    # Ensure VAE dtype matches the target computation dtype
    if state_loading_device == "cpu":
        runner.vae.to(device)
    
    # For GGUF models, ensure VAE uses BFloat16 for compatibility with quantized models
    if checkpoint_path.endswith('.gguf'):
        runner.vae = runner.vae.to(torch.bfloat16)
        if debug:
            print(f"🔧 Converted VAE to BFloat16 for GGUF model compatibility")
    
    if 'state' in locals():
        del state
    if debug:
        print(f"🔄 CONFIG VAE : VAE LOAD STATE DICT TIME: {time.time() - t} seconds")

    # Set causal slicing if available
    t = time.time()
    if hasattr(runner.vae, "set_causal_slicing") and hasattr(config.vae, "slicing"):
        runner.vae.set_causal_slicing(**config.vae.slicing)

    if debug:
        print(f"🔄 CONFIG VAE : VAE SET CAUSAL SLICING TIME: {time.time() - t} seconds")

    return runner
    #runner.vae.to("cpu")