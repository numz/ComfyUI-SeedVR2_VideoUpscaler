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
import gc
import time
import torch
from src.utils.constants import get_script_directory
from omegaconf import DictConfig, OmegaConf

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️ SafeTensors not available, recommended install: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

try:
    import gguf
    import warnings
    import traceback
    from src.optimization.gguf_dequant import dequantize_tensor as fast_dequantize
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
from src.common.distributed import get_device

# Import GGUF ops for quantized model support
try:
    from src.optimization.gguf_ops import apply_quantized_ops
    GGUF_OPS_AVAILABLE = True
except ImportError:
    GGUF_OPS_AVAILABLE = False

# Get script directory for config paths
script_directory = get_script_directory()


def configure_runner(model, base_cache_dir, preserve_vram=False, debug=None, 
                    cache_model=False, block_swap_config=None, cached_runner=None, vae_tiling_enabled=False,
                    vae_tile_size=None, vae_tile_overlap=None):
    """
    Configure and create a VideoDiffusionInfer runner for the specified model
    
    Args:
        model (str): Model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir (str): Base directory containing model files
        preserve_vram (bool): Whether to preserve VRAM
        debug: Debug instance for logging
        cache_model (bool): Enable model caching
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
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Check if we can reuse the cached runner
    if cached_runner and cache_model:
        # Update all runtime parameters dynamically
        runtime_params = {
            'vae_tiling_enabled': vae_tiling_enabled,
            'vae_tile_size': vae_tile_size,
            'vae_tile_overlap': vae_tile_overlap
        }
        for key, value in runtime_params.items():
            setattr(cached_runner, key, value)
        
        # Clear RoPE caches before reuse
        if hasattr(cached_runner, 'dit'):
            dit_model = cached_runner.dit
            if hasattr(dit_model, 'dit_model'):
                dit_model = dit_model.dit_model
            clear_rope_lru_caches(dit_model)
        
        # Clear runner cache to prevent accumulation
        if hasattr(cached_runner, 'cache') and hasattr(cached_runner.cache, 'cache'):
            cached_runner.cache.cache.clear()
        
        debug.log(f"Cache hit: Reusing runner for model {model}", category="reuse", force=True)
        
        # Check if blockswap needs to be applied
        blockswap_needed = block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0

        # Sets _blockswap_active for CLI usage
        if blockswap_needed and not hasattr(cached_runner, "_blockswap_active"):
            cached_runner._blockswap_active = True

        if blockswap_needed:
            # Check if configuration changed (compare entire dicts)
            cached_config = getattr(cached_runner, "_cached_blockswap_config", None)
            config_matches = (cached_config == block_swap_config)
            
            if config_matches:
                # Configuration matches - fast re-application
                debug.log("BlockSwap config matches, performing fast re-application", category="reuse", force=True)
                cached_runner._blockswap_active = True
                apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
            else:
                # Configuration changed or new - apply config
                debug.log("Applying BlockSwap to cached runner", category="blockswap", force=True)
                apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
                cached_runner._cached_blockswap_config = block_swap_config.copy() if block_swap_config else None
            
            # Store debug instance on runner
            cached_runner.debug = debug
            return cached_runner
        else:
            # No BlockSwap needed
            cached_runner.debug = debug
            return cached_runner
        
    else:
        debug.log(f"Cache miss: Creating new runner for model {model}", category="cache", force=True)
    
    # If we reach here, create a new runner
    debug.start_timer("config_load")
    vram_info = get_basic_vram_info()

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
    debug.end_timer("config_load", "Config loaded")
    # DiT model configuration is now handled directly in the YAML config files
    # No need for dynamic path resolution here anymore!

    # Load and configure VAE with additional parameters
    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    debug.start_timer("vae_config_load")
    vae_config = OmegaConf.load(vae_config_path)
    debug.end_timer("vae_config_load", "VAE configuration YAML parsed from disk")
    
    debug.start_timer("vae_config_set")
    # Configure VAE parameters
    spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    debug.end_timer("vae_config_set", f"VAE downsample factors configured (spatial: {spatial_downsample_factor}x, temporal: {temporal_downsample_factor}x)")
    
    # Merge additional VAE config with main config (preserving __object__ from main config)
    debug.start_timer("vae_config_merge")
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    debug.end_timer("vae_config_merge", "VAE config merged with main pipeline config")
    
    debug.start_timer("runner_video_infer")
    # Create runner
    runner = VideoDiffusionInfer(config, debug, vae_tiling_enabled=vae_tiling_enabled, vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap)
    OmegaConf.set_readonly(runner.config, False)
    # Store model name for cache validation
    runner._model_name = model
    runner._is_gguf_model = model.endswith('.gguf')
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialized")
    
    # Set device
    device = str(get_device())
    #if torch.mps.is_available():
    #    device = "mps"
    
    # Configure models
    checkpoint_path = os.path.join(base_cache_dir, f'./{model}')
    debug.start_timer("dit_model_infer")
    runner = configure_dit_model_inference(runner, device, checkpoint_path, config, preserve_vram, model_weight, vram_info, debug, block_swap_config)

    debug.end_timer("dit_model_infer", "DiT model configured")
    
    debug.start_timer("vae_model_infer")
    checkpoint_path = os.path.join(base_cache_dir, f'./{config.vae.checkpoint}')
    runner = configure_vae_model_inference(runner, device, checkpoint_path, config, preserve_vram, model_weight, vram_info, debug)

    debug.end_timer("vae_model_infer", "VAE model configured")
    
    debug.start_timer("vae_memory_limit")
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    debug.end_timer("vae_memory_limit", "VAE memory limit set")
    
    # Check if BlockSwap is active
    blockswap_active = (
        block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    )
    
    # Pre-initialize RoPE cache for optimal performance if BlockSwap is NOT active
    if not blockswap_active:
        debug.start_timer("rope_cache_preinit")
        preinitialize_rope_cache(runner, debug)
        debug.end_timer("rope_cache_preinit", "RoPE cache pre-initialized")
    else:
        debug.log("Skipping RoPE cache pre-initialization (BlockSwap handles RoPE on-demand to save memory)", category="info")
    
    # Apply BlockSwap if configured
    if blockswap_active:
        apply_block_swap_to_dit(runner, block_swap_config, debug)
    #clear_vram_cache()
    
    # Store debug instance on runner for consistent access
    runner.debug = debug
    
    return runner


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
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.debug = getattr(self, "debug", None)
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
            result = fast_dequantize(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", category="warning")
            self.debug.log(f"Falling back to numpy dequantization", category="model")
            
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
            self.debug.log(f"Numpy fallback also failed: {e}", category="warning")
            self.debug.log(f"   Tensor type: {self.tensor_type}")
            self.debug.log(f"   Shape: {self.shape}")
            self.debug.log(f"   Target shape: {self.tensor_shape}")
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
                    self.debug.log(f"Error in linear dequantization: {e}", category="warning")
                    self.debug.log(f"  Function: {func}")
                    self.debug.log(f"  Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}")
                    raise
                    
        # For tensor operations that might need dequantization, dequantize first
        if func in [torch.matmul, torch.mm, torch.bmm, torch.addmm]:
            try:
                dequantized_self = self.dequantize()
                # Replace self with dequantized version in args
                new_args = tuple(dequantized_self if arg is self else arg for arg in args)
                return func(*new_args, **kwargs)
            except Exception as e:
                self.debug.log(f"Error in {func.__name__} dequantization: {e}", category="warning")
                raise
                
        # For other operations, delegate to parent without dequantization
        return super().__torch_function__(func, types, args, kwargs)
    
    
def load_gguf_state_dict(path, handle_prefix="model.diffusion_model.", device="cpu", debug=None):
    """
    Load GGUF state dict keeping tensors quantized for VRAM efficiency
    """
    if not GGUF_AVAILABLE:
        raise ImportError("GGUF required to load this model. Install with: pip install gguf")
    
    if debug is None:
        raise ValueError("Debug instance must be provided to load_gguf_state_dict")
        
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
    debug.log(f"Loading {len(tensors)} tensors directly to {device}...", category="model")
    
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
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting and memory management
        if i % 100 == 0:
            debug.log(f"   Loaded {i+1}/{len(tensors)} tensors...")
            # Force garbage collection to minimize RAM accumulation
            gc.collect()
            if device != "cpu":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.mps.is_available():
                    torch.mps.empty_cache()
                
    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device}", category="success")
    return state_dict


def load_quantized_state_dict(checkpoint_path, device="cpu", keep_native_fp8=True, debug=None):
    """
    Load state dict from SafeTensors, PyTorch, or GGUF with optimal format support
    
    Args:
        checkpoint_path (str): Path to model checkpoint (.safetensors, .pth, or .gguf)
        device (str): Target device for loading
        keep_native_fp8 (bool): Whether to preserve native FP8 format for performance
        
    Returns:
        dict: State dictionary with optimal dtype handling
        
    Features:
        - Automatic format detection (SafeTensors vs PyTorch vs GGUF)
        - Native FP8 preservation for 2x speedup and 50% VRAM reduction
        - GGUF quantization support without type conversion
        - Intelligent dtype conversion when needed for compatibility
        - Memory-mapped loading for large models
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to load_quantized_state_dict")
        
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
        state = load_gguf_state_dict(checkpoint_path, handle_prefix="model.diffusion_model.", device=device, debug=debug)
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
    
    return state



def configure_dit_model_inference(runner, device, checkpoint, config, 
                                 preserve_vram=False, model_weight=None, 
                                 vram_info=None, debug=None, block_swap_config=None):
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
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_dit_model_inference")
    
    # Create dit model
    debug.start_timer("dit_model_create")

    # Check if BlockSwap is active
    blockswap_active = (
        block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    )

    is_gguf = checkpoint.endswith('.gguf')
    loading_device = "cpu" if (preserve_vram or blockswap_active or is_gguf) else device
    if blockswap_active:
        debug.log("Creating DiT model on CPU for BlockSwap (will swap blocks to GPU during inference)", category="model", force=True)
    else:
        debug.log(f"Creating DiT model on {loading_device}", category="model")

    with torch.device(loading_device):
        runner.dit = create_object(config.dit.model)

    debug.end_timer("dit_model_create", f"DiT model instantiated on {loading_device} - weights not loaded yet")

    runner.dit.set_gradient_checkpointing(config.dit.gradient_checkpoint)

    # Detect and log model format
    debug.log(f"Loading model_weight: {model_weight}", category="model", force=True)

    debug.start_timer("dit_load_state_dict")
    if checkpoint.endswith('.gguf'):
        state_loading_device = device  # Load directly to GPU
    else:
        state_loading_device = "cpu" if "7b" in model_weight and vram_info['total_gb'] < 25 else device
        
    # For GGUF models, keep_native_fp8 should be False since they have their own quantization
    keep_native_fp8 = not checkpoint.endswith('.gguf')
    state = load_quantized_state_dict(checkpoint, state_loading_device, keep_native_fp8=keep_native_fp8, debug=debug)

    debug.end_timer("dit_load_state_dict", "DiT state dict loaded")
    debug.start_timer("dit_load")
    # Handle GGUF models with custom loading approach
    if checkpoint.endswith('.gguf'):
        debug.log("Loading GGUF model - keeping tensors quantized...", category="model")
        
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
                    #debug.log(f"Shape mismatch detected!", category="error")
                    #debug.log(f"   Parameter: {key}")
                    #debug.log(f"   Model expects: {model_shape}")
                    #debug.log(f"   GGUF provides: {gguf_shape}")
                    
                    # Check if this is a quantization issue rather than architecture mismatch
                    if hasattr(gguf_param, 'tensor_type') and gguf_param.tensor_type not in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                        #debug.log(f"   This is a quantized tensor - the shape difference might be due to quantization")
                        #debug.log(f"   Raw tensor shape: {gguf_param.shape}")
                        #debug.log(f"   Logical tensor shape: {gguf_shape}")
                        
                        # For quantized tensors, we should use the logical shape, not the raw shape
                        if hasattr(gguf_param, 'tensor_shape') and gguf_param.tensor_shape == model_shape:
                            #debug.log(f"   ✅ Logical shapes match, continuing with quantized tensor")
                            continue
                        
                    architecture_mismatch = True
                    break
                
        debug.log(f"Architecture check complete. Mismatch: {architecture_mismatch}", category="info")
        
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
                    #if hasattr(param, 'tensor_type') and debug:
                        #debug.log(f"Loading quantized parameter: {name}", category="info")
                        #debug.log(f"   Shape: {param_shape}")
                        #debug.log(f"   Type: {param.tensor_type}")
                        
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
                                            numpy_data = original_tensor.cpu().numpy()
                                            dequantized = gguf.quants.dequantize(numpy_data, original_tensor.tensor_type)
                                            result = torch.from_numpy(dequantized).to(device, dtype)
                                            result.requires_grad_(False)
                                            result = result.reshape(original_tensor.tensor_shape)
                                            return result
                                        except Exception as e:
                                            debug.log(f"Warning: Could not dequantize tensor: {e}", category="warning")
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
                    debug.log(f"Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}", category="error")
                    raise ValueError(f"Shape mismatch for parameter {name}")
                    
        debug.log(f"GGUF loading complete: {len(loaded_params)} parameters loaded", category="success")
        debug.log(f"Quantized parameters: {quantized_params}", category="memory")
        debug.log(f"VRAM savings: Tensors kept in quantized format", category="memory")
        
        # Debug: check for shape mismatches
        unmatched_params = []
        for name, param in state.items():
            if name not in model_state:
                unmatched_params.append(name)
        if unmatched_params:
            debug.log(f"Warning: {len(unmatched_params)} parameters from GGUF not found in model", category="warning")
            debug.log(f"   First few unmatched: {unmatched_params[:5]}")
            
        missing_params = []
        for name in model_state:
            if name not in loaded_params:
                missing_params.append(name)
        if missing_params:
            debug.log(f"Warning: {len(missing_params)} model parameters not loaded from GGUF", category="warning")
            debug.log(f"   First few missing: {missing_params[:5]}")
        success_rate = 1.0  # If we get here, loading was successful
        
        # Move model to GPU after GGUF loading (if not preserving VRAM and no BlockSwap)
        if not preserve_vram and not blockswap_active:
            debug.log("Moving GGUF model from CPU to GPU...", category="model")
            runner.dit = runner.dit.to(device)
    else:
        runner.dit.load_state_dict(state, strict=True, assign=True)

    if 'state' in locals():
        del state
            
    debug.end_timer("dit_load", "DiT load")
    #state.to("cpu")
    #runner.dit = runner.dit.to(device)
    
    # Apply quantized operations for GGUF models
    if checkpoint.endswith('.gguf') and GGUF_OPS_AVAILABLE:
        t = time.time()
        debug.log("Skipping quantized operations replacement - using direct parameter approach", category="config", force=True)
        debug.log(f"CONFIG DIT : GGUF quantized ops time: {time.time() - t} seconds", category="config")
    elif checkpoint.endswith('.gguf'):
        debug.log("GGUF ops not available - model will run with standard operations", category="warning")
        debug.log("   Install missing dependencies for optimal GGUF performance")

    # Apply universal compatibility wrapper to ALL models
    # This ensures RoPE compatibility and optimal performance across all architectures
    debug.start_timer("FP8CompatibleDiT")
    # Check if already wrapped to avoid double wrapping
    if not isinstance(runner.dit, FP8CompatibleDiT):
        runner.dit = FP8CompatibleDiT(runner.dit, skip_conversion=False, blockswap_active=blockswap_active, debug=debug)
    debug.end_timer("FP8CompatibleDiT", "FP8/RoPE compatibility wrapper applied to DiT model")

    # Move DiT to CPU to prevent VRAM leaks (especially for 3B model with complex RoPE)
    if preserve_vram and not blockswap_active:
        debug.log("Moving DiT model to CPU (preserve_vram enabled)", category="memory")
        runner.dit = runner.dit.to("cpu")
        if "7b" in model_weight:
            clear_vram_cache(debug)
    else:
        if state_loading_device == "cpu" and not blockswap_active:
            runner.dit.to(device)

    # Log BlockSwap status if active
    if blockswap_active:
        debug.log(f"BlockSwap active ({block_swap_config.get('blocks_to_swap', 0)} blocks) - placement handled by BlockSwap", category="blockswap")

    return runner


def configure_vae_model_inference(runner, device, checkpoint_path, config, 
                                 preserve_vram=False, model_weight=None, 
                                 vram_info=None, debug=None):
    """
    Configure VAE model for inference without distributed decorators
    
    Args:
        runner: VideoDiffusionInfer instance  
        config: Model configuration
        device (str): Target device
        
    Features:
        - Dynamic path resolution for VAE checkpoints
        - SafeTensors and PyTorch format support
        - FP8 and FP16 VAE handling
        - Causal slicing configuration
    """
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_vae_model_inference")
    
    # Create vae model
    if torch.mps.is_available():
        config.vae.dtype = "float16"
        if "fp8_e4m3fn" in runner._model_name or ".gguf" in runner._model_name:
            config.vae.dtype = "bfloat16"
            
    
    dtype = getattr(torch, config.vae.dtype)
    debug.start_timer("vae_model_create")
    loading_device = "cpu" if preserve_vram else device

    with torch.device(device):
        runner.vae = create_object(config.vae.model)
    debug.end_timer("vae_model_create", f"VAE model created on {device} with dtype {dtype}")

    debug.start_timer("model_requires_grad")
    runner.vae.requires_grad_(False).eval()
    debug.end_timer("model_requires_grad", f"VAE model set to eval mode (gradients disabled)")
    
    # t = time.time()
    #runner.vae.to(device=loading_device, dtype=dtype)
    #debug.log(f"🔄 CONFIG VAE : TO CPU TIME: {time.time() - t} seconds device: {device} dtype: {dtype}", category="timing")
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
             debug.log(f"CONFIG VAE : Found VAE checkpoint at: {vae_checkpoint_path}", category="vae")
            break
    debug.log(f"CONFIG VAE : VAE CHECKPOINT PATH TIME: {time.time() - t} seconds", category="timing")
    if vae_checkpoint_path is None:
        raise FileNotFoundError(f"VAE checkpoint not found. Tried paths: {possible_paths}")
    '''
    # Load VAE with format detection
    debug.start_timer("vae_load")
    state_loading_device = "cpu" if "7b" in model_weight and vram_info['total_gb'] < 25 else device
    debug.log(f"Loading VAE SafeTensors: {checkpoint_path}", category="vae", force=True)
    # Use optimized loading for all SafeTensors formats
    if "fp8_e4m3fn" in checkpoint_path:
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=True, debug=debug)
    else:
        # For FP16 SafeTensors and GGUF models, disable native FP8
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=False, debug=debug)

    debug.end_timer("vae_load", "VAE loaded")
    debug.start_timer("vae_load_state_dict")
    runner.vae.load_state_dict(state)

    if torch.mps.is_available():
        runner.vae = runner.vae.to(dtype=getattr(torch, config.vae.dtype))
    
    if state_loading_device == "cpu":
        runner.vae.to(device)
    
    # For GGUF models, ensure VAE uses BFloat16 for compatibility with quantized models
    if checkpoint_path.endswith('.gguf'):
        runner.vae = runner.vae.to(torch.bfloat16)
        debug.log(f"Converted VAE to BFloat16 for GGUF model compatibility", category="config")
            
    if 'state' in locals():
        del state
    debug.end_timer("vae_load_state_dict", "VAE state dict loaded")

    # Set causal slicing if available
    debug.start_timer("vae_set_causal_slicing")
    if hasattr(runner.vae, "set_causal_slicing") and hasattr(config.vae, "slicing"):
        runner.vae.set_causal_slicing(**config.vae.slicing)

    debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configured")

    # Attach debug to VAE
    runner.vae.debug = debug

    # Propagate debug to all modules efficiently
    for module in runner.vae.modules():
        module.debug = debug

    return runner
    #runner.vae.to("cpu")