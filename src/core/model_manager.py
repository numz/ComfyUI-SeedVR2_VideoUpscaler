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
from src.utils.constants import get_script_directory
from omegaconf import DictConfig, OmegaConf

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️ SafeTensors not available, recommended install: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

from src.optimization.memory_manager import get_basic_vram_info, clear_vram_cache
from src.optimization.compatibility import FP8CompatibleDiT
from src.optimization.memory_manager import preinitialize_rope_cache, clear_rope_lru_caches
from src.common.config import load_config, create_object
from src.core.infer import VideoDiffusionInfer
from src.optimization.blockswap import apply_block_swap_to_dit

# Get script directory for config paths
script_directory = get_script_directory()


def configure_runner(model, base_cache_dir, preserve_vram=False, debug=None, 
                    cache_model=False, block_swap_config=None, cached_runner=None):
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
                    debug.log("BlockSwap config matches, performing fast re-application", category="reuse", force=True)
                    
                    # Mark as active before applying
                    cached_runner._blockswap_active = True
                    
                    # Apply BlockSwap (will be fast since model structure is intact)
                    apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
                else:
                    # Configuration changed - apply new config
                    debug.log("BlockSwap configuration changed, applying new config", category="blockswap", force=True)
                    apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
            else:
                # No cached config - apply fresh
                debug.log("Applying BlockSwap to cached runner", category="blockswap", force=True)
                apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
            
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
        model_weight = "7b_fp8" if "fp8" in model else "7b_fp16"
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
        model_weight = "3b_fp8" if "fp8" in model else "3b_fp16"
    
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
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    # Store model name for cache validation
    runner._model_name = model
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialized")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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


def load_quantized_state_dict(checkpoint_path, device="cpu", keep_native_fp8=True):
    """
    Load state dict from SafeTensors or PyTorch with optimal FP8 native support
    
    Args:
        checkpoint_path (str): Path to model checkpoint (.safetensors or .pth)
        device (str): Target device for loading
        keep_native_fp8 (bool): Whether to preserve native FP8 format for performance
        
    Returns:
        dict: State dictionary with optimal dtype handling
        
    Features:
        - Automatic format detection (SafeTensors vs PyTorch)
        - Native FP8 preservation for 2x speedup and 50% VRAM reduction
        - Intelligent dtype conversion when needed for compatibility
        - Memory-mapped loading for large models
    """
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors required to load this model. Install with: pip install safetensors")
        state = load_safetensors_file(checkpoint_path, device=device)
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    else:
        raise ValueError(f"Unsupported format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
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

    loading_device = "cpu" if (preserve_vram or blockswap_active) else device

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
    state_loading_device = "cpu" if "7b" in model_weight and vram_info['total_gb'] < 25 else device
    state = load_quantized_state_dict(checkpoint, state_loading_device, keep_native_fp8=True)

    debug.end_timer("dit_load_state_dict", "DiT state dict loaded")
    debug.start_timer("dit_load")
    runner.dit.load_state_dict(state, strict=True, assign=True)

    if 'state' in locals():
        del state
            
    debug.end_timer("dit_load", "DiT load")
    #state.to("cpu")
    #runner.dit = runner.dit.to(device)

    # Apply universal compatibility wrapper to ALL models
    # This ensures RoPE compatibility and optimal performance across all architectures
    debug.start_timer("FP8CompatibleDiT")
    # Check if already wrapped to avoid double wrapping
    if not isinstance(runner.dit, FP8CompatibleDiT):
        runner.dit = FP8CompatibleDiT(runner.dit, skip_conversion=False, debug=debug)
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
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=True)
    else:
        # For FP16 SafeTensors, disable native FP8
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=False)

    debug.end_timer("vae_load", "VAE loaded")
    debug.start_timer("vae_load_state_dict")
    runner.vae.load_state_dict(state)
    if state_loading_device == "cpu":
        runner.vae.to(device)
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