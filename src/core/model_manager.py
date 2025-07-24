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
        model_weight = "7b_fp8" if "fp8" in model else "7b_fp16"
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
        model_weight = "3b_fp8" if "fp8" in model else "3b_fp16"
    
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
    #clear_vram_cache()
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

    loading_device = "cpu" if (preserve_vram or blockswap_active) else device
    
    if blockswap_active and debug:
        print(f"🔄 CONFIG DIT : BlockSwap active - creating model on CPU")

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
    state_loading_device = "cpu" if "7b" in model_weight and vram_info['total_gb'] < 25 else device
    state = load_quantized_state_dict(checkpoint, state_loading_device, keep_native_fp8=True)

    if debug:
        print(f"🔄 CONFIG DIT : DiT load state dict time: {time.time() - t} seconds")
    t = time.time()
    runner.dit.load_state_dict(state, strict=True, assign=True)

    if 'state' in locals():
        del state
            
    if debug:
        print(f"🔄 CONFIG DIT : DiT load time: {time.time() - t} seconds")
    #state.to("cpu")
    #runner.dit = runner.dit.to(device)

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
    print(f"🚀 Loading VAE SafeTensors: {checkpoint_path}")
    # Use optimized loading for all SafeTensors formats
    if "fp8_e4m3fn" in checkpoint_path:
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=True)
    else:
        # For FP16 SafeTensors, disable native FP8
        state = load_quantized_state_dict(checkpoint_path, state_loading_device, keep_native_fp8=False)

    if debug:
        print(f"🔄 CONFIG VAE : VAE LOAD TIME: {time.time() - t} seconds")
    t = time.time()
    runner.vae.load_state_dict(state)
    if state_loading_device == "cpu":
        runner.vae.to(device)
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