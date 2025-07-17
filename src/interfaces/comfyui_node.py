# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

import os
import time
import torch
from typing import Tuple, Dict, Any

from src.utils.downloads import download_weight, get_base_cache_dir
from src.core.model_manager import configure_runner
from src.core.generation import generation_loop
from src.optimization.memory_manager import fast_model_cleanup, fast_ram_cleanup
from src.optimization.blockswap import cleanup_blockswap
from src.optimization.memory_manager import (
    clear_rope_lru_caches, 
    fast_model_cleanup, 
    fast_ram_cleanup, 
    clear_all_caches
)

# Import ComfyUI progress reporting
from server import PromptServer

script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SeedVR2:
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    - Real-time progress reporting
    """
    
    def __init__(self):
        """Initialize SeedVR2 node"""
        self.runner = None
        self.text_pos_embeds = None
        self.text_neg_embeds = None
        self.current_model_name = ""


    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define ComfyUI input parameter types and constraints
        
        Returns:
            Dictionary defining input parameters, types, and validation
        """
        return {
            "required": {
                "images": ("IMAGE", ),
                "model": ([
                    "seedvr2_ema_3b_fp16.safetensors", 
                    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_fp16.safetensors",
                    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_sharp_fp16.safetensors",
                    "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
                ], {
                    "default": "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
                }),
                "seed": ("INT", {
                    "default": 100, 
                    "min": 0, 
                    "max": 2**32 - 1, 
                    "step": 1,
                    "tooltip": "Random seed for generation reproducibility"
                }),
                "new_resolution": ("INT", {
                    "default": 1072, 
                    "min": 16, 
                    "max": 4320, 
                    "step": 16,
                    "tooltip": "Target new resolution for upscaled video"
                }),
                "batch_size": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 4,
                    "tooltip": "Number of frames to process per batch (recommend 4n+1 format)"
                }),
                "preserve_vram": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "block_swap_config": (
                    "block_swap_config",
                    {"tooltip": "Optional BlockSwap configuration for low VRAM mode"},
                ),
            },
        }
    
    # Define return types for ComfyUI
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images: torch.Tensor, model: str, seed: int, new_resolution: int, 
        batch_size: int, preserve_vram: bool, block_swap_config=None) -> Tuple[torch.Tensor]:
        """Execute SeedVR2 video upscaling with progress reporting"""
        
        temporal_overlap = 0 
        print(f"🔄 Preparing model: {model}")
        
        download_weight(model)
        debug = False
        cfg_scale = 1.0
        try:
            return self._internal_execute(images, model, seed, new_resolution, cfg_scale, batch_size, preserve_vram, temporal_overlap, debug, block_swap_config)
        except Exception as e:
            self.cleanup(force_ram_cleanup=True)
            raise e



    def cleanup(self, force_ram_cleanup: bool = True, keep_model_cached: bool = False, block_swap_config=None):
        """
        Comprehensive cleanup with memory tracking

        Args:
            force_ram_cleanup (bool): Whether to perform aggressive RAM cleanup
            keep_model_cached (bool): Whether to keep the model in RAM (only applies with BlockSwap)
            block_swap_config: Block swap configuration with enable_debug flag
        """
        # Determine if we should keep model cached
        should_keep_model = False
        if self.runner and keep_model_cached:
            is_blockswap_active = (
                hasattr(self.runner, "_blockswap_active") 
                and self.runner._blockswap_active
            )
            # Check if cache_model is enabled in config
            cache_model_enabled = block_swap_config and block_swap_config.get("cache_model", False)
            should_keep_model = is_blockswap_active and cache_model_enabled
        
        # Use existing debugger if available
        debugger = None
        if self.runner and hasattr(self.runner, '_blockswap_debugger'):
            debugger = self.runner._blockswap_debugger
            debugger.clear_history()
        
        # Perform partial or full cleanup based on model caching
        if should_keep_model:
            debugger.log("🧹 Partial cleanup - keeping model in RAM")
            
            # Clean BlockSwap with state preservation
            if hasattr(self.runner, "_blockswap_active") and self.runner._blockswap_active:
                cleanup_blockswap(self.runner, keep_state_for_cache=True)
            
            # Clear all caches
            if self.runner:              
                clear_all_caches(self.runner, debugger)
            
        else:
            # Full cleanup - existing implementation
            if debugger:
                debugger.log("🧹 Full cleanup - clearing everything")

            if self.runner:
                # Clean BlockSwap if active
                if hasattr(self.runner, "_blockswap_active") and self.runner._blockswap_active:
                    cleanup_blockswap(self.runner, keep_state_for_cache=False)
                
                # Clear cache
                if hasattr(self.runner, 'cache') and hasattr(self.runner.cache, 'cache'):
                    for key, value in list(self.runner.cache.cache.items()):
                        if hasattr(value, 'cpu'):
                            value.cpu()
                        if hasattr(value, 'detach'):
                            value.detach()
                        del value
                    self.runner.cache.cache.clear()
                
                # Clear DiT model
                if hasattr(self.runner, 'dit') and self.runner.dit is not None:
                    # Handle FP8CompatibleDiT wrapper
                    if hasattr(self.runner.dit, 'dit_model'):
                        # Clean inner model first
                        clear_rope_lru_caches(self.runner.dit.dit_model)
                        fast_model_cleanup(self.runner.dit.dit_model)
                        # Break reference from wrapper to model
                        self.runner.dit.dit_model = None
                    else:
                        # Direct model cleanup
                        clear_rope_lru_caches(self.runner.dit)
                        fast_model_cleanup(self.runner.dit)
                    
                    del self.runner.dit
                    self.runner.dit = None
                
                # Clear VAE model  
                if hasattr(self.runner, 'vae') and self.runner.vae is not None:
                    #from src.optimization.memory_manager import fast_model_cleanup
                    fast_model_cleanup(self.runner.vae)
                    del self.runner.vae
                    self.runner.vae = None
                
                # Clear other components
                for component in ['sampler', 'sampling_timesteps', 'schedule', 'config']:
                    if hasattr(self.runner, component):
                        setattr(self.runner, component, None)
                
                del self.runner
                self.runner = None
            
        # Clear embeddings
        if self.text_pos_embeds is not None:
            if hasattr(self.text_pos_embeds, 'cpu'):
                self.text_pos_embeds.cpu()
            del self.text_pos_embeds
            self.text_pos_embeds = None
        
        if self.text_neg_embeds is not None:
            if hasattr(self.text_neg_embeds, 'cpu'):
                self.text_neg_embeds.cpu()
            del self.text_neg_embeds
            self.text_neg_embeds = None
        
        self.current_model_name = ""
        
        # Fast RAM cleanup
        if force_ram_cleanup:
            fast_ram_cleanup()
        
        # BlockSwap debugger memory state
        if debugger is not None:
            cleanup_stage = "partial" if should_keep_model else "full"
            debugger.log_memory_state(f"After {cleanup_stage} cleanup", show_tensors=False)


    def _internal_execute(self, images, model, seed, new_resolution, cfg_scale, batch_size, preserve_vram, temporal_overlap, debug, block_swap_config):
        """Internal execution logic with progress tracking"""
        total_start_time = time.time()

        # Check if we should use model caching
        use_cache = (
            block_swap_config
            and block_swap_config.get("blocks_to_swap", 0) > 0
            and block_swap_config.get("cache_model", False)
        )

        if self.runner is not None:
            current_model = getattr(self.runner, '_model_name', None)
            model_changed = current_model != model

            if model_changed and self.runner is not None:
                print(
                    f"🔄 Model changed from {self.current_model_name} to {model}, clearing cache..."
                )
                self.cleanup(
                    force_ram_cleanup=True,
                    keep_model_cached=False,
                    block_swap_config=block_swap_config,
                )
                self.runner = None

        # Configure runner
        if debug:
            print("🔄 Configuring inference runner...")
        runner_start = time.time()
        self.runner = configure_runner(
            model, get_base_cache_dir(), preserve_vram, debug, 
            block_swap_config=block_swap_config,
            cached_runner=self.runner  # Pass existing runner if any
        )
        
        self.current_model_name = model
        
        if debug:
            print(f"🔄 Runner configuration time: {time.time() - runner_start:.2f}s")
        
        if debug:
            print("🚀 Starting video upscaling generation...")

        # Execute generation with progress callback
        sample = generation_loop(
            self.runner, images, cfg_scale, seed, new_resolution, 
            batch_size, preserve_vram, temporal_overlap, debug,
            block_swap_config=block_swap_config,
            progress_callback=self._progress_callback
        )
        
        
        print(f"✅ Video upscaling completed successfully!")       
        # Cleanup
        print(f"🔄 Total execution time: {time.time() - total_start_time:.2f}s")           
        self.cleanup(force_ram_cleanup=True, keep_model_cached=use_cache, block_swap_config=block_swap_config)
        return (sample,)

    def _progress_callback(self, batch_idx, total_batches, current_batch_frames, message=""):
        """Progress callback for generation loop"""
            
        # Send numerical progress
        progress_value = int((batch_idx / total_batches) * 100)
        progress_data = {
            "value": progress_value,
            "max": 100,
            "node": "seedvr2_node"
        }
        PromptServer.instance.send_sync("progress", progress_data, None)

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup(force_ram_cleanup=True, keep_model_cached=False, block_swap_config=None)
        except:
            pass


class SeedVR2BlockSwap:
    """Configure block swapping to reduce VRAM usage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 36,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU. Start with 16 and increase until OOM errors stop. 0=disabled",
                    },
                ),
                "use_non_blocking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use non-blocking GPU transfers for better performance.",
                    },
                ),
                "offload_io_components": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload embeddings and I/O layers to CPU. Enable if you need additional VRAM savings beyond block swapping",
                    },
                ),
                "cache_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep model in RAM between runs to avoid model loading time. Useful for batch processing",
                    },
                ),
                "enable_debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Show detailed memory usage and timing information during inference",
                    },
                ),
            }
        }

    RETURN_TYPES = ("block_swap_config",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = """Configure block swapping to reduce VRAM usage during video upscaling.

BlockSwap dynamically moves transformer blocks between GPU and CPU/RAM during inference, enabling large models to run on limited VRAM systems with minimal performance impact.

Configuration Guidelines:
    - blocks_to_swap=0: Disabled (fastest, highest VRAM usage)
    - blocks_to_swap=16: Balanced mode (moderate speed/VRAM trade-off)
    - blocks_to_swap=32-36: Maximum savings (slowest, lowest VRAM)

Advanced Options:
    - use_non_blocking: Enables asynchronous GPU transfers for better performance
    - offload_io_components: Moves embeddings and I/O layers to CPU for additional VRAM savings (slower)
    - cache_model: Keeps model in RAM between runs (avoids model loading time on subsequent generations)
    - enable_debug: Shows detailed memory usage and timing information

Performance Tips:
    - Start with blocks_to_swap=16 and increase until you no longer get OOM errors or decrease if you have spare VRAM
    - Enable offload_io_components if you still need additional VRAM savings
    - Note: Even if inference succeeds, you may still OOM during VAE decoding - combine BlockSwap with VAE tiling if needed (feature in development)
    - Enable cache_model for batch processing to skip model reloading between runs
    - Keep non_blocking=True for better performance (default)
    - Combine with smaller batch_size for maximum VRAM savings

The actual memory savings depend on your specific model architecture and will be shown in the debug output when enabled.
    """

    def create_config(
        self,
        blocks_to_swap,
        use_non_blocking,
        offload_io_components,
        cache_model,
        enable_debug,
    ):
        if blocks_to_swap > 0 or offload_io_components:
            configs = []
            if blocks_to_swap > 0:
                configs.append(f"{blocks_to_swap} blocks")
            if use_non_blocking:
                configs.append("non blocking")
            if offload_io_components:
                configs.append("I/O components")
            if cache_model and blocks_to_swap > 0:
                configs.append("model caching")
            print(f"🔄 BlockSwap configured: {', '.join(configs)}")
        return (
            {
                "blocks_to_swap": blocks_to_swap,
                "use_non_blocking": use_non_blocking,
                "offload_io_components": offload_io_components,
                "cache_model": cache_model,
                "enable_debug": enable_debug,
            },
        )


# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
    "SeedVR2BlockSwap": SeedVR2BlockSwap,
}

# Human-readable node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
    "SeedVR2BlockSwap": "SeedVR2 BlockSwap Config",
}

# Export version and metadata
__version__ = "2.0.0-modular"
__author__ = "SeedVR2 Team"
__description__ = "High-quality video upscaling using advanced diffusion models"

# Additional exports for introspection
__all__ = [
    'SeedVR2',
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS'
] 
