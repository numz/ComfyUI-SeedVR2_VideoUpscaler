# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

import os
import time
import torch
from typing import Tuple, Dict, Any

from src.utils.constants import get_base_cache_dir
from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_models, DEFAULT_MODEL
from src.utils.constants import get_script_directory
from src.utils.debug import Debug
from src.core.model_manager import configure_runner
from src.core.generation import generation_loop
from src.optimization.memory_manager import fast_model_cleanup, fast_ram_cleanup, get_vram_usage
from src.optimization.blockswap import cleanup_blockswap
from src.optimization.memory_manager import (
    clear_rope_lru_caches, 
    fast_model_cleanup, 
    fast_ram_cleanup, 
    clear_all_caches
)

# Import ComfyUI progress reporting
from server import PromptServer

script_directory = get_script_directory()

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
        self.debug = None


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
                "model": (get_available_models(), {
                    "default": DEFAULT_MODEL,
                    "tooltip": "Model variants with different sizes and precisions. Models will automatically download on first use. Additional models can be added to the ComfyUI models folder."
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2**32 - 1, 
                    "step": 1,
                    "tooltip": "Random seed for generation. Same seed = same output."
                }),
                "new_resolution": ("INT", {
                    "default": 1072, 
                    "min": 16, 
                    "max": 4320, 
                    "step": 16,
                    "tooltip": "Target resolution for the shortest edge. Maintains aspect ratio."
                }),
                "batch_size": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 4,
                    "tooltip": "Number of frames to process per batch (recommend 4n+1 format)"
                }),
                "preserve_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload models between steps to save VRAM. Slower but uses less memory."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model and VAE in RAM between runs. Speeds up batch processing."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed memory usage and timing information during generation."
                }),
            },
            "optional": {
                "block_swap_config": ("block_swap_config", {
                    "tooltip": "Optional BlockSwap configuration for additional VRAM savings"
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images: torch.Tensor, model: str, seed: int, new_resolution: int, 
        batch_size: int, preserve_vram: bool, cache_model: bool, enable_debug: bool,
        block_swap_config=None) -> Tuple[torch.Tensor]:
        """Execute SeedVR2 video upscaling with progress reporting"""
        
        temporal_overlap = 0 
        
        # Initialize or reuse debug instance
        if self.debug is None:
            self.debug = Debug(enabled=enable_debug)
        else:
            self.debug.enabled = enable_debug
        
        self.debug.start_timer("total_execution")
        self.debug.log("\n─── Model Preparation ───", category="none")
        self.debug.start_timer("model_preparation")
        self.debug.log_memory_state("Execution start")
        self.debug.log(f"Preparing model: {model}", category="general", force=True)
        
        # Check if download succeeded
        if not download_weight(model, debug=self.debug):
            raise RuntimeError(
                f"Required files for {model} are not available. "
                "Please check the console output for manual download instructions."
            )

        cfg_scale = 1.0
        try:
            return self._internal_execute(images, model, seed, new_resolution, cfg_scale, 
                                        batch_size, preserve_vram, temporal_overlap, 
                                        cache_model, block_swap_config)
        except Exception as e:
            self.cleanup(force_ram_cleanup=True, cache_model=cache_model, debug=self.debug)
            raise e
        

    def cleanup(self, force_ram_cleanup: bool = True, cache_model: bool = False, debug=None):
        """
        Comprehensive cleanup with memory tracking

        Args:
            force_ram_cleanup (bool): Whether to perform aggressive RAM cleanup
            cache_model (bool): Whether to keep the model in RAM
            debug: Optional Debug instance for logging
        """
        # Determine if we should keep model cached
        should_keep_model = cache_model and self.runner is not None
        
        # Get debug from runner if not provided (for __del__ case)
        if debug is None and self.runner and hasattr(self.runner, 'debug'):
            debug = self.runner.debug
        
        if debug is None:
            # Silent cleanup for destructor case
            return
        
        cleanup_type = "partial" if should_keep_model else "full"
        debug.log(f"Starting {cleanup_type} cleanup", category="cleanup")
        debug.log_memory_state(f"Before {cleanup_type} cleanup")
        
        # Perform partial or full cleanup based on model caching
        if should_keep_model:            
            # Clean BlockSwap with state preservation
            if hasattr(self.runner, "_blockswap_active") and self.runner._blockswap_active:
                cleanup_blockswap(self.runner, keep_state_for_cache=True)
            
            # Clear caches but keep models
            if self.runner:              
                clear_all_caches(self.runner, debug)
                
                # Move VAE to CPU and clear intermediate tensors
                if hasattr(self.runner, 'vae') and self.runner.vae is not None:
                    debug.log("Moving VAE to CPU and clearing intermediate tensors", category="cleanup")
                    
                    # Clear any intermediate tensors/buffers in VAE
                    for module in self.runner.vae.modules():
                        # Clear module-specific caches
                        if hasattr(module, '_temp_cache'):
                            delattr(module, '_temp_cache')
                        if hasattr(module, '_intermediate_cache'):
                            delattr(module, '_intermediate_cache')
                        
                        # Clear any CUDA tensors in module attributes
                        for attr_name in list(vars(module).keys()):
                            attr = getattr(module, attr_name, None)
                            if torch.is_tensor(attr) and attr.is_cuda:
                                # Move tensor to CPU if it's not a parameter/buffer
                                if attr_name not in module._parameters and attr_name not in module._buffers:
                                    setattr(module, attr_name, attr.cpu())
                                    del attr
                    
                    # Move entire VAE to CPU (preserves model for reuse)
                    self.runner.vae = self.runner.vae.to('cpu')
                    
                    # Clear CUDA cache to free VRAM immediately
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                    debug.log("VAE moved to CPU, intermediate tensors cleared", category="success")

            debug.log("Models kept in RAM for next run", category="store")
            
        else:
            # Full cleanup - existing implementation
            debug.log("Performing full cleanup", category="cleanup")

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
                        # Ensure RoPE modules are on CPU
                        for name, module in self.runner.dit.dit_model.named_modules():
                            if hasattr(module, 'rope') and hasattr(module.rope, 'to'):
                                module.rope = module.rope.to('cpu')
                                if hasattr(module.rope, 'freqs'):
                                    module.rope.freqs = module.rope.freqs.to('cpu')
                        fast_model_cleanup(self.runner.dit.dit_model)
                        # Aggressively clear the wrapper too
                        self.runner.dit.dit_model = None
                        # Delete the wrapper's __dict__ to break any circular refs
                        self.runner.dit.__dict__.clear()
                        # Break all references
                        self.runner.dit.dit_model = None
                        if hasattr(self.runner.dit, 'debug'):
                            self.runner.dit.debug = None
                    else:
                        # Direct model cleanup
                        clear_rope_lru_caches(self.runner.dit)
                        # Ensure RoPE modules are on CPU
                        for name, module in self.runner.dit.named_modules():
                            if hasattr(module, 'rope') and hasattr(module.rope, 'to'):
                                module.rope = module.rope.to('cpu')
                                if hasattr(module.rope, 'freqs'):
                                    module.rope.freqs = module.rope.freqs.to('cpu')
                        fast_model_cleanup(self.runner.dit)
                    
                    del self.runner.dit
                    self.runner.dit = None
                
                # Clear VAE model  
                if hasattr(self.runner, 'vae') and self.runner.vae is not None:
                    #from src.optimization.memory_manager import fast_model_cleanup
                    fast_model_cleanup(self.runner.vae)
                    # Clear VAE's internal dict
                    self.runner.vae.__dict__.clear()
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


    def _internal_execute(self, images, model, seed, new_resolution, cfg_scale, batch_size, 
                 preserve_vram, temporal_overlap, cache_model, block_swap_config):
        """Internal execution logic with progress tracking"""
        
        debug = self.debug

        if self.runner is not None:
            current_model = getattr(self.runner, '_model_name', None)
            model_changed = current_model != model

            if model_changed and self.runner is not None:
                debug.log(f"Model changed from {self.current_model_name} to {model}, clearing cache...", category="cache")
                self.cleanup(
                    force_ram_cleanup=True,
                    cache_model=False,  # Don't keep old model
                    debug=debug,
                )
                self.runner = None

        # Configure runner
        debug.log("Configuring inference runner...", category="runner")
        
        self.runner = configure_runner(
            model, get_base_cache_dir(), preserve_vram, debug,
            cache_model=cache_model,
            block_swap_config=block_swap_config,
            cached_runner=self.runner if cache_model else None
        )
        
        self.current_model_name = model
        debug.log_memory_state("Model preparation completed")

        debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)

        debug.log("", category="none", force=True)
        debug.log("Starting video upscaling generation...\n", category="generation", force=True)
        debug.start_timer("generation_loop")

        # Execute generation with debug
        sample = generation_loop(
            self.runner, images, cfg_scale, seed, new_resolution, 
            batch_size, preserve_vram, temporal_overlap, debug,
            block_swap_config=block_swap_config,
            progress_callback=self._progress_callback
        )
        
        debug.log("", category="none", force=True)
        debug.log("Video upscaling completed successfully!", category="generation", force=True)
        # Log performance summary before clearing
        if debug.enabled:            
            # Log BlockSwap summary if it was used
            if hasattr(self.runner, '_blockswap_active') and self.runner._blockswap_active:
                swap_summary = debug.get_swap_summary()
                if swap_summary and swap_summary.get('total_swaps', 0) > 0:
                    total_time = swap_summary.get('block_total_ms', 0) + swap_summary.get('io_total_ms', 0)
                    debug.log(f"BlockSwap overhead: {total_time:.1f}ms across {swap_summary['total_swaps']} swaps", category="blockswap")
            
            # Log memory usage summary
            allocated, reserved, peak = get_vram_usage()
            debug.log(f"Final VRAM usage - Allocated: {allocated:.2f}GB, Peak: {peak:.2f}GB", category="memory")
        debug.log_memory_state("Video generation - Memory")
        debug.end_timer("generation_loop", "Video generation completed", show_breakdown=True)
       
        debug.log("\n─── Final Cleanup ───", category="none")
        debug.start_timer("final_cleanup")
        self.cleanup(force_ram_cleanup=True, cache_model=cache_model, debug=debug)
        debug.log_memory_state("Final cleanup - Memory", detailed_tensors=False)
        debug.end_timer("final_cleanup", "Final cleanup completed", show_breakdown=True)
        # Cleanup  
        debug.log("\n─────────", category="none")
        child_times = {
            "Model preparation": debug.timer_durations.get("model_preparation", 0),
            "Video generation": debug.timer_durations.get("generation_loop", 0),
            "Final cleanup": debug.timer_durations.get("final_cleanup", 0)
        }
        debug.end_timer("total_execution", "Total execution", show_breakdown=True, custom_children=child_times)
        debug.log("─────────", category="none")
        # Clear history for next run
        debug.clear_history()

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

Performance Tips:
    - Start with blocks_to_swap=16 and increase until you no longer get OOM errors or decrease if you have spare VRAM
    - Enable offload_io_components if you still need additional VRAM savings
    - Note: Even if inference succeeds, you may still OOM during VAE decoding - combine BlockSwap with VAE tiling if needed (feature in development)
    - Keep non_blocking=True for better performance (default)
    - Combine with smaller batch_size for maximum VRAM savings

The actual memory savings depend on your specific model architecture and will be shown in the debug output when enabled.
    """

    def create_config(self, blocks_to_swap, use_non_blocking, offload_io_components):
        """Create BlockSwap configuration"""
        if blocks_to_swap == 0:
            return (None,)
        
        config = {
            "blocks_to_swap": blocks_to_swap,
            "use_non_blocking": use_non_blocking,
            "offload_io_components": offload_io_components,
        }
        
        return (config,)


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