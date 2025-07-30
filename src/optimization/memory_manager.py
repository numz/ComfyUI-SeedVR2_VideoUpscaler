"""
Memory management module for SeedVR2
Handles VRAM usage, cache management, and memory optimization

Extracted from: seedvr2.py (lines 373-405, 607-626, 1016-1044)
"""

import os
import torch
import gc
import time
from typing import Tuple, Optional
from src.common.cache import Cache

def get_basic_vram_info():
    """🔍 Méthode basique avec PyTorch natif"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Mémoire libre et totale (en bytes)
    free_memory, total_memory = torch.cuda.mem_get_info()
    
    # Conversion en GB
    free_gb = free_memory / (1024**3)
    total_gb = total_memory / (1024**3)
    
    return {
        "free_gb": free_gb,
        "total_gb": total_gb
    }

# Initial VRAM check at module load
vram_info = get_basic_vram_info()
if "error" not in vram_info:
    print(f"📊 Initial VRAM status: {vram_info['free_gb']:.2f}GB free / {vram_info['total_gb']:.2f}GB total")
else:
    print(f"⚠️ VRAM check: {vram_info['error']} - SeedVR2 requires an NVIDIA GPU")

def get_vram_usage() -> Tuple[float, float, float]:
    """
    Get current VRAM usage (allocated, reserved, peak)
    
    Returns:
        tuple: (allocated_gb, reserved_gb, max_allocated_gb)
               Returns (0, 0, 0) if CUDA not available
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return allocated, reserved, max_allocated
    return 0, 0, 0


def clear_vram_cache(debug) -> None:
    """Clear VRAM cache and run garbage collection"""
        
    debug.log("Clearing VRAM cache...", category="cleanup")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def reset_vram_peak(debug) -> None:
    """
    Reset VRAM peak counter for new tracking
    """
    debug.log("Resetting VRAM peak memory statistics", category="memory")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def preinitialize_rope_cache(runner, debug) -> None:
    """
    🚀 Pre-initialize RoPE cache to avoid OOM at first launch
    
    Args:
        runner: The model runner containing DiT and VAE models
        debug: Optional Debug instance
    """
    
    debug.log("Pre-initializing RoPE cache to avoid OOM...", category="setup")

    try:
        # Create dummy tensors to simulate common shapes
        # Format: [batch, channels, frames, height, width] for vid_shape
        # Format: [batch, seq_len] for txt_shape
        common_shapes = [
            # Common video resolutions
            (torch.tensor([[1, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 1 frame, 77 tokens
            (torch.tensor([[4, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 4 frames
            (torch.tensor([[5, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 5 frames (4n+1 format)
            (torch.tensor([[1, 4, 4]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # Higher resolution
        ]
        
        # Create mock cache for pre-initialization
            
        temp_cache = Cache()
        
        # Access RoPE modules in DiT (recursive search)
        def find_rope_modules(module):
            rope_modules = []
            for name, child in module.named_modules():
                if hasattr(child, 'get_freqs') and callable(getattr(child, 'get_freqs')):
                    rope_modules.append((name, child))
            return rope_modules
        
        rope_modules = find_rope_modules(runner.dit)
        
        # Pre-calculate for each RoPE module found
        for name, rope_module in rope_modules:
            # Temporarily move module to CPU if necessary
            original_device = next(rope_module.parameters()).device if list(rope_module.parameters()) else torch.device('cpu')
            rope_module.to('cpu')
            
            try:
                for vid_shape, txt_shape in common_shapes:
                    cache_key = f"720pswin_by_size_bysize_{tuple(vid_shape[0].tolist())}_sd3.mmrope_freqs_3d"
                    
                    def compute_freqs():
                        try:
                            # Calculate with reduced dimensions to avoid OOM
                            with torch.no_grad():
                                # Detect RoPE module type
                                module_type = type(rope_module).__name__
                                
                                if module_type == 'NaRotaryEmbedding3d':
                                    # NaRotaryEmbedding3d: only takes shape (vid_shape)
                                    return rope_module.get_freqs(vid_shape.cpu())
                                else:
                                    # Standard RoPE: takes vid_shape and txt_shape
                                    return rope_module.get_freqs(vid_shape.cpu(), txt_shape.cpu())
                                    
                        except Exception as e:
                            debug.log(f"Failed for {cache_key}: {e}", level="WARNING", category="cache")
                            # Return empty tensors as fallback
                            time.sleep(1)
                            clear_vram_cache(debug)

                            return torch.zeros(1, 64)
                    
                    # Store in cache
                    temp_cache(cache_key, compute_freqs)
                
            except Exception as e:
                debug.log(f"Error in module {name}: {e}", level="ERROR", category="cache")
            finally:
                # Restore to original device
                rope_module.to(original_device)
        
        # Copy temporary cache to runner cache
        if hasattr(runner, 'cache'):
            runner.cache.cache.update(temp_cache.cache)
        else:
            runner.cache = temp_cache
        
    except Exception as e:
        debug.log(f"Error during RoPE pre-init: {e}", level="WARNING", category="setup", force=True)
        debug.log("Model will work but could have OOM at first launch", level="WARNING", category="setup", force=True)


def clear_rope_lru_caches(model) -> int:
    """Clear ALL LRU caches from RoPE modules"""
    cleared_count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'get_axial_freqs') and hasattr(module.get_axial_freqs, 'cache_clear'):
            module.get_axial_freqs.cache_clear()
            cleared_count += 1
    
    return cleared_count


def fast_model_cleanup(model):
    """Fast model cleanup without logs"""
    if model is None:
        return
    
    # Move to CPU
    model.to("cpu")
    
    # Clear parameters and buffers recursively
    def clear_recursive(m):
        for child in m.children():
            clear_recursive(child)
        for param in m.parameters():
            if param is not None:
                param.data = param.data.cpu()
                param.grad = None
        for buffer in m.buffers():
            if buffer is not None:
                buffer.data = buffer.data.cpu()
    
    clear_recursive(model)


def fast_ram_cleanup():
    """Fast RAM cleanup without excessive logging"""
    # Garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    
    # Clear PyTorch internal caches
    try:
        torch._C._clear_cache()
    except:
        pass
    

def clear_all_caches(runner, debug, offload_vae=False) -> int:
    """
    Aggressively clear all caches from runner and model.
    Optimized to only process what's necessary.
    
    Args:
        runner: The runner instance to clear caches from
        debug: Optional Debug instance for logging
        offload_vae: If True, also moves VAE to CPU and clears its caches
    """
    if not runner:
        return 0
    
    cleaned_items = 0
    
    # Early exit if no caches to clear
    has_cache = hasattr(runner, 'cache') and hasattr(runner.cache, 'cache')
    if not has_cache and not hasattr(runner, 'dit') and not (offload_vae and hasattr(runner, 'vae')):
        return 0
    
    # Clear main runner cache efficiently
    if has_cache and runner.cache.cache:
        cache_entries = len(runner.cache.cache)
        
        # Process all cache items to properly free memory
        for key, value in list(runner.cache.cache.items()):
            if torch.is_tensor(value):
                # Force deallocation of tensor storage
                if value.is_cuda:
                    value.data = value.data.cpu()
                value.grad = None
                if value.numel() > 0:
                    value.data.set_()  # Release underlying storage
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if torch.is_tensor(item):
                        if item.is_cuda:
                            item.data = item.data.cpu()
                        item.grad = None
                        if item.numel() > 0:
                            item.data.set_()
        
        # Clear the cache after processing
        runner.cache.cache.clear()
        cleaned_items += cache_entries
        debug.log(f"Cleared {cache_entries} cache entries", category="success")
    
    # Clear any accumulated state in blocks
    if hasattr(runner, 'dit'):
        model = runner.dit
        if hasattr(model, 'dit_model'):
            model = model.dit_model

        # Clear RoPE LRU caches
        rope_caches_cleared = clear_rope_lru_caches(model)
        cleaned_items += rope_caches_cleared
        if rope_caches_cleared > 0:
            debug.log(f"Cleared {rope_caches_cleared} RoPE LRU caches", category="success")
            
        # Clear block attributes if needed
        if hasattr(model, 'blocks'):
            block_attrs_cleared = 0
            
            # Define PyTorch's essential attributes that must NOT be deleted
            essential_attrs = {
                '_modules', '_parameters', '_buffers', 
                '_forward_hooks', '_forward_pre_hooks', 
                '_backward_hooks', '_backward_pre_hooks',
                '_state_dict_hooks', '_state_dict_pre_hooks',
                '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks',
                '_non_persistent_buffers_set', '_version',
                '_is_full_backward_hook', 'training',
                '_original_forward',  # BlockSwap attribute
                '_is_io_wrapped',     # BlockSwap attribute
                '_block_idx',         # BlockSwap attribute
            }
            
            for idx, block in enumerate(model.blocks):
                # Get all attributes that look like caches
                attrs_to_remove = []
                for attr_name in list(block.__dict__.keys()):
                    # Only remove cache-like attributes, not essential PyTorch attributes
                    if (attr_name not in essential_attrs and 
                        ('cache' in attr_name or 
                         'temp' in attr_name or 
                         (attr_name.startswith('_') and 
                          not attr_name.startswith('__') and 
                          attr_name not in essential_attrs))):
                        attrs_to_remove.append(attr_name)
                
                # Remove the identified attributes
                for attr_name in attrs_to_remove:
                    try:
                        delattr(block, attr_name)
                        block_attrs_cleared += 1
                    except AttributeError:
                        pass  # Already deleted or doesn't exist
                            
            if block_attrs_cleared > 0:
                debug.log(f"Cleared {block_attrs_cleared} temporary attributes from blocks", category="success")
    
    # Clear any temporary attributes that might accumulate
    temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                  '_rope_cache', '_intermediate_cache', '_backward_cache']
    
    # Check both runner and model for these attributes
    for obj in [runner, getattr(runner, 'dit', None)]:
        if obj is None:
            continue
            
        # Handle wrapped models
        if hasattr(obj, 'dit_model'):
            obj = obj.dit_model
            
        for attr in temp_attrs:
            if hasattr(obj, attr):
                delattr(obj, attr)
                cleaned_items += 1
                debug.log(f"Cleared {attr} from {type(obj).__name__}", category="success")
    
    # Handle VAE offloading if requested
    if offload_vae and hasattr(runner, 'vae') and runner.vae is not None:
        debug.log("Moving VAE to CPU and clearing intermediate tensors", category="cleanup")
        
        # Clear any intermediate tensors/buffers in VAE
        vae_caches_cleared = 0
        for module in runner.vae.modules():
            # Clear module-specific caches
            for cache_attr in ['_temp_cache', '_intermediate_cache']:
                if hasattr(module, cache_attr):
                    delattr(module, cache_attr)
                    vae_caches_cleared += 1
            
            # Clear any CUDA tensors in module attributes
            for attr_name in list(vars(module).keys()):
                attr = getattr(module, attr_name, None)
                if torch.is_tensor(attr) and attr.is_cuda:
                    # Move tensor to CPU if it's not a parameter/buffer
                    if attr_name not in module._parameters and attr_name not in module._buffers:
                        setattr(module, attr_name, attr.cpu())
                        vae_caches_cleared += 1
        
        if vae_caches_cleared > 0:
            debug.log(f"Cleared {vae_caches_cleared} VAE caches", category="success")
        
        # Move entire VAE to CPU (preserves model for reuse)
        runner.vae = runner.vae.to('cpu')
        debug.log("VAE moved to CPU, intermediate tensors cleared", category="success")
        cleaned_items += vae_caches_cleared
                
    # Force garbage collection
    gc.collect(2)  # Collect all generations
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return cleaned_items