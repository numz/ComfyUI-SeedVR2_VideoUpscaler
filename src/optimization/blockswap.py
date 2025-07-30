"""
BlockSwap Module for SeedVR2

This module implements dynamic block swapping between GPU and CPU memory
to enable running large models on limited VRAM systems.

Key Features:
- Dynamic transformer block offloading during inference
- Non-blocking GPU transfers for optimal performance
- RoPE computation fallback to CPU on OOM
- Minimal performance overhead with intelligent caching
- I/O component offloading for maximum memory savings
"""

import time
import types
import torch
import weakref
import gc
from typing import Dict, Any, List, Tuple, Optional, Union
from src.optimization.memory_manager import get_vram_usage
from src.optimization.compatibility import call_rope_with_stability


def get_module_memory_mb(module: torch.nn.Module) -> float:
    """
    Calculate memory usage of a module in MB.
    
    Args:
        module: PyTorch module to measure
        
    Returns:
        Memory usage in megabytes
    """
    total_bytes = sum(
        param.nelement() * param.element_size() 
        for param in module.parameters() 
        if param.data is not None
    )
    return total_bytes / (1024 * 1024)


def apply_block_swap_to_dit(runner, block_swap_config: Dict[str, Any], debug) -> None:
    """
    Apply block swapping configuration to a DIT model with OOM protection.
    
    This is the main entry point for configuring block swapping on a model.
    It handles block selection, I/O component offloading, and device placement.
    
    Args:
        runner: VideoDiffusionInfer instance containing the model
        block_swap_config: Configuration dictionary with keys:
            - blocks_to_swap: Number of blocks to swap (from the start)
            - offload_io_components: Whether to offload I/O components
            - use_non_blocking: Whether to use non-blocking transfers
            - enable_debug: Whether to enable debug logging
    """
    if not block_swap_config:
        return

    blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
    if blocks_to_swap <= 0:
        return
    
    if debug is None:
        if hasattr(runner, 'debug') and runner.debug is not None:
            debug = runner.debug
        else:
            raise ValueError("Debug instance must be provided to apply_block_swap_to_dit")
    
    debug.start_timer("apply_blockswap")
    debug.log_memory_state("Before BlockSwap")

    # Get the actual model (handle FP8CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model
    
    # Determine devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    offload_device = "cpu"
    use_non_blocking = block_swap_config.get("use_non_blocking", True)

    configs = []
    blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
    if blocks_to_swap > 0:
        configs.append(f"{blocks_to_swap} blocks")
    if block_swap_config.get("use_non_blocking", False):
        configs.append("non-blocking")
    if block_swap_config.get("offload_io_components", False):
        configs.append("I/O components")
    debug.log(f"BlockSwap configured: {', '.join(configs)}", category="blockswap", force=True)
        
    # Validate model structure
    if not hasattr(model, "blocks"):
        debug.log("Model doesn't have 'blocks' attribute for BlockSwap", level="WARNING", category="blockswap")
        return

    total_blocks = len(model.blocks)
    debug.log(f"Model has {total_blocks} blocks total", category="blockswap")
    blocks_to_swap = min(blocks_to_swap, total_blocks)

    # Configure model with blockswap attributes
    model.blocks_to_swap = blocks_to_swap - 1  # Convert to 0-indexed
    model.main_device = device
    model.offload_device = offload_device
    model.use_non_blocking = use_non_blocking

    debug.log(f"Configuring: {blocks_to_swap}/{total_blocks} blocks for swapping", category="blockswap")
    
    # Configure I/O components
    offload_io_components = block_swap_config.get("offload_io_components", False)
    io_components_offloaded = _configure_io_components(model, device, offload_device, use_non_blocking, 
                                                       offload_io_components, debug)

    # Configure block placement and memory tracking
    memory_stats = _configure_blocks(model, device, offload_device, use_non_blocking, debug)
    memory_stats['io_components'] = io_components_offloaded

     # Log memory summary
    _log_memory_summary(memory_stats, offload_device, device, offload_io_components, 
                       use_non_blocking, debug)
    
    # Wrap block forward methods for dynamic swapping
    for b, block in enumerate(model.blocks):
        if b <= model.blocks_to_swap:
            _wrap_block_forward(block, b, model, debug)

    # Patch RoPE modules for robust error handling
    _patch_rope_for_blockswap(model, debug)

    # Mark BlockSwap as active
    runner._blockswap_active = True

    # Store configuration for debugging and cleanup
    runner._block_swap_config = {
        "blocks_swapped": blocks_to_swap,
        "offload_io_components": offload_io_components,
        "total_blocks": total_blocks,
        "use_non_blocking": use_non_blocking,
        "offload_device": offload_device,
        "main_device": device,
        "enable_debug": block_swap_config.get("enable_debug", False),
        "offload_memory": memory_stats['offload_memory'],
        "main_memory": memory_stats['main_memory']
    }

    # Protect model from being moved entirely
    _protect_model_from_move(model, runner, debug)

    debug.log("BlockSwap configuration complete", category="success")
    debug.log_memory_state("After BlockSwap")
    debug.end_timer("apply_blockswap", "BlockSwap configuration applied")
    

def _configure_io_components(model, device: str, offload_device: str, 
                            use_non_blocking: bool, offload_io_components: bool,
                            debug) -> List[str]:
    """Configure I/O component placement and wrapping."""
    io_components_offloaded = []
    
    # Process non-block parameters
    for name, param in model.named_parameters():
        if "block" not in name:
            target_device = offload_device if offload_io_components else device
            param.data = param.data.to(target_device, non_blocking=use_non_blocking)
            status = "(offloaded)" if offload_io_components else ""
            debug.log(f"  {name} → {target_device} {status}", category="blockswap")

    # Handle I/O modules with dynamic swapping
    for name, module in model.named_children():
        if name != "blocks":
            if offload_io_components:
                module.to(offload_device)
                _wrap_io_forward(module, name, model, debug)
                io_components_offloaded.append(name)
                debug.log(f"  {name} → {offload_device} (with dynamic swapping)", category="blockswap")
            else:
                module.to(device)
                debug.log(f"  {name} → {device}", category="blockswap")

    return io_components_offloaded


def _configure_blocks(model, device: str, offload_device: str, 
                     use_non_blocking: bool, debug) -> Dict[str, float]:
    """Configure block placement and calculate memory statistics."""
    total_offload_memory = 0.0
    total_main_memory = 0.0

    # Move blocks based on swap configuration
    for b, block in enumerate(model.blocks):
        block_memory = get_module_memory_mb(block)

        if b > model.blocks_to_swap:
            block.to(device)
            total_main_memory += block_memory
        else:
            block.to(offload_device, non_blocking=use_non_blocking)
            total_offload_memory += block_memory

    # Ensure all buffers match their containing module's device
    for b, block in enumerate(model.blocks):
        target_device = device if b > model.blocks_to_swap else offload_device
        for name, buffer in block.named_buffers():
            if buffer.device != torch.device(target_device):
                buffer.data = buffer.data.to(target_device)

    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    return {
        "offload_memory": total_offload_memory,
        "main_memory": total_main_memory,
        "io_components": []  # Will be populated by caller
    }


def _log_memory_summary(memory_stats: Dict[str, float], offload_device: str, 
                       device: str, offload_io_components: bool,
                       use_non_blocking: bool, debug) -> None:
    """Log memory usage summary."""
    debug.log("BlockSwap memory configuration:", category="blockswap")
    if memory_stats['main_memory'] == 0:
        debug.log(f"  All {memory_stats['offload_memory']:.2f}MB transformer blocks offloaded to {offload_device}", category="blockswap")
        debug.log(f"  VRAM usage: {memory_stats['main_memory']:.2f}MB (blocks loaded on-demand during inference)", category="blockswap")
    else:
        debug.log(f"  Transformer blocks on {offload_device}: {memory_stats['offload_memory']:.2f}MB", category="blockswap")
        debug.log(f"  Transformer blocks on {device}: {memory_stats['main_memory']:.2f}MB", category="blockswap")
    
    total_memory = memory_stats['offload_memory'] + memory_stats['main_memory']
    debug.log(f"  Total transformer blocks memory: {total_memory:.2f}MB", category="blockswap")
    
    if offload_io_components and memory_stats.get('io_components'):
        debug.log(f"  I/O components offloaded: {', '.join(memory_stats['io_components'])}", category="blockswap")
    
    debug.log(f"  Non-blocking GPU transfers: {'Enabled' if use_non_blocking else 'Disabled'}", category="blockswap")



def _wrap_block_forward(block: torch.nn.Module, block_idx: int, model: torch.nn.Module, debug) -> None:
    """Wrap individual block forward to handle device movement using weak references to prevent leaks."""
    
    if hasattr(block, '_original_forward'):
        return  # Already wrapped

    # Store original forward method
    original_forward = block.forward
    
    # Create weak references
    model_ref = weakref.ref(model)
    debug_ref = weakref.ref(debug)
    
    # Store block_idx on the block itself to avoid closure issues
    block._block_idx = block_idx
    
    def wrapped_forward(self, *args, **kwargs):
        # Retrieve weak references
        model = model_ref()
        debug = debug_ref()
        
        if not model:
            # Model has been garbage collected, fall back to original
            return original_forward(*args, **kwargs)

        # Check if block swap is active for this block
        if hasattr(model, 'blocks_to_swap') and self._block_idx <= model.blocks_to_swap:
            t_start = time.time() if debug and debug.enabled else None

            # Only move to GPU if necessary
            current_device = next(self.parameters()).device
            target_device = torch.device(model.main_device)
            
            if current_device != target_device:
                self.to(model.main_device, non_blocking=model.use_non_blocking)
                
            # Synchronize if needed
            if hasattr(model, 'use_non_blocking') and not model.use_non_blocking:
                torch.cuda.synchronize()

            # Execute forward pass with OOM protection
            output = original_forward(*args, **kwargs)

            # Move back to offload device
            self.to(model.offload_device, non_blocking=model.use_non_blocking)
            
            # Log timing if debug is available
            if debug and t_start is not None:
                debug.log_swap_time(
                    component_id=self._block_idx,
                    duration=time.time() - t_start,
                    component_type="block"
                )

            # Only clear cache under memory pressure
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.9:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            output = original_forward(*args, **kwargs)

        return output

    # Bind the wrapped function as a method to the block
    block.forward = types.MethodType(wrapped_forward, block)
    
    # Store reference to original forward for cleanup
    block._original_forward = original_forward


def _wrap_io_forward(module: torch.nn.Module, module_name: str, model: torch.nn.Module, debug) -> None:
    """Wrap I/O component forward using weak references to prevent memory leaks."""
    
    if hasattr(module, '_is_io_wrapped') and module._is_io_wrapped:
        debug.log(f"Reusing existing I/O wrapper for {module_name}", category="reuse")
        return  # Already wrapped

    # Store original forward method
    original_forward = module.forward
    
    # Create weak references
    model_ref = weakref.ref(model)
    debug_ref = weakref.ref(debug) if debug else lambda: None
    
    # Store module name on the module itself
    module._module_name = module_name
    module._original_forward = original_forward
    
    def wrapped_io_forward(self, *args, **kwargs):
        # Retrieve weak references
        model = model_ref()
        debug = debug_ref()
        
        if not model:
            # Model has been garbage collected, fall back to original
            return self._original_forward(*args, **kwargs)

        t_start = time.time() if debug and debug.enabled else None
        
        # Check current device to avoid unnecessary moves
        current_device = next(self.parameters()).device
        target_device = torch.device(model.main_device)
        
        # Move to GPU for computation if needed
        if current_device != target_device:
            self.to(model.main_device)
            
        # Synchronize if not using non-blocking transfers
        if hasattr(model, 'use_non_blocking') and not model.use_non_blocking:
            torch.cuda.synchronize()

        # Execute forward pass
        output = self._original_forward(*args, **kwargs)

        # Move back to offload device
        self.to(model.offload_device, non_blocking=model.use_non_blocking)
        
        # Log timing if debug is available
        if debug and t_start is not None:
            debug.log_swap_time(
                component_id=self._module_name,
                duration=time.time() - t_start,
                component_type="io"
            )

        # Only clear cache under memory pressure
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.9:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return output
    
    # Bind as a method
    module.forward = types.MethodType(wrapped_io_forward, module)
    module._is_io_wrapped = True
    
    # Store module reference for restoration
    if not hasattr(model, '_io_swappers'):
        model._io_swappers = []
    model._io_swappers.append((module, module_name))


def _patch_rope_for_blockswap(model, debug) -> None:
    """
    Patch RoPE modules to handle device mismatches gracefully.
    
    This complements the stability wrapper from compatibility.py by adding
    device-aware error handling. Only handles device/memory errors, letting
    other exceptions bubble up to the stability wrapper if present.
    """
    rope_patches = []
    
    for name, module in model.named_modules():
        if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
            # Skip if already wrapped by blockswap
            if hasattr(module, '_blockswap_wrapped') and module._blockswap_wrapped:
                continue
            
            # Get current method (might be stability-wrapped)
            current_method = module.get_axial_freqs
            
            # Create device-aware wrapper with proper closure handling
            def make_device_aware_wrapper(module_name, current_fn):
                def device_aware_rope_wrapper(self, *args, **kwargs):
                    try:
                        # Try current method (original or stability-wrapped)
                        return current_fn(*args, **kwargs)
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        error_msg = str(e).lower()
                        # Only handle device/memory specific errors
                        if any(x in error_msg for x in ["device", "memory", "allocation"]):
                            debug.log(f"RoPE device issue for {module_name}: {e}", level="WARNING", category="blockswap")
                            
                            # Get current device from parameters
                            current_device = next(self.parameters()).device if list(self.parameters()) else torch.device("cuda")
                            
                            # Try clearing cache first (non-invasive fix)
                            if hasattr(current_fn, 'cache_clear'):
                                current_fn.cache_clear()
                                try:
                                    return current_fn(*args, **kwargs)
                                except:
                                    pass
                            
                            # Fallback to CPU computation with stability
                            debug.log(f"RoPE fallback to CPU for {module_name} (attempting cache reuse)", category="reuse")
                            self.cpu()
                            
                            try:
                                # Use call_rope_with_stability for CPU computation
                                # This ensures cache is cleared and autocast disabled
                                original_fn = getattr(self, '_original_get_axial_freqs', current_fn)
                                result = call_rope_with_stability(original_fn, *args, **kwargs)
                                
                                # Move module back to original device
                                self.to(current_device)
                                
                                # Move result to appropriate device if it's a tensor
                                if hasattr(result, 'to'):
                                    target_device = args[0].device if len(args) > 0 and hasattr(args[0], 'device') else current_device
                                    return result.to(target_device)
                                return result
                                
                            except Exception as cpu_error:
                                # Always restore device even on error
                                self.to(current_device)
                                raise cpu_error
                        else:
                            # Not a device error, let it bubble up
                            raise
                
                return device_aware_rope_wrapper
            
            # Apply wrapper
            module.get_axial_freqs = types.MethodType(
                make_device_aware_wrapper(name, current_method), 
                module
            )
            module._blockswap_wrapped = True
            
            # Store for cleanup (use original or previously stored)
            original_method = getattr(module, '_original_get_axial_freqs', current_method)
            rope_patches.append((module, original_method))
    
    if rope_patches:
        model._rope_patches = rope_patches
        debug.log(f"Patched {len(rope_patches)} RoPE modules with device handling", category="success")


def _protect_model_from_move(model, runner, debug) -> None:
    """
    Protect model from being moved entirely to GPU when BlockSwap is active.
    
    This prevents other code from accidentally moving the entire model to GPU
    which would defeat the purpose of block swapping.
    """
    if not hasattr(model, '_original_to'):
        # Store runner reference as weak reference to avoid circular refs
        model._blockswap_runner_ref = weakref.ref(runner)
        model._original_to = model.to
        
        # Define the protected method without closures
        def protected_model_to(self, device, *args, **kwargs):
            # Check blockswap status using weak reference
            if str(device) != "cpu":
                runner_ref = getattr(self, '_blockswap_runner_ref', None)
                if runner_ref:
                    runner_obj = runner_ref()
                    if runner_obj and hasattr(runner_obj, "_blockswap_active") and runner_obj._blockswap_active:
                        debug.log("Blocked attempt to move blockswapped model to GPU", level="WARNING", category="blockswap", force=True)
                        return self
            
            # Use original method stored as attribute
            if hasattr(self, '_original_to'):
                return self._original_to(device, *args, **kwargs)
            else:
                # This shouldn't happen, but fallback to super().to()
                return super(type(self), self).to(device, *args, **kwargs)
        
        # Bind as a method to the model instance
        model.to = types.MethodType(protected_model_to, model)


def cleanup_blockswap(runner, keep_state_for_cache: bool = False) -> None:
    """
    Clean up BlockSwap configurations and restore original methods.
    
    This should be called when BlockSwap is no longer needed to restore
    the model to its original state and free up any resources.
    
    Args:
        runner: VideoDiffusionInfer instance to clean up
        keep_state_for_cache: If True, stores configuration for fast re-application
    """
    # Get debug instance from runner
    if not hasattr(runner, 'debug') or runner.debug is None:
        raise ValueError("Debug instance must be available on runner for cleanup_blockswap")
    
    debug = runner.debug
    
    # Early return if BlockSwap not active
    if not hasattr(runner, "_blockswap_active") or not runner._blockswap_active:
        if debug:
            debug.log("BlockSwap not active, skipping cleanup", level="WARNING", category="blockswap")
        return

    debug.log("Starting BlockSwap cleanup", category="cleanup")

    # Get the actual model (handle FP8CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model

    # Store configuration BEFORE cleanup if caching
    cached_config = None
    if keep_state_for_cache and hasattr(runner, "_block_swap_config"):
        cached_config = {
            "blocks_to_swap": runner._block_swap_config.get("blocks_swapped"),
            "offload_io_components": runner._block_swap_config.get("offload_io_components"),
            "use_non_blocking": runner._block_swap_config.get("use_non_blocking"),
            "offload_device": runner._block_swap_config.get("offload_device"),
            "main_device": runner._block_swap_config.get("main_device"),
            "enable_debug": runner._block_swap_config.get("enable_debug", False),
        }
        runner._cached_blockswap_config = cached_config
        debug.log("Storing configuration for fast re-application", category="store")

    # Restore block forward methods
    if hasattr(model, 'blocks'):
        restored_count = 0
        for idx, block in enumerate(model.blocks):
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
                restored_count += 1
                
                # Clean up ALL wrapper attributes
                attrs_to_clean = ['_block_idx', '_model_ref', '_debug_ref', '_blockswap_wrapped']
                for attr in attrs_to_clean:
                    if hasattr(block, attr):
                        delattr(block, attr)
                
                # Clear gradients to free memory
                block.zero_grad(set_to_none=True)
                
                # Move block to CPU and ensure all buffers follow
                if not keep_state_for_cache:
                    block.to("cpu")
                    # Force memory deallocation for all parameters and buffers
                    for param in block.parameters():
                        if param.data.numel() > 0:
                            param.data.set_()
                    for buffer in block.buffers():
                        if buffer.data.numel() > 0:
                            buffer.data.set_()
        
        if restored_count > 0:
            debug.log(f"Restored original forward for {restored_count} blocks", category="success")

    # Restore RoPE methods and clear LRU caches
    if hasattr(model, '_rope_patches'):
        if keep_state_for_cache:
            # Just clear caches but keep the device-aware wrappers
            for module, original_method in model._rope_patches:
                if hasattr(module.get_axial_freqs, 'cache_clear'):
                    module.get_axial_freqs.cache_clear()
                if hasattr(original_method, 'cache_clear'):
                    original_method.cache_clear()
            debug.log(f"Cleared {len(model._rope_patches)} RoPE caches (kept device-aware wrappers)", category="success")
        else:
            # Full cleanup - restore original methods
            for module, original_method in model._rope_patches:
                if hasattr(module.get_axial_freqs, 'cache_clear'):
                    module.get_axial_freqs.cache_clear()
                if hasattr(original_method, 'cache_clear'):
                    original_method.cache_clear()
                module.get_axial_freqs = original_method
                # Clean up wrapper attributes
                if hasattr(module, '_rope_wrapped'):
                    delattr(module, '_rope_wrapped')
                if hasattr(module, '_original_get_axial_freqs'):
                    delattr(module, '_original_get_axial_freqs')
            debug.log(f"Restored {len(model._rope_patches)} RoPE modules", category="success")
            delattr(model, '_rope_patches')
    else:
        # Fallback: Clear RoPE caches without restoration
        cleared_count = 0
        for module in model.modules():
            if hasattr(module, 'get_axial_freqs') and hasattr(module.get_axial_freqs, 'cache_clear'):
                module.get_axial_freqs.cache_clear()
                cleared_count += 1
        if cleared_count > 0:
            debug.log(f"Cleared {cleared_count} RoPE LRU caches", category="success")

    # Restore I/O component forward methods
    if hasattr(model, '_io_swappers'):
        for module, module_name in model._io_swappers:
            if hasattr(module, '_is_io_wrapped') and hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                # Clean up wrapper attributes
                attrs_to_clean = ['_original_forward', '_model_ref', '_debug_ref', 
                                 '_module_name', '_is_io_wrapped']
                for attr in attrs_to_clean:
                    if hasattr(module, attr):
                        delattr(module, attr)
        debug.log(f"Restored {len(model._io_swappers)} I/O component wrappers", category="success")
        delattr(model, '_io_swappers')

    # Restore original .to() method
    if hasattr(model, '_original_to'):
        model.to = model._original_to
        delattr(model, '_original_to')
        debug.log("Restored original .to() method", category="success")

    # Clean up weak reference on model
    if hasattr(model, '_blockswap_runner_ref'):
        delattr(model, '_blockswap_runner_ref')

    # Clean up BlockSwap attributes from model
    attrs_to_remove = ["blocks_to_swap", "main_device", "offload_device", "use_non_blocking"]
    for attr in attrs_to_remove:
        if hasattr(model, attr):
            delattr(model, attr)

    # Mark model as not configured
    if hasattr(model, '_blockswap_configured'):
        delattr(model, '_blockswap_configured')

    # Move model to CPU to free VRAM
    if not keep_state_for_cache:
        model.to("cpu")
        debug.log("Moved model to CPU", category="store")

    # Clean up runner attributes
    runner._blockswap_active = False
    
    # Remove all config attributes if not caching
    if not cached_config:
        if hasattr(runner, "_cached_blockswap_config"):
            delattr(runner, "_cached_blockswap_config")
        if hasattr(runner, "_block_swap_config"):
            delattr(runner, "_block_swap_config")

    # Clear debug reference (only if not caching)
    if not keep_state_for_cache and hasattr(runner, '_blockswap_debug'):
        delattr(runner, '_blockswap_debug')

    # Clear local debug reference
    debug = None

    # Force garbage collection (multiple passes for thorough cleanup)
    gc.collect(2)  # Full collection including oldest generation
    gc.collect()
    gc.collect()

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()