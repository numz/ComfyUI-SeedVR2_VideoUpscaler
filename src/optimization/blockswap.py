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

from typing import Dict, Any, List, Optional
from .memory_manager import clear_memory
from .compatibility import call_rope_with_stability
from ..common.distributed import get_device


def is_blockswap_enabled(config: Optional[Dict[str, Any]]) -> bool:
    """
    Check if BlockSwap configuration indicates BlockSwap should be enabled.
    
    BlockSwap is enabled if either blocks_to_swap > 0 OR swap_io_components is True.
    This is the authoritative function for determining BlockSwap status from configuration.
    
    Args:
        config: BlockSwap configuration dictionary with optional keys:
            - blocks_to_swap: Number of blocks to offload (0 = disabled)
            - swap_io_components: Whether to offload I/O components
        
    Returns:
        True if BlockSwap should be active, False otherwise
    """
    if not config:
        return False
    
    blocks_to_swap = config.get("blocks_to_swap", 0)
    swap_io_components = config.get("swap_io_components", False)
    
    return blocks_to_swap > 0 or swap_io_components


def validate_blockswap_config(
    block_swap_config: Optional[Dict[str, Any]],
    dit_device: 'torch.device',
    dit_offload_device: Optional['torch.device'],
    debug: 'Debug'
) -> Optional[Dict[str, Any]]:
    """
    Validate and potentially modify BlockSwap configuration.
    
    Performs platform-specific validation and configuration adjustment:
    - On macOS (MPS): Auto-disables BlockSwap since unified memory makes it meaningless
    - On other platforms: Validates that offload_device is properly configured
    
    This is the single authoritative validation point for BlockSwap configuration,
    called early in configure_runner() before any model loading.
    
    Args:
        block_swap_config: BlockSwap configuration dictionary (may be None)
        dit_device: Target device for DiT model inference
        dit_offload_device: Device for offloading DiT blocks (may be None)
        debug: Debug instance for logging warnings/errors
        
    Returns:
        Validated/modified block_swap_config (may be None or modified copy)
        
    Raises:
        ValueError: If BlockSwap is enabled but offload_device is invalid (non-MPS only)
    """
    if not is_blockswap_enabled(block_swap_config):
        return block_swap_config
    
    blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
    swap_io_components = block_swap_config.get("swap_io_components", False)
    
    # Check for macOS unified memory - BlockSwap is meaningless there
    if dit_device.type == "mps":
        debug.log(
            f"BlockSwap disabled: macOS uses unified memory (no separate VRAM/RAM). "
            f"Ignoring blocks_to_swap={blocks_to_swap}, swap_io_components={swap_io_components}",
            level="WARNING", category="blockswap", force=True
        )
        # Return disabled config
        return {
            **block_swap_config,
            "blocks_to_swap": 0,
            "swap_io_components": False
        }
    
    # Validate offload_device is set and different from dit_device
    offload_device_valid = (
        dit_offload_device is not None and 
        str(dit_offload_device) != str(dit_device)
    )
    
    if not offload_device_valid:
        config_details = []
        if blocks_to_swap > 0:
            config_details.append(f"blocks_to_swap={blocks_to_swap}")
        if swap_io_components:
            config_details.append("swap_io_components=True")
        
        offload_str = str(dit_offload_device) if dit_offload_device else "none"
        raise ValueError(
            f"BlockSwap enabled ({', '.join(config_details)}) but dit_offload_device is invalid. "
            f"Current: device='{dit_device}', dit_offload_device='{offload_str}'. "
            f"BlockSwap requires offload_device on the DiT Model to be set and different from device. "
            f"Set --dit_offload_device cpu or disable BlockSwap."
        )
    
    return block_swap_config


# Timing helpers marked to skip torch.compile tracing
# These functions are excluded from Dynamo's graph tracing to avoid warnings
# about non-traceable builtins like time.time(), but they still execute normally
@torch._dynamo.disable
def _get_swap_start_time(debug, enabled: bool) -> Optional[float]:
    """Get start time for swap operation if debug is enabled."""
    return time.time() if debug and enabled else None


@torch._dynamo.disable  
def _log_swap_timing(debug, t_start: Optional[float], component_id, component_type: str) -> None:
    """Log swap timing if start time was captured."""
    if debug and t_start is not None:
        debug.log_swap_time(
            component_id=component_id,
            duration=time.time() - t_start,
            component_type=component_type
        )


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


def apply_block_swap_to_dit(
    runner: 'VideoDiffusionInfer',
    block_swap_config: Dict[str, Any],
    debug: 'Debug'
) -> None:
    """
    Apply block swapping configuration to a DiT model with OOM protection.
    
    This is the main entry point for configuring block swapping on a model.
    Handles block selection, I/O component offloading, device placement, and
    forward method wrapping for dynamic memory management.
    
    Args:
        runner: VideoDiffusionInfer instance containing the model
        block_swap_config: Configuration dictionary with keys:
            - blocks_to_swap: Number of blocks to swap (from the start)
            - swap_io_components: Whether to offload I/O components  
            - enable_debug: Whether to enable debug logging
            - offload_device: Device to offload to (default: 'cpu')
        debug: Debug instance for logging (required)
    """
    # Early return if BlockSwap not enabled
    if not is_blockswap_enabled(block_swap_config):
        return

    blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
    swap_io_components = block_swap_config.get("swap_io_components", False)
    
    # Early return only if both block swap and I/O swap are disabled
    if blocks_to_swap <= 0 and not swap_io_components:
        return
    
    if debug is None:
        if hasattr(runner, 'debug') and runner.debug is not None:
            debug = runner.debug
        else:
            raise ValueError("Debug instance must be provided to apply_block_swap_to_dit")
    
    debug.start_timer("apply_blockswap")

    # Get the actual model (handle CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model
    
    # Determine devices
    if hasattr(runner, '_dit_device'):
        device = runner._dit_device
    else:
        device = get_device()
    offload_device = block_swap_config.get("offload_device", torch.device('cpu'))

    # Validate model structure
    if not hasattr(model, "blocks"):
        debug.log("Model doesn't have 'blocks' attribute for BlockSwap", level="ERROR", category="blockswap", force=True)
        return

    total_blocks = len(model.blocks)
    
    # Clamp blocks_to_swap to available blocks BEFORE logging
    effective_blocks = min(blocks_to_swap, total_blocks) if blocks_to_swap > 0 else 0
    
    # Log configuration clearly based on what's enabled
    block_text = "block" if effective_blocks <= 1 else "blocks"
    if effective_blocks > 0 and swap_io_components:
        debug.log(f"BlockSwap: {effective_blocks}/{total_blocks} transformer {block_text} + I/O components offloaded to {str(offload_device).upper()}", category="blockswap", force=True)
    elif effective_blocks > 0:
        debug.log(f"BlockSwap: {effective_blocks}/{total_blocks} transformer {block_text} offloaded to {str(offload_device).upper()}", category="blockswap", force=True)
    elif swap_io_components:
        debug.log(f"BlockSwap: I/O components offloaded to {str(offload_device).upper()} (0/{total_blocks} blocks swapped)", category="blockswap", force=True)
    
    # Configure model with blockswap attributes
    if blocks_to_swap > 0:
        model.blocks_to_swap = effective_blocks - 1  # Convert to 0-indexed
    else:
        # No block swapping, set to -1 so no blocks match the swap condition
        model.blocks_to_swap = -1
    
    model.main_device = device
    model.offload_device = offload_device

    # Configure I/O components
    io_config = _configure_io_components(model, device, offload_device, 
                                        swap_io_components, debug)
    memory_stats = _configure_blocks(model, device, offload_device, debug)
    memory_stats['io_components'] = io_config['components']
    memory_stats['io_memory_mb'] = io_config['memory_mb']
    memory_stats['gpu_components'] = io_config['gpu_components']
    memory_stats['io_gpu_memory_mb'] = io_config['gpu_memory_mb']

    # Log memory summary
    _log_memory_summary(memory_stats, offload_device, device, swap_io_components, 
                       debug)
    
    # Wrap block forward methods for dynamic swapping (only if blocks_to_swap > 0)
    if blocks_to_swap > 0:
        for b, block in enumerate(model.blocks):
            if b <= model.blocks_to_swap:
                _wrap_block_forward(block, b, model, debug)

    # Patch RoPE modules for robust error handling
    _patch_rope_for_blockswap(model, debug)

    # Mark BlockSwap as active
    runner._blockswap_active = True

    # Store configuration for debugging and cleanup
    model._block_swap_config = {
        "blocks_swapped": blocks_to_swap,
        "swap_io_components": swap_io_components,
        "total_blocks": total_blocks,
        "offload_device": offload_device,
        "main_device": device,
        "offload_memory": memory_stats['offload_memory'],
        "main_memory": memory_stats['main_memory']
    }

    # Protect model from being moved entirely
    _protect_model_from_move(model, runner, debug)

    debug.log("BlockSwap configuration complete", category="success")
    debug.end_timer("apply_blockswap", "BlockSwap configuration application")
    

def _configure_io_components(
    model: torch.nn.Module,
    device: torch.device,
    offload_device: torch.device,
    swap_io_components: bool,
    debug: 'Debug'
) -> Dict[str, Any]:
    """
    Configure I/O component placement and wrapping with memory tracking.
    
    Handles all non-block modules (embeddings, normalization layers, etc.) by
    either keeping them on GPU or offloading them with dynamic swapping wrappers.
    
    Args:
        model: DiT model containing named children to configure
        device: Main computation device (typically GPU)
        offload_device: Device for offloaded components (typically CPU)
        swap_io_components: If True, offload I/O components with dynamic swapping
        debug: Debug instance for logging (required)
        
    Returns:
        Dictionary containing:
            - components: List of offloaded component names
            - memory_mb: Total memory of offloaded components in MB
            - gpu_components: List of components remaining on GPU
            - gpu_memory_mb: Total memory of GPU components in MB
    """
    io_components_offloaded = []
    io_components_on_gpu = []
    io_memory_mb = 0.0
    io_gpu_memory_mb = 0.0
    
    # Handle I/O modules with dynamic swapping
    for name, module in model.named_children():
        if name != "blocks":
            module_memory = get_module_memory_mb(module)
            
            if swap_io_components:
                module.to(offload_device)
                _wrap_io_forward(module, name, model, debug)
                io_components_offloaded.append(name)
                io_memory_mb += module_memory
                debug.log(f"{name} → {str(offload_device).upper()} ({module_memory:.2f}MB, dynamic swapping)", category="blockswap", indent_level=1)
            else:
                module.to(device)
                io_components_on_gpu.append(name)
                io_gpu_memory_mb += module_memory
                debug.log(f"{name} → {str(device).upper()} ({module_memory:.2f}MB)", category="blockswap", indent_level=1)

    return {
        'components': io_components_offloaded,
        'memory_mb': io_memory_mb,
        'gpu_components': io_components_on_gpu,
        'gpu_memory_mb': io_gpu_memory_mb
    }


def _configure_blocks(
    model: torch.nn.Module,
    device: torch.device,
    offload_device: torch.device,
    debug: 'Debug'
) -> Dict[str, float]:
    """
    Configure transformer block placement and calculate memory statistics.
    
    Moves blocks to their designated devices based on model.blocks_to_swap
    attribute. Blocks with index <= blocks_to_swap go to offload device,
    others stay on main device.
    
    Args:
        model: DiT model with blocks attribute and blocks_to_swap configured
        device: Main computation device for non-swapped blocks
        offload_device: Device for swapped blocks  
        debug: Debug instance for logging (required)
        
    Returns:
        Dictionary containing:
            - offload_memory: Total memory of offloaded blocks in MB
            - main_memory: Total memory of blocks on main device in MB
            - io_components: Empty list (populated by caller)
    """
    total_offload_memory = 0.0
    total_main_memory = 0.0

    # Move blocks based on swap configuration
    for b, block in enumerate(model.blocks):
        block_memory = get_module_memory_mb(block)

        if b > model.blocks_to_swap:
            block.to(device)
            total_main_memory += block_memory
        else:
            block.to(offload_device, non_blocking=False)
            total_offload_memory += block_memory

    # Ensure all buffers match their containing module's device
    for b, block in enumerate(model.blocks):
        target_device = device if b > model.blocks_to_swap else offload_device
        for name, buffer in block.named_buffers():
            if buffer.device != torch.device(target_device):
                buffer.data = buffer.data.to(target_device, non_blocking=False)

    return {
        "offload_memory": total_offload_memory,
        "main_memory": total_main_memory,
        "io_components": []  # Will be populated by caller
    }


def _log_memory_summary(
    memory_stats: Dict[str, float],
    offload_device: torch.device,
    device: torch.device,
    swap_io_components: bool,
    debug: 'Debug'
) -> None:
    """
    Log comprehensive memory usage summary for BlockSwap configuration.
    
    Displays detailed breakdown of memory distribution across devices,
    including transformer blocks and I/O components.
    
    Args:
        memory_stats: Dictionary containing:
            - offload_memory: Memory offloaded from blocks (MB)
            - main_memory: Memory remaining on main device (MB)
            - io_memory_mb: Memory from offloaded I/O components (MB)
            - io_gpu_memory_mb: Memory from I/O components on GPU (MB)
        offload_device: Device used for offloading
        device: Main computation device
        swap_io_components: Whether I/O components are being swapped
        debug: Debug instance for logging (required)
    """
    debug.log("BlockSwap memory configuration:", category="blockswap")
    
    # Log transformer blocks memory
    blocks_offloaded = memory_stats['offload_memory']
    blocks_on_gpu = memory_stats['main_memory']
    
    offload_str = str(offload_device)
    device_str = str(device)
    
    if blocks_on_gpu == 0:
        debug.log(f"Transformer blocks: {blocks_offloaded:.2f}MB on {offload_str} (dynamic swapping)", category="blockswap", indent_level=1)
    else:
        debug.log(f"Transformer blocks: {blocks_on_gpu:.2f}MB on {device_str}, {blocks_offloaded:.2f}MB on {offload_str}", category="blockswap", indent_level=1)
    
    # Always log I/O components (whether swapping or not)
    io_memory = memory_stats.get('io_memory_mb', 0.0)
    io_gpu_memory = memory_stats.get('io_gpu_memory_mb', 0.0)
    
    if swap_io_components and io_memory > 0:
        io_components = memory_stats.get('io_components', [])
        debug.log(f"I/O components: {io_memory:.2f}MB on {offload_str} (dynamic swapping)", category="blockswap", indent_level=1)
        debug.log(f"{', '.join(io_components)}", category="blockswap", indent_level=2)
    elif io_gpu_memory > 0:
        io_gpu_components = memory_stats.get('gpu_components', [])
        debug.log(f"I/O components: {io_gpu_memory:.2f}MB on {device_str}", category="blockswap", indent_level=1)
        debug.log(f"{', '.join(io_gpu_components)}", category="blockswap", indent_level=2)
    
    # Log total VRAM savings
    total_offloaded = blocks_offloaded + (io_memory if swap_io_components else 0)
    if total_offloaded > 0:
        debug.log(f"Total VRAM saved: {total_offloaded:.2f}MB (~{total_offloaded/1024:.2f}GB)", category="blockswap", indent_level=1)


def _wrap_block_forward(
    block: torch.nn.Module,
    block_idx: int,
    model: torch.nn.Module,
    debug: 'Debug'
) -> None:
    """
    Wrap individual transformer block forward for dynamic device swapping.
    
    Creates a wrapped forward method that automatically:
    1. Moves block to GPU before computation
    2. Executes original forward pass
    3. Moves block back to offload device after computation
    4. Logs timing and manages memory pressure
    
    Uses weak references to prevent memory leaks from closure retention.
    
    Args:
        block: Individual transformer block to wrap
        block_idx: Index of this block in model.blocks
        model: Parent DiT model (used for device references)
        debug: Debug instance for logging (required)
    """
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
            # Use dynamo-disabled helper to get start time (avoids compilation warnings)
            t_start = _get_swap_start_time(debug, debug.enabled if debug else False)

            # Only move to GPU if necessary
            current_device = next(self.parameters()).device
            target_device = torch.device(model.main_device)
            
            if current_device != target_device:
                self.to(model.main_device, non_blocking=False)

            # Execute forward pass with OOM protection
            output = original_forward(*args, **kwargs)

            # Move back to offload device
            self.to(model.offload_device, non_blocking=False)
            
            # Use dynamo-disabled helper to log timing (avoids compilation warnings)
            _log_swap_timing(debug, t_start, self._block_idx, "block")

            # Only clear cache under memory pressure
            clear_memory(debug=debug, deep=False, force=False, timer_name="wrap_block_forward")
        else:
            output = original_forward(*args, **kwargs)

        return output

    # Bind the wrapped function as a method to the block
    block.forward = types.MethodType(wrapped_forward, block)
    
    # Store reference to original forward for cleanup
    block._original_forward = original_forward


def _wrap_io_forward(
    module: torch.nn.Module,
    module_name: str,
    model: torch.nn.Module,
    debug: 'Debug'
) -> None:
    """
    Wrap I/O component forward for dynamic device swapping.
    
    Similar to _wrap_block_forward but for I/O components (embeddings,
    normalization layers, etc.). Handles swapping between GPU and CPU
    during forward passes.
    
    Uses weak references to prevent circular dependencies and memory leaks.
    
    Args:
        module: I/O component module to wrap
        module_name: Name identifier for logging (e.g., 'x_embedder')
        model: Parent DiT model (used for device references)
        debug: Debug instance for logging (required)
    """
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

        # Use dynamo-disabled helper to get start time (avoids compilation warnings)
        t_start = _get_swap_start_time(debug, debug.enabled if debug else False)
        
        # Check current device to avoid unnecessary moves
        current_device = next(self.parameters()).device
        target_device = torch.device(model.main_device)
        
        # Move to GPU for computation if needed
        if current_device != target_device:
            self.to(model.main_device, non_blocking=False)

        # Execute forward pass
        output = self._original_forward(*args, **kwargs)

        # Move back to offload device
        self.to(model.offload_device, non_blocking=False)
        
        # Use dynamo-disabled helper to log timing (avoids compilation warnings)
        _log_swap_timing(debug, t_start, self._module_name, "I/O")

        # Only clear cache under memory pressure
        clear_memory(debug=debug, deep=False, force=False, timer_name="wrap_block_forward")

        return output
    
    # Bind as a method
    module.forward = types.MethodType(wrapped_io_forward, module)
    module._is_io_wrapped = True
    
    # Store module reference for restoration
    if not hasattr(model, '_io_swappers'):
        model._io_swappers = []
    model._io_swappers.append((module, module_name))


def _patch_rope_for_blockswap(
    model: torch.nn.Module,
    debug: 'Debug'
) -> None:
    """
    Patch RoPE (Rotary Position Embedding) modules for device-aware fallback.
    
    Adds CPU fallback logic to RoPE modules to handle device mismatch errors
    that can occur during BlockSwap operations. Complements the stability
    wrapper from compatibility.py with device-specific error handling.
    
    Args:
        model: DiT model containing RoPE modules to patch
        debug: Debug instance for logging (required)
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
                            debug.log(f"RoPE OOM for {module_name}", level="WARNING", category="rope", force=True)
                            debug.log(f"Clearing RoPE cache and retrying", category="info", force=True)
                            
                            # Get current device from parameters
                            try:
                                current_device = next(self.parameters()).device
                            except StopIteration:
                                # Fallback: use model's main_device if BlockSwap has set it, else use offload_device
                                if hasattr(model, 'main_device'):
                                    current_device = torch.device(model.main_device)
                                elif hasattr(model, 'offload_device'):
                                    current_device = torch.device(model.offload_device)
                            
                            # Try clearing cache first (non-invasive fix)
                            if hasattr(current_fn, 'cache_clear'):
                                current_fn.cache_clear()
                                try:
                                    # Retry on same device after clearing cache
                                    return current_fn(*args, **kwargs)
                                except Exception as retry_error:
                                    # Cache clear wasn't enough, need more drastic measures
                                    debug.log(f"Cache clear insufficient for {module_name}, falling back to CPU", level="WARNING", category="rope", force=True)
                            
                            # Fallback to CPU computation with stability
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


def _protect_model_from_move(
    model: torch.nn.Module,
    runner: 'VideoDiffusionInfer',
    debug: 'Debug'
) -> None:
    """
    Protect model from unintended full device movement during BlockSwap.
    
    Wraps model.to() method to prevent other code from accidentally moving
    the entire model to GPU, which would defeat BlockSwap's memory savings.
    Allows movement only when explicitly bypassed via model flag.
    
    Args:
        model: DiT model to protect
        runner: VideoDiffusionInfer instance (for active status check)
        debug: Debug instance for logging (required)
    """
    if not hasattr(model, '_original_to'):
        # Store runner reference as weak reference to avoid circular refs
        model._blockswap_runner_ref = weakref.ref(runner)
        model._original_to = model.to
        
        # Define the protected method without closures
        def protected_model_to(self, device, *args, **kwargs):
            # Check if protection is temporarily bypassed for offloading
            # Flag is stored on model itself (not runner) to survive runner recreation
            if getattr(self, "_blockswap_bypass_protection", False):
                # Protection bypassed, allow movement
                if hasattr(self, '_original_to'):
                    return self._original_to(device, *args, **kwargs)
            
            # Get configured offload device directly from model
            blockswap_offload_device = "cpu"  # default
            if hasattr(self, "_block_swap_config"):
                blockswap_offload_device = self._block_swap_config.get("offload_device", "cpu")
            
            # Check if BlockSwap is currently active via runner weak reference
            runner_ref = getattr(self, '_blockswap_runner_ref', None)
            blockswap_is_active = False
            if runner_ref:
                runner_obj = runner_ref()
                if runner_obj and hasattr(runner_obj, "_blockswap_active"):
                    blockswap_is_active = runner_obj._blockswap_active
            
            # Block attempts to move model away from configured offload device when active
            if blockswap_is_active and str(device) != str(blockswap_offload_device):
                # Get debug instance from runner if available
                debug_instance = None
                if runner_ref:
                    runner_obj = runner_ref()
                    if runner_obj and hasattr(runner_obj, 'debug'):
                        debug_instance = runner_obj.debug
                
                if debug_instance:
                    debug_instance.log(
                        f"Blocked attempt to move BlockSwap model from {blockswap_offload_device} to {device}",
                        level="WARNING", category="blockswap", force=True
                    )
                return self
            
            # Allow movement (either bypass is enabled or target is offload device)
            if hasattr(self, '_original_to'):
                return self._original_to(device, *args, **kwargs)
            else:
                # Fallback - shouldn't happen
                return super(type(self), self).to(device, *args, **kwargs)
        
        # Bind as a method to the model instance
        model.to = types.MethodType(protected_model_to, model)


def set_blockswap_bypass(runner, bypass: bool, debug):
    """
    Set or unset bypass flag for BlockSwap protection.
    Used for offloading to temporarily allow model movement.
    
    Args:
        runner: Runner instance with BlockSwap
        bypass: True to bypass protection, False to enforce it
        debug: Debug instance for logging
    """
    if not hasattr(runner, "_blockswap_active") or not runner._blockswap_active:
        return
    
    # Get the actual model (handle CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model
    
    # Store on model so it survives runner recreation during caching
    model._blockswap_bypass_protection = bypass
    
    if bypass:
        debug.log("BlockSwap protection disabled to allow model DiT offloading", category="success")
    else:
        debug.log("BlockSwap protection renabled to avoid accidentally offloading the entire DiT model", category="success")


def cleanup_blockswap(runner, keep_state_for_cache=False):
    """
    Clean up BlockSwap configuration based on caching mode.
    
    When caching (keep_state_for_cache=True):
    - Keep all BlockSwap configuration intact
    - Only mark as inactive for safety during non-inference operations
    
    When not caching (keep_state_for_cache=False):
    - Full cleanup of all BlockSwap state
    
    Args:
        runner: VideoDiffusionInfer instance to clean up
        keep_state_for_cache: If True, preserve BlockSwap state for reuse
    """
    # Get debug instance from runner
    if not hasattr(runner, 'debug') or runner.debug is None:
        raise ValueError("Debug instance must be available on runner for cleanup_blockswap")
    
    debug = runner.debug
    
    # Get the actual model (handle CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model
    
    # Check if there's any BlockSwap state to clean up (check both runner and model)
    has_blockswap_state = (
        hasattr(runner, "_blockswap_active") or 
        hasattr(model, "_block_swap_config") or
        hasattr(model, "_blockswap_bypass_protection")
    )
    
    if not has_blockswap_state:
        return

    debug.log("Starting BlockSwap cleanup", category="cleanup")

    if keep_state_for_cache:
        # Minimal cleanup for caching - just mark as inactive and allow offloading
        # Everything else stays intact for fast reactivation
        if hasattr(runner, "_blockswap_active") and runner._blockswap_active:
            if not getattr(model, "_blockswap_bypass_protection", False):
                set_blockswap_bypass(runner=runner, bypass=True, debug=debug)
            runner._blockswap_active = False
        debug.log("BlockSwap deactivated for caching (configuration preserved)", category="success")
        return

    # Full cleanup when not caching
    # Get the actual model (handle CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model

    # 1. Restore block forward methods
    if hasattr(model, 'blocks'):
        restored_count = 0
        for block in model.blocks:
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
                restored_count += 1
                
                # Clean up wrapper attributes
                for attr in ['_block_idx', '_model_ref', '_debug_ref', '_blockswap_wrapped']:
                    if hasattr(block, attr):
                        delattr(block, attr)
        
        if restored_count > 0:
            debug.log(f"Restored {restored_count} block forward methods", category="success")

    # 2. Restore RoPE patches
    if hasattr(model, '_rope_patches'):
        for module, original_method in model._rope_patches:
            module.get_axial_freqs = original_method
            # Clean up wrapper attributes
            for attr in ['_rope_wrapped', '_original_get_axial_freqs']:
                if hasattr(module, attr):
                    delattr(module, attr)
        debug.log(f"Restored {len(model._rope_patches)} RoPE methods", category="success")
        delattr(model, '_rope_patches')

    # 3. Restore I/O component forward methods and move to offload device
    if hasattr(model, '_io_swappers'):
        for module, module_name in model._io_swappers:
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                # Clean up wrapper attributes
                for attr in ['_original_forward', '_model_ref', '_debug_ref', 
                           '_module_name', '_is_io_wrapped']:
                    if hasattr(module, attr):
                        delattr(module, attr)
        debug.log(f"Restored {len(model._io_swappers)} I/O components", category="success")
        delattr(model, '_io_swappers')
    
    # Move all IO components to offload device during full cleanup
    if hasattr(model, 'offload_device'):
        offload_device = model.offload_device
        moved_count = 0
        for name, module in model.named_children():
            if name != "blocks":
                module.to(offload_device)
                moved_count += 1
        if moved_count > 0:
            debug.log(f"Moved {moved_count} IO components to offload device", category="success")

    # 4. Restore original .to() method
    if hasattr(model, '_original_to'):
        model.to = model._original_to
        delattr(model, '_original_to')
        debug.log("Restored original .to() method", category="success")

    # 5. Clean up BlockSwap-specific attributes
    for attr in ['_blockswap_runner_ref', 'blocks_to_swap', 'main_device', 
                 'offload_device']:
        if hasattr(model, attr):
            delattr(model, attr)

    # 6. Clean up runner attributes
    runner._blockswap_active = False
    
    # Remove all config attributes
    for attr in ['_cached_blockswap_config', '_block_swap_config', '_blockswap_debug']:
        if hasattr(runner, attr):
            delattr(runner, attr)
    
    debug.log("BlockSwap cleanup complete", category="success")
