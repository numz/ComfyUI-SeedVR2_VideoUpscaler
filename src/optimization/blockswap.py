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
import psutil
import gc
import comfy.model_management as mm
from typing import Dict, Any, List, Tuple, Callable

from src.optimization.memory_manager import get_vram_usage


class BlockSwapDebugger:
    """
    Debug logger for BlockSwap operations.
    
    Tracks memory usage, swap timings, and provides detailed logging
    for debugging and performance analysis of block swapping operations.
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize the debugger.
        
        Args:
            enabled: Whether debug logging is enabled
        """
        self.enabled = enabled
        self.swap_times: List[Tuple[int, float, str]] = []
        self.vram_history: List[float] = []
        self.device_map: Dict[int, str] = {}

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message if debugging is enabled.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARN, SWAP, etc.)
        """
        if self.enabled:
            print(f"[{level}] {message}")

    def log_swap_time(self, block_idx: int, duration: float, direction: str) -> None:
        """
        Log block swap timing information.
        
        Args:
            block_idx: Index of the block being swapped
            duration: Time taken for the swap in seconds
            direction: Direction of swap ("compute" for GPU->CPU->GPU cycle)
        """
        self.swap_times.append((block_idx, duration, direction))
        if self.enabled:
            self.log(
                f"Block {block_idx} swap {direction}: {duration*1000:.1f}ms", "SWAP"
            )

    def log_memory_state(self, stage: str, show_tensors: bool = False) -> None:
        """
        Log current memory state for debugging.

        Args:
            stage: Description of current stage
            show_tensors: Whether to count and display tensor count
        """
        # GPU Memory
        if torch.cuda.is_available():
            allocated_gb, reserved_gb, peak_gb = get_vram_usage()
            vram_info = (
                f"VRAM: {allocated_gb:.2f}/{reserved_gb:.2f}GB (peak: {peak_gb:.2f}GB)"
            )
            self.vram_history.append(allocated_gb)
        else:
            vram_info = "VRAM: CPU mode"

        # RAM Memory
        ram_info = ""
        if psutil:
            try:
                process = psutil.Process()
                ram_gb = process.memory_info().rss / (1024**3)
                ram_info = f" | RAM: {ram_gb:.1f}GB"
            except Exception:
                # psutil might fail in some environments
                pass

        # Tensor count (optional - expensive operation)
        tensor_info = ""
        if show_tensors:
            tensor_count = sum(1 for obj in gc.get_objects() if torch.is_tensor(obj))
            tensor_info = f" | Tensors: {tensor_count}"

        self.log(f"🧮 {stage}: {vram_info}{ram_info}{tensor_info}")


class BlockSwapper:
    """
    Handles dynamic block swapping between GPU and CPU.
    
    This class wraps a transformer block's forward method to automatically
    move the block to GPU for computation and back to CPU afterwards.
    """

    def __init__(
        self,
        block_idx: int,
        original_forward: Callable,
        offload_device: str,
        use_non_blocking: bool,
        debugger: BlockSwapDebugger,
    ):
        """
        Initialize the block swapper.
        
        Args:
            block_idx: Index of the block being wrapped
            original_forward: Original forward method of the block
            offload_device: Device to offload to (typically "cpu")
            use_non_blocking: Whether to use non-blocking transfers
            debugger: BlockSwapDebugger instance for logging
        """
        self.block_idx = block_idx
        self.original_forward = original_forward
        self.offload_device = offload_device
        self.use_non_blocking = use_non_blocking
        self.debugger = debugger
        self._swap_count = 0
        self._total_swap_time = 0.0

    def __call__(self, block_self, *args, **kwargs):
        """
        Execute the block with automatic GPU/CPU swapping.
        
        Args:
            block_self: The block instance (self from the method binding)
            *args: Positional arguments for the forward method
            **kwargs: Keyword arguments for the forward method
            
        Returns:
            Output from the block's forward method
        """
        self._swap_count += 1
        t_start = time.time() if self.debugger.enabled else None

        # Move block to GPU
        block_self.to("cuda", non_blocking=self.use_non_blocking)
        if self.use_non_blocking and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Execute forward pass
        output = self.original_forward(*args, **kwargs)

        # Log timing if enabled
        if self.debugger.enabled and t_start is not None:
            self.debugger.log_swap_time(
                self.block_idx, time.time() - t_start, "compute"
            )

        # Move block back to CPU
        block_self.to(self.offload_device, non_blocking=self.use_non_blocking)
        if self.use_non_blocking and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Periodic cache cleanup to prevent memory fragmentation
        if self.block_idx % 4 == 0:
            torch.cuda.empty_cache()

        if t_start:
            self._total_swap_time += time.time() - t_start

        return output


class IOSwapper:
    """
    Handles dynamic I/O component swapping between GPU and CPU.
    
    Similar to BlockSwapper but designed for I/O components like
    embeddings, patch_embed, etc. that need different handling.
    """

    def __init__(
        self,
        component_name: str,
        original_forward: Callable,
        offload_device: str,
        use_non_blocking: bool,
        debugger: BlockSwapDebugger,
    ):
        """
        Initialize the I/O component swapper.
        
        Args:
            component_name: Name of the component being wrapped
            original_forward: Original forward method
            offload_device: Device to offload to (typically "cpu")
            use_non_blocking: Whether to use non-blocking transfers
            debugger: BlockSwapDebugger instance for logging
        """
        self.component_name = component_name
        self.original_forward = original_forward
        self.offload_device = offload_device
        self.use_non_blocking = use_non_blocking
        self.debugger = debugger
        self._swap_count = 0

    def __call__(self, component_self, *args, **kwargs):
        """
        Execute the component with automatic GPU/CPU swapping.
        
        Args:
            component_self: The component instance
            *args: Positional arguments for the forward method
            **kwargs: Keyword arguments for the forward method
            
        Returns:
            Output from the component's forward method
        """
        self._swap_count += 1

        # Move to GPU
        component_self.to("cuda", non_blocking=self.use_non_blocking)
        if self.use_non_blocking:
            torch.cuda.synchronize()

        # Forward pass
        output = self.original_forward(*args, **kwargs)

        # Move back to CPU
        component_self.to(self.offload_device, non_blocking=self.use_non_blocking)
        if self.use_non_blocking:
            torch.cuda.synchronize()

        # Force VRAM cleanup for I/O components
        torch.cuda.empty_cache()

        return output


def apply_block_swap_to_dit(runner, block_swap_config: Dict[str, Any]) -> None:
    """
    Apply block swap configuration to a DiT model.
    
    This is the main entry point for configuring block swapping on a model.
    It handles block selection, I/O component offloading, RoPE patching,
    and device placement.
    
    Args:
        runner: VideoDiffusionInfer instance containing the model
        block_swap_config: Configuration dictionary with keys:
            - blocks_to_swap: Number of blocks to swap (from the end)
            - offload_io_components: Whether to offload I/O components
            - use_non_blocking: Whether to use non-blocking transfers
            - enable_debug: Whether to enable debug logging
    """
    if not block_swap_config:
        return

    blocks_to_swap = block_swap_config.get("blocks_to_swap", 0)
    offload_io_components = block_swap_config.get("offload_io_components", False)

    # Early exit if nothing to swap
    if blocks_to_swap <= 0 and not offload_io_components:
        return
    
    # Initialize debugger
    debugger = BlockSwapDebugger(enabled=block_swap_config.get("enable_debug", False))
    debugger.log("Starting BlockSwap configuration...")

    # Determine devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    offload_device = str(mm.unet_offload_device())
    use_non_blocking = block_swap_config.get("use_non_blocking", True)

    # Get the actual model (handle FP8CompatibleDiT wrapper)
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model

    # Validate model has blocks
    if blocks_to_swap > 0 and not hasattr(model, "blocks"):
        debugger.log("Model doesn't have 'blocks' attribute for BlockSwap", "WARN")
        return

    # Get total blocks and validate request
    total_blocks = len(model.blocks) if hasattr(model, "blocks") else 0
    if blocks_to_swap > 0:
        debugger.log(f"Model has {total_blocks} blocks total")
        if blocks_to_swap > total_blocks:
            debugger.log(
                f"WARNING: Requested {blocks_to_swap} blocks but model only has {total_blocks}",
                "WARN",
            )
        blocks_to_swap = min(blocks_to_swap, total_blocks)

    debugger.log(
        f"Configuring: {blocks_to_swap}/{total_blocks} blocks, I/O offload: {offload_io_components}"
    )
    debugger.log_memory_state("Before BlockSwap", show_tensors=True)

    # Handle I/O components offloading
    io_swap_count = 0
    io_patches: List[Tuple[Any, Callable]] = []
    
    for name, module in model.named_children():
        if name != "blocks":  # Everything except transformer blocks
            if offload_io_components:
                # Offload to CPU with dynamic swapping
                module.to(offload_device)
                original_forward = module.forward
                swapper = IOSwapper(
                    name,
                    original_forward,
                    offload_device,
                    use_non_blocking,
                    debugger,
                )
                module.forward = types.MethodType(swapper, module)
                module._io_swapper = swapper
                io_patches.append((module, original_forward))
                io_swap_count += 1
                debugger.log(f"  {name} → {offload_device} (with dynamic swapping)")
            else:
                # Normal behavior: keep on GPU
                module.to(device)
                debugger.log(f"  {name} → {device}")

    # Store I/O patches for cleanup
    if io_swap_count > 0:
        runner._io_patches = io_patches
        debugger.log(f"✅ Configured {io_swap_count} I/O components for offloading")

    # Configure block placement
    if blocks_to_swap > 0:
        blocks_to_keep_on_gpu = total_blocks - blocks_to_swap
        
        # Move non-swapped blocks to GPU
        debugger.log(f"Moving first {blocks_to_keep_on_gpu} blocks to GPU...")
        for idx in range(blocks_to_keep_on_gpu):
            model.blocks[idx].to(device)
            debugger.log(f"  Block {idx} → {device}")

        # Ensure all buffers match device placement
        # This prevents device mismatch errors with RoPE and other buffers
        for idx in range(blocks_to_keep_on_gpu):
            for module in model.blocks[idx].modules():
                if hasattr(module, "_buffers"):
                    for buffer_name, buffer in module._buffers.items():
                        if buffer is not None and buffer.device != torch.device(device):
                            module._buffers[buffer_name] = buffer.to(device)
        
        debugger.log(f"✅ Ensured all buffers match block devices")

        # Patch RoPE modules for robust OOM handling
        rope_patches = _patch_rope_for_blockswap(model, debugger)
        runner._rope_patches = rope_patches

        # Configure block swapping for the last N blocks
        swap_count = 0
        for idx in range(blocks_to_keep_on_gpu, total_blocks):
            block = model.blocks[idx]
            block._original_forward = block.forward
            
            # Create and bind swapper
            swapper = BlockSwapper(
                idx,
                block._original_forward,
                offload_device,
                use_non_blocking,
                debugger,
            )
            block.forward = types.MethodType(swapper, block)
            block._swapper = swapper
            swap_count += 1

        debugger.log(f"✅ Configured {swap_count} blocks for swapping")

    # Mark BlockSwap as active and store configuration
    runner._blockswap_active = True
    runner._block_swap_config = {
        "blocks_swapped": blocks_to_swap,
        "io_components_offloaded": io_swap_count,
        "total_blocks": total_blocks,
        "use_non_blocking": use_non_blocking,
        "offload_device": offload_device,
        "main_device": device,
    }

    # Protect model from being moved entirely
    _protect_model_from_move(runner)

    # Final memory state and cleanup
    torch.cuda.empty_cache()
    debugger.log_memory_state("After BlockSwap", show_tensors=True)
    debugger.log("✅ BlockSwap configuration complete")


def _patch_rope_for_blockswap(model, debugger: BlockSwapDebugger) -> List[Tuple[Any, Callable]]:
    """
    Patch RoPE modules to handle OOM gracefully during block swapping.
    
    RoPE (Rotary Position Embeddings) can cause device mismatches when
    blocks are on different devices. This patches the get_axial_freqs
    method to handle these cases robustly.
    
    Args:
        model: The model containing RoPE modules
        debugger: BlockSwapDebugger instance for logging
        
    Returns:
        List of (module, original_method) tuples for cleanup
    """
    rope_patches: List[Tuple[Any, Callable]] = []
    rope_patched = 0

    # Map RoPE modules to their containing blocks
    rope_to_block: Dict[Any, int] = {}
    if hasattr(model, "blocks"):
        for block_idx, block in enumerate(model.blocks):
            for name, module in block.named_modules():
                if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
                    rope_to_block[module] = block_idx

    # Patch all RoPE modules
    for name, module in model.named_modules():
        if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
            # Store original methods
            original_method = module.get_axial_freqs
            original_unbound = module.__class__.get_axial_freqs
            block_idx = rope_to_block.get(module, -1)

            def make_robust_wrapper(
                rope_module, module_name: str, bound_method: Callable, 
                unbound_method: Callable, parent_block_idx: int
            ) -> Callable:
                """Create a robust wrapper for RoPE get_axial_freqs method."""
                def robust_method(*args, **kwargs):
                    try:
                        # Try the normal bound method first
                        return bound_method(*args, **kwargs)
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Handle device mismatches and OOM errors
                        if (
                            "device" in error_msg
                            or "memory" in error_msg
                            or isinstance(e, (RuntimeError, KeyError))
                        ):
                            debugger.log(f"RoPE issue for {module_name}: {e}")

                            # Get current device
                            current_device = "cuda"
                            if list(rope_module.parameters()):
                                current_device = next(rope_module.parameters()).device

                            # Try with unbound method (helps with cache context)
                            try:
                                return unbound_method(rope_module, *args, **kwargs)
                            except Exception:
                                # Final fallback: compute on CPU
                                debugger.log(f"RoPE fallback to CPU for {module_name}")
                                rope_module.cpu()
                                try:
                                    # Clear any LRU cache
                                    if hasattr(bound_method, "cache_clear"):
                                        bound_method.cache_clear()
                                    
                                    result = unbound_method(rope_module, *args, **kwargs)
                                    rope_module.to(current_device)
                                    
                                    # Move result back if it's a tensor
                                    if hasattr(result, "to"):
                                        return result.to(current_device)
                                    return result
                                except Exception as cpu_error:
                                    # Restore device even on error
                                    rope_module.to(current_device)
                                    raise cpu_error
                        # Re-raise if not a handled error type
                        raise

                return robust_method

            # Create and apply patched method
            patched_method = make_robust_wrapper(
                module, name, original_method, original_unbound, block_idx
            )
            module.get_axial_freqs = patched_method

            # Store for cleanup
            rope_patches.append((module, original_method))
            rope_patched += 1

    if rope_patched > 0:
        debugger.log(
            f"✅ Patched {rope_patched} RoPE modules with robust device handling"
        )

    return rope_patches


def _protect_model_from_move(runner) -> None:
    """
    Protect model from being moved entirely to GPU when BlockSwap is active.
    
    This prevents other code from accidentally moving the entire model to GPU
    which would defeat the purpose of block swapping.
    
    Args:
        runner: VideoDiffusionInfer instance containing the model
    """
    if hasattr(runner.dit, "dit_model"):
        # Handle FP8CompatibleDiT wrapper case
        original_dit_to = runner.dit.dit_model.to
        runner._original_dit_to = original_dit_to

        def protected_dit_to(device, *args, **kwargs):
            """Protected .to() method that blocks GPU moves during BlockSwap."""
            if str(device) != "cpu" and hasattr(runner, "_blockswap_active"):
                print("⚠️ Blocked attempt to move blockswapped model to GPU")
                return runner.dit.dit_model
            return original_dit_to(device, *args, **kwargs)

        runner.dit.dit_model.to = protected_dit_to
    else:
        # Direct model case
        original_dit_to = runner.dit.to
        runner._original_dit_to = original_dit_to

        def protected_dit_to(device, *args, **kwargs):
            """Protected .to() method that blocks GPU moves during BlockSwap."""
            if str(device) != "cpu" and hasattr(runner, "_blockswap_active"):
                print("⚠️ Blocked attempt to move blockswapped model to GPU")
                return runner.dit
            return original_dit_to(device, *args, **kwargs)

        runner.dit.to = protected_dit_to


def cleanup_blockswap(runner) -> None:
    """
    Clean up BlockSwap configurations and restore original methods.
    
    This should be called when BlockSwap is no longer needed to restore
    the model to its original state and free up any resources.
    
    Args:
        runner: VideoDiffusionInfer instance to clean up
    """
    if not hasattr(runner, "_blockswap_active"):
        return

    # Restore original .to() method
    if hasattr(runner, "_original_dit_to"):
        if hasattr(runner.dit, "dit_model"):
            runner.dit.dit_model.to = runner._original_dit_to
        else:
            runner.dit.to = runner._original_dit_to
        delattr(runner, "_original_dit_to")

    # Restore original RoPE methods and clear their caches
    if hasattr(runner, "_rope_patches"):
        for module, original_method in runner._rope_patches:
            if hasattr(original_method, "cache_clear"):
                original_method.cache_clear()
            module.get_axial_freqs = original_method
        runner._rope_patches.clear()
        del runner._rope_patches

    # Restore original I/O component methods
    if hasattr(runner, "_io_patches"):
        for module, original_forward in runner._io_patches:
            module.forward = original_forward
            if hasattr(module, "_io_swapper"):
                delattr(module, "_io_swapper")
        runner._io_patches.clear()
        del runner._io_patches

    # Clean up BlockSwap markers
    if hasattr(runner, "_blockswap_active"):
        delattr(runner, "_blockswap_active")
    if hasattr(runner, "_block_swap_config"):
        delattr(runner, "_block_swap_config")

    # Clean up swapper references from blocks
    model = runner.dit
    if hasattr(model, "dit_model"):
        model = model.dit_model
    
    if hasattr(model, "blocks"):
        for block in model.blocks:
            block.to("cpu")
            if hasattr(block, "_swapper"):
                delattr(block, "_swapper")
            if hasattr(block, "_original_forward"):
                block.forward = block._original_forward
                delattr(block, "_original_forward")