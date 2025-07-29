"""
Unified debugging system for SeedVR2 generation pipeline

Provides structured logging, memory tracking, and performance monitoring
for all pipeline stages, including BlockSwap operations.
"""

import time
import torch
import psutil
import gc
from typing import Optional, List, Dict, Any, Tuple, Union, Set
from src.optimization.memory_manager import get_vram_usage, get_basic_vram_info


class Debug:
    """
    Unified debug logging for generation pipeline and BlockSwap monitoring
    
    Features:
    - Structured logging with categories
    - Memory tracking (VRAM/RAM)
    - Timing utilities
    - BlockSwap operation tracking
    - Minimal overhead when disabled
    """
    
    # Icon mapping for different categories
    CATEGORY_ICONS = {
        "general": "🔄",      # General operations/processing
        "timing": "⚡",       # Performance timing
        "memory": "📊",       # Memory usage tracking
        "cache": "💾",        # Cache operations
        "cleanup": "🧹",      # Cleanup operations
        "setup": "🔧",        # Configuration/setup
        "generation": "🎬",   # Generation process
        "model": "🚀",        # Model loading/operations
        "blockswap": "🔀",    # BlockSwap operations
        "download": "📥",     # Download operations
        "success": "✅",      # Successful completion
        "warning": "⚠️",      # Warnings
        "error": "❌",        # Errors
        "info": "ℹ️",         # Statistics/info
        "tip" :"💡",           # Tip/suggestion
        "video": "📹",        # Video/sequence info
        "reuse": "♻️",        # Reusing/recycling
        "runner": "🏃",       # Runner operations
        "vae": "🎨",          # VAE operations
        "store": "📦",        # Storing
        "precision": "🎯",    # Precision
        "none" : "",
    }
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.timers: Dict[str, float] = {}
        self.memory_checkpoints: List[Dict[str, Any]] = []
        self.timer_hierarchy: Dict[str, List[str]] = {}
        self.timer_durations: Dict[str, float] = {}
        self.timer_messages: Dict[str, str] = {} 
        self.swap_times: List[Dict[str, Any]] = []
        self.vram_history: List[float] = []
        self.active_timer_stack: List[str] = [] 
        
    def log(self, message: str, level: str = "INFO", category: str = "general", force: bool = False) -> None:
        """
        Log a categorized message
        
        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR)
            category: Category for the message
            force: If True, always log regardless of enabled state (for generic messages)
        """
        # Always log forced messages (generic messages that were previously print statements)
        # or log if debugging is enabled
        if force or self.enabled:
            # Get icon for category, fallback to general icon
            icon = self.CATEGORY_ICONS.get(category, self.CATEGORY_ICONS["general"])
            
            # Format prefix based on level
            if level == "WARNING":
                icon = self.CATEGORY_ICONS["warning"]
            elif level == "ERROR":
                icon = self.CATEGORY_ICONS["error"]
            
            # Build the log message
            prefix = f"{icon}"
            if level != "INFO":
                prefix += f" [{level}]"
            
            print(f"{prefix} {message}")
    
    def start_timer(self, name: str, force: bool = False) -> None:
        """
        Start a named timer
        
        Args:
            name: Timer name
            force: If True, start timer even when debug is disabled
        """
        if self.enabled or force:
            self.timers[name] = time.time()
            
            # Auto-hierarchy: if there's an active timer, this is a child
            if self.active_timer_stack:
                parent = self.active_timer_stack[-1]
                if parent not in self.timer_hierarchy:
                    self.timer_hierarchy[parent] = []
                # Only add if not already a child (prevents duplicates)
                if name not in self.timer_hierarchy[parent]:
                    self.timer_hierarchy[parent].append(name)
            
            # Push to stack
            self.active_timer_stack.append(name)
    
    def end_timer(self, name: str, message: Optional[str] = None, 
              force: bool = False, show_breakdown: bool = False,
              custom_children: Optional[Dict[str, float]] = None) -> float:
        """
        End a timer and optionally log its duration
        
        Args:
            name: Timer name
            message: Optional message to log with the duration
            force: If True, log even when debug is disabled (for critical timings)
            show_breakdown: If True, show breakdown of child timers
            custom_children: Optional dict of child timer names and durations to override automatic hierarchy
        
        Returns:
            Duration in seconds (0.0 if timer not found)
        """
        # Check if timer exists
        if name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.timer_durations[name] = duration
        # Store the message for later use in summary
        if message:
            self.timer_messages[name] = message
        del self.timers[name]

        # Pop from stack if this is the current active timer
        if self.active_timer_stack and self.active_timer_stack[-1] == name:
            self.active_timer_stack.pop()

        # If debug is disabled and not forcing, return early
        if not self.enabled and not force:
            return duration
        
        # ONLY log if show_breakdown is True - this means it's a major summary timer
        if message and show_breakdown:
            # Use custom children if provided, otherwise use automatic hierarchy
            if custom_children:
                children = custom_children
                child_total = sum(children.values())
                unaccounted = duration - child_total
                
                self.log(f"{message}: {duration:.2f}s", category="timing", force=force)
                
                # Sort custom children by duration for better readability
                sorted_children = sorted(children.items(), key=lambda x: x[1], reverse=True)
                
                for child_name, child_duration in sorted_children:
                    if child_duration >= 0.01:  # Only show if >= 10ms
                        self.log(f"  └─ {child_name}: {child_duration:.2f}s", category="timing", force=force)
            else:
                # Use automatic hierarchy tracking
                children = self.timer_hierarchy.get(name, [])
                child_total = sum(self.timer_durations.get(child, 0) for child in children)
                unaccounted = duration - child_total
                
                self.log(f"{message}: {duration:.2f}s", category="timing", force=force)
                
                # Sort children by duration for better readability
                sorted_children = sorted(children, key=lambda c: self.timer_durations.get(c, 0), reverse=True)

                for child in sorted_children:
                    child_duration = self.timer_durations.get(child, 0)
                    if child_duration >= 0.01:  # Only show if >= 10ms
                        child_message = self.timer_messages.get(child, child)
                        self.log(f"  └─ {child_message}: {child_duration:.2f}s", category="timing", force=force)
                        
                        # Recursively show grandchildren
                        if child in self.timer_hierarchy:
                            grandchildren = self.timer_hierarchy[child]
                            sorted_grandchildren = sorted(grandchildren, key=lambda c: self.timer_durations.get(c, 0), reverse=True)
                            
                            for grandchild in sorted_grandchildren:
                                grandchild_duration = self.timer_durations.get(grandchild, 0)
                                if grandchild_duration >= 0.01:  # Only show if >= 10ms
                                    grandchild_message = self.timer_messages.get(grandchild, grandchild)
                                    self.log(f"      └─ {grandchild_message}: {grandchild_duration:.2f}s", category="timing", force=force)
            
            if unaccounted > 0.01:  # Show if more than 10ms unaccounted
                self.log(f"  └─ (other operations): {unaccounted:.2f}s", category="timing", force=force)
        
        return duration
        
    def log_memory_state(self, label: str, show_diff: bool = True, show_tensors: bool = False) -> None:
        """Log current memory usage with optional diff and tensor count"""
        if not self.enabled:
            return
        
        # GPU Memory
        if torch.cuda.is_available():
            vram_allocated, vram_reserved, vram_max_allocated = get_vram_usage()
            vram_basic_info = get_basic_vram_info()
            
            if "error" not in vram_basic_info:
                vram_free = vram_basic_info["free_gb"]
                vram_total = vram_basic_info["total_gb"]
                vram_used = vram_total - vram_free
                
                # Clear, concise VRAM format
                vram_info = (f"[VRAM] {vram_allocated:.2f}GB allocated / "
                            f"{vram_reserved:.2f}GB reserved / "
                            f"{vram_free:.2f}GB free / "
                            f"{vram_total:.2f}GB total")
                self.vram_history.append(vram_allocated)
            else:
                vram_used = 0
                vram_free = 0
                vram_info = "VRAM: CPU mode"
        else:
            vram_used = 0
            vram_free = 0
            vram_info = "VRAM: CPU mode"
        
        # RAM Memory - Clear and informative
        ram_info = ""
        ram_process_gb = 0
        if psutil:
            try:
                # Process-specific memory
                process = psutil.Process()
                mem_info = process.memory_info()
                ram_process_gb = mem_info.rss / (1024**3)  # Physical memory used by our process
                
                # System-wide memory
                sys_mem = psutil.virtual_memory()
                ram_total_gb = sys_mem.total / (1024**3)
                ram_available_gb = sys_mem.available / (1024**3)
                
                # Calculate what's used by other processes
                ram_others_gb = ram_total_gb - ram_available_gb - ram_process_gb
                
                # Clear format matching user's request
                ram_info = (f" --- [RAM] {ram_process_gb:.1f}GB SeedVR2 / "
                        f"{ram_others_gb:.1f}GB other processes / "
                        f"{ram_available_gb:.1f}GB free / "
                        f"{ram_total_gb:.1f}GB total ")
                        
            except Exception:
                # Fallback to basic info
                try:
                    process = psutil.Process()
                    ram_process_gb = process.memory_info().rss / (1024**3)
                    ram_info = f" | RAM: {ram_process_gb:.1f}GB used"
                except:
                    pass
        
        # Tensor count (optional)
        tensor_info = ""
        if show_tensors:
            tensor_count = sum(1 for obj in gc.get_objects() if torch.is_tensor(obj))
            cuda_tensor_count = sum(1 for obj in gc.get_objects() 
                                if torch.is_tensor(obj) and obj.is_cuda)
            tensor_info = f" --- [Tensors] {cuda_tensor_count} on GPU / {tensor_count} total"
        
        # Build checkpoint
        checkpoint = {
            "label": label,
            "vram_used_gb": vram_used,
            "vram_allocated_gb": vram_allocated if torch.cuda.is_available() else 0,
            "vram_reserved_gb": vram_reserved if torch.cuda.is_available() else 0,
            "vram_free_gb": vram_free if torch.cuda.is_available() else 0,
            "ram_process_gb": ram_process_gb,
            "timestamp": time.time()
        }
        
        # Log the state
        self.log(f"{label}: {vram_info}{ram_info}{tensor_info}", category="memory")
        
        # Show diff from last checkpoint
        if show_diff and self.memory_checkpoints:
            last = self.memory_checkpoints[-1]
            vram_diff = vram_used - last["vram_used_gb"]
            ram_diff = ram_process_gb - last.get("ram_process_gb", ram_process_gb)
            
            diffs = []
            if abs(vram_diff) > 0.1:  # Significant VRAM change
                sign = "+" if vram_diff > 0 else ""
                diffs.append(f"VRAM {sign}{vram_diff:.2f}GB")
            if abs(ram_diff) > 0.1:  # Significant RAM change
                sign = "+" if ram_diff > 0 else ""
                diffs.append(f"RAM {sign}{ram_diff:.2f}GB")
            
            if diffs:
                self.log(f"  Memory changes: {', '.join(diffs)}", category="memory")
        
        self.memory_checkpoints.append(checkpoint)
    
    def log_swap_time(self, component_id: Union[int, str], duration: float, 
                 component_type: str = "block") -> None:
        """Log swap timing information for BlockSwap operations"""
        if self.enabled:
            # Store timing data
            self.swap_times.append({
                'component_id': component_id,
                'component_type': component_type,
                'duration': duration,
            })
            
            # Format message based on component type
            if component_type == "block":
                message = f"Block {component_id} swap: {duration*1000:.1f}ms"
            else:
                message = f"{component_type.capitalize()} {component_id} swap: {duration*1000:.1f}ms"
            
            self.log(message, category="blockswap")
    
    def clear_history(self) -> None:
        """Clear all history tracking"""
        self.timers.clear()
        self.memory_checkpoints.clear()
        self.swap_times.clear()
        self.vram_history.clear()
        self.timer_hierarchy.clear()
        self.timer_durations.clear()
        self.timer_messages.clear()
        self.active_timer_stack.clear()
    
    def get_swap_summary(self) -> Dict[str, Any]:
        """Get summary of swap operations for analysis"""
        if not self.swap_times:
            return {}
        
        # Group by component type
        block_swaps = [s for s in self.swap_times if s['component_type'] == 'block']
        io_swaps = [s for s in self.swap_times if s['component_type'] != 'block']
        
        # Calculate statistics
        summary = {
            'total_swaps': len(self.swap_times),
            'block_swaps': len(block_swaps),
            'io_swaps': len(io_swaps),
        }
        
        if block_swaps:
            block_times = [s['duration'] for s in block_swaps]
            summary['block_avg_ms'] = sum(block_times) * 1000 / len(block_times)
            summary['block_total_ms'] = sum(block_times) * 1000
            summary['block_min_ms'] = min(block_times) * 1000
            summary['block_max_ms'] = max(block_times) * 1000
            
            # Track which blocks are swapped most frequently
            block_frequency = {}
            for swap in block_swaps:
                block_id = swap['component_id']
                block_frequency[block_id] = block_frequency.get(block_id, 0) + 1
            summary['most_swapped_block'] = max(block_frequency, key=block_frequency.get)
            summary['most_swapped_count'] = block_frequency[summary['most_swapped_block']]
        
        if io_swaps:
            io_times = [s['duration'] for s in io_swaps]
            summary['io_avg_ms'] = sum(io_times) * 1000 / len(io_times)
            summary['io_total_ms'] = sum(io_times) * 1000
            
            # Track which I/O components are swapped
            io_components = list(set(s['component_id'] for s in io_swaps))
            summary['io_components_swapped'] = io_components
        
        # VRAM efficiency metrics
        if self.vram_history:
            summary['peak_vram_gb'] = max(self.vram_history)
            summary['avg_vram_gb'] = sum(self.vram_history) / len(self.vram_history)
            summary['vram_variation_gb'] = max(self.vram_history) - min(self.vram_history)
        
        return summary