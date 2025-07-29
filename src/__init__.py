"""
SeedVR2 Video Upscaler - Modular Architecture
Refactored from monolithic seedvr2.py for better maintainability

Author: Refactored codebase
Version: 2.0.0 - Modular

Available Modules:
- utils: Download and path utilities
- optimization: Memory, performance, and compatibility optimizations  
- core: Model management and generation pipeline (NEW)
- processing: Video and tensor processing (coming next)
- interfaces: ComfyUI integration
"""
'''
# Track which modules are available for progressive migration
MODULES_AVAILABLE = {
    'downloads': True,          # ✅ Module 1 - Downloads and model management
    'memory_manager': True,     # ✅ Module 2 - Memory optimization
    'performance': True,        # ✅ Module 3 - Performance optimizations  
    'compatibility': True,      # ✅ Module 4 - FP8/FP16 compatibility
    'model_manager': True,      # ✅ Module 5 - Model configuration and loading
    'generation': True,         # ✅ Module 6 - Generation loop and inference
    'video_transforms': True,   # ✅ Module 7 - Video processing and transforms
    'comfyui_node': True,       # ✅ Module 8 - ComfyUI node interface (COMPLETE!)
    'infer': True,              # ✅ Module 9 - Infer
}
'''
# Core imports (always available)
import os
import sys

# Add current directory to path for fallback imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
'''
# Progressive import system with fallback
# ===== MODULE 0: Constants =====
if MODULES_AVAILABLE['downloads']:
    from src.utils.constants import (
        get_base_cache_dir,
    )

# ===== MODULE 1: Downloads =====
if MODULES_AVAILABLE['downloads']:
    from src.utils.downloads import (
        download_weight,
    )
    

# ===== MODULE 2: Memory Manager ===== 
if MODULES_AVAILABLE['memory_manager']:
    from src.optimization.memory_manager import (
        get_vram_usage,
        clear_vram_cache,
        reset_vram_peak,
        preinitialize_rope_cache,
    )
    

# ===== MODULE 3: Performance =====
if MODULES_AVAILABLE['performance']:
    from src.optimization.performance import (
        optimized_video_rearrange,
        optimized_single_video_rearrange,
        optimized_sample_to_image_format,
        temporal_latent_blending,
    )
    

# ===== MODULE 4: Compatibility =====
if MODULES_AVAILABLE['compatibility']:
    from src.optimization.compatibility import (
        FP8CompatibleDiT,
    )
    

# ===== MODULE 5: Model Manager =====
if MODULES_AVAILABLE['model_manager']:
    from src.core.model_manager import (
        configure_runner,
        load_quantized_state_dict,
        configure_dit_model_inference,
        configure_vae_model_inference,
    )
    

# ===== MODULE 6: Generation =====
if MODULES_AVAILABLE['generation']:
    from src.core.generation import (
        generation_step,
        generation_loop,
        load_text_embeddings,
        calculate_optimal_batch_params,
        prepare_video_transforms
    )

# ===== MODULE 7: Video Transforms =====
if MODULES_AVAILABLE['infer']:
    from src.core.infer import VideoDiffusionInfer
    

# ===== MODULE 8: ComfyUI Node ===== 
if MODULES_AVAILABLE['comfyui_node']:
    try:
        from src.interfaces.comfyui_node import (
            SeedVR2,
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS
        )
    except:
        pass

# Export all available functions
__all__ = [
    # Constants
    'get_base_cache_dir',

    # Utils
    'download_weight', 
    
    # Memory Management
    'get_vram_usage', 'clear_vram_cache', 'reset_vram_peak', 
    'preinitialize_rope_cache', 
    
    # Performance & Video Processing
    'optimized_video_rearrange', 'optimized_single_video_rearrange', 'optimized_sample_to_image_format',
    'temporal_latent_blending', 
    'validate_video_format', 'ensure_4n_plus_1_format', 'calculate_padding_requirements', 'apply_wavelet_reconstruction', 'temporal_consistency_check',
    
    # Compatibility
    'FP8CompatibleDiT',
    
    # Core Model & Generation & Infer
    'configure_runner', 'load_quantized_state_dict', 'configure_dit_model_inference', 'configure_vae_model_inference',
    'generation_step', 'generation_loop', 'load_text_embeddings', 'calculate_optimal_batch_params',
    'prepare_video_transforms', 'VideoDiffusionInfer',
    
    # ComfyUI Interface
    'SeedVR2', 'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'format_execution_results',
    
    # Progress tracking
    'get_refactoring_progress'
] 
'''