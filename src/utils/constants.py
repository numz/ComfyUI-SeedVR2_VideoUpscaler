"""
Shared constants and utilities for SeedVR2
Only includes constants actually used in the codebase
"""

import os

# Model folder names
SEEDVR2_FOLDER_NAME = "SEEDVR2"  # Physical folder name on disk
SEEDVR2_MODEL_TYPE = "seedvr2"   # Model type identifier for ComfyUI

# Supported model file formats
#SUPPORTED_MODEL_EXTENSIONS = {'.safetensors', '.gguf'}
SUPPORTED_MODEL_EXTENSIONS = {'.safetensors'}

def get_script_directory():
    """Get the root script directory path (3 levels up from this file)"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_base_cache_dir():
    """Get or create the model cache directory"""
    try:
        import folder_paths
        cache_dir = os.path.join(folder_paths.models_dir, SEEDVR2_FOLDER_NAME)
        folder_paths.add_model_folder_path(SEEDVR2_MODEL_TYPE, cache_dir)
    except:
        cache_dir = f"./{SEEDVR2_MODEL_TYPE}_models"
    
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def is_supported_model_file(filename: str) -> bool:
    """Check if a file has a supported model extension"""
    return any(filename.endswith(ext) for ext in SUPPORTED_MODEL_EXTENSIONS)