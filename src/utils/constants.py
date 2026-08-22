"""
Shared constants and utilities for SeedVR2
Only includes constants actually used in the codebase
"""

# Version information
__version__ = "2.5.24"

import os
import warnings
import inspect
from typing import Optional

# Model folder names and ComfyUI model categories
SEEDVR2_FOLDER_NAME = "SEEDVR2"  # Standalone fallback folder name
SEEDVR2_MODEL_TYPE = "seedvr2"  # Legacy model type identifier
DIT_MODEL_TYPE = "diffusion_models"
VAE_MODEL_TYPE = "vae"

# Supported model file formats
SUPPORTED_MODEL_EXTENSIONS = {'.safetensors', '.gguf'}

# GGUF Quantization Constants
QK_K = 256
K_SCALE_SIZE = 12
GGUF_BLOCK_SIZE = 32
GGUF_TYPE_SIZE = 64

# Download configuration
HUGGINGFACE_BASE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"
DOWNLOAD_CHUNK_SIZE = 8192 * 1024  # 8MB chunks for hash calculation
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 2  # seconds

def get_script_directory() -> str:
    """Get the root script directory path (3 levels up from this file)"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_legacy_model_dir() -> str:
    """Return the legacy SeedVR2 directory used by standalone installations."""
    return os.path.join(".", "models", SEEDVR2_FOLDER_NAME)


def get_legacy_model_paths() -> list:
    """Return legacy directories used by standalone and ComfyUI installations."""
    paths = []
    try:
        import folder_paths
        # Preserve paths registered by the original SeedVR2 integration or by
        # extra_model_paths.yaml under either spelling of the legacy category.
        if hasattr(folder_paths, "folder_names_and_paths"):
            for folder_type, (registered, _) in folder_paths.folder_names_and_paths.items():
                if folder_type.lower() in {SEEDVR2_MODEL_TYPE.lower(), SEEDVR2_FOLDER_NAME.lower()}:
                    paths.extend(registered)
        try:
            paths.extend(folder_paths.get_folder_paths(SEEDVR2_MODEL_TYPE))
        except Exception:
            pass
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            paths.append(os.path.join(models_dir, SEEDVR2_FOLDER_NAME))
    except Exception:
        pass
    paths.append(get_legacy_model_dir())
    return list(dict.fromkeys(paths))


def _register_legacy_model_paths() -> list:
    """Register the legacy directory in both ComfyUI model categories."""
    paths = get_legacy_model_paths()
    try:
        import folder_paths
        for path in paths:
            folder_paths.add_model_folder_path(DIT_MODEL_TYPE, path)
            folder_paths.add_model_folder_path(VAE_MODEL_TYPE, path)
    except Exception:
        pass
    return paths


def _get_canonical_model_dir(model_type: str) -> Optional[str]:
    """Return the current ComfyUI instance directory for a model category."""
    try:
        import folder_paths
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return os.path.join(models_dir, model_type)
    except Exception:
        pass
    return None


def _get_registered_model_paths(model_type: str) -> list:
    """Return all paths registered by ComfyUI for a model category."""
    try:
        import folder_paths
        return list(folder_paths.get_folder_paths(model_type))
    except Exception:
        return []


def is_valid_model_path(path: str, model_type: str) -> bool:
    """Return whether a resolved path is usable for the requested category."""
    return bool(path) and os.path.isfile(path)


def get_model_paths(model_type: str, fallback_dir: Optional[str] = None) -> list:
    """Return model paths for a category, including the legacy read fallback."""
    legacy_paths = _register_legacy_model_paths()
    registered_paths = _get_registered_model_paths(model_type)
    if registered_paths:
        paths = registered_paths
        canonical_dir = _get_canonical_model_dir(model_type)
        if canonical_dir and canonical_dir not in paths:
            paths.append(canonical_dir)
    elif fallback_dir:
        paths = [fallback_dir]
    else:
        canonical_dir = _get_canonical_model_dir(model_type)
        paths = [canonical_dir] if canonical_dir else []

    for legacy_dir in legacy_paths:
        if legacy_dir not in paths:
            paths.append(legacy_dir)
    return paths or legacy_paths


def get_model_dir(model_type: str, fallback_dir: Optional[str] = None) -> str:
    """Return the download directory for a ComfyUI model category."""
    if fallback_dir:
        return fallback_dir

    registered_paths = _get_registered_model_paths(model_type)
    canonical_dir = _get_canonical_model_dir(model_type)
    if (
        model_type == DIT_MODEL_TYPE
        and canonical_dir
        and registered_paths
        and os.path.basename(os.path.normpath(registered_paths[0])).lower() == "unet"
    ):
        return canonical_dir
    if registered_paths:
        return registered_paths[0]

    if canonical_dir:
        return canonical_dir
    return get_legacy_model_dir()


def get_base_cache_dir() -> str:
    """
    Get the default model cache directory path.
    
    Returns the path without creating the directory.
    
    Returns:
        str: Path to default cache directory
    """
    try:
        import folder_paths  # Only works if ComfyUI is available
        return get_model_dir(DIT_MODEL_TYPE)
    except Exception:
        return get_legacy_model_dir()


def get_all_model_paths(model_type: str = DIT_MODEL_TYPE) -> list:
    """Get all paths registered for a ComfyUI model category."""
    paths = get_model_paths(model_type)
    unique_paths = []
    seen = set()
    for path in paths:
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(path)
    return unique_paths


def get_all_model_files(model_type: str = DIT_MODEL_TYPE) -> dict:
    """
    Get a mapping of all model files to their full paths across all registered directories.
    
    Returns:
        dict: Mapping of filename -> full path for all discovered model files
    """
    model_files = {}
    all_paths = get_all_model_paths(model_type)
    
    for path in all_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if is_supported_model_file(file):
                    # Only keep first occurrence of each file (priority order)
                    if file not in model_files:
                        model_files[file] = os.path.join(path, file)
    
    return model_files


def find_model_file(
    filename: str,
    fallback_dir: Optional[str] = None,
    model_type: str = DIT_MODEL_TYPE,
) -> str:
    """
    Find a model file in any registered path.
    
    Args:
        filename: Name of the model file to find
        fallback_dir: Directory to use if file not found in any registered path
        
    Returns:
        str: Full path to the model file
    """
    # Register legacy paths before asking folder_paths to resolve the name.
    model_files = get_all_model_files(model_type)

    try:
        import folder_paths
        full_path = folder_paths.get_full_path(model_type, filename)
        if full_path and is_valid_model_path(full_path, model_type):
            return full_path
    except Exception:
        pass

    # Return path if found
    if filename in model_files:
        return model_files[filename]
    
    # Fallback to specified directory or base cache dir
    if fallback_dir:
        return os.path.join(fallback_dir, filename)
    else:
        return os.path.join(get_model_dir(model_type), filename)


def get_validation_cache_path(cache_dir: Optional[str] = None) -> str:
    """
    Get path to model validation cache file.
    
    Args:
        cache_dir: Optional directory for cache file. If None, uses default base cache dir.
        
    Returns:
        Full path to validation cache JSON file
    """
    if cache_dir is None:
        cache_dir = get_base_cache_dir()
    return os.path.join(cache_dir, ".validation_cache.json")


def is_supported_model_file(filename: str) -> bool:
    """Check if a file has a supported model extension"""
    return any(filename.endswith(ext) for ext in SUPPORTED_MODEL_EXTENSIONS)


def suppress_tensor_warnings() -> None:
    """
    Suppress common tensor conversion and numpy array warnings that are expected behavior
    when working with GGUF tensors and numpy arrays.
    """
    warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)