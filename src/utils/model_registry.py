"""
Model Registry for SeedVR2
Central registry for model definitions, repositories, and metadata
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.constants import SEEDVR2_MODEL_TYPE, is_supported_model_file, get_base_cache_dir

@dataclass
class ModelInfo:
    """Model metadata"""
    repo: str = "numz/SeedVR2_comfyUI"
    category: str = "model"  # 'model' or 'vae'
    precision: str = "fp16"  # 'fp16', 'fp8_e4m3fn', 'Q4_K_M', etc.
    size: str = "3B"  # '3B', '7B', etc.
    variant: Optional[str] = None  # 'sharp', etc.

# Model registry with metadata
MODEL_REGISTRY = {
    # 3B models
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": ModelInfo(size="3B", precision="fp8_e4m3fn"),
    "seedvr2_ema_3b_fp16.safetensors": ModelInfo(size="3B", precision="fp16"),
    
    # 7B models
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16"),
    "seedvr2_ema_7b_fp16.safetensors": ModelInfo(size="7B", precision="fp16"),
    
    # 7B sharp variants
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp"),
    "seedvr2_ema_7b_sharp_fp16.safetensors": ModelInfo(size="7B", precision="fp16", variant="sharp"),
    
    # VAE models
    "ema_vae_fp16.safetensors": ModelInfo(category="vae", precision="fp16"),
}

GGUF_MODEL_REGISTRY = {
    # 3B models
    "seedvr2_ema_3b-Q3_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="3B", precision="Q3_K_M"),
    "seedvr2_ema_3b-Q4_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="3B", precision="Q4_K_M"),
    "seedvr2_ema_3b-Q5_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="3B", precision="Q5_K_M"),
    "seedvr2_ema_3b-Q6_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="3B", precision="Q6_K_M"),
    "seedvr2_ema_3b-Q8_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="3B", precision="Q8_K_M"),
    
    # 7B models
    "seedvr2_ema_7b-Q3_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q3_K_M"),
    "seedvr2_ema_7b-Q4_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q4_K_M"),
    "seedvr2_ema_7b-Q5_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q5_K_M"),
    "seedvr2_ema_7b-Q6_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q6_K_M"),
    "seedvr2_ema_7b-Q8_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q8_K_M"),
    
    # 7B sharp variants
    "seedvr2_ema_7b_sharp-Q3_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q3_K_M"),
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q4_K_M"),
    "seedvr2_ema_7b_sharp-Q5_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q5_K_M"),
    "seedvr2_ema_7b_sharp-Q6_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q6_K_M"),
    "seedvr2_ema_7b_sharp-Q8_K_M.gguf": ModelInfo(repo="cmeka/SeedVR2-GGUF", size="7B", precision="Q8_K_M"),
    
    # VAE models
    "ema_vae_fp16.safetensors": ModelInfo(category="vae", precision="fp16"),
}

# Configuration constants
DEFAULT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_GGUF_MODEL = "seedvr2_ema_3b-Q4_K_M.gguf"
DEFAULT_VAE = "ema_vae_fp16.safetensors"

def get_default_models(gguf=False) -> List[str]:
    """Get list of default models (non-VAE)"""
    if gguf:
        return [name for name, info in GGUF_MODEL_REGISTRY.items() if info.category == "model"]
    return [name for name, info in MODEL_REGISTRY.items() if info.category == "model"]

def get_model_repo(model_name: str, gguf=False) -> str:
    """Get repository for a specific model"""
    if gguf:
        return GGUF_MODEL_REGISTRY.get(model_name, ModelInfo()).repo
    return MODEL_REGISTRY.get(model_name, ModelInfo()).repo

def get_available_models(gguf=False) -> List[str]:
    """Get all available models including those discovered on disk"""
    model_list = get_default_models(gguf)
    
    try:
        import folder_paths # only works if comfyui is available
        # Ensure the folder is registered before trying to list files
        get_base_cache_dir()
        # Get all models from the SEEDVR2 folder using centralized constant
        available_models = folder_paths.get_filename_list(SEEDVR2_MODEL_TYPE)
        
        # Add any models not in the registry with supported extensions
        for model in available_models:
            if is_supported_model_file(model) and model not in MODEL_REGISTRY:
                model_list.append(model)
    except:
        pass
    
    return model_list