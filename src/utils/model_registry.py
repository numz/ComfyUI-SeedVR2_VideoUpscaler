"""
Model Registry for SeedVR2
Central registry for model definitions, repositories, and metadata
"""

import os
from typing import List, Optional
from dataclasses import dataclass
from .constants import get_all_model_files

# Model class imports using relative imports
from ..models.dit_3b.nadit import NaDiT as NaDiT3B
from ..models.dit_7b.nadit import NaDiT as NaDiT7B
from ..models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

# Model classes - simple registry with clear keys
MODEL_CLASSES = {
    "dit_3b.nadit": NaDiT3B,
    "dit_7b.nadit": NaDiT7B,
    "video_vae_v3.modules.attn_video_vae": VideoAutoencoderKLWrapper,
}

@dataclass
class ModelInfo:
    """Model metadata"""
    repo: str = "numz/SeedVR2_comfyUI"
    category: str = "dit" # 'model' or 'vae'
    precision: str = "fp16" # 'fp16', 'fp8_e4m3fn', 'Q4_K_M', etc.
    size: str = "3B" # '3B', '7B', etc.
    variant: Optional[str] = None # 'sharp', etc.
    subfolder: Optional[str] = None # HF repo subfolder, e.g. 'diffusion_models' for Comfy-Org
    sha256: Optional[str] = None # Cached hash

# Model registry with metadata
MODEL_REGISTRY = {
    # 3B models
    "seedvr2_ema_3b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q4_K_M", sha256="e665e3909de1a8c88a69c609bca9d43ff5a134647face2ce4497640cc3597f0e"),
    "seedvr2_ema_3b-Q8_0.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q8_0", sha256="be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96"),
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": ModelInfo(size="3B", precision="fp8_e4m3fn", sha256="3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"),
    "seedvr2_ema_3b_fp16.safetensors": ModelInfo(size="3B", precision="fp16", sha256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"),
    
    # 7B models
    "seedvr2_ema_7b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", sha256="db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789"),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", sha256="3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2"),
    "seedvr2_ema_7b_fp16.safetensors": ModelInfo(size="7B", precision="fp16", sha256="7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"),
    
    # 7B sharp variants
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", variant="sharp", sha256="7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69"),
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp", sha256="0d2c5b8be0fda94351149c5115da26aef4f4932a7a2a928c6f184dda9186e0be"),
    "seedvr2_ema_7b_sharp_fp16.safetensors": ModelInfo(size="7B", precision="fp16", variant="sharp", sha256="20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd"),
    
    # Comfy-Org official models (https://huggingface.co/Comfy-Org/SeedVR2)
    # Same DiT weights as above (plus baked-in conditioning tensors, ignored on load).
    # Only the drop-in formats (fp16, fp8_e4m3fn) are registered here; the quantized
    # formats (int8_convrot, mxfp8, nvfp4) require comfy_kitchen dequantization support.
    "seedvr2_3b_fp16.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="3B", precision="fp16", sha256="98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660"),
    "seedvr2_3b_fp8_e4m3fn.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="3B", precision="fp8_e4m3fn", sha256="a0226eaa2c3e6f47ae5ce83225120f16479da890ced1a3bc32b1a14619787914"),
    "seedvr2_7b_fp16.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="7B", precision="fp16", sha256="2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252"),
    "seedvr2_7b_fp8_e4m3fn.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="7B", precision="fp8_e4m3fn", sha256="5065e77d647dd553d9090a81e20d6de590d931a61df79d785e008433926ee418"),
    "seedvr2_7b_sharp_fp16.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="7B", precision="fp16", variant="sharp", sha256="70823bca54b9c24eeb56e1c452697c7c2a430867e58db0e376c6e260f3a4489d"),
    "seedvr2_7b_sharp_fp8_e4m3fn.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="7B", precision="fp8_e4m3fn", variant="sharp", sha256="7602c5f70868d28e7730035e4e9d745b05d661c8f0a7eb758e63f9c8603596ef"),

    # Comfy-Org INT8 convrot models (comfy_kitchen int8_tensorwise + convolution rotation).
    # Dequantized on load via comfy.quant_ops (see model_loader._dequantize_comfy_quant_state).
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="3B", precision="int8_convrot", sha256="c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157"),
    "seedvr2_7b_int8_convrot.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="7B", precision="int8_convrot", sha256="5aa0d25fc9d35e449b659d0c9a5dcb22e2a4fa04032101b95a39da42b32c1be6"),
    "seedvr2_7b_sharp_int8_convrot.safetensors": ModelInfo(repo="Comfy-Org/SeedVR2", subfolder="diffusion_models", size="7B", precision="int8_convrot", variant="sharp", sha256="db48be2f1cc7e36b01a2aa529810f5d9c6a971edd29be225cf1b0eb18d51c366"),

    # VAE models
    "ema_vae_fp16.safetensors": ModelInfo(category="vae", precision="fp16", sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}

# Configuration constants
DEFAULT_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"

def get_default_models(category: str) -> List[str]:
    """Get list of default models"""
    return [name for name, info in MODEL_REGISTRY.items() if info.category == category]

def get_model_repo(model_name: str) -> str:
    """Get repository for a specific model"""
    return MODEL_REGISTRY.get(model_name, ModelInfo()).repo

def get_available_dit_models() -> List[str]:
    """Get all available DiT models including those discovered on disk"""
    model_list = get_default_models("dit")
    
    try:
        # Get all model files from all paths
        model_files = get_all_model_files()
        
        # Add files not in registry
        discovered_models = [
            filename for filename in model_files
            if filename not in MODEL_REGISTRY
        ]
        
        # Add discovered models to the list
        model_list.extend(sorted(discovered_models))
    except:
        pass
    
    return model_list

def get_available_vae_models() -> List[str]:
    """Get all available VAE models from the registry"""
    model_list = get_default_models("vae")
    return model_list