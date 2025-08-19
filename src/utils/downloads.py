"""
Downloads utility module for SeedVR2
Handles model and VAE downloads from HuggingFace repositories
"""

import os
import urllib.error
from typing import Optional
from torchvision.datasets.utils import download_url

from src.utils.model_registry import get_model_repo, DEFAULT_VAE
from src.utils.constants import get_base_cache_dir

# HuggingFace URL template
HUGGINGFACE_BASE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"

def download_weight(model: str, model_dir: Optional[str] = None, debug=None) -> bool:
    """
    Download a SeedVR2 model and its associated VAE from HuggingFace Hub
    
    Args:
        model: Model filename to download
        model_dir: Optional custom directory for models
        debug: Optional Debug instance for logging
    """    
    # Setup paths
    cache_dir = model_dir or get_base_cache_dir()
    model_path = os.path.join(cache_dir, model)
    vae_path = os.path.join(cache_dir, DEFAULT_VAE)
    is_gguf = model.endswith('.gguf')
    # Download model if not exists
    if not os.path.exists(model_path):
        repo = get_model_repo(model, gguf=is_gguf)
        url = HUGGINGFACE_BASE_URL.format(repo=repo, filename=model)
        
        if debug:
            debug.log(f"Downloading {model} from HF {repo}...", category="download", force=True)
        try:
            download_url(url, cache_dir, filename=model)
            if debug:
                debug.log(f"Downloaded: {model}", category="success", force=True)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if debug:
                debug.log(f"Model download failed: {e}", level="ERROR", category="download", force=True)
                debug.log(f"Please download model manually from: https://huggingface.co/{repo}", category="info", force=True)
                debug.log(f"and place it in: {cache_dir}", category="info", force=True)
            return False
    
    # Download VAE if not exists
    if not os.path.exists(vae_path):
        vae_repo = get_model_repo(DEFAULT_VAE, gguf=is_gguf)
        vae_url = HUGGINGFACE_BASE_URL.format(repo=vae_repo, filename=DEFAULT_VAE)
        
        if debug:
            debug.log(f"Downloading VAE: {DEFAULT_VAE} from HF {vae_repo}...", category="download", force=True)
        try:
            download_url(vae_url, cache_dir, filename=DEFAULT_VAE)
            if debug:
                debug.log(f"Downloaded: {DEFAULT_VAE}", category="success", force=True)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if debug:
                debug.log(f"VAE download failed: {e}", level="ERROR", category="download", force=True)
                debug.log(f"Please download VAE manually from: https://huggingface.co/{vae_repo}", category="info", force=True)
                debug.log(f"and place it in: {cache_dir}", category="info", force=True)
            return False
    
    return True