"""
Downloads utility module for SeedVR2
Handles model and VAE downloads from HuggingFace repositories
"""

import os
import urllib.error
from typing import Optional
from torchvision.datasets.utils import download_url

from src.utils.model_registry import MODEL_REGISTRY, get_model_repo, DEFAULT_VAE
from src.utils.constants import get_base_cache_dir

# HuggingFace URL template
HUGGINGFACE_BASE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"

def download_weight(model: str, model_dir: Optional[str] = None) -> None:
    """
    Download a SeedVR2 model and its associated VAE from HuggingFace Hub
    
    Args:
        model: Model filename to download
        model_dir: Optional custom directory for models
    """
    # Setup paths
    cache_dir = model_dir or get_base_cache_dir()
    model_path = os.path.join(cache_dir, model)
    vae_path = os.path.join(cache_dir, DEFAULT_VAE)
    
    # Download model if not exists
    if not os.path.exists(model_path):
        repo = get_model_repo(model)
        url = HUGGINGFACE_BASE_URL.format(repo=repo, filename=model)
        
        print(f"📥 Downloading {model} from HF {repo}...")
        try:
            download_url(url, cache_dir, filename=model)
            print(f"✅ Downloaded: {model}")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"❌ Model download failed: {e}")
            print(f"📎 Please download model manually from: https://huggingface.co/{repo}")
            print(f"   and place it in: {cache_dir}")
            return False
    
    # Download VAE if not exists
    if not os.path.exists(vae_path):
        vae_repo = get_model_repo(DEFAULT_VAE)
        vae_url = HUGGINGFACE_BASE_URL.format(repo=vae_repo, filename=DEFAULT_VAE)
        
        print(f"📥 Downloading VAE: {DEFAULT_VAE} from HF {vae_repo}...")
        try:
            download_url(vae_url, cache_dir, filename=DEFAULT_VAE)
            print(f"✅ Downloaded: {DEFAULT_VAE}")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"❌ VAE download failed: {e}")
            print(f"📎 Please download VAE manually from: https://huggingface.co/{vae_repo}")
            print(f"   and place it in: {cache_dir}")
            return False
    
    return True