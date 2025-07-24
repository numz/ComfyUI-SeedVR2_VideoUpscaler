"""
Downloads utility module for SeedVR2
Handles model and VAE downloads from HuggingFace

Extracted from: seedvr2.py (line 968-1015)
"""

import os
from torchvision.datasets.utils import download_url
try:
    import folder_paths
    # Configuration des chemins
    base_cache_dir = os.path.join(folder_paths.models_dir, "SEEDVR2")

    # S'assurer que le dossier de cache existe
    folder_paths.add_model_folder_path("seedvr2", os.path.join(folder_paths.models_dir, "SEEDVR2"))
except:
    base_cache_dir = "./seedvr2_models"

def download_weight(model, model_dir=None):
    """
    Télécharge un modèle SeedVR2 et son VAE associé depuis HuggingFace Hub
    
    Args:
        model (str): Nom du fichier modèle à télécharger
                    (ex: "seedvr2_ema_3b_fp16.safetensors")
    
    Gère automatiquement:
    - Téléchargement du modèle principal
    - Téléchargement du VAE avec fallbacks:
      1. ema_vae_fp16.safetensors (priorité)
      2. ema_vae_fp8_e4m3fn.safetensors (fallback)  
      3. ema_vae.pth (legacy fallback)
    """
    if model_dir is None:
        model_path = os.path.join(base_cache_dir, model)
        vae_fp16_path = os.path.join(base_cache_dir, "ema_vae_fp16.safetensors")
        cache_dir = base_cache_dir
    else:
        model_path = os.path.join(model_dir, model)
        vae_fp16_path = os.path.join(model_dir, "ema_vae_fp16.safetensors")
        cache_dir = model_dir
   
    # Configuration HuggingFace
    repo_id = "numz/SeedVR2_comfyUI"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    
    # 🚀 Téléchargement du modèle principal
    if not os.path.exists(model_path):
        print(f"📥 Downloading model: {model}")
        download_url(f"{base_url}/{model}", cache_dir, filename=model)
        print(f"✅ Downloaded: {model}")
    
    # 🚀 Téléchargement du VAE avec stratégie de fallback
    if not os.path.exists(vae_fp16_path):
        print("📥 Downloading FP16 VAE SafeTensors...")
        try:
            download_url(f"{base_url}/ema_vae_fp16.safetensors", cache_dir, filename="ema_vae_fp16.safetensors")
            print("✅ Downloaded: ema_vae_fp16.safetensors (FP16 SafeTensors)")
        except Exception as e:
            print(f"⚠️ FP16 SafeTensors VAE not available: {e}")
    
    return


def get_base_cache_dir():
    """
    Retourne le répertoire de cache base pour les modèles SeedVR2
    
    Returns:
        str: Chemin du répertoire de cache
    """
    return base_cache_dir
