"""
Downloads utility module for SeedVR2
Handles model and VAE downloads from HuggingFace repositories with integrity validation
"""

import os
import hashlib
import json
import urllib.request
from typing import Optional
from tqdm import tqdm
import time

from .model_registry import MODEL_REGISTRY, DEFAULT_VAE
from .constants import (
    get_base_cache_dir,
    get_all_model_paths,
    find_model_file,
    get_validation_cache_path,
    HUGGINGFACE_BASE_URL,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_RETRY_DELAY
)

def load_validation_cache(cache_dir: Optional[str] = None):
    """
    Load validation cache.
    
    Args:
        cache_dir: Optional directory containing cache file. If None, uses default.
    """
    cache_path = get_validation_cache_path(cache_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_validation_cache(cache, cache_dir: Optional[str] = None):
    """
    Save validation cache.
    
    Args:
        cache: Cache dictionary to save
        cache_dir: Optional directory for cache file. If None, uses default.
    """
    cache_path = get_validation_cache_path(cache_dir)
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except:
        pass


def is_file_validated_cached(filepath: str, cache_dir: Optional[str] = None) -> bool:
    """
    Check if file is validated using cache (fast).
    
    Args:
        filepath: Path to file to check
        cache_dir: Optional directory containing cache file. If None, uses default.
    """
    if not os.path.exists(filepath):
        return False
    
    cache = load_validation_cache(cache_dir)
    filename = os.path.basename(filepath)
    
    if filename in cache:
        cached = cache[filename]
        # Check if file hasn't changed since validation
        if (cached.get('size') == os.path.getsize(filepath) and 
            abs(cached.get('mtime', 0) - os.path.getmtime(filepath)) < 2):
            return True
    
    return False


def validate_file(filepath: str, expected_hash: Optional[str] = None, cache_dir: Optional[str] = None) -> bool:
    """
    File validation with hash check and cache update.
    
    Args:
        filepath: Path to file to validate
        expected_hash: Optional SHA256 hash to verify against
        cache_dir: Optional directory for cache file. If None, uses default.
    """
    if not os.path.exists(filepath):
        return False
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False
    
    # Quick safetensors header check
    if filepath.endswith('.safetensors'):
        try:
            with open(filepath, 'rb') as f:
                header_size = int.from_bytes(f.read(8), 'little')
                if header_size <= 0 or header_size > file_size:
                    return False
        except:
            return False
    
    # Only calculate hash if we have an expected hash to compare
    if expected_hash:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(DOWNLOAD_CHUNK_SIZE):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        
        if actual_hash != expected_hash:
            return False
        
        # Update cache with validated info
        cache = load_validation_cache(cache_dir)
        cache[os.path.basename(filepath)] = {
            "size": os.path.getsize(filepath),
            "mtime": os.path.getmtime(filepath),
            "hash": expected_hash
        }
        save_validation_cache(cache, cache_dir)
    
    return True


def download_with_resume(url: str, filepath: str, debug=None) -> bool:
    """Download with resume support and progress bar"""
    temp_file = f"{filepath}.download"
    existing_size = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0
    
    headers = {'Range': f'bytes={existing_size}-'} if existing_size > 0 else {}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content_length = int(response.headers.get('Content-Length', 0))
            total_size = existing_size + content_length
            
            pbar = tqdm(total=total_size, initial=existing_size, unit='B', 
                       unit_scale=True, unit_divisor=1024, 
                       desc=os.path.basename(filepath))
            
            with open(temp_file, 'ab' if existing_size else 'wb') as f:
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
        
        os.replace(temp_file, filepath)
        return True
        
    except Exception as e:
        if debug:
            debug.log(f"Download error: {e}", level="ERROR", category="download", force=True)
        return False


def download_weight(dit_model: str, vae_model: str, model_dir: Optional[str] = None, debug=None) -> bool:
    """Download SeedVR2 DiT and VAE models with integrity checking"""
    cache_dir = model_dir or get_base_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    files_to_download = [
        (dit_model, MODEL_REGISTRY.get(dit_model)),
        (vae_model, MODEL_REGISTRY.get(vae_model))
    ]
    
    for filename, model_info in files_to_download:
        if not model_info:
            if debug:
                debug.log(f"{filename} not in registry, skipping validation", 
                         level="WARNING", category="setup")
            continue
        
        # Get model type for logging
        model_type = "VAE" if model_info.category == "vae" else "DiT"
        
        # Check if file exists in any registered path first
        existing_filepath = find_model_file(filename, fallback_dir=cache_dir)
        
        # Use existing file path if it exists, otherwise download to cache_dir
        if os.path.exists(existing_filepath):
            filepath = existing_filepath
            # Debug log: Model found
            if debug:
                debug.log(f"{model_type} model found: {filepath}", category="setup")
        else:
            filepath = os.path.join(cache_dir, filename)
            # Debug log: Model not found, will need to download
            if debug:
                searched_paths = get_all_model_paths()
                debug.log(f"{model_type} model not found: {filename}", category="setup")
                debug.log(f"Searched in {len(searched_paths)} location(s):", category="setup")
                for i, path in enumerate(searched_paths, 1):
                    debug.log(f"[{i}] {path}", category="setup", indent_level=1)
            
        expected_hash = model_info.sha256
        repo = model_info.repo
        
        ## Quick cache check first
        if is_file_validated_cached(filepath, cache_dir):
            # Debug log: Model already validated (using cache)
            if debug:
                debug.log(f"{model_type} model already validated (cache): {filepath}", category="setup")
            continue
        
        # File exists - validate it
        if os.path.exists(filepath):
            if debug:
                debug.log(f"Validating {filename}...", category="setup", force=True)
            
            if validate_file(filepath, expected_hash, cache_dir):
                # Debug log: Model validated successfully
                if debug:
                    debug.log(f"{model_type} model validated successfully: {filepath}", category="setup")
                continue
            else:
                # File is corrupted
                if debug:
                    debug.log(f"File corrupted: {filename}, re-downloading...", 
                            level="WARNING", category="download", force=True)
                os.remove(filepath)
                # Clear from cache
                cache = load_validation_cache(cache_dir)
                if filename in cache:
                    del cache[filename]
                    save_validation_cache(cache, cache_dir)
        
        # Download file
        url = HUGGINGFACE_BASE_URL.format(repo=repo, filename=filename)
        temp_file = f"{filepath}.download"
        
        if os.path.exists(temp_file) and debug:
            size_gb = os.path.getsize(temp_file) / (1024**3)
            debug.log(f"Resuming {filepath} from {size_gb:.2f}GB (source: {url})", 
                     category="download", force=True)
        elif debug:
            debug.log(f"Downloading {filepath} from {url}...", 
                     category="download", force=True)
        
        # Download with retries
        success = False
        for attempt in range(DOWNLOAD_MAX_RETRIES):
            if attempt > 0:
                time.sleep(DOWNLOAD_RETRY_DELAY * attempt)
                if debug:
                    debug.log(f"Retry {attempt}/{DOWNLOAD_MAX_RETRIES}", 
                             category="download", force=True)
            
            if download_with_resume(url, filepath, debug):
                # Validate downloaded file
                if validate_file(filepath, expected_hash):
                    if debug:
                        debug.log(f"Downloaded and validated: {filename}", 
                                 category="success", force=True)
                    success = True
                    break
                else:
                    # Remove corrupted download
                    for f in [filepath, temp_file]:
                        if os.path.exists(f):
                            os.remove(f)
                    if debug:
                        debug.log(f"Downloaded file failed validation, retrying...", 
                                 level="WARNING", category="download", force=True)
        
        if not success:
            if debug:
                debug.log(f"Failed to download {filename} after {DOWNLOAD_MAX_RETRIES} attempts", 
                         level="ERROR", category="download", force=True)
                debug.log(f"Manual download: https://huggingface.co/{repo}/blob/main/{filename}", 
                         category="info", force=True)
                debug.log(f"Save to: {filepath}", category="info", force=True)
            return False
    
    return True