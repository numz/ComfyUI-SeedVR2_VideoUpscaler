"""
Global Model Cache for SeedVR2
Enables independent DiT and VAE model sharing across multiple upscaler node instances
"""

from typing import Dict, Any, Optional, Tuple
from ..optimization.memory_manager import release_model_memory


class GlobalModelCache:
    """
    Global cache for sharing DiT and VAE models independently across upscaler instances.
    Each upscaler gets its own runner but can share the underlying models.
    Also caches runner templates for instant runner creation.
    """
    
    def __init__(self):
        # Storage for cached DiT models: node_id -> (model, config)
        self._dit_models: Dict[str, Tuple[Any, Dict]] = {}
        # Storage for cached VAE models: node_id -> (model, config)
        self._vae_models: Dict[str, Tuple[Any, Dict]] = {}
        # Storage for runner templates: "dit_id+vae_id" -> runner
        self._runner_templates: Dict[str, Any] = {}
    
    def get_dit(self, dit_config: Dict[str, Any], debug: Optional['Debug'] = None) -> Optional[Any]:
        """
        Get cached DiT model if available.
        
        Args:
            dit_config: Configuration dictionary with 'cache_model' and 'node_id' keys
            debug: Optional debug instance for logging
            
        Returns:
            Cached DiT model instance if found, None otherwise
        """
        if not dit_config.get('cache_model', False):
            return None
            
        node_id = dit_config.get('node_id')
        if node_id in self._dit_models:
            model, stored_config = self._dit_models[node_id]
            return model
        return None
    
    def get_vae(self, vae_config: Dict[str, Any], debug: Optional['Debug'] = None) -> Optional[Any]:
        """
        Get cached VAE model if available.
        
        Args:
            vae_config: Configuration dictionary with 'cache_model' and 'node_id' keys
            debug: Optional debug instance for logging
            
        Returns:
            Cached VAE model instance if found, None otherwise
        """
        if not vae_config.get('cache_model', False):
            return None
            
        node_id = vae_config.get('node_id')
        if node_id in self._vae_models:
            model, stored_config = self._vae_models[node_id]
            return model
        return None
    
    def get_runner(self, dit_id: Optional[int], vae_id: Optional[int],
                        debug: Optional['Debug'] = None) -> Optional[Any]:
        """
        Get cached runner template if available.
        
        Args:
            dit_id: DiT node ID for lookup
            vae_id: VAE node ID for lookup
            debug: Optional debug instance for logging
            
        Returns:
            Cached runner template if found, None if either ID is None or not cached
        """
        if dit_id is None or vae_id is None:
            return None
            
        runner_key = f"{dit_id}+{vae_id}"
        if runner_key in self._runner_templates:
            return self._runner_templates[runner_key]
        return None
    
    def set_dit(self, dit_config: Dict[str, Any], model: Any, model_name: str, debug: Optional['Debug'] = None) -> Optional[str]:
        """
        Store DiT model in cache.
        
        Args:
            dit_config: Configuration dictionary with 'cache_model' and 'node_id' keys
            model: DiT model instance to cache
            model_name: Name identifier for the model
            debug: Optional debug instance for logging
            
        Returns:
            Node ID string if cached successfully, None if caching disabled
        """
        if not dit_config.get('cache_model', False):
            return None
            
        node_id = dit_config.get('node_id')
        self._dit_models[node_id] = (model, dit_config)
        
        if debug:
            debug.log(f"DiT model cached in memory (node {node_id}): {model_name}", 
                     category="cache", force=True)
        
        return node_id

    def set_vae(self, vae_config: Dict[str, Any], model: Any, model_name: str, debug: Optional['Debug'] = None) -> Optional[str]:
        """
        Store VAE model in cache.
        
        Args:
            vae_config: Configuration dictionary with 'cache_model' and 'node_id' keys
            model: VAE model instance to cache
            model_name: Name identifier for the model
            debug: Optional debug instance for logging
            
        Returns:
            Node ID string if cached successfully, None if caching disabled
        """
        if not vae_config.get('cache_model', False):
           return None
            
        node_id = vae_config.get('node_id')
        self._vae_models[node_id] = (model, vae_config)
        
        if debug:
            debug.log(f"VAE model cached in memory (node {node_id}): {model_name}", 
                     category="cache", force=True)
        
        return node_id
    
    def set_runner(self, dit_id: Optional[int], vae_id: Optional[int],
                    runner: Any, debug: Optional['Debug'] = None) -> Optional[str]:
        """
        Store runner template in cache.
        
        Args:
            dit_id: DiT node ID for the runner template
            vae_id: VAE node ID for the runner template
            runner: Runner instance to cache as template
            debug: Optional debug instance for logging
            
        Returns:
            Runner key string (format: "dit_id+vae_id") if cached successfully, 
            None if either ID is None or runner already cached
        """
        if dit_id is None or vae_id is None:
            return None
            
        runner_key = f"{dit_id}+{vae_id}"
        if runner_key not in self._runner_templates:
            self._runner_templates[runner_key] = runner
            if debug:
                debug.log(f"Runner template cached in memory: nodes {runner_key}", category="cache", force=True)
            return runner_key
        
        return None
    
    def remove_dit(self, dit_config: Dict[str, Any], debug: Optional['Debug'] = None) -> bool:
        """
        Remove DiT model from cache if it exists.
        
        Args:
            dit_config: Configuration dictionary with 'node_id' key
            debug: Optional debug instance for logging
            
        Returns:
            True if model was removed, False if not found in cache
            
        Note:
            Also removes any runner templates that used this DiT model
        """
        node_id = dit_config.get('node_id')
        if node_id in self._dit_models:
            if debug:
                debug.log(f"Removing cached DiT: {node_id}", category="cache", force=True)

            model, stored_config = self._dit_models[node_id]
            
            # Release model memory
            if model is not None:
                release_model_memory(model=model, debug=debug)
            
            del self._dit_models[node_id]
            
            # Remove any runner templates that used this DiT
            templates_to_remove = [k for k in self._runner_templates.keys() if k.startswith(str(node_id) + "+")]
            for template_key in templates_to_remove:
                del self._runner_templates[template_key]         
            
            return True
        return False
    
    def remove_vae(self, vae_config: Dict[str, Any], debug: Optional['Debug'] = None) -> bool:
        """
        Remove VAE model from cache if it exists.
        
        Args:
            vae_config: Configuration dictionary with 'node_id' key
            debug: Optional debug instance for logging
            
        Returns:
            True if model was removed, False if not found in cache
            
        Note:
            Also removes any runner templates that used this VAE model
        """
        node_id = vae_config.get('node_id')
        if node_id in self._vae_models:
            if debug:
                debug.log(f"Removing cached VAE: {node_id}", category="cache", force=True)

            model, stored_config = self._vae_models[node_id]
            
            # Release model memory directly
            if model is not None:
                release_model_memory(model=model, debug=debug)
            
            del self._vae_models[node_id]
            
            # Remove any runner templates that used this VAE
            templates_to_remove = [k for k in self._runner_templates.keys() if k.endswith("+" + str(node_id))]
            for template_key in templates_to_remove:
                del self._runner_templates[template_key]
            
            return True
        return False


# Global singleton instance
_global_cache = GlobalModelCache()

def get_global_cache() -> GlobalModelCache:
    """Get the global model cache instance."""
    return _global_cache