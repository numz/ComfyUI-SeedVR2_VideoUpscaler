"""
Global Model Cache for SeedVR2
Enables independent DiT and VAE model sharing across multiple upscaler node instances
"""

import threading
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from ..optimization.memory_manager import (
    is_model_cache_claimed,
    iter_model_wrapper_chain,
    release_model_memory,
    set_model_cache_claimed_state,
    set_model_cache_cold_state,
)

if TYPE_CHECKING:
    from ..utils.debug import Debug


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
        # Synchronizes DiT/VAE model cache claim/set/replace/remove operations
        self._model_cache_lock = threading.RLock()
        # Synchronizes runner-template claim/set/remove operations
        self._runner_templates_lock = threading.RLock()

    def _models_share_identity(self, cached_model: Any, expected_model: Any) -> bool:
        """Return True when two model references point into the same wrapper/base chain."""
        if cached_model is None or expected_model is None:
            return False

        cached_ids = {id(model) for model in iter_model_wrapper_chain(cached_model)}
        expected_ids = {id(model) for model in iter_model_wrapper_chain(expected_model)}
        return bool(cached_ids & expected_ids)
    
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
        with self._model_cache_lock:
            if node_id in self._dit_models:
                model, stored_config = self._dit_models[node_id]
                if is_model_cache_claimed(model):
                    if debug:
                        debug.log(
                            f"Cached DiT is already claimed by another execution; skipping reuse (node {node_id})",
                            category="cache",
                            force=True,
                        )
                    return None
                set_model_cache_claimed_state(model, True)
                return model
        return None

    def peek_dit(self, dit_config: Dict[str, Any]) -> Optional[Any]:
        """Return the cached DiT model without claiming it."""
        node_id = dit_config.get('node_id')
        if node_id is None:
            return None

        with self._model_cache_lock:
            entry = self._dit_models.get(node_id)
            return None if entry is None else entry[0]
    
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
        with self._model_cache_lock:
            if node_id in self._vae_models:
                model, stored_config = self._vae_models[node_id]
                if is_model_cache_claimed(model):
                    if debug:
                        debug.log(
                            f"Cached VAE is already claimed by another execution; skipping reuse (node {node_id})",
                            category="cache",
                            force=True,
                        )
                    return None
                set_model_cache_claimed_state(model, True)
                return model
        return None

    def peek_vae(self, vae_config: Dict[str, Any]) -> Optional[Any]:
        """Return the cached VAE model without claiming it."""
        node_id = vae_config.get('node_id')
        if node_id is None:
            return None

        with self._model_cache_lock:
            entry = self._vae_models.get(node_id)
            return None if entry is None else entry[0]
    
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
        with self._runner_templates_lock:
            return self._runner_templates.get(runner_key)

    def claim_runner(self, dit_id: Optional[int], vae_id: Optional[int],
                     dit_model: str, vae_model: str) -> Tuple[Optional[Any], str]:
        """
        Atomically inspect and claim a cached runner template for exclusive reuse.

        Returns:
            (template, status) where status is one of:
            - "missing": no cached template exists
            - "active": template exists but is already in use
            - "tainted": template exists but was marked failed/interrupted
            - "mismatch": template exists but was built for different DiT/VAE names
            - "claimed": template was successfully claimed for reuse
        """
        if dit_id is None or vae_id is None:
            return None, "missing"

        runner_key = f"{dit_id}+{vae_id}"
        with self._runner_templates_lock:
            template = self._runner_templates.get(runner_key)
            if template is None:
                return None, "missing"

            if getattr(template, '_seedvr2_execution_active', False):
                return template, "active"

            if getattr(template, '_seedvr2_runner_tainted', False):
                return template, "tainted"

            current_dit = getattr(template, '_dit_model_name', None)
            current_vae = getattr(template, '_vae_model_name', None)
            if current_dit != dit_model or current_vae != vae_model:
                return template, "mismatch"

            template._seedvr2_execution_active = True
            return template, "claimed"
    
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
        with self._model_cache_lock:
            existing = self._dit_models.get(node_id)
            if existing is not None:
                existing_model, _ = existing
                if not self._models_share_identity(existing_model, model) and is_model_cache_claimed(existing_model):
                    if debug:
                        debug.log(
                            f"Skipped caching DiT model for node {node_id}: cache entry is currently claimed by another execution",
                            level="WARNING",
                            category="cache",
                            force=True,
                        )
                    return None
            set_model_cache_cold_state(model, False)
            set_model_cache_claimed_state(model, True)
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
        with self._model_cache_lock:
            existing = self._vae_models.get(node_id)
            if existing is not None:
                existing_model, _ = existing
                if not self._models_share_identity(existing_model, model) and is_model_cache_claimed(existing_model):
                    if debug:
                        debug.log(
                            f"Skipped caching VAE model for node {node_id}: cache entry is currently claimed by another execution",
                            level="WARNING",
                            category="cache",
                            force=True,
                        )
                    return None
            set_model_cache_cold_state(model, False)
            set_model_cache_claimed_state(model, True)
            self._vae_models[node_id] = (model, vae_config)
        
        if debug:
            debug.log(f"VAE model cached in memory (node {node_id}): {model_name}", 
                     category="cache", force=True)
        
        return node_id

    def replace_dit(
        self,
        dit_config: Dict[str, Any],
        model: Any,
        debug: Optional['Debug'] = None,
        expected_model: Optional[Any] = None,
    ) -> bool:
        """Rewrite a cached DiT entry to the latest claimed model object."""
        node_id = dit_config.get('node_id')
        with self._model_cache_lock:
            if node_id not in self._dit_models:
                return False

            cached_model, stored_config = self._dit_models[node_id]
            if expected_model is not None and not self._models_share_identity(cached_model, expected_model):
                if debug:
                    debug.log(
                        f"Skipped cached DiT rewrite for node {node_id}: cache entry no longer matches the claimed model",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False

            self._dit_models[node_id] = (model, stored_config)
        if debug:
            debug.log(f"Refreshed cached DiT entry to the latest claimed model object (node {node_id})", category="cache", force=True)
        return True

    def replace_vae(
        self,
        vae_config: Dict[str, Any],
        model: Any,
        debug: Optional['Debug'] = None,
        expected_model: Optional[Any] = None,
    ) -> bool:
        """Rewrite a cached VAE entry to the latest claimed model object."""
        node_id = vae_config.get('node_id')
        with self._model_cache_lock:
            if node_id not in self._vae_models:
                return False

            cached_model, stored_config = self._vae_models[node_id]
            if expected_model is not None and not self._models_share_identity(cached_model, expected_model):
                if debug:
                    debug.log(
                        f"Skipped cached VAE rewrite for node {node_id}: cache entry no longer matches the claimed model",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False

            self._vae_models[node_id] = (model, stored_config)
        if debug:
            debug.log(f"Refreshed cached VAE entry to the latest claimed model object (node {node_id})", category="cache", force=True)
        return True
    
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
            Runner key string (format: "dit_id+vae_id") if this call cached or
            replaced the template, None if either ID is None or an existing
            non-tainted template is intentionally kept.
        """
        if dit_id is None or vae_id is None:
            return None
            
        runner_key = f"{dit_id}+{vae_id}"
        with self._runner_templates_lock:
            existing = self._runner_templates.get(runner_key)
            if existing is runner:
                return runner_key

            replace_existing = False
            if existing is not None:
                replace_existing = getattr(existing, '_seedvr2_runner_tainted', False)

            if existing is None or replace_existing:
                self._runner_templates[runner_key] = runner
                if debug:
                    action = "replaced" if replace_existing else "cached"
                    debug.log(f"Runner template {action} in memory: nodes {runner_key}", category="cache", force=True)
                return runner_key
        
        return None

    def remove_runner(self, dit_id: Optional[int], vae_id: Optional[int],
                      debug: Optional['Debug'] = None,
                      expected_runner: Optional[Any] = None) -> bool:
        """Remove a cached runner template for the given DiT/VAE node pair.

        If expected_runner is provided, only remove the cache entry when the
        currently stored runner is that exact object.
        """
        if dit_id is None or vae_id is None:
            return False

        with self._runner_templates_lock:
            runner_key = f"{dit_id}+{vae_id}"
            cached_runner = self._runner_templates.get(runner_key)
            if cached_runner is None:
                return False

            if expected_runner is not None and cached_runner is not expected_runner:
                if debug:
                    debug.log(
                        f"Skipped cached runner removal for nodes {runner_key}: cache entry no longer matches expected runner",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False

            del self._runner_templates[runner_key]
        if debug:
            debug.log(f"Removed cached runner template: nodes {runner_key}", category="cache", force=True)
        return True

    def taint_and_remove_runner(self,
                                dit_id: Optional[int],
                                vae_id: Optional[int],
                                debug: Optional['Debug'] = None,
                                expected_runner: Optional[Any] = None) -> bool:
        """Atomically mark a cached runner template tainted/inactive and remove it."""
        if dit_id is None or vae_id is None:
            return False

        runner_key = f"{dit_id}+{vae_id}"
        with self._runner_templates_lock:
            cached_runner = self._runner_templates.get(runner_key)
            if cached_runner is None:
                return False

            if expected_runner is not None and cached_runner is not expected_runner:
                if debug:
                    debug.log(
                        f"Skipped taint+remove for cached runner nodes {runner_key}: cache entry no longer matches expected runner",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False

            cached_runner._seedvr2_runner_tainted = True
            cached_runner._seedvr2_execution_active = False
            del self._runner_templates[runner_key]

        if debug:
            debug.log(f"Tainted and removed cached runner template: nodes {runner_key}", category="cache", force=True)
        return True
    
    def remove_dit(
        self,
        dit_config: Dict[str, Any],
        debug: Optional['Debug'] = None,
        expected_model: Optional[Any] = None,
    ) -> bool:
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
        with self._model_cache_lock:
            if node_id not in self._dit_models:
                return False

            cached_model, stored_config = self._dit_models[node_id]
            if expected_model is None and is_model_cache_claimed(cached_model):
                if debug:
                    debug.log(
                        f"Skipped cached DiT removal for node {node_id}: cache entry is currently claimed by another execution",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False
            if expected_model is not None and not self._models_share_identity(cached_model, expected_model):
                if debug:
                    debug.log(
                        f"Skipped cached DiT removal for node {node_id}: cache entry no longer matches the claimed model",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False

            if debug:
                debug.log(f"Removing cached DiT: {node_id}", category="cache", force=True)

            model = cached_model
            del self._dit_models[node_id]

        if model is not None:
            release_model_memory(model=model, debug=debug)

        with self._runner_templates_lock:
            templates_to_remove = [k for k in self._runner_templates.keys() if k.startswith(str(node_id) + "+")]
            for template_key in templates_to_remove:
                del self._runner_templates[template_key]

        return True
    
    def remove_vae(
        self,
        vae_config: Dict[str, Any],
        debug: Optional['Debug'] = None,
        expected_model: Optional[Any] = None,
    ) -> bool:
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
        with self._model_cache_lock:
            if node_id not in self._vae_models:
                return False

            cached_model, stored_config = self._vae_models[node_id]
            if expected_model is None and is_model_cache_claimed(cached_model):
                if debug:
                    debug.log(
                        f"Skipped cached VAE removal for node {node_id}: cache entry is currently claimed by another execution",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False
            if expected_model is not None and not self._models_share_identity(cached_model, expected_model):
                if debug:
                    debug.log(
                        f"Skipped cached VAE removal for node {node_id}: cache entry no longer matches the claimed model",
                        level="WARNING",
                        category="cache",
                        force=True,
                    )
                return False

            if debug:
                debug.log(f"Removing cached VAE: {node_id}", category="cache", force=True)

            model = cached_model
            del self._vae_models[node_id]

        if model is not None:
            release_model_memory(model=model, debug=debug)

        with self._runner_templates_lock:
            templates_to_remove = [k for k in self._runner_templates.keys() if k.endswith("+" + str(node_id))]
            for template_key in templates_to_remove:
                del self._runner_templates[template_key]

        return True


# Global singleton instance
_global_cache = GlobalModelCache()

def get_global_cache() -> GlobalModelCache:
    """Get the global model cache instance."""
    return _global_cache
