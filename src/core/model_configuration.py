"""
Model Configuration and Setup for SeedVR2

This module orchestrates model configuration, caching, and runtime settings:
- Runner configuration and lifecycle management
- Model caching with validation and updates
- DiT and VAE model setup with meta device initialization
- Configuration change detection and updates
- torch.compile integration
- BlockSwap configuration
- VAE tiling configuration

Key Features:
- Unified runner configuration via configure_runner()
- Global model caching with name validation
- Configuration change detection to avoid unnecessary reloads
- Meta device initialization for memory efficiency
- Dynamic configuration updates (BlockSwap, torch.compile, attention mode)
- Automatic cleanup of stale cache entries

Main Functions:
- configure_runner: Main entry point for runner setup
- apply_model_specific_config: Apply BlockSwap and torch.compile to models
- _setup_models: Setup DiT and VAE models (cached or new)
- _update_dit_config: Update DiT configuration on cached models
- _update_vae_config: Update VAE configuration on cached models

Cache Management:
- _initialize_cache_context: Initialize cache with validation
- _acquire_runner: Get or create runner instance
- _create_new_runner: Create new runner with config
- _configure_runner_settings: Configure runner settings for tiling/compile/BlockSwap

Configuration Helpers:
- _configs_equal: Compare configurations for equality
- _describe_blockswap_config: Human-readable BlockSwap description
- _describe_compile_config: Human-readable torch.compile description
- _describe_attention_mode: Human-readable attention mode description
- _describe_tiling_config: Human-readable VAE tiling description
- _update_model_config: Generic config update handler with change detection

Model Setup:
- _setup_dit_model: Setup DiT model from cache or create new structure
- _setup_vae_model: Setup VAE model from cache or create new structure

torch.compile Support:
- _configure_torch_compile: Configure torch.compile settings
- _apply_torch_compile: Apply torch.compile to full model
- _apply_vae_submodule_compile: Apply torch.compile to VAE submodules
- _disable_compile_for_dynamic_modules: Mark dynamic modules no-compile
- _propagate_debug_to_modules: Propagate debug instance to submodules

BlockSwap Management:
- _handle_blockswap_change: Handle BlockSwap configuration changes

This module uses model_loader for actual weight loading operations.
"""

import os
import torch
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, Union, Callable

from .model_loader import (
    prepare_model_structure,
    script_directory
)
from .infer import VideoDiffusionInfer
from .model_cache import get_global_cache
from ..common.config import load_config
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from ..optimization.compatibility import (
    CompatibleDiT,
    TRITON_AVAILABLE,
    validate_attention_mode
)
from ..optimization.blockswap import is_blockswap_enabled, validate_blockswap_config, apply_block_swap_to_dit, cleanup_blockswap
from ..optimization.memory_manager import cleanup_dit, cleanup_vae
from ..utils.constants import find_model_file


def _configs_equal(config1: Optional[Dict[str, Any]], config2: Optional[Dict[str, Any]]) -> bool:
    """
    Compare two configuration dictionaries for equality.
    Handles None values properly.
    
    Args:
        config1: First configuration dict (can be None)
        config2: Second configuration dict (can be None)
        
    Returns:
        True if configs are equivalent, False otherwise
    """
    # Both None = equal
    if config1 is None and config2 is None:
        return True
    
    # One None, one not = different
    if (config1 is None) != (config2 is None):
        return False
    
    # Compare dictionary contents
    return config1 == config2


def _describe_blockswap_config(config: Optional[Dict[str, Any]]) -> str:
    """
    Generate human-readable description of BlockSwap configuration.
    
    Args:
        config: BlockSwap configuration dictionary
        
    Returns:
        Human-readable description string
    """
    if not is_blockswap_enabled(config):
        return "disabled"
    
    blocks_to_swap = config.get("blocks_to_swap", 0)
    swap_io_components = config.get("swap_io_components", False)
    
    block_text = "block" if blocks_to_swap <= 1 else "blocks"
    parts = [f"{blocks_to_swap} {block_text}"]
    if swap_io_components:
        parts.append("I/O offload")
    
    return f"enabled ({', '.join(parts)})"


def _describe_compile_config(config: Optional[Dict[str, Any]]) -> str:
    """
    Generate human-readable description of torch.compile configuration.
    
    Args:
        config: torch.compile configuration dictionary
        
    Returns:
        Human-readable description string
    """
    if config is None or not config:
        return "disabled"
    
    # Core parameters
    mode = config.get("mode", "default")
    backend = config.get("backend", "inductor")
    
    parts = [f"{mode} mode"]
    
    # Optional flags
    if backend != "inductor":
        parts.append(f"{backend} backend")
    if config.get("fullgraph", False):
        parts.append("fullgraph")
    if config.get("dynamic", False):
        parts.append("dynamic")
    
    # Dynamo tuning parameters (show if non-default)
    cache_limit = config.get("dynamo_cache_size_limit", 64)
    recompile_limit = config.get("dynamo_recompile_limit", 128)
    
    if cache_limit != 64:
        parts.append(f"cache_limit={cache_limit}")
    if recompile_limit != 128:
        parts.append(f"recompile_limit={recompile_limit}")
    
    return f"enabled ({', '.join(parts)})"


def _describe_attention_mode(attention_mode: Optional[str]) -> str:
    """
    Generate human-readable description of attention mode configuration.
    
    Args:
        attention_mode: Attention mode string ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
        
    Returns:
        Human-readable description string
    """
    if attention_mode is None:
        return "sdpa (default)"
    
    mode_descriptions = {
        'sdpa': 'PyTorch SDPA',
        'flash_attn_2': 'Flash Attention 2',
        'flash_attn_3': 'Flash Attention 3',
        'sageattn_2': 'SageAttention 2',
        'sageattn_3': 'SageAttention 3 (Blackwell)'
    }
    
    return mode_descriptions.get(attention_mode, attention_mode)
    

def _describe_tiling_config(encode_tiled: bool, encode_tile_size: Optional[Tuple[int, int]], 
                           encode_tile_overlap: Optional[Tuple[int, int]],
                           decode_tiled: bool, decode_tile_size: Optional[Tuple[int, int]], 
                           decode_tile_overlap: Optional[Tuple[int, int]]) -> str:
    """
    Generate human-readable description of VAE tiling configuration.
    
    Args:
        encode_tiled: Whether encode tiling is enabled
        encode_tile_size: Tile size for encoding
        encode_tile_overlap: Tile overlap for encoding
        decode_tiled: Whether decode tiling is enabled
        decode_tile_size: Tile size for decoding
        decode_tile_overlap: Tile overlap for decoding
        
    Returns:
        Human-readable description string
    """
    if not encode_tiled and not decode_tiled:
        return "disabled"
    
    parts = []
    if encode_tiled:
        parts.append(f"encode Tile: {encode_tile_size}, Overlap: {encode_tile_overlap}")
    if decode_tiled:
        parts.append(f"decode Tile: {decode_tile_size}, Overlap: {decode_tile_overlap}")
    
    return "; ".join(parts)

    
def _update_model_config(
    runner: 'VideoDiffusionInfer',
    model_attr: str,
    model_type: str,
    new_configs: Dict[str, Any],
    cached_config_attrs: Dict[str, str],
    model_config_attrs: Dict[str, str],
    config_describers: Dict[str, Callable],
    special_handlers: Optional[Dict[str, Callable]] = None,
    debug: Optional['Debug'] = None
) -> bool:
    """
    Update model configuration uniformly for DiT or VAE.
    
    This generic function handles config comparison, logging, and attribute updates
    for both DiT and VAE models, reducing code duplication.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_attr: Attribute name for the model ('dit' or 'vae')
        model_type: Model type string for logging ('DiT' or 'VAE')
        new_configs: Dict mapping config names to new values
                    e.g., {'torch_compile': ..., 'block_swap': ...}
        cached_config_attrs: Dict mapping config names to runner attribute names
                           e.g., {'torch_compile': '_dit_compile_args', ...}
        model_config_attrs: Dict mapping config names to model storage attribute names.
                            e.g., {'torch_compile': '_config_compile', ...}
        config_describers: Dict mapping config names to description functions
                          e.g., {'torch_compile': _describe_compile_config, ...}
        special_handlers: Optional dict of config-specific handlers
                         e.g., {'block_swap': handle_blockswap_change}
        debug: Debug instance
        
    Returns:
        True if successful
    """
    model = getattr(runner, model_attr)
    
    # Check if model is on meta device (not materialized yet)
    try:
        param_device = next(model.parameters()).device
        if param_device.type == 'meta':
            # Model not yet materialized - just update config attributes
            for config_name, attr_name in cached_config_attrs.items():
                setattr(runner, attr_name, new_configs.get(config_name))
            # Store on model so config travels with the model when cached
            for config_name, attr_name in cached_config_attrs.items():
                model_attr_name = f'_config_{config_name.split("_")[-1]}'
                setattr(model, model_attr_name, new_configs.get(config_name))
            return True
    except StopIteration:
        pass
    
    # Get cached configurations and check what changed
    config_changes = []
    changes_detected = {}
    
    for config_name, attr_name in cached_config_attrs.items():
        cached_value = getattr(runner, attr_name, None)
        new_value = new_configs.get(config_name)
        changed = not _configs_equal(cached_value, new_value)
        changes_detected[config_name] = changed
        
        if changed:
            # Get description function for this config type
            desc_func = config_describers.get(config_name)
            if desc_func:
                if config_name == 'tiling':
                    # Special case for tiling which needs multiple params
                    old_cfg = cached_value or {}
                    new_cfg = new_value or {}
                    old_desc = desc_func(
                        old_cfg.get('encode_tiled', False),
                        old_cfg.get('encode_tile_size'),
                        old_cfg.get('encode_tile_overlap'),
                        old_cfg.get('decode_tiled', False),
                        old_cfg.get('decode_tile_size'),
                        old_cfg.get('decode_tile_overlap')
                    )
                    new_desc = desc_func(
                        new_cfg.get('encode_tiled', False),
                        new_cfg.get('encode_tile_size'),
                        new_cfg.get('encode_tile_overlap'),
                        new_cfg.get('decode_tiled', False),
                        new_cfg.get('decode_tile_size'),
                        new_cfg.get('decode_tile_overlap')
                    )
                else:
                    old_desc = desc_func(cached_value)
                    new_desc = desc_func(new_value)
                
                # Format config name for display
                display_name = config_name.replace('_', ' ').title()
                if config_name == 'torch_compile':
                    display_name = 'torch.compile'
                elif config_name == 'block_swap':
                    display_name = 'BlockSwap'
                elif config_name == 'attention_mode':
                    display_name = 'Attention Mode'
                    
                config_changes.append(f"{display_name}: {old_desc} → {new_desc}")
    
    # If nothing changed, reuse model as-is
    if not any(changes_detected.values()):
        debug.log(f"{model_type} configuration unchanged, reusing cached model", category=model_type.lower())
        # Still update attributes to ensure consistency
        for config_name, attr_name in cached_config_attrs.items():
            setattr(runner, attr_name, new_configs.get(config_name))
        return True
    
    # Log configuration changes
    debug.log(f"{model_type} configuration changed:", category=model_type.lower(), force=True)
    for change in config_changes:
        debug.log(f"{change}", category=model_type.lower(), force=True, indent_level=1)
    
    # Handle torch.compile unwrapping if needed
    if changes_detected.get('torch_compile', False):
        if model_attr == 'dit' and hasattr(model, '_orig_mod'):
            debug.log(f"Removing torch.compile from {model_type}", category="setup")
            model = model._orig_mod
            setattr(runner, model_attr, model)
        elif model_attr == 'vae':
            # Unwrap compiled VAE submodules if present
            if hasattr(model, 'encoder') and hasattr(model.encoder, '_orig_mod'):
                debug.log(f"Removing torch.compile from {model_type} encoder", category="setup")
                model.encoder = model.encoder._orig_mod
            if hasattr(model, 'decoder') and hasattr(model.decoder, '_orig_mod'):
                debug.log(f"Removing torch.compile from {model_type} decoder", category="setup")
                model.decoder = model.decoder._orig_mod
    
    # Execute special handlers for config-specific logic
    if special_handlers:
        for config_name, handler in special_handlers.items():
            if changes_detected.get(config_name, False):
                handler(runner, cached_config_attrs[config_name], 
                       new_configs[config_name], debug)
    
    # Update config attributes
    setattr(runner, model_attr, model)
    for config_name, attr_name in cached_config_attrs.items():
        setattr(runner, attr_name, new_configs.get(config_name))
    
    # Store on model so config travels with the model when cached
    for config_name, attr_name in cached_config_attrs.items():
        # Use explicit mapping if provided, otherwise derive from config name
        if model_config_attrs and config_name in model_config_attrs:
            model_attr_name = model_config_attrs[config_name]
        else:
            model_attr_name = f'_config_{config_name.split("_")[-1]}'
        setattr(model, model_attr_name, new_configs.get(config_name))
    
    # Mark that configs need to be applied
    config_needs_app_attr = f'_{model_attr}_config_needs_application'
    setattr(runner, config_needs_app_attr, True)
    
    return True


def _handle_blockswap_change(
    runner: 'VideoDiffusionInfer',
    attr_name: str,
    new_config: Optional[Dict[str, Any]],
    debug: Optional['Debug'] = None
) -> None:
    """
    Handle BlockSwap-specific configuration changes with proper cleanup.
    
    Called by _update_model_config when BlockSwap configuration changes are detected.
    Manages transition between BlockSwap states (enabled/disabled/reconfigured) with
    proper cleanup to avoid memory leaks and state corruption.
    
    Args:
        runner: VideoDiffusionInfer instance with DiT model
        attr_name: Runner attribute name storing cached config (e.g., '_dit_block_swap_config')
        new_config: New BlockSwap configuration dict or None to disable
        debug: Debug instance for logging
    """
    cached_config = getattr(runner, attr_name, None)
    
    # Determine BlockSwap status from configs
    had_blockswap = is_blockswap_enabled(cached_config)
    has_blockswap = is_blockswap_enabled(new_config)
    
    # If old config had BlockSwap features, clean them up first
    if had_blockswap and not has_blockswap:
        # Disabling BlockSwap completely
        debug.log("Disabling BlockSwap completely", category="blockswap")
        cleanup_blockswap(runner=runner, keep_state_for_cache=False)
    
    # Mark as inactive so the new config can be applied
    runner._blockswap_active = False


def _update_dit_config(
    runner: 'VideoDiffusionInfer',
    block_swap_config: Optional[Dict[str, Any]],
    torch_compile_args: Optional[Dict[str, Any]],
    attention_mode: Optional[str],
    debug: Optional['Debug'] = None
) -> bool:
    """
    Update DiT model configuration when reusing cached model.
    
    Compares new configuration settings against cached config to detect changes.
    Handles BlockSwap, torch.compile, and attention_mode configuration updates with 
    proper cleanup and reapplication when settings change.
    
    Args:
        runner: VideoDiffusionInfer instance with cached DiT model
        block_swap_config: New BlockSwap configuration dict with keys:
            - blocks_to_swap: int - Number of transformer blocks to offload
            - swap_io_components: bool - Whether to offload I/O components
        torch_compile_args: New torch.compile configuration dict with keys:
            - backend: str - Compiler backend (default: "inductor")
            - mode: str - Compilation mode (default: "default")
            - fullgraph: bool - Require single graph compilation
            - dynamic: bool - Enable dynamic shapes
            - dynamo_cache_size_limit: int - Cache size limit
            - dynamo_recompile_limit: int - Recompilation limit
        attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
        debug: Debug instance for logging
        
    Returns:
        bool: True if configuration was successfully updated
    """
    return _update_model_config(
        runner=runner,
        model_attr='dit',
        model_type='DiT',
        new_configs={
            'torch_compile': torch_compile_args,
            'block_swap': block_swap_config,
            'attention_mode': attention_mode
        },
        cached_config_attrs={
            'torch_compile': '_dit_compile_args',
            'block_swap': '_dit_block_swap_config',
            'attention_mode': '_dit_attention_mode'
        },
        model_config_attrs={
            'torch_compile': '_config_compile',
            'block_swap': '_config_swap',
            'attention_mode': '_config_attn'
        },
        config_describers={
            'torch_compile': _describe_compile_config,
            'block_swap': _describe_blockswap_config,
            'attention_mode': _describe_attention_mode
        },
        special_handlers={
            'block_swap': _handle_blockswap_change
        },
        debug=debug
    )


def _update_vae_config(
    runner: 'VideoDiffusionInfer',
    torch_compile_args: Optional[Dict[str, Any]],
    debug: Optional['Debug'] = None
) -> bool:
    """
    Update VAE model configuration when reusing cached model.
    
    Compares new configuration settings against cached config to detect changes.
    Handles torch.compile and tiling configuration updates with proper cleanup
    and reapplication when settings change.
    
    Args:
        runner: VideoDiffusionInfer instance with cached VAE model
        torch_compile_args: New torch.compile configuration dict with keys:
            - backend: str - Compiler backend (default: "inductor")
            - mode: str - Compilation mode (default: "default")
            - fullgraph: bool - Require single graph compilation
            - dynamic: bool - Enable dynamic shapes
            - dynamo_cache_size_limit: int - Cache size limit
            - dynamo_recompile_limit: int - Recompilation limit
        debug: Debug instance for logging
        
    Returns:
        bool: True if configuration was successfully updated
    """
    new_tiling_config = getattr(runner, '_new_vae_tiling_config', None)
    
    return _update_model_config(
        runner=runner,
        model_attr='vae',
        model_type='VAE',
        new_configs={
            'torch_compile': torch_compile_args,
            'tiling': new_tiling_config
        },
        cached_config_attrs={
            'torch_compile': '_vae_compile_args',
            'tiling': '_vae_tiling_config'
        },
        model_config_attrs={
            'torch_compile': '_config_compile',
            'tiling': '_config_tiling'
        },
        config_describers={
            'torch_compile': _describe_compile_config,
            'tiling': _describe_tiling_config
        },
        special_handlers=None,
        debug=debug
    )


def _initialize_cache_context(
    dit_cache: bool,
    vae_cache: bool,
    dit_id: Optional[int],
    vae_id: Optional[int],
    dit_model: str,
    vae_model: str,
    debug: Optional['Debug'] = None
) -> Dict[str, Any]:
    """
    Initialize cache context with global cache lookups and model name validation.
    
    Checks the global cache for existing DiT/VAE models and validates that cached
    models match the requested model names. Removes stale cache entries when model
    names don't match.
    
    Args:
        dit_cache: Whether DiT caching is enabled
        vae_cache: Whether VAE caching is enabled
        dit_id: Node ID for DiT model lookup
        vae_id: Node ID for VAE model lookup
        dit_model: Requested DiT model filename for validation
        vae_model: Requested VAE model filename for validation
        debug: Debug instance for logging
        
    Returns:
        Dict[str, Any]: Cache context dictionary containing:
            - global_cache: GlobalModelCache instance
            - dit_cache: DiT caching enabled flag
            - vae_cache: VAE caching enabled flag
            - dit_id: DiT node ID
            - vae_id: VAE node ID
            - dit_model: DiT model name
            - vae_model: VAE model name
            - cached_dit: Cached DiT model instance (if found and valid)
            - cached_vae: Cached VAE model instance (if found and valid)
            - dit_newly_cached: Flag indicating if DiT was just cached
            - vae_newly_cached: Flag indicating if VAE was just cached
            - reusing_runner: Flag indicating if runner template is reused
            
    Note:
        Automatically removes stale cache entries when model names don't match
        the cached model's stored name (_model_name attribute).
    """
    global_cache = get_global_cache()
    context = {
        'global_cache': global_cache,
        'dit_cache': dit_cache,
        'vae_cache': vae_cache,
        'dit_id': dit_id,
        'vae_id': vae_id,
        'dit_model': dit_model,
        'vae_model': vae_model,
        'cached_dit': None,
        'cached_vae': None,
        'dit_newly_cached': False,
        'vae_newly_cached': False,
        'reusing_runner': False
    }
    
    # Check for cached DiT model with model name validation
    # Model name validation prevents stale cache when user switches models in UI
    if dit_cache and dit_model and dit_id is not None:
        cached_model = global_cache.get_dit({'node_id': dit_id, 'cache_model': True}, debug)
        if cached_model:
            # Verify cached model matches requested model by checking _model_name attribute
            cached_model_name = getattr(cached_model, '_model_name', None)
            if cached_model_name == dit_model:
                # Cache hit with valid model - reuse it
                context['cached_dit'] = cached_model
            else:
                # Model changed - remove stale cache and log the change
                if cached_model_name:
                    debug.log(f"DiT model changed in cache ({cached_model_name} → {dit_model}), "
                             f"removing stale cached model", category="cache", force=True)
                global_cache.remove_dit({'node_id': dit_id}, debug)
    else:
        # Caching disabled or no ID - clean up any existing cache for this node
        if dit_id is not None:
            global_cache.remove_dit({'node_id': dit_id}, debug)

    # Check for cached VAE model with model name validation
    if vae_cache and vae_model and vae_id is not None:
        cached_model = global_cache.get_vae({'node_id': vae_id, 'cache_model': True}, debug)
        if cached_model:
            # Verify cached model matches requested model by checking _model_name attribute
            cached_model_name = getattr(cached_model, '_model_name', None)
            if cached_model_name == vae_model:
                context['cached_vae'] = cached_model
            else:
                # Model changed - remove stale cache and log the change
                if cached_model_name:
                    debug.log(f"VAE model changed in cache ({cached_model_name} → {vae_model}), "
                             f"removing stale cached model", category="cache", force=True)
                global_cache.remove_vae({'node_id': vae_id}, debug)
    else:
        if vae_id is not None:
            global_cache.remove_vae({'node_id': vae_id}, debug)
    
    return context


def _acquire_runner(
    cache_context: Dict[str, Any],
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Optional['Debug'] = None
) -> VideoDiffusionInfer:
    """
    Get or create VideoDiffusionInfer runner instance using unified caching approach.
    
    Checks global cache for existing runner template matching the DiT/VAE node ID pair.
    If found and model names match, reuses the template. Otherwise creates new runner.
    
    Args:
        cache_context: Cache context dict from _initialize_cache_context containing:
            - global_cache: GlobalModelCache instance
            - dit_id: DiT node ID for cache lookup
            - vae_id: VAE node ID for cache lookup
            - reusing_runner: Flag to be updated if template is reused
        dit_model: DiT model filename for validation (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename for validation (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory for model files
        debug: Debug instance for logging
        
    Returns:
        VideoDiffusionInfer: Runner instance (cached template or newly created)
    """
    # Try to get runner template from global cache
    template = cache_context['global_cache'].get_runner(
        cache_context['dit_id'], cache_context['vae_id'], debug
    )
    
    if template:
        # We have a template - check if we can use it
        current_dit = getattr(template, '_dit_model_name', None)
        current_vae = getattr(template, '_vae_model_name', None)
        models_match = (current_dit == dit_model and current_vae == vae_model)
        
        if models_match:
            # Perfect match - reuse template directly
            runner_key = f"{cache_context['dit_id']}+{cache_context['vae_id']}"
            debug.log(f"Reusing cached runner template: nodes {runner_key}", category="reuse", force=True)
            cache_context['reusing_runner'] = True
            return template
        else:
            # Template exists but models changed and no cached models - create new
            return _create_new_runner(dit_model, vae_model, base_cache_dir, debug)
    else:
        # No template - create new runner
        return _create_new_runner(dit_model, vae_model, base_cache_dir, debug)


def _create_new_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Optional['Debug'] = None
) -> VideoDiffusionInfer:
    """
    Create a new VideoDiffusionInfer runner instance from scratch.
    
    Loads appropriate configuration file based on model size (3B or 7B), creates
    runner instance, and initializes with default settings. Called when no cached
    runner template is available or when model selection changes.
    
    Args:
        dit_model: DiT model filename (determines config selection)
                  - Contains "7b" → loads configs_7b/main.yaml
                  - Otherwise → loads configs_3b/main.yaml
        vae_model: VAE model filename (stored for reference, not used in config selection)
        base_cache_dir: Base directory for model files (not used directly but passed for context)
        debug: Debug instance for logging and timing
        
    Returns:
        VideoDiffusionInfer: Newly created runner with:
            - Loaded OmegaConf configuration
            - Initialized diffusion sampler and schedule
            - Config set to mutable (readonly=False)
            - No models loaded (structure only)
    """
    debug.log(f"Creating new runner: DiT={dit_model}, VAE={vae_model}", 
             category="runner", force=True)
    
    debug.start_timer("config_load")
    config_path = os.path.join(script_directory, 
                              './configs_7b' if "7b" in dit_model else './configs_3b', 
                              'main.yaml')
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")
    
    debug.start_timer("runner_video_infer")
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    return runner


def configure_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: 'Debug',
    ctx: Dict[str, Any],
    dit_cache: bool = False,
    vae_cache: bool = False,
    dit_id: Optional[int] = None,
    vae_id: Optional[int] = None,
    block_swap_config: Optional[Dict[str, Any]] = None,
    encode_tiled: bool = False,
    encode_tile_size: Optional[Tuple[int, int]] = None,
    encode_tile_overlap: Optional[Tuple[int, int]] = None,
    decode_tiled: bool = False,
    decode_tile_size: Optional[Tuple[int, int]] = None,
    decode_tile_overlap: Optional[Tuple[int, int]] = None,
    tile_debug: str = "false",
    attention_mode: str = 'sdpa',
    torch_compile_args_dit: Optional[Dict[str, Any]] = None,
    torch_compile_args_vae: Optional[Dict[str, Any]] = None
) -> Tuple[VideoDiffusionInfer, Dict[str, Any]]:
    """
    Configure VideoDiffusionInfer runner with model loading and settings.
    
    Handles model changes and caching logic with independent DiT/VAE caching support.
    
    Args:
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        debug: Debug instance for logging (required)
        ctx: Generation context from setup_generation_context
        dit_cache: Whether to cache DiT model between runs
        vae_cache: Whether to cache VAE model between runs
        dit_id: Node instance ID for DiT model caching (required if dit_cache=True)
        vae_id: Node instance ID for VAE model caching (required if vae_cache=True)
        block_swap_config: Optional BlockSwap configuration for DiT memory optimization
        encode_tiled: Enable tiled encoding to reduce VRAM during VAE encoding
        encode_tile_size: Tile size for encoding (height, width)
        encode_tile_overlap: Tile overlap for encoding (height, width)
        decode_tiled: Enable tiled decoding to reduce VRAM during VAE decoding
        decode_tile_size: Tile size for decoding (height, width)
        decode_tile_overlap: Tile overlap for decoding (height, width)
        tile_debug: Tile visualization mode (false/encode/decode)
        attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
        torch_compile_args_dit: Optional torch.compile configuration for DiT model
        torch_compile_args_vae: Optional torch.compile configuration for VAE model
        
    Returns:
        Tuple[VideoDiffusionInfer, Dict[str, Any]]: (configured runner, cache context dict)
        
    Features:
        - Independent DiT and VAE caching for flexible memory management
        - Dynamic model reloading when models change
        - Optional torch.compile optimization for inference speedup
        - Separate encode/decode tiling configuration for optimal performance
        - Memory optimization and BlockSwap integration
        
    Raises:
        ValueError: If debug instance is not provided
    """
    
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Validate BlockSwap configuration early (before any model loading)
    block_swap_config = validate_blockswap_config(
        block_swap_config=block_swap_config,
        dit_device=ctx['dit_device'],
        dit_offload_device=ctx.get('dit_offload_device'),
        debug=debug
    )
    
    # Phase 1: Initialize cache and get cached models
    cache_context = _initialize_cache_context(
        dit_cache, vae_cache, dit_id, vae_id, 
        dit_model, vae_model, debug
    )
    
    # Phase 2: Get or create runner
    runner = _acquire_runner(
        cache_context, dit_model, vae_model, 
        base_cache_dir, debug
    )
    
    # Phase 3: Configure runner settings
    _configure_runner_settings(
        runner, ctx,
        encode_tiled, encode_tile_size, encode_tile_overlap,
        decode_tiled, decode_tile_size, decode_tile_overlap,
        tile_debug, attention_mode,
        torch_compile_args_dit, torch_compile_args_vae,
        block_swap_config, debug
    )
    
    # Phase 4: Setup models (load from cache or create new)
    _setup_models(
        runner, cache_context, dit_model, vae_model, 
        base_cache_dir, block_swap_config, debug
    )
    
    return runner, cache_context


def _configure_runner_settings(
    runner: VideoDiffusionInfer,
    ctx: Dict[str, Any],
    encode_tiled: bool,
    encode_tile_size: Optional[Tuple[int, int]],
    encode_tile_overlap: Optional[Tuple[int, int]],
    decode_tiled: bool,
    decode_tile_size: Optional[Tuple[int, int]],
    decode_tile_overlap: Optional[Tuple[int, int]],
    tile_debug: str,
    attention_mode: str,
    torch_compile_args_dit: Optional[Dict[str, Any]],
    torch_compile_args_vae: Optional[Dict[str, Any]],
    block_swap_config: Optional[Dict[str, Any]],
    debug: Optional['Debug'] = None
) -> None:
    """
    Configure runner settings for VAE tiling, torch.compile, and BlockSwap.
    
    Stores configuration settings on runner for later comparison and application.
    Settings are stored in temporary "_new_*" attributes and later validated/applied
    in _setup_models phase. This separation allows configuration change detection
    when reusing cached models.
    
    Args:
        runner: VideoDiffusionInfer instance to configure
        ctx: Generation context from setup_generation_context
        encode_tiled: Enable tiled VAE encoding to reduce VRAM during encoding
        encode_tile_size: Tile dimensions (height, width) for encoding in pixels
        encode_tile_overlap: Overlap dimensions (height, width) between encoding tiles
        decode_tiled: Enable tiled VAE decoding to reduce VRAM during decoding
        decode_tile_size: Tile dimensions (height, width) for decoding in pixels
        decode_tile_overlap: Overlap dimensions (height, width) between decoding tiles
        tile_debug: Tile visualization mode (false/encode/decode)
        attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
        torch_compile_args_dit: torch.compile configuration for DiT model or None
        torch_compile_args_vae: torch.compile configuration for VAE model or None
        block_swap_config: BlockSwap configuration for DiT model or None
        debug: Debug instance (stored on runner for model access)
    """
    # VAE tiling settings
    runner.encode_tiled = encode_tiled
    runner.encode_tile_size = encode_tile_size
    runner.encode_tile_overlap = encode_tile_overlap
    runner.decode_tiled = decode_tiled
    runner.decode_tile_size = decode_tile_size
    runner.decode_tile_overlap = decode_tile_overlap
    runner.tile_debug = tile_debug
    
    # Store the new configs temporarily for later comparison
    # Don't set them as attributes yet - let the update functions handle that
    runner._new_dit_compile_args = torch_compile_args_dit
    runner._new_vae_compile_args = torch_compile_args_vae
    runner._new_dit_block_swap_config = block_swap_config
    runner._new_dit_attention_mode = attention_mode
    runner._new_vae_tiling_config = {
        'encode_tiled': encode_tiled,
        'encode_tile_size': encode_tile_size,
        'encode_tile_overlap': encode_tile_overlap,
        'decode_tiled': decode_tiled,
        'decode_tile_size': decode_tile_size,
        'decode_tile_overlap': decode_tile_overlap
    }
    
    # Store device configuration on runner for submodule access (e.g., BlockSwap, Cleanup)
    runner._dit_device = ctx['dit_device']
    runner._vae_device = ctx['vae_device']
    runner._dit_offload_device = ctx['dit_offload_device']
    runner._vae_offload_device = ctx['vae_offload_device']
    runner._tensor_offload_device = ctx['tensor_offload_device']
    runner._compute_dtype = ctx['compute_dtype']

    runner.debug = debug


def _setup_models(
    runner: VideoDiffusionInfer,
    cache_context: Dict[str, Any],
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    block_swap_config: Optional[Dict[str, Any]],
    debug: Optional['Debug'] = None
) -> None:
    """
    Setup DiT and VAE models from cache or create new structures.
    
    Central orchestration function that:
    1. Sets up DiT model (cached or new structure)
    2. Sets up VAE model (cached or new structure)
    3. Validates and updates configurations for cached models
    4. Stores initial configurations for new models
    5. Cleans up temporary configuration attributes
    
    This function coordinates the complex interaction between model caching,
    configuration management, and the meta device initialization strategy.
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cache_context: Cache context from _initialize_cache_context with keys:
            - cached_dit, cached_vae: Cached model instances (or None)
            - dit_id, vae_id: Node IDs for caching
            - Other cache state flags
        dit_model: DiT model filename for loading/validation
        vae_model: VAE model filename for loading/validation
        base_cache_dir: Base directory containing model files
        block_swap_config: BlockSwap configuration (passed to DiT setup)
        debug: Debug instance for logging and timing
    """
    debug.start_timer("model_structures")
    
    # Setup DiT
    dit_created = _setup_dit_model(runner, cache_context, dit_model, base_cache_dir, 
                                   block_swap_config, debug)
    
    # Only update DiT config if model was cached/reused (not newly created)
    if not dit_created and hasattr(runner, 'dit') and runner.dit is not None:
        _update_dit_config(runner, runner._new_dit_block_swap_config, 
                         runner._new_dit_compile_args, runner._new_dit_attention_mode, debug)
    elif dit_created:
        # For newly created models, just set initial config attributes (no comparison needed)
        runner._dit_compile_args = runner._new_dit_compile_args
        runner._dit_block_swap_config = runner._new_dit_block_swap_config
        runner._dit_attention_mode = runner._new_dit_attention_mode
        # Also store on model so config travels with the model when cached
        if hasattr(runner, 'dit') and runner.dit:
            runner.dit._config_compile = runner._new_dit_compile_args
            runner.dit._config_swap = runner._new_dit_block_swap_config
            runner.dit._config_attn = runner._new_dit_attention_mode
    
    # Setup VAE
    vae_created = _setup_vae_model(runner, cache_context, vae_model, base_cache_dir, debug)
    
    # Only update VAE config if model was cached/reused (not newly created)
    if not vae_created and hasattr(runner, 'vae') and runner.vae is not None:
        _update_vae_config(runner, runner._new_vae_compile_args, debug)
    elif vae_created:
        # For newly created models, just set initial config attributes (no comparison needed)
        runner._vae_compile_args = runner._new_vae_compile_args
        runner._vae_tiling_config = runner._new_vae_tiling_config
        # Also store on model so config travels with the model when cached
        if hasattr(runner, 'vae') and runner.vae:
            runner.vae._config_compile = runner._new_vae_compile_args
            runner.vae._config_tiling = runner._new_vae_tiling_config
    
    # Clean up temporary attributes
    for attr in ['_new_dit_compile_args', '_new_vae_compile_args', '_new_dit_block_swap_config', '_new_dit_attention_mode', '_new_vae_tiling_config']:
        if hasattr(runner, attr):
            delattr(runner, attr)
    
    debug.end_timer("model_structures", "Model structures prepared")


def _setup_dit_model(
    runner: VideoDiffusionInfer,
    cache_context: Dict[str, Any],
    dit_model: str,
    base_cache_dir: str,
    block_swap_config: Optional[Dict[str, Any]],
    debug: Optional['Debug'] = None
) -> bool:
    """
    Setup DiT model from cache or create new meta device structure.
    
    Handles three scenarios:
    1. Model changed: Cleanup old model, create new structure
    2. Cached model available: Reuse cached model, restore config
    3. No model exists: Create new meta device structure
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cache_context: Cache context dict with keys:
            - cached_dit: Cached DiT model instance (or None)
            - dit_id: Node ID for cache operations
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        block_swap_config: BlockSwap configuration to store (not applied here)
        debug: Debug instance for logging
        
    Returns:
        bool: True if new model structure was created, False if cached model reused
    """
    
    # Check if model changed - clean up old model if different
    current_dit_name = getattr(runner, '_dit_model_name', None)
    if current_dit_name and current_dit_name != dit_model:
        if hasattr(runner, 'dit') and runner.dit is not None:
            debug.log(f"DiT model changed ({current_dit_name} → {dit_model}), cleaning old model", 
                     category="cache", force=True)
            cleanup_dit(runner=runner, debug=debug, cache_model=False)
    
    if cache_context['cached_dit'] is not None:
        # Reuse cached DiT model
        debug.log(f"Reusing cached DiT ({cache_context['dit_id']}): {dit_model}", 
                category="reuse", force=True)
        runner.dit = cache_context['cached_dit']
        runner._dit_checkpoint = find_model_file(dit_model, base_cache_dir)
        runner._dit_model_name = dit_model
        
        # Restore config attributes from model to runner (config travels with model)
        runner._dit_compile_args = getattr(runner.dit, '_config_compile', None)
        runner._dit_block_swap_config = getattr(runner.dit, '_config_swap', None)
        runner._dit_attention_mode = getattr(runner.dit, '_config_attn', None)
        
        # blockswap_active will be set by apply_block_swap_to_dit
        # when the model is materialized to the inference device
        runner._blockswap_active = False
        
        return False
    elif not hasattr(runner, 'dit') or runner.dit is None:
        # Create new DiT model
        # Set DiT dtype from runner's compute_dtype
        # compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)
        # dit_dtype_str = str(compute_dtype).split('.')[-1]
        # runner.config.dit.dtype = dit_dtype_str
        # runner._dit_dtype_override = compute_dtype
        
        dit_checkpoint_path = find_model_file(dit_model, base_cache_dir)
        runner = prepare_model_structure(runner, "dit", dit_checkpoint_path, 
                                        runner.config, debug, block_swap_config)
        runner._dit_model_name = dit_model
        return True
    else:
        # Model already exists from previous run with same runner
        runner._dit_model_name = dit_model
        return False


def _setup_vae_model(
    runner: VideoDiffusionInfer,
    cache_context: Dict[str, Any],
    vae_model: str,
    base_cache_dir: str,
    debug: Optional['Debug'] = None
) -> bool:
    """
    Setup VAE model from cache or create new meta device structure.
    
    Handles three scenarios:
    1. Model changed: Cleanup old model, configure and create new structure
    2. Cached model available: Reuse cached model, restore config
    3. No model exists: Configure VAE settings, create new meta device structure
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cache_context: Cache context dict with keys:
            - cached_vae: Cached VAE model instance (or None)
            - vae_id: Node ID for cache operations
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        debug: Debug instance for logging
        
    Returns:
        bool: True if new model structure was created, False if cached model reused
    """
    
    # Check if model changed - clean up old model if different
    current_vae_name = getattr(runner, '_vae_model_name', None)
    if current_vae_name and current_vae_name != vae_model:
        if hasattr(runner, 'vae') and runner.vae is not None:
            debug.log(f"VAE model changed ({current_vae_name} → {vae_model}), cleaning old model", 
                     category="cache", force=True)
            cleanup_vae(runner=runner, debug=debug, cache_model=False)
    
    if cache_context['cached_vae'] is not None:
        # Reuse cached VAE model
        debug.log(f"Reusing cached VAE ({cache_context['vae_id']}): {vae_model}", 
                category="reuse", force=True)
        runner.vae = cache_context['cached_vae']
        runner._vae_checkpoint = find_model_file(vae_model, base_cache_dir)
        runner._vae_model_name = vae_model
        
        # Restore config attributes from model to runner (config travels with model)
        runner._vae_compile_args = getattr(runner.vae, '_config_compile', None)
        runner._vae_tiling_config = getattr(runner.vae, '_config_tiling', None)
        
        return False
    elif not hasattr(runner, 'vae') or runner.vae is None:
        # Create new VAE model
        # Configure VAE
        vae_config_path = os.path.join(script_directory, 
                                      'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
        vae_config = load_config(vae_config_path)
        
        spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
        temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
        vae_config.spatial_downsample_factor = spatial_downsample_factor
        vae_config.temporal_downsample_factor = temporal_downsample_factor

        runner.config.vae.model = OmegaConf.merge(runner.config.vae.model, vae_config)
        
        # Set VAE dtype from runner's compute_dtype
        compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)
        vae_dtype_str = str(compute_dtype).split('.')[-1]
        runner.config.vae.dtype = vae_dtype_str
        runner._vae_dtype_override = compute_dtype
        
        vae_checkpoint_path = find_model_file(vae_model, base_cache_dir)
        runner = prepare_model_structure(runner, "vae", vae_checkpoint_path, 
                                        runner.config, debug, None)
        
        debug.log(
            f"VAE downsample factors configured "
            f"(spatial: {spatial_downsample_factor}x, "
            f"temporal: {temporal_downsample_factor}x)",
            category="vae"
        )

        runner._vae_model_name = vae_model
        return True
    else:
        # Model already exists from previous run with same runner
        runner._vae_model_name = vae_model
        return False


def apply_model_specific_config(model: torch.nn.Module, runner: VideoDiffusionInfer, 
                                config: OmegaConf, is_dit: bool, 
                                debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Apply model-specific configurations (FP8, BlockSwap, torch.compile).
    
    This function is idempotent and can be safely called on both newly materialized
    and already-configured models. It checks state flags to determine what needs
    to be applied.
    
    Critical: For DiT, BlockSwap must be applied BEFORE torch.compile.
    torch.compile captures the computational graph, so any wrapping done
    after compilation (like BlockSwap's forward wrapping) won't work.
    
    Args:
        model: Loaded model instance
        runner: Runner to attach model to
        config: Full configuration object
        is_dit: Whether this is a DiT model (vs VAE)
        debug: Debug instance
        
    Returns:
        Configured model with BlockSwap and torch.compile applied if configured
    """
    if is_dit:
        # DiT-specific
        # Apply compatibility wrapper with compute_dtype
        if not isinstance(model, CompatibleDiT):
            debug.log("Applying DiT compatibility wrapper", category="setup")
            debug.start_timer("CompatibleDiT")
            # Get compute_dtype from runner if available, fallback to bfloat16
            compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)
            model = CompatibleDiT(model, debug, compute_dtype=compute_dtype, skip_conversion=False)
            debug.end_timer("CompatibleDiT", "Compatibility wrapper application")
        else:
            debug.log("Reusing existing DiT compatibility wrapper", category="reuse")
        
        # Apply attention mode and compute_dtype to all FlashAttentionVarlen modules
        if hasattr(runner, '_dit_attention_mode'):
            requested_attention_mode = runner._dit_attention_mode or 'sdpa'
            
            # Validate and get final attention_mode (with warning if fallback needed)
            attention_mode = validate_attention_mode(requested_attention_mode, debug)
            
            # Get compute_dtype from runner
            compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)            
            debug.log(f"Applying {attention_mode} attention mode and {compute_dtype} compute dtype to model", category="setup")
            
            # Get the actual model (unwrap if needed)
            actual_model = model.dit_model if hasattr(model, 'dit_model') else model
            
            # Update all FlashAttentionVarlen instances
            updated_count = 0
            for module in actual_model.modules():
                if type(module).__name__ == 'FlashAttentionVarlen':
                    module.attention_mode = attention_mode
                    module.compute_dtype = compute_dtype
                    updated_count += 1
            
            if updated_count > 0:
                debug.log(f"Applied {attention_mode} and compute_dtype={compute_dtype} to {updated_count} modules", category="success")

        # Apply BlockSwap before torch.compile (only if not already active)
        # BlockSwap wraps forward methods, and torch.compile needs to capture the wrapped version
        if hasattr(runner, '_dit_block_swap_config') and runner._dit_block_swap_config:
            # Check if BlockSwap needs to be applied
            needs_blockswap = not (hasattr(runner, '_blockswap_active') and runner._blockswap_active)
            if needs_blockswap:
                runner.dit = model
                apply_block_swap_to_dit(runner, runner._dit_block_swap_config, debug)
                # Mark as active after successful application
                runner._blockswap_active = True
        
        # Apply torch.compile after BlockSwap (only if not already compiled)
        # Check: model has _orig_mod attribute means it's already torch.compiled
        if hasattr(runner, '_dit_compile_args') and runner._dit_compile_args:
            if not hasattr(model, '_orig_mod'):
                model = _apply_torch_compile(model, runner._dit_compile_args, "DiT", debug)
        
        runner.dit = model

        # Clear the config application flag after successful application
        if hasattr(runner, '_dit_config_needs_application'):
            runner._dit_config_needs_application = False
        
    else:
        # VAE-specific configurations
        # Set to eval mode (no gradients needed for inference)
        if model.training:
            debug.log("VAE model set to eval mode (gradients disabled)", category="vae")
            debug.start_timer("model_requires_grad")
            model.requires_grad_(False).eval()
            debug.end_timer("model_requires_grad", "VAE model set to eval mode")
        
        # Configure causal slicing if available - always apply as it's lightweight
        if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
            debug.log("Configuring VAE causal slicing for temporal processing", category="vae")
            debug.start_timer("vae_set_causal_slicing")
            model.set_causal_slicing(**config.vae.slicing)
            debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configuration")
        
        # Set memory limits if available - always apply to ensure limits are set
        if hasattr(model, "set_memory_limit") and hasattr(config.vae, "memory_limit"):
            debug.log("Configuring VAE memory limits for causal convolutions", category="vae")
            debug.start_timer("vae_set_memory_limit")
            model.set_memory_limit(**config.vae.memory_limit)
            debug.end_timer("vae_set_memory_limit", "VAE memory limits configured")

        # Apply torch.compile if configured (only if not already compiled)
        if hasattr(runner, '_vae_compile_args') and runner._vae_compile_args:
            # Check if encoder/decoder already compiled (have _orig_mod)
            encoder_compiled = hasattr(model, 'encoder') and hasattr(model.encoder, '_orig_mod')
            decoder_compiled = hasattr(model, 'decoder') and hasattr(model.decoder, '_orig_mod')
            
            if not (encoder_compiled and decoder_compiled):
                model = _apply_vae_submodule_compile(model, runner._vae_compile_args, debug)
            else:
                debug.log("Reusing existing torch.compile for VAE submodules", category="reuse")
        
        # Propagate debug and tensor_offload_device to submodules
        model.debug = debug
        model.tensor_offload_device = runner._tensor_offload_device
        _propagate_debug_to_modules(model, debug)
        runner.vae = model
        
        # Clear the config application flag after successful application
        if hasattr(runner, '_vae_config_needs_application'):
            runner._vae_config_needs_application = False
    
    return model


def _configure_torch_compile(compile_args: Dict[str, Any], model_type: str, 
                            debug: Optional['Debug'] = None) -> Tuple[Dict[str, Any], bool]:
    """
    Extract and configure torch.compile settings with dependency validation.
    
    Centralizes common configuration logic for both full model and submodule compilation.
    Validates that required dependencies (like Triton for inductor backend) are available
    before attempting compilation to provide early, clear error messages.
    
    Args:
        compile_args: Compilation configuration dictionary
        model_type: Model type string for logging (e.g., "DiT", "VAE")
        debug: Debug instance for logging
        
    Returns:
        Tuple of (compile_settings dict, success boolean)
        compile_settings contains: backend, mode, fullgraph, dynamic
        
    Raises:
        RuntimeError: If inductor backend is requested but Triton is not available
    """
    # Extract settings with defaults
    settings = {
        'backend': compile_args.get('backend', 'inductor'),
        'mode': compile_args.get('mode', 'default'),
        'fullgraph': compile_args.get('fullgraph', False),
        'dynamic': compile_args.get('dynamic', False)
    }
    
    dynamo_cache_size_limit = compile_args.get('dynamo_cache_size_limit', 64)
    dynamo_recompile_limit = compile_args.get('dynamo_recompile_limit', 128)
    
    # Check Triton availability for inductor backend BEFORE attempting compilation
    if settings['backend'] == 'inductor':
        if not TRITON_AVAILABLE:
            error_msg = (
                f"Cannot use torch.compile with 'inductor' backend: Triton is not installed.\n"
                f"\n"
                f"Triton is required for the inductor backend which performs kernel fusion and optimization.\n"
                f"\n"
                f"To fix this issue:\n"
                f"  1. Install Triton: pip install triton\n"
                f"  2. OR change backend to 'cudagraphs' (lightweight, no Triton needed)\n"
                f"  3. OR disable torch.compile\n"
                f"\n"
                f"For more info: https://github.com/triton-lang/triton"
            )
            debug.log(error_msg, level="ERROR", category="setup", force=True)
            raise RuntimeError(
                "torch.compile with inductor backend requires Triton. "
                "Install with: pip install triton"
            )
    
    # Log compilation configuration
    debug.log(f"Configuring torch.compile for {model_type}...", category="setup", force=True)
    debug.log(f"Backend: {settings['backend']} | Mode: {settings['mode']} | "
             f"Fullgraph: {settings['fullgraph']} | Dynamic: {settings['dynamic']}", 
             category="setup", indent_level=1)
    debug.log(f"Dynamo cache_size_limit: {dynamo_cache_size_limit} | "
             f"recompile_limit: {dynamo_recompile_limit}", category="setup", indent_level=1)
    
    # Configure torch._dynamo settings
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        torch._dynamo.config.recompile_limit = dynamo_recompile_limit
        debug.log(f"torch._dynamo configured successfully", category="success", indent_level=1)
        return settings, True
    except Exception as e:
        debug.log(f"Could not configure torch._dynamo settings: {e}", 
                 level="WARNING", category="setup", force=True, indent_level=1)
        return settings, False


def _apply_torch_compile(model: torch.nn.Module, compile_args: Dict[str, Any], 
                        model_type: str, debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Apply torch.compile to entire model with configured settings.
    
    Args:
        model: Model to compile
        compile_args: Compilation configuration
        model_type: "DiT" or "VAE" for logging
        debug: Debug instance
        
    Returns:
        Compiled model, or original model if compilation fails
    """
    try:
        # Configure compilation settings
        settings, _ = _configure_torch_compile(compile_args, model_type, debug)
        
        # Compile entire model
        debug.start_timer(f"{model_type.lower()}_compile")
        compiled_model = torch.compile(model, **settings)
        debug.end_timer(f"{model_type.lower()}_compile", 
                       f"{model_type} model wrapped for compilation", force=True)
        debug.log(f"Actual compilation will happen on first batch (expect initial delay, then speedup)", category="info", indent_level=1)
        
        return compiled_model
        
    except Exception as e:
        debug.log(f"torch.compile failed for {model_type}: {e}", 
                 level="WARNING", category="setup", force=True)
        debug.log(f"Falling back to uncompiled model", 
                 level="WARNING", category="setup", force=True, indent_level=1)
        return model


def _disable_compile_for_dynamic_modules(module: torch.nn.Module) -> None:
    """
    Mark modules with dynamic shapes to be excluded from torch.compile.
    This prevents recompilation issues with variable tensor sizes.
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, InflatedCausalConv3d):
            # Mark module to skip compilation
            submodule._dynamo_disable = True


def _apply_vae_submodule_compile(model: torch.nn.Module, compile_args: Dict[str, Any], 
                                 debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Apply torch.compile to VAE core submodules instead of entire model.
    
    The VAE's high-level encode/decode methods contain complex control flow
    (temporal slicing, tiling, stateful memory management) that prevents 
    torch.compile from optimizing effectively. Instead, we compile only the 
    core neural networks (encoder, decoder) which have straightforward forward 
    passes suitable for compilation.
    
    Note: quant_conv and post_quant_conv are disabled (None) in the current VAE 
    architecture/yaml, so they are not compiled.
    
    Args:
        model: VAE model instance
        compile_args: Compilation configuration
        debug: Debug instance
        
    Returns:
        VAE model with compiled submodules
    """
    try:
        # Configure compilation settings
        settings, _ = _configure_torch_compile(compile_args, "VAE submodules", debug)
        
        # Compile submodules
        compiled_modules = []
        debug.start_timer("vae_submodule_compile")
        
        if hasattr(model, 'encoder') and model.encoder is not None:
            # Disable compilation for InflatedCausalConv3d modules due to dynamic shapes
            _disable_compile_for_dynamic_modules(model.encoder)
            model.encoder = torch.compile(model.encoder, **settings)
            compiled_modules.append('encoder')
            debug.log(f"VAE encoder found and added to compilation queue", category="success", indent_level=1)

        if hasattr(model, 'decoder') and model.decoder is not None:
            # Disable compilation for InflatedCausalConv3d modules due to dynamic shapes  
            _disable_compile_for_dynamic_modules(model.decoder)
            model.decoder = torch.compile(model.decoder, **settings)
            compiled_modules.append('decoder')
            debug.log(f"VAE decoder found and added to compilation queue", category="success", indent_level=1)
        
        debug.end_timer("vae_submodule_compile", 
                       f"VAE submodules compiled: {', '.join(compiled_modules)}", force=True)
        debug.log(f"Actual compilation will happen on first batch (expect initial delay, then speedup)", category="info", indent_level=1)
        
        return model
        
    except Exception as e:
        debug.log(f"torch.compile failed for VAE submodules: {e}", 
                 level="WARNING", category="setup", force=True)
        debug.log(f"Falling back to uncompiled VAE", 
                 level="WARNING", category="setup", force=True, indent_level=1)
        return model


def _propagate_debug_to_modules(module: torch.nn.Module, debug: 'Debug') -> None:
    """
    Propagate debug instance to specific submodules that need it.
    Only targets modules that actually use debug to avoid unnecessary memory overhead.
    
    Args:
        module: Parent module to propagate through
        debug: Debug instance to attach
    """
    if debug is None:
        return  # Early exit if no debug instance
        
    target_modules = {'ResnetBlock3D', 'Upsample3D', 'InflatedCausalConv3d', 'GroupNorm'}
    
    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ in target_modules:
            if not hasattr(submodule, 'debug'):  # Only set if not already present
                submodule.debug = debug