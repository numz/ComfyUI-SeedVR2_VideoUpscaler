"""
Generation Logic Module for SeedVR2

This module implements a four-phase batch processing pipeline for video upscaling:
- Phase 1: Batch VAE encoding of all input frames
- Phase 2: Batch DiT upscaling of all encoded latents
- Phase 3: Batch VAE decoding of all upscaled latents
- Phase 4: Post-processing and final video assembly

This architecture minimizes model swapping overhead by completing each phase
for all batches before moving to the next phase, significantly improving
performance especially when using model offloading.

Key Features:
- Four-phase pipeline (encode-all → upscale-all → decode-all → postprocess-all) for efficiency
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Temporal overlap support for smooth transitions between batches
- Adaptive dtype detection and configuration
- Memory-efficient pre-allocated batch processing
- Stream-based assembly eliminates memory spikes for long videos
- Advanced video format handling (4n+1 constraint)
- Clean separation of concerns with phase-specific resource management
- Each phase handles its own cleanup in finally blocks
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable

from .generation_utils import (
    setup_video_transform,
    pad_video_temporal,
    check_interrupt,
    ensure_precision_initialized,
    _draw_tile_boundaries,
    load_text_embeddings,
    blend_overlapping_frames,
    calculate_optimal_batch_params,
    script_directory
)
from .model_configuration import apply_model_specific_config
from .model_loader import materialize_model
from .alpha_upscaling import process_alpha_for_batch
from .infer import VideoDiffusionInfer
from ..common.seed import set_seed
from ..optimization.memory_manager import (
    cleanup_dit,
    cleanup_vae,
    cleanup_text_embeddings,
    manage_tensor,
    manage_model_device,
    release_tensor_memory,
    release_tensor_collection
)
from ..optimization.performance import (
    optimized_video_rearrange, 
    optimized_single_video_rearrange, 
    optimized_sample_to_image_format
)
from ..utils.color_fix import (
    lab_color_transfer,
    wavelet_adaptive_color_correction,
    hsv_saturation_histogram_match, 
    wavelet_reconstruction,
    adaptive_instance_normalization
)


def _prepare_video_batch(
    images: torch.Tensor,
    start_idx: int,
    end_idx: int,
    uniform_padding: int = 0,
    debug: Optional['Debug'] = None,
    log_info: bool = False
) -> torch.Tensor:
    """
    Extract and prepare video batch with uniform padding and permutation.
    
    Args:
        images: Source video frames [T, H, W, C]
        start_idx: Start frame index
        end_idx: End frame index (exclusive)
        uniform_padding: Number of frames to pad (0 = no padding)
        debug: Debug instance for optional logging
        log_info: If True, log padding operations (used during encoding only)
        
    Returns:
        Prepared video in TCHW format
    """
    # Extract frames (view/slice, not copy)
    video = images[start_idx:end_idx]
    
    # Apply uniform padding if needed
    if uniform_padding > 0:
        if log_info and debug:
            current_frames = end_idx - start_idx
            debug.log(f"Sequence of {current_frames} frames", category="video", force=True, indent_level=1)
            debug.log(f"Padding batch: {uniform_padding} frame{'s' if uniform_padding != 1 else ''} added ({current_frames} → {current_frames + uniform_padding}) for uniform batches", 
                     category="video", force=True, indent_level=1)
        video = pad_video_temporal(video, count=uniform_padding, temporal_dim=0, prepend=False, debug=None)
    
    # Permute to TCHW format
    video = video.permute(0, 3, 1, 2)
    
    return video


def _apply_4n1_padding(video: torch.Tensor) -> torch.Tensor:
    """
    Apply 4n+1 temporal padding constraint required by VAE.
    
    Args:
        video: Video tensor in TCHW format
        
    Returns:
        Padded video in TCHW format
    """
    t = video.size(0)
    if t % 4 != 1:
        video = optimized_single_video_rearrange(video)  # TCHW -> CTHW
        video = pad_video_temporal(video, temporal_dim=1, prepend=False, debug=None)
        video = optimized_single_video_rearrange(video)  # CTHW -> TCHW
    return video


def _reconstruct_and_transform_batch(
    ctx: Dict[str, Any],
    batch_idx: int,
    debug: Optional['Debug'] = None
) -> torch.Tensor:
    """
    Reconstruct and transform a video batch for color correction (Phase 4).
    
    Args:
        ctx: Context with input_images, batch_metadata, video_transform
        batch_idx: Index of batch to reconstruct
        debug: Debug instance for logging
        
    Returns:
        Transformed video in CTHW format, ready for color correction
    """
    start_idx, end_idx, uniform_padding = ctx['batch_metadata'][batch_idx]
    
    # Prepare video batch
    video = _prepare_video_batch(
        images=ctx['input_images'],
        start_idx=start_idx,
        end_idx=end_idx,
        uniform_padding=uniform_padding,
        debug=None,
        log_info=False
    )
    
    # Apply 4n+1 padding using shared helper
    video = _apply_4n1_padding(video)
    
    # Extract RGB and transform
    if ctx.get('is_rgba', False):
        rgb_video = video[:, :3, :, :]
    else:
        rgb_video = video
    
    transformed_video = ctx['video_transform'](rgb_video)
    
    del video
    
    return transformed_video


def encode_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    images: torch.Tensor,
    debug: 'Debug',
    batch_size: int = 5,
    uniform_batch_size: bool = False,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    temporal_overlap: int = 0,
    resolution: int = 1080,
    max_resolution: int = 0,
    input_noise_scale: float = 0.0,
    color_correction: str = "wavelet"
) -> Dict[str, Any]:
    """
    Phase 1: VAE Encoding for all batches
    
    Encodes video frames to latents in batches, handling temporal overlap and 
    memory optimization. Creates context automatically if not provided.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Generation context from setup_generation_context (required)
        images: Input frames tensor [T, H, W, C] range [0,1] (required) 
        debug: Debug instance for logging (required)
        batch_size: Frames per batch (4n+1 format: 1, 5, 9, 13...)
        uniform_batch_size: Pad final batch to match batch_size for uniform batches
        seed: Random seed for deterministic VAE sampling (default: 42)
        progress_callback: Optional callback(current, total, frames, phase_name)
        temporal_overlap: Overlapping frames between batches for continuity
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        input_noise_scale: Scale for input noise (0.0-1.0). Adds noise to input images
                          before VAE encoding to reduce artifacts at high resolutions.
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
                         Determines if transformed videos need to be stored for later use.
        
    Returns:
        dict: Context containing:
            - batch_metadata: Lightweight indices for on-demand transform reconstruction
            - all_latents: List of encoded latents ready for upscaling
            - Other state for subsequent phases
            
    Raises:
        ValueError: If required inputs are missing or invalid
        RuntimeError: If encoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to encode_all_batches")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 1: VAE encoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase1_encoding")

    # Context must be provided
    if ctx is None:
        raise ValueError("Generation context must be provided to encode_all_batches")
    
    # Validate and store inputs
    if images is None:
        raise ValueError("Images to encode must be provided")
    else:
        # MPS: keep on device to avoid sync overhead in Phase 4 color correction
        if ctx['vae_device'].type == 'mps' and images.device.type != 'mps':
            ctx['input_images'] = images.to(ctx['vae_device'])
        else:
            ctx['input_images'] = images
    
    # Get total frame count from context (set in video_upscaler before encoding)
    total_frames = ctx.get('total_frames', len(images))
    
    # Set it if not already set (for standalone/CLI usage)
    if 'total_frames' not in ctx:
        ctx['total_frames'] = total_frames
    
    if total_frames == 0:
        raise ValueError("No frames to process")
    
    # Setup video transformation pipeline and compute dimensions if not already done
    if 'true_target_dims' not in ctx:
        sample_frame = images[0].permute(2, 0, 1).unsqueeze(0)
        setup_video_transform(ctx, resolution, max_resolution, debug, sample_frame)
        del sample_frame
    else:
        setup_video_transform(ctx, resolution, max_resolution, debug)
    
    # Detect if input is RGBA (4 channels)
    ctx['is_rgba'] = images[0].shape[-1] == 4
    
    # Display batch optimization tip if applicable
    if total_frames > 0:
        batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
        if batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames:
            debug.log("", category="none", force=True)
            debug.log(f"Tip: For {total_frames} frames, batch_size={batch_params['best_batch']} matches video length optimally", category="tip", force=True)
            debug.log(f"Matching batch_size to shot length improves temporal coherence", category="tip", force=True, indent_level=1)
            debug.log("", category="none", force=True)
    
    # Calculate batching parameters
    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
        debug.log(f"temporal_overlap >= batch_size, resetting to 0", level="WARNING", category="setup", force=True)
    
    # Store actual temporal overlap used (may differ from parameter if reset)
    ctx['actual_temporal_overlap'] = temporal_overlap
    
    # Calculate number of batches
    num_encode_batches = 0
    for idx in range(0, total_frames, step):
        end_idx = min(idx + batch_size, total_frames)
        if idx > 0 and end_idx - idx <= temporal_overlap:
            break
        num_encode_batches += 1
    
    # Pre-allocate lists for memory efficiency
    ctx['all_latents'] = [None] * num_encode_batches
    ctx['all_ori_lengths'] = [None] * num_encode_batches
    if color_correction != "none":
        ctx['batch_metadata'] = [None] * num_encode_batches
    
    encode_idx = 0
    
    try:
        # Materialize VAE if still on meta device
        if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)
        else:
            # Model already materialized (cached) - apply any pending configs if needed
            if getattr(runner, '_vae_config_needs_application', False):
                debug.log("Applying updated VAE configuration", category="vae", force=True)
                apply_model_specific_config(runner.vae, runner, runner.config, False, debug)
        
        # Initialize precision after VAE is materialized with actual weights
        ensure_precision_initialized(ctx, runner, debug)

        # Cache VAE now that it's fully configured and ready for inference
        if ctx['cache_context']['vae_cache'] and not ctx['cache_context']['cached_vae']:
            runner.vae._model_name = ctx['cache_context']['vae_model']
            ctx['cache_context']['global_cache'].set_vae(
                {'node_id': ctx['cache_context']['vae_id'], 'cache_model': True}, 
                runner.vae, ctx['cache_context']['vae_model'], debug
            )
            ctx['cache_context']['vae_newly_cached'] = True
            
            # If both models now cached, cache runner template
            dit_is_cached = ctx['cache_context']['cached_dit'] or ctx['cache_context']['dit_newly_cached']
            if dit_is_cached:
                ctx['cache_context']['global_cache'].set_runner(
                    ctx['cache_context']['dit_id'], ctx['cache_context']['vae_id'], 
                    runner, debug
                )
        
        # Set deterministic seed for VAE encoding (separate from diffusion noise)
        # Uses seed + 1,000,000 to avoid collision with upscaling batch seeds
        # This ensures VAE sampling is deterministic while maintaining quality
        seed_vae = seed + 1000000
        set_seed(seed_vae)
        debug.log(f"Using seed: {seed_vae} (VAE uses seed+1000000 for deterministic sampling)", category="vae")
        
        # Move VAE to GPU for encoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=ctx['vae_device'], 
                          model_name="VAE", debug=debug, runner=runner)
        
        debug.log_memory_state("After VAE loading for encoding", detailed_tensors=False)

        # Initialize tile_boundaries for encoding debug
        if runner.tile_debug == "encode" and runner.encode_tiled:
            debug.encode_tile_boundaries = []
            debug.log("Tile debug enabled: encode tile boundaries will be visualized", category="vae", force=True)
            debug.log("Remember to disable --tile_debug in production to remove overlay visualization", category="tip", indent_level=1, force=True)
        
        # Process encoding
        for batch_idx in range(0, total_frames, step):
            check_interrupt(ctx)
            
            # Calculate indices with temporal overlap
            if batch_idx == 0:
                start_idx = 0
                end_idx = min(batch_size, total_frames)
            else:
                start_idx = batch_idx
                end_idx = min(start_idx + batch_size, total_frames)
                if end_idx - start_idx <= temporal_overlap:
                    break
            
            current_frames = end_idx - start_idx
            is_uniform_padding = uniform_batch_size and current_frames < batch_size
            
            debug.log(f"Encoding batch {encode_idx+1}/{num_encode_batches}", category="vae", force=True)
            debug.start_timer(f"encode_batch_{encode_idx+1}")
            
            # Save original length before any padding
            ori_length = current_frames
            
            # Prepare video batch with uniform padding
            video = _prepare_video_batch(
                images=images,
                start_idx=start_idx,
                end_idx=end_idx,
                uniform_padding=batch_size - current_frames if is_uniform_padding else 0,
                debug=debug,
                log_info=True
            )
            if is_uniform_padding:
                current_frames = batch_size
            
            video = manage_tensor(
                tensor=video,
                target_device=ctx['vae_device'],
                tensor_name=f"video_batch_{encode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE encoding",
                indent_level=1
            )

            # Check temporal dimension for 4n+1 padding
            t = video.size(0)
            
            # Log sequence size if not already logged (for non-uniform batches)
            if not is_uniform_padding:
                debug.log(f"Sequence of {t} frames", category="video", force=True, indent_level=1)

            # Apply 4n+1 padding using shared helper
            if t % 4 != 1:
                target = ((t-1)//4+1)*4+1
                padding_frames = target - t
                debug.log(f"Padding batch: {padding_frames} frame{'s' if padding_frames != 1 else ''} added ({t} → {target}) to meet 4n+1 constraint", 
                         category="video", force=True, indent_level=1)
                # Apply 4n+1 padding to match exact frame count from encoding
                video = _apply_4n1_padding(video)

            # Apply transformations (matches reconstruction logic)
            if ctx.get('is_rgba', False):
                debug.log(f"Extracted Alpha channel for edge-guided upscaling", category="alpha", indent_level=1)
                rgb_video = video[:, :3, :, :]
            else:
                rgb_video = video

            transformed_video = ctx['video_transform'](rgb_video)

            # Apply input noise if requested (to reduce artifacts at high resolutions)
            if input_noise_scale > 0:
                debug.log(f"Applying input noise (scale: {input_noise_scale:.2f})", category="video", indent_level=1)
                
                # Generate noise matching the video shape
                noise = torch.randn_like(transformed_video)
                
                # Subtle noise amplitude
                noise = noise * 0.05
                
                # Linear blend factor: 0 at scale=0, 0.5 at scale=1
                blend_factor = input_noise_scale * 0.5
                
                # Apply blend
                transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor
                
                del noise

            # Store original length for proper trimming later
            ctx['all_ori_lengths'][encode_idx] = ori_length

            # Store batch frame indices for on-demand reconstruction
            if color_correction != "none":
                ctx['batch_metadata'][encode_idx] = (start_idx, end_idx, batch_size - ori_length if is_uniform_padding else 0)
            
            # Extract and store Alpha and RGB from padded original video (before encoding)
            if ctx.get('is_rgba', False):
                if 'all_alpha_channels' not in ctx:
                    ctx['all_alpha_channels'] = [None] * num_encode_batches
                if 'all_input_rgb' not in ctx:
                    ctx['all_input_rgb'] = [None] * num_encode_batches
                
                # Extract from padded RGBA video (format: T, 4, H, W)
                alpha_channel = video[:, 3:4, :, :]
                rgb_video_original = video[:, :3, :, :]
                
                # Store on tensor_offload_device to save VRAM (or keep on device if none)
                if ctx['tensor_offload_device'] is not None:
                    ctx['all_alpha_channels'][encode_idx] = manage_tensor(
                        tensor=alpha_channel,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"alpha_channel_{encode_idx+1}",
                        debug=debug,
                        reason="storing Alpha channel for upscaling",
                        indent_level=1
                    )
                    ctx['all_input_rgb'][encode_idx] = manage_tensor(
                        tensor=rgb_video_original,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"rgb_original_{encode_idx+1}",
                        debug=debug,
                        reason="storing RGB edge guidance for Alpha upscaling",
                        indent_level=1
                    )
                else:
                    ctx['all_alpha_channels'][encode_idx] = alpha_channel
                    ctx['all_input_rgb'][encode_idx] = rgb_video_original
                
                del alpha_channel, rgb_video_original

            del video

            # Move to VAE device with correct dtype for encoding
            transformed_video = manage_tensor(
                tensor=transformed_video,
                target_device=ctx['vae_device'],
                tensor_name=f"transformed_video_{encode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE encoding",
                indent_level=1
            )
            
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])

            # Don't store transformed_video - will reconstruct on-demand in Phase 4
            del transformed_video, rgb_video
            
            # Convert from VAE dtype to compute dtype and offload to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None and (cond_latents[0].is_cuda or cond_latents[0].is_mps):
                ctx['all_latents'][encode_idx] = manage_tensor(
                    tensor=cond_latents[0],
                    target_device=ctx['tensor_offload_device'],
                    tensor_name=f"latent_{encode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="storing encoded latents for upscaling",
                    indent_level=1
                )
            else:
                # Stay on current device but convert to compute dtype
                ctx['all_latents'][encode_idx] = manage_tensor(
                    tensor=cond_latents[0],
                    target_device=cond_latents[0].device,
                    tensor_name=f"latent_{encode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="VAE dtype → compute dtype",
                    indent_level=1
                )
            
            del cond_latents
            
            debug.end_timer(f"encode_batch_{encode_idx+1}", f"Encoded batch {encode_idx+1}")
            
            if progress_callback:
                progress_callback(encode_idx+1, num_encode_batches, 
                                current_frames, "Phase 1: Encoding")
            
            encode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 1 (Encoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Offload VAE to configured offload device if specified
        if ctx['vae_offload_device'] is not None:
            manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'], 
                                model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    debug.end_timer("phase1_encoding", "Phase 1: VAE encoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 1 (VAE encoding)", show_tensors=False)
    
    return ctx


def upscale_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    seed: int = 42,
    latent_noise_scale: float = 0.0,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 2: DiT Upscaling for all encoded batches.
    
    Processes all encoded latents through the diffusion model for upscaling.
    Requires context from encode_all_batches with encoded latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from encode_all_batches containing latents (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        seed: Random seed for reproducible generation
        latent_noise_scale: Noise scale for latent space augmentation (0.0-1.0).
                           Adds noise during diffusion conditioning. Can soften details
                           but may help with certain artifacts. 0.0 = no noise (crisp),
                           1.0 = maximum noise (softer)
        cache_model: If True, keep DiT model for reuse instead of deleting it
        
    Returns:
        dict: Updated context containing:
            - all_upscaled_latents: List of upscaled latents ready for decoding
            - Preserved state from encoding phase
            
    Raises:
        ValueError: If context is missing or has no encoded latents
        RuntimeError: If upscaling fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to upscale_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for upscale_all_batches. Run encode_all_batches first.")
        
    # Validate we have encoded latents
    if 'all_latents' not in ctx or not ctx['all_latents']:
        raise ValueError("No encoded latents found. Run encode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 2: DiT upscaling ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscaling")
    
    # Load text embeddings if not already loaded
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)
        debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Configure diffusion parameters
    # Force cfg_scale = 1.0 for one-step distilled models (CFG is incompatible with distillation)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=ctx['dit_device'], dtype=ctx['compute_dtype'])

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_latents'] if l is not None])

    # Safety check for empty latents
    if num_valid_latents == 0:
        debug.log("No valid latents to upscale", level="WARNING", category="dit", force=True)
        ctx['all_upscaled_latents'] = []
        return ctx
    
    # Pre-allocate list for upscaled latents
    ctx['all_upscaled_latents'] = [None] * num_valid_latents
    
    upscale_idx = 0
    
    try:
        # Materialize DiT if still on meta device
        if runner.dit and next(runner.dit.parameters()).device.type == 'meta':
            materialize_model(runner, "dit", ctx['dit_device'], runner.config, debug)
        else:
            # Model already materialized (cached) - apply any pending configs if needed
            if getattr(runner, '_dit_config_needs_application', False):
                debug.log("Applying updated DiT configuration", category="dit", force=True)
                apply_model_specific_config(runner.dit, runner, runner.config, True, debug)
    
        # Initialize precision after DiT is materialized with actual weights
        ensure_precision_initialized(ctx, runner, debug)

        # Cache DiT now that it's fully configured and ready for inference
        if ctx['cache_context']['dit_cache'] and not ctx['cache_context']['cached_dit']:
            runner.dit._model_name = ctx['cache_context']['dit_model']
            ctx['cache_context']['global_cache'].set_dit(
                {'node_id': ctx['cache_context']['dit_id'], 'cache_model': True}, 
                runner.dit, ctx['cache_context']['dit_model'], debug
            )
            ctx['cache_context']['dit_newly_cached'] = True
            
            # If both models now cached, cache runner template
            vae_is_cached = ctx['cache_context']['cached_vae'] or ctx['cache_context']['vae_newly_cached']
            if vae_is_cached:
                ctx['cache_context']['global_cache'].set_runner(
                    ctx['cache_context']['dit_id'], ctx['cache_context']['vae_id'], 
                    runner, debug
                )
        
        # Move DiT to GPU for upscaling (no-op if already there)
        manage_model_device(model=runner.dit, target_device=ctx['dit_device'], 
                            model_name="DiT", debug=debug, runner=runner)

        debug.log_memory_state("After DiT loading for upscaling", detailed_tensors=False)

        for batch_idx, latent in enumerate(ctx['all_latents']):
            if latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Upscaling batch {upscale_idx+1}/{num_valid_latents}", category="generation", force=True)
            # Reset seed for each batch to ensure identical RNG state
            # This ensures identical inputs produce identical outputs regardless of batch position
            set_seed(seed)
            debug.log(f"Using seed: {seed} for deterministic generation", category="dit")

            debug.start_timer(f"upscale_batch_{upscale_idx+1}")
            
            # Move to DiT device with correct dtype for upscaling (no-op if already there)
            latent = manage_tensor(
                tensor=latent,
                target_device=ctx['dit_device'],
                tensor_name=f"latent_{upscale_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="DiT upscaling",
                indent_level=1
            )

            # Generate noise (randn_like automatically uses latent's device)
            base_noise = torch.randn_like(latent, dtype=ctx['compute_dtype'])
            
            noises = [base_noise]
            aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
            
            # Log latent noise application if enabled
            if latent_noise_scale > 0:
                debug.log(f"Applying latent noise (scale: {latent_noise_scale:.3f})", category="generation")
            
            def _add_noise(x, aug_noise):
                if latent_noise_scale == 0.0:
                    return x
                t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
                shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
                t = runner.timestep_transform(t, shape)
                x = runner.schedule.forward(x, aug_noise, t)
                del t, shape
                return x
            
            # Generate condition
            condition = runner.get_condition(
                noises[0],
                task="sr",
                latent_blur=_add_noise(latent, aug_noises[0]),
            )
            conditions = [condition]
            
            # Detect DiT model dtype (handle CompatibleDiT wrapper)
            dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit
            try:
                dit_dtype = next(dit_model.parameters()).dtype
            except StopIteration:
                dit_dtype = ctx['compute_dtype']  # Fallback for meta device or empty model
            
            # Use autocast if DiT dtype differs from compute dtype
            # Skip autocast on MPS (CompatibleDiT already handles dtype conversion)
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                if dit_dtype != ctx['compute_dtype'] and ctx['dit_device'].type != 'mps':
                    with torch.autocast(ctx['dit_device'].type, ctx['compute_dtype'], enabled=True):
                        upscaled_latents = runner.inference(
                            noises=noises,
                            conditions=conditions,
                            **ctx['text_embeds'],
                        )
                else:
                    upscaled_latents = runner.inference(
                        noises=noises,
                        conditions=conditions,
                        **ctx['text_embeds'],
                    )
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")
            
            # Offload upscaled latents to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None and (upscaled_latents[0].is_cuda or upscaled_latents[0].is_mps):
                ctx['all_upscaled_latents'][upscale_idx] = manage_tensor(
                    tensor=upscaled_latents[0],
                    target_device=ctx['tensor_offload_device'],
                    tensor_name=f"upscaled_latent_{upscale_idx+1}",
                    debug=debug,
                    reason="storing upscaled latents for decoding",
                    indent_level=1
                )
            else:
                ctx['all_upscaled_latents'][upscale_idx] = upscaled_latents[0]
            
            # Free original latent - release tensor memory first
            release_tensor_memory(ctx['all_latents'][batch_idx])
            ctx['all_latents'][batch_idx] = None
            
            del noises, aug_noises, latent, conditions, condition, base_noise, upscaled_latents
            
            debug.end_timer(f"upscale_batch_{upscale_idx+1}", f"Upscaled batch {upscale_idx+1}")
            
            if progress_callback:
                progress_callback(upscale_idx+1, num_valid_latents,
                                1, "Phase 2: Upscaling")
            
            upscale_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 2 (Upscaling): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Log BlockSwap summary if it was used
        if hasattr(runner, '_blockswap_active') and runner._blockswap_active:
            swap_summary = debug.get_swap_summary()
            if swap_summary and swap_summary.get('total_swaps', 0) > 0:
                total_time = swap_summary.get('block_total_ms', 0) + swap_summary.get('io_total_ms', 0)
                debug.log("BlockSwap Summary", category="blockswap")
                debug.log(f"BlockSwap overhead: {total_time:.2f}ms", category="blockswap", indent_level=1)
                debug.log(f"Total swaps: {swap_summary['total_swaps']}", category="blockswap", indent_level=1)
                
                # Show block swap details
                if 'block_swaps' in swap_summary and swap_summary['block_swaps'] > 0:
                    avg_ms = swap_summary.get('block_avg_ms', 0)
                    total_ms = swap_summary.get('block_total_ms', 0)
                    min_ms = swap_summary.get('block_min_ms', 0)
                    max_ms = swap_summary.get('block_max_ms', 0)
                    
                    debug.log(f"Block swaps: {swap_summary['block_swaps']} "
                            f"(avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms, max: {max_ms:.2f}ms, total: {total_ms:.2f}ms)", 
                            category="blockswap", indent_level=1)
                    
                    # Show most frequently swapped block
                    if 'most_swapped_block' in swap_summary:
                        debug.log(f"Most swapped: Block {swap_summary['most_swapped_block']} "
                                f"({swap_summary['most_swapped_count']} times)", category="blockswap", indent_level=1)
                
                # Show I/O swap details if present
                if 'io_swaps' in swap_summary and swap_summary['io_swaps'] > 0:
                    debug.log(f"I/O swaps: {swap_summary['io_swaps']} "
                            f"(avg: {swap_summary.get('io_avg_ms', 0):.2f}ms, total: {swap_summary.get('io_total_ms', 0):.2f}ms)", 
                            category="blockswap", indent_level=1)

        # Cleanup DiT as it's no longer needed after upscaling
        cleanup_dit(runner=runner, debug=debug, cache_model=cache_model)
        
        # Cleanup text embeddings as they're no longer needed after upscaling
        cleanup_text_embeddings(ctx, debug)
    
    debug.end_timer("phase2_upscaling", "Phase 2: DiT upscaling complete", show_breakdown=True)
    debug.log_memory_state("After phase 2 (DiT upscaling)", show_tensors=False)
    
    return ctx


def decode_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 3: VAE Decoding.
    
    Decodes all upscaled latents back to pixel space and writes directly to
    pre-allocated final_video tensor. This avoids memory duplication by not
    storing intermediate batch_samples.
    
    Requires context from upscale_all_batches with upscaled latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from upscale_all_batches containing upscaled latents (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cache_model: If True, keep VAE model for reuse instead of deleting it
        
    Returns:
        dict: Updated context containing:
            - final_video: Pre-allocated tensor with decoded samples (unnormalized, in [-1,1])
            - decode_batch_info: List of (start_idx, end_idx, ori_length) for Phase 4 processing
            - VAE cleanup completed
            
    Raises:
        ValueError: If context is missing or has no upscaled latents
        RuntimeError: If decoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to decode_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for decode_all_batches. Run upscale_all_batches first.")
    
    # Validate we have upscaled latents
    if 'all_upscaled_latents' not in ctx or not ctx['all_upscaled_latents']:
        raise ValueError("No upscaled latents found. Run upscale_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 3: VAE decoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase3_decoding")

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_upscaled_latents'] if l is not None])
    num_batches = len([l for l in ctx['all_ori_lengths'] if l is not None])
    
    # Get output dimensions from context (set during Phase 1)
    if 'true_target_dims' not in ctx:
        raise ValueError("true_target_dims not found in context. Run encode_all_batches first.")
    true_h, true_w = ctx['true_target_dims']
    total_frames = ctx.get('total_frames', 0)
    C = 4 if ctx.get('is_rgba', False) else 3
    
    # Pre-allocate final_video at the START of decode phase (before any batch processing)
    # This ensures we only need memory for final_video + 1 batch, not final_video + all batch_samples
    # MPS: keep on device (unified memory, no benefit to CPU offload)
    if ctx['tensor_offload_device'] is not None:
        target_device = ctx['tensor_offload_device']
    elif ctx['vae_device'].type == 'mps':
        target_device = ctx['vae_device']
    else:
        target_device = 'cpu'
    channels_str = "RGBA" if C == 4 else "RGB"
    required_gb = (total_frames * true_h * true_w * C * 2) / (1024**3)
    debug.log(f"Pre-allocating output tensor: {total_frames} frames, {true_w}x{true_h}px, {channels_str} ({required_gb:.2f}GB)", 
              category="setup", force=True)
    
    ctx['final_video'] = torch.empty((total_frames, true_h, true_w, C), dtype=ctx['compute_dtype'], device=target_device)
    
    # Track batch write positions for Phase 4 processing
    # Each entry: (write_start, write_end, batch_idx, ori_length)
    ctx['decode_batch_info'] = []
    
    # Get temporal overlap from context (set during Phase 1)
    temporal_overlap = ctx.get('actual_temporal_overlap', 0)
    
    # Track padding removed for final summary
    total_padding_removed = 0
    
    current_write_idx = 0
    decode_idx = 0
    
    try:
        # VAE should already be materialized from encoding phase
        if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)

        # Precision should already be initialized from encoding phase
        ensure_precision_initialized(ctx, runner, debug)

        # Move VAE to GPU for decoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=ctx['vae_device'], 
                          model_name="VAE", debug=debug, runner=runner)
        
        debug.log_memory_state("After VAE loading for decoding", detailed_tensors=False)

        # Initialize tile_boundaries for decoding debug
        if runner.tile_debug == "decode" and runner.decode_tiled:
            debug.decode_tile_boundaries = []
            debug.log("Tile debug enabled: decode tile boundaries will be visualized", category="vae", force=True)
            debug.log("Remember to disable --tile_debug in production to remove overlay visualization", category="tip", indent_level=1, force=True)
        
        # Process decoding
        for batch_idx, upscaled_latent in enumerate(ctx['all_upscaled_latents']):
            if upscaled_latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Decoding batch {decode_idx+1}/{num_valid_latents}", category="vae", force=True)
            debug.start_timer(f"decode_batch_{decode_idx+1}")
            
            # Move to VAE device with correct dtype for decoding (no-op if already there)
            upscaled_latent = manage_tensor(
                tensor=upscaled_latent,
                target_device=ctx['vae_device'],
                tensor_name=f"upscaled_latent_{decode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE decoding",
                indent_level=1
            )
            
            # Decode latent
            debug.start_timer("vae_decode")
            samples = runner.vae_decode([upscaled_latent])
            debug.end_timer("vae_decode", "VAE decode")
            
            # Process samples - get the single decoded sample
            debug.start_timer("optimized_video_rearrange")
            samples = optimized_video_rearrange(samples)
            debug.end_timer("optimized_video_rearrange", "Video rearrange")
            
            # Get the decoded sample (always single-element list)
            sample = samples[0]
            del samples
            
            # Get original length for this batch (before any padding was added)
            ori_length = ctx['all_ori_lengths'][decode_idx] if decode_idx < len(ctx['all_ori_lengths']) else sample.shape[0]
            
            # Trim temporal padding: sample is in [T, C, H, W] format after rearrange
            if ori_length < sample.shape[0]:
                padding_removed = sample.shape[0] - ori_length
                debug.log(f"Trimming temporal padding: {padding_removed} frames removed ({sample.shape[0]} → {ori_length})", 
                         category="video", indent_level=1)
                sample = sample[:ori_length]
                total_padding_removed += padding_removed
            
            # Trim spatial padding to true target dimensions
            current_h, current_w = sample.shape[-2:]
            if current_h != true_h or current_w != true_w:
                debug.log(f"Trimming spatial padding: {current_w}x{current_h} → {true_w}x{true_h}", 
                         category="video", indent_level=1)
                sample = sample[:, :, :true_h, :true_w]
            
            # Convert to output format: [T, C, H, W] → [T, H, W, C]
            # Note: We keep values in [-1, 1] range - normalization happens in Phase 4
            sample = optimized_sample_to_image_format(sample)  # T, C, H, W → T, H, W, C
            
            # Calculate write position with temporal overlap handling
            batch_frames = sample.shape[0]
            if decode_idx == 0 or temporal_overlap == 0:
                # First batch or no overlap: write all frames
                write_start = current_write_idx
                write_end = current_write_idx + batch_frames
            else:
                # Subsequent batches with overlap: blend overlapping region
                if temporal_overlap < batch_frames and current_write_idx >= temporal_overlap:
                    # Blend overlapping region in-place on final_video
                    prev_tail = ctx['final_video'][current_write_idx - temporal_overlap:current_write_idx]
                    cur_head = sample[:temporal_overlap]
                    
                    # Move to same device for blending if needed
                    if prev_tail.device != cur_head.device:
                        cur_head = cur_head.to(prev_tail.device)
                    
                    blended = blend_overlapping_frames(prev_tail, cur_head, temporal_overlap)
                    ctx['final_video'][current_write_idx - temporal_overlap:current_write_idx] = blended
                    
                    debug.log(f"Blended {temporal_overlap} overlapping frames at positions {current_write_idx - temporal_overlap}-{current_write_idx}", 
                             category="video", indent_level=1)
                    
                    # Write only non-overlapping part
                    sample = sample[temporal_overlap:]
                    batch_frames = sample.shape[0]
                    del prev_tail, cur_head, blended
                
                write_start = current_write_idx
                write_end = current_write_idx + batch_frames
            
            # Move sample to target device and write directly to final_video
            sample = manage_tensor(
                tensor=sample,
                target_device=target_device,
                tensor_name=f"sample_{decode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="writing to final_video",
                indent_level=1
            )
            
            # Write to final_video - for RGBA, write only RGB channels (VAE outputs 3 channels)
            if ctx.get('is_rgba', False):
                ctx['final_video'][write_start:write_end, :, :, :3] = sample
            else:
                ctx['final_video'][write_start:write_end] = sample
            
            # Store batch info for Phase 4 processing
            ctx['decode_batch_info'].append((write_start, write_end, decode_idx, ori_length))
            current_write_idx = write_end
            
            debug.log(f"Wrote {batch_frames} frames to positions {write_start}-{write_end}", 
                     category="video", indent_level=1)
            
            # Free memory immediately - no batch_samples storage
            release_tensor_memory(ctx['all_upscaled_latents'][batch_idx])
            ctx['all_upscaled_latents'][batch_idx] = None
            del upscaled_latent, sample
            
            debug.end_timer(f"decode_batch_{decode_idx+1}", f"Decoded batch {decode_idx+1}")
            
            if progress_callback:
                progress_callback(decode_idx+1, num_valid_latents,
                                1, "Phase 3: Decoding")
            
            decode_idx += 1
        
        # Store padding stats for Phase 4 final summary
        ctx['total_padding_removed'] = total_padding_removed
            
    except Exception as e:
        debug.log(f"Error in Phase 3 (Decoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Cleanup VAE as it's no longer needed
        cleanup_vae(runner=runner, debug=debug, cache_model=cache_model)
        
        # Clean up upscaled latents storage
        if 'all_upscaled_latents' in ctx:
            release_tensor_collection(ctx['all_upscaled_latents'])
            del ctx['all_upscaled_latents']
        
    debug.end_timer("phase3_decoding", "Phase 3: VAE decoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 3 (VAE decoding)", show_tensors=False)
    
    return ctx


def postprocess_all_batches(
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    color_correction: str = "wavelet",
    prepend_frames: int = 0,
    temporal_overlap: int = 0,
    batch_size: int = 5
) -> Dict[str, Any]:
    """
    Phase 4: Post-processing and Final Assembly.
    
    Processes final_video slices in-place: applies alpha upscaling, color correction,
    and normalization. Reads from and writes back to the same final_video tensor
    to avoid memory duplication.
    
    Args:
        ctx: Context from decode_all_batches containing final_video (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
        prepend_frames: Number of prepended frames to remove from final output (default: 0)
        temporal_overlap: Number of overlapping frames between batches for blending (default: 0)
        batch_size: Frames per batch used during encoding for overlap calculation (default: 5)
        
    Returns:
        dict: Updated context containing:
            - final_video: Assembled video tensor [T, H, W, C] range [0,1] with overlap blended and prepended frames removed
            - All intermediate storage cleared for memory efficiency
            
    Raises:
        ValueError: If context is missing or has no final_video
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to postprocess_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for postprocess_all_batches. Run decode_all_batches first.")
    
    # Validate we have final_video (pre-allocated in decode_all_batches)
    if 'final_video' not in ctx or ctx['final_video'] is None:
        raise ValueError("final_video not found. Run decode_all_batches first.")
    
    # Validate we have batch info for processing
    if 'decode_batch_info' not in ctx or not ctx['decode_batch_info']:
        raise ValueError("decode_batch_info not found. Run decode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 4: Post-processing ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase4_postprocessing")
    
    # Total_frames represents the original input frame count (set in Phase 1)
    total_frames = ctx.get('total_frames', 0)
    
    # Early exit if no frames to process
    if total_frames == 0:
        ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=ctx['compute_dtype'])
        debug.log("No frames to process", level="WARNING", category="generation", force=True)
        return ctx
    
    # Get batch info from decode phase
    batch_info_list = ctx['decode_batch_info']
    num_valid_samples = len(batch_info_list)
    
    # Calculate total post-processing work units
    # For RGBA: each batch needs 2 steps (alpha processing + color correction/assembly)
    # For RGB: each batch needs 1 step (color correction/assembly only)
    has_alpha_processing = (ctx.get('is_rgba', False) and 
                           'all_alpha_channels' in ctx and 
                           'all_input_rgb' in ctx and
                           isinstance(ctx.get('all_alpha_channels'), list))
    
    if has_alpha_processing:
        total_postprocessing_steps = num_valid_samples * 2  # Alpha + main processing
    else:
        total_postprocessing_steps = num_valid_samples  # Main processing only
    
    current_postprocessing_step = 0
    
    # Get padding stats from Phase 3
    total_padding_removed = ctx.get('total_padding_removed', 0)
    
    # Alpha processing - handle RGBA inputs with edge-guided upscaling
    # Process alpha on final_video slices in-place
    if has_alpha_processing:
        debug.log("Processing Alpha channel with edge-guided upscaling...", category="alpha")
        
        # Validate alpha channel data exists
        if not isinstance(ctx.get('all_alpha_channels'), list) or not isinstance(ctx.get('all_input_rgb'), list):
            debug.log("WARNING: Alpha channel data malformed, skipping alpha processing", 
                     level="WARNING", category="alpha", force=True)
        else:
            for write_start, write_end, batch_idx, ori_length in batch_info_list:
                # Bounds checking for alpha channel lists
                if batch_idx >= len(ctx['all_alpha_channels']) or ctx['all_alpha_channels'][batch_idx] is None:
                    continue
                    
                # Validate alpha channel tensor integrity
                if not isinstance(ctx['all_alpha_channels'][batch_idx], torch.Tensor):
                    debug.log(f"WARNING: Alpha channel {batch_idx} is not a tensor, skipping", 
                             level="WARNING", category="alpha", force=True)
                    continue
                
                debug.log(f"Processing Alpha batch {batch_idx+1}/{num_valid_samples}", category="alpha", force=True)
                debug.start_timer(f"alpha_batch_{batch_idx+1}")

                # Get RGB slice from final_video for alpha processing
                # final_video is [T, H, W, C], process_alpha_for_batch expects list of [T, C, H, W]
                rgb_slice = ctx['final_video'][write_start:write_end, :, :, :3]  # Only RGB
                rgb_tchw = rgb_slice.permute(0, 3, 1, 2)  # [T, H, W, 3] → [T, 3, H, W]
                
                # Process Alpha and merge with RGB
                processed_samples = process_alpha_for_batch(
                    rgb_samples=[rgb_tchw],
                    alpha_original=ctx['all_alpha_channels'][batch_idx],
                    rgb_original=ctx['all_input_rgb'][batch_idx],
                    device=ctx['vae_device'],
                    compute_dtype=ctx['compute_dtype'],
                    debug=debug
                )
                
                # processed_samples[0] is [T, 4, H, W] (RGBA)
                # Extract only the alpha channel and write to final_video's alpha slot
                processed_rgba = processed_samples[0]  # [T, 4, H, W]
                alpha_channel = processed_rgba[:, 3:4, :, :]  # [T, 1, H, W]
                alpha_thwc = alpha_channel.permute(0, 2, 3, 1)  # [T, 1, H, W] → [T, H, W, 1]
                
                alpha_thwc = manage_tensor(
                    tensor=alpha_thwc,
                    target_device=ctx['final_video'].device,
                    tensor_name=f"alpha_channel_{batch_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="writing alpha channel to final_video",
                    indent_level=1
                )
                
                # Write only the alpha channel to the 4th channel slot
                ctx['final_video'][write_start:write_end, :, :, 3:4] = alpha_thwc
                
                del rgb_slice, rgb_tchw, processed_samples, processed_rgba, alpha_channel, alpha_thwc
                
                # Free memory immediately
                release_tensor_memory(ctx['all_alpha_channels'][batch_idx])
                ctx['all_alpha_channels'][batch_idx] = None

                release_tensor_memory(ctx['all_input_rgb'][batch_idx])
                ctx['all_input_rgb'][batch_idx] = None
            
                debug.end_timer(f"alpha_batch_{batch_idx+1}", f"Alpha batch {batch_idx+1}")
                
                # Update progress for alpha processing step
                current_postprocessing_step += 1
                if progress_callback:
                    progress_callback(current_postprocessing_step, total_postprocessing_steps,
                                    1, "Phase 4: Post-processing")

        debug.log("Alpha processing complete for all batches", category="alpha")
    
    try:
        # Process each batch slice in final_video in-place
        for info_idx, (write_start, write_end, batch_idx, ori_length) in enumerate(batch_info_list):
            check_interrupt(ctx)
            
            debug.log(f"Post-processing batch {info_idx+1}/{num_valid_samples}", category="video", force=True)
            debug.start_timer(f"postprocess_batch_{info_idx+1}")
            
            # Get slice from final_video - currently in [T, H, W, C] format, values in [-1, 1]
            sample_thwc = ctx['final_video'][write_start:write_end]
            
            # For RGBA, we only process RGB channels for color correction
            # Alpha was already written during alpha processing above
            if ctx.get('is_rgba', False) and sample_thwc.shape[-1] == 4:
                sample_thwc_rgb = sample_thwc[..., :3]  # [T, H, W, 3]
                sample = sample_thwc_rgb.permute(0, 3, 1, 2)  # [T, H, W, 3] → [T, 3, H, W]
            else:
                sample = sample_thwc.permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]
            
            # Move to VAE device for processing
            sample = manage_tensor(
                tensor=sample,
                target_device=ctx['vae_device'],
                tensor_name=f"sample_{info_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="post-processing",
                indent_level=1
            )
            
            # Reconstruct transformed video on-demand for color correction
            input_video = None
            if color_correction != "none" and ctx.get('batch_metadata') is not None:
                if batch_idx < len(ctx['batch_metadata']) and ctx['batch_metadata'][batch_idx] is not None:
                    # Reconstruct transformation
                    transformed_video = _reconstruct_and_transform_batch(ctx, batch_idx, debug)
                    input_video = optimized_single_video_rearrange(transformed_video)
                    del transformed_video
                    
                    # For batches after the first with temporal overlap, the overlap frames
                    # were blended in Phase 3 and are not part of this slice. Skip them.
                    actual_overlap = ctx.get('actual_temporal_overlap', 0)
                    if info_idx > 0 and actual_overlap > 0:
                        input_video = input_video[actual_overlap:]
                    
                    # Trim input_video to match sample length (handles padding differences)
                    if input_video.shape[0] > sample.shape[0]:
                        input_video = input_video[:sample.shape[0]]
                    
                    # Trim spatial dimensions to true target size
                    if 'true_target_dims' in ctx:
                        true_h, true_w = ctx['true_target_dims']
                        if input_video.shape[-2] != true_h or input_video.shape[-1] != true_w:
                            input_video = input_video[:, :, :true_h, :true_w]
            
            # Apply color correction if enabled (RGB only)
            if color_correction != "none" and input_video is not None:
                # Check if RGBA (samples are in T, C, H, W format at this point)
                has_alpha = ctx.get('is_rgba', False)
                alpha_channel = None
                
                if has_alpha:
                    # Check actual channel count
                    if sample.shape[1] == 4:
                        # Extract and temporarily store alpha for reattachment after color correction
                        alpha_channel = sample[:, 3:4, :, :]  # (T, 1, H, W)
                        sample = sample[:, :3, :, :]  # Keep only RGB (T, 3, H, W)
                
                # Ensure both tensors are on same device (GPU) for color correction
                if input_video.device != sample.device:
                    input_video = manage_tensor(
                        tensor=input_video,
                        target_device=sample.device,
                        tensor_name=f"input_video_{info_idx+1}",
                        debug=debug,
                        reason="color correction",
                        indent_level=1
                    )
                    
                # Apply selected color correction method
                debug.start_timer(f"color_correction_{color_correction}")
                
                if color_correction == "lab":
                    debug.log("Applying LAB perceptual color transfer", category="video", force=True, indent_level=1)
                    sample = lab_color_transfer(sample, input_video, debug, luminance_weight=0.8)
                elif color_correction == "wavelet_adaptive":
                    debug.log("Applying wavelet with adaptive saturation correction", category="video", force=True, indent_level=1)
                    sample = wavelet_adaptive_color_correction(sample, input_video, debug)
                elif color_correction == "wavelet":
                    debug.log("Applying wavelet color reconstruction", category="video", force=True, indent_level=1)
                    sample = wavelet_reconstruction(sample, input_video, debug)
                elif color_correction == "hsv":
                    debug.log("Applying HSV hue-conditional saturation matching", category="video", force=True, indent_level=1)
                    sample = hsv_saturation_histogram_match(sample, input_video, debug)
                elif color_correction == "adain":
                    debug.log("Applying AdaIN color correction", category="video", force=True, indent_level=1)
                    sample = adaptive_instance_normalization(sample, input_video)
                else:
                    debug.log(f"Unknown color correction method: {color_correction}", level="WARNING", category="video", force=True, indent_level=1)
                
                debug.end_timer(f"color_correction_{color_correction}", f"Color correction ({color_correction})")
                
                # Free the reconstructed transformed video
                del input_video

                # Recombine with Alpha if it was present in input
                if has_alpha and alpha_channel is not None:
                    # Concatenate in channels-first: (T, 3, H, W) + (T, 1, H, W) -> (T, 4, H, W)
                    sample = torch.cat([sample, alpha_channel], dim=1)
            
            else:
                debug.log("Color correction disabled (set to none)", category="video", indent_level=1)
            
            # Convert to final format: [T, C, H, W] → [T, H, W, C]
            sample = optimized_sample_to_image_format(sample)
            
            # Apply normalization only to RGB channels, preserve Alpha as-is
            if ctx.get('is_rgba', False) and sample.shape[-1] == 4:
                # Split RGBA: sample is (T, H, W, C) format after optimized_sample_to_image_format
                rgb_channels = sample[..., :3]  # (T, H, W, 3)
                alpha_channel = sample[..., 3:4]  # (T, H, W, 1)
                
                # Normalize only RGB from [-1, 1] to [0, 1]
                rgb_channels.clamp_(-1, 1).mul_(0.5).add_(0.5)
                
                # Merge back with unchanged Alpha
                sample = torch.cat([rgb_channels, alpha_channel], dim=-1)
            else:
                # RGB only: apply normalization as usual
                sample.clamp_(-1, 1).mul_(0.5).add_(0.5)
            
            # Draw tile boundaries for debugging (if tile info available)
            for phase, attr in [('encode', 'encode_tile_boundaries'), ('decode', 'decode_tile_boundaries')]:
                tiles = getattr(debug, attr, None)
                if tiles:
                    sample = _draw_tile_boundaries(sample, debug, tiles, phase)
                    break
            
            # Move to final_video device and write back in-place
            sample = manage_tensor(
                tensor=sample,
                target_device=ctx['final_video'].device,
                tensor_name=f"sample_{info_idx+1}_final",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="writing processed result to final_video",
                indent_level=1
            )
            
            # Write back to final_video in-place
            # For RGBA, write only RGB channels (alpha already written during alpha processing)
            if ctx.get('is_rgba', False) and ctx['final_video'].shape[-1] == 4:
                ctx['final_video'][write_start:write_end, :, :, :3] = sample
            else:
                ctx['final_video'][write_start:write_end] = sample
            
            # Free sample memory
            del sample, sample_thwc
            
            debug.end_timer(f"postprocess_batch_{info_idx+1}", f"Post-processed batch {info_idx+1}")
            
            # Update progress for main processing step
            current_postprocessing_step += 1
            if progress_callback:
                progress_callback(current_postprocessing_step, total_postprocessing_steps,
                                1, "Phase 4: Post-processing")

        # Verify final assembly
        if ctx['final_video'] is not None:
            # Remove prepended frames if any were added at the start
            frames_before_removal = ctx['final_video'].shape[0]
            
            if prepend_frames > 0:
                if prepend_frames < ctx['final_video'].shape[0]:
                    debug.log(f"Removing {prepend_frames} prepended frames from output", category="video", force=True)
                    ctx['final_video'] = ctx['final_video'][prepend_frames:]
                else:
                    debug.log(f"Warning: prepend_frames ({prepend_frames}) >= total frames ({ctx['final_video'].shape[0]}), skipping removal", 
                            level="WARNING", category="video", force=True)

            final_shape = ctx['final_video'].shape
            Tf, Hf, Wf, Cf = final_shape[0], final_shape[1], final_shape[2], final_shape[3]
            channels_str = "RGBA" if Cf == 4 else "RGB" if Cf == 3 else f"{Cf}-channel"
            
            # Build message showing prepend and/or padding removal if applicable
            frame_info = f"{Tf} frames"
            adjustments = []

            if prepend_frames > 0 and prepend_frames < frames_before_removal:
                adjustments.append(f"{prepend_frames} prepend")

            if total_padding_removed > 0:
                adjustments.append(f"{total_padding_removed} padding")
            
            # Use actual temporal overlap from encoding (may have been reset)
            actual_overlap = ctx.get('actual_temporal_overlap', temporal_overlap)
            
            # Calculate and include temporal overlap blending info
            if actual_overlap > 0:
                frames_blended = (num_valid_samples - 1) * actual_overlap
                adjustments.append(f"{frames_blended} overlap")

            if adjustments:
                # Add back all removed/blended frames to get true computed count
                total_computed = frames_before_removal + total_padding_removed
                if actual_overlap > 0:
                    total_computed += (num_valid_samples - 1) * actual_overlap
                frame_info += f" ({total_computed} computed with {' + '.join(adjustments)} removed)"
            
            debug.log(f"Output assembled: {frame_info}, Resolution: {Wf}x{Hf}px, Channels: {channels_str}", 
                    category="generation", force=True)
        else:
            ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=ctx['compute_dtype'])
            debug.log("No frames were processed", level="WARNING", category="generation", force=True)
            
    except Exception as e:
        debug.log(f"Error in Phase 4 (Post-processing): {e}", level="ERROR", category="generation", force=True)
        raise
    finally:
        # 1. Clean up decode_batch_info and padding stats
        if 'decode_batch_info' in ctx:
            del ctx['decode_batch_info']
        if 'total_padding_removed' in ctx:
            del ctx['total_padding_removed']
        
        # 2. Clean up video transform caches
        if 'video_transform' in ctx and ctx['video_transform'] is not None:
            if hasattr(ctx['video_transform'], 'transforms'):
                for transform in ctx['video_transform'].transforms:
                    # Clear cache attributes
                    for cache_attr in ['cache', '_cache']:
                        if hasattr(transform, cache_attr):
                            setattr(transform, cache_attr, None)
                    # Clear remaining attributes
                    if hasattr(transform, '__dict__'):
                        transform.__dict__.clear()
            del ctx['video_transform']
        
        # 3. Clean up storage lists (all_latents, all_alpha_channels, etc.)
        tensor_storage_keys = ['all_latents', 'all_alpha_channels', 'all_input_rgb']
        for key in tensor_storage_keys:
            if key in ctx and ctx[key]:
                release_tensor_collection(ctx[key])
                del ctx[key]
        
        # 4. Clean up non-tensor storage
        if 'all_ori_lengths' in ctx:
            del ctx['all_ori_lengths']
        if 'true_target_dims' in ctx:
            del ctx['true_target_dims']
        if 'batch_metadata' in ctx:
            del ctx['batch_metadata']
        if 'input_images' in ctx:
            release_tensor_memory(ctx['input_images'])
            del ctx['input_images']

    debug.end_timer("phase4_postprocessing", "Phase 4: Post-processing complete", show_breakdown=True)
    debug.log_memory_state("After phase 4 (Post-processing)", show_tensors=False)
    
    return ctx
