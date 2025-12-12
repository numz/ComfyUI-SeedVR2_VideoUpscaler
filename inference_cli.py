#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler - Standalone CLI Interface

Command-line interface for high-quality upscaling using SeedVR2 diffusion models.
Supports single and multi-GPU processing with advanced memory optimization.

Key Features:
    • Multi-GPU Processing: Automatic workload distribution across multiple GPUs with
      temporal overlap blending for seamless transitions
    • Streaming Mode: Memory-efficient processing of long videos in chunks, avoiding
      full video loading into RAM while maintaining temporal consistency
    • Memory Optimization: BlockSwap for limited VRAM, VAE tiling for large resolutions,
      intelligent tensor offloading between processing phases
    • Performance: Torch.compile integration, BFloat16 compute pipeline,
      efficient model caching for batch and streaming processing
    • Flexibility: Multiple output formats (MP4/PNG), advanced color correction methods,
      directory batch processing with auto-format detection
    • Quality Control: Temporal overlap blending, frame prepending for artifact reduction,
      configurable noise scales for detail preservation

Architecture:
    The CLI implements a 4-phase processing pipeline:
    1. Encode: VAE encoding with optional input noise and tiling
    2. Upscale: DiT transformer upscaling with latent space diffusion
    3. Decode: VAE decoding with optional tiling
    4. Postprocess: Color correction and temporal blending

Usage:
    python inference_cli.py video.mp4 --resolution 1080
    For complete usage examples, run: python inference_cli.py --help

Requirements:
    • Python 3.10+
    • PyTorch 2.4+ with CUDA 12.1+ (NVIDIA) or MPS (Apple Silicon)
    • 16GB+ VRAM recommended (8GB minimum with BlockSwap)
    • OpenCV, NumPy for video I/O

Model Support:
    • 3B models: seedvr2_ema_3b_fp16.safetensors (default), _fp8_e4m3fn/GGUF variants
    • 7B models: seedvr2_ema_7b_fp16.safetensors, _fp8_e4m3fn/GGUF variants
    • VAE: ema_vae_fp16.safetensors (shared across all models)
    • Auto-downloads from HuggingFace on first run with SHA256 validation
"""

# Standard library imports
import sys
import os
import argparse
import time
import platform
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple, Literal, Generator
from datetime import datetime
from pathlib import Path

# Set up path before any other imports to fix module resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set environment variable so all spawned processes can find modules
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Configure platform-specific memory management before heavy imports
# Must be set BEFORE import torch
if platform.system() == "Darwin":
    # MPS allocator requires: low_watermark <= high_watermark
    # Setting both to 0.0 disables PyTorch memory limits, letting macOS manage memory
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    # Pre-parse arguments that must be handled before torch import
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--cuda_device", type=str, default=None)
    _pre_args, _ = _pre_parser.parse_known_args()
    
    if _pre_args.cuda_device is not None:
        device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
        
        # Skip validation if CUDA_VISIBLE_DEVICES is already set (worker process)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            # Temporary torch import for CUDA device validation only
            # Must happen before setting CUDA_VISIBLE_DEVICES and before main torch import
            import torch as _torch_check
            if _torch_check.cuda.is_available():
                available_count = _torch_check.cuda.device_count()
                invalid_devices = [d for d in device_list_env if not d.isdigit() or int(d) >= available_count]
                if invalid_devices:
                    print(f"❌ [ERROR] Invalid CUDA device ID(s): {', '.join(invalid_devices)}. "
                        f"Available devices: 0-{available_count-1} (total: {available_count})")
                    sys.exit(1)
            else:
                print("❌ [ERROR] CUDA is not available on this system. Cannot use --cuda_device argument.")
                sys.exit(1)
            
            # Set CUDA_VISIBLE_DEVICES for single GPU after validation
            if len(device_list_env) == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# Heavy dependency imports after environment configuration
import torch
import cv2
import numpy as np
import subprocess

# Project imports
from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_dit_models, DEFAULT_DIT, DEFAULT_VAE
from src.utils.constants import SEEDVR2_FOLDER_NAME
from src.core.generation_utils import (
    setup_generation_context, 
    prepare_runner, 
    compute_generation_info, 
    log_generation_start,
    blend_overlapping_frames,
    load_text_embeddings,
    script_directory
)
from src.core.generation_phases import (
    encode_all_batches, 
    upscale_all_batches, 
    decode_all_batches, 
    postprocess_all_batches
)
from src.utils.debug import Debug
from src.optimization.memory_manager import clear_memory, get_gpu_backend, is_cuda_available
debug = Debug(enabled=False)  # Will be enabled via --debug CLI flag

# =============================================================================
# Device Management Helpers
# =============================================================================

def _device_id_to_name(device_id: str, platform_type: str = None) -> str:
    """
    Convert device ID to full device name.
    
    Args:
        device_id: Device ID ("0", "1") or special value ("cpu", "none")
        platform_type: Override platform type ("cuda", "mps", "cpu")
    
    Returns:
        Full device name ("cuda:0", "mps:0", "cpu", "none")
    """
    if device_id in ("cpu", "none"):
        return device_id
    
    if platform_type is None:
        platform_type = get_gpu_backend()
    
    # MPS typically doesn't use indices
    if platform_type == "mps":
        return "mps"
    
    return f"{platform_type}:{device_id}"


def _parse_offload_device(offload_arg: str, platform_type: str = None, cache_enabled: bool = False) -> Optional[str]:
    """
    Parse offload device argument to full device name.
    
    Args:
        offload_arg: Offload device argument ("none", "cpu", "0", "1", or "cuda:1")
        platform_type: Override platform type
        cache_enabled: If True and offload_arg is "none", default to "cpu"
    
    Returns:
        Full device name or None
    """
    if offload_arg == "none":
        # If caching enabled but no offload device specified, default to CPU
        return "cpu" if cache_enabled else None
    
    if offload_arg == "cpu":
        return "cpu"
    
    # If already a full device name (cuda:1, mps:0), return as-is
    if ":" in offload_arg:
        return offload_arg
    
    # Otherwise treat as device ID
    return _device_id_to_name(offload_arg, platform_type)


# =============================================================================
# Constants
# =============================================================================

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# =============================================================================
# Video I/O Functions
# =============================================================================

def get_media_files(directory: str) -> List[str]:
    """
    Get all video and image files from directory, sorted alphabetically.
    
    Args:
        directory: Path to directory to scan
        
    Returns:
        Sorted list of file paths (strings) matching video or image extensions
    """
    valid_extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    path = Path(directory)
    
    # Get all files and filter by extension (case-insensitive)
    files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    return sorted([str(f) for f in files])


def extract_frames_from_image(image_path: str) -> Tuple[torch.Tensor, float]:
    """
    Extract single frame from image file and convert to tensor format.
    
    Reads image using OpenCV, converts BGR to RGB, normalizes to [0,1] range,
    and formats as single-frame video tensor for consistent processing.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Tuple containing:
            - frames_tensor: Single frame as tensor [1, H, W, C], Float16, range [0,1] (C=3 for RGB, C=4 for RGBA)
            - fps: Default FPS value (30.0) for image-to-video conversion
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be opened
    """
    debug.log(f"Loading image: {image_path}", category="file")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image with alpha channel preserved
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")
    
    # Convert BGR(A) to RGB(A) based on channel count
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        debug.log(f"Detected RGBA image (alpha channel preserved)", category="file")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize
    frame = frame.astype(np.float32) / 255.0
    
    # Convert to tensor [1, H, W, C]
    frames_tensor = torch.from_numpy(frame[None, ...]).to(torch.float16)
    
    debug.log(f"Image tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")
    
    return frames_tensor, 30.0  # Default FPS for images


def get_input_type(input_path: str) -> Literal['video', 'image', 'directory', 'unknown']:
    """
    Determine input type from file path.
    
    Args:
        input_path: Path to input file or directory
        
    Returns:
        Input type: 'video', 'image', 'directory', or 'unknown'
        
    Raises:
        FileNotFoundError: If input path doesn't exist
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if path.is_dir():
        return 'directory'
    
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    else:
        return "unknown"


def generate_output_path(input_path: str, output_format: str, output_dir: Optional[str] = None, 
                        input_type: Optional[str] = None, from_directory: bool = False) -> str:
    """
    Generate output path based on input path and format.
    
    Args:
        input_path: Source file path
        output_format: "mp4" or "png"
        output_dir: Optional output directory (overrides default behavior)
        input_type: Optional input type ("image", "video", "directory")
        from_directory: True if processing files from a directory (batch mode)
    
    Returns:
        Absolute output path (file for single image/video, directory for sequences)
    """
    input_path_obj = Path(input_path)
    input_name = input_path_obj.stem
    
    # Determine base directory and whether to add suffix
    if output_dir:
        # User specified output directory - use as-is, no suffix
        base_dir = Path(output_dir)
        add_suffix = False
    elif from_directory:
        # Batch mode: create sibling folder with _upscaled, keep original filenames
        original_dir = input_path_obj.parent
        base_dir = original_dir.parent / f"{original_dir.name}_upscaled"
        add_suffix = False
    else:
        # Single file mode: output to same directory with _upscaled suffix
        base_dir = input_path_obj.parent
        add_suffix = True
    
    # Build filename with optional suffix
    file_suffix = "_upscaled" if add_suffix else ""
    
    # Generate output path based on format
    if output_format == "png":
        if input_type == "image":
            output_path = base_dir / f"{input_name}{file_suffix}.png"
        else:
            output_path = base_dir / f"{input_name}{file_suffix}"
    else:
        output_path = base_dir / f"{input_name}{file_suffix}.mp4"
    
    return str(output_path.resolve())


def process_single_file(input_path: str, args: argparse.Namespace, device_list: List[str], 
                       output_path: Optional[str] = None, format_auto_detected: bool = False,
                       runner_cache: Optional[Dict[str, Any]] = None) -> int:
    """
    Process a single video or image file with optional model caching.
    
    For videos, supports streaming mode (chunk_size > 0) which processes in memory-bounded
    chunks with temporal overlap for seamless transitions between chunks.
    
    Args:
        input_path: Path to input file
        args: Command-line arguments with all processing settings
        device_list: List of GPU device IDs as strings
        output_path: Optional explicit output path (auto-generated if None)
        format_auto_detected: Whether output format was auto-detected
        runner_cache: Optional cache dict for model reuse across multiple files
    
    Returns:
        Number of frames written to output
    """
    input_type = get_input_type(input_path)
    
    if input_type == "unknown":
        debug.log(f"Skipping unsupported file: {input_path}", level="WARNING", category="file", force=True)
        return 0
    
    debug.log(f"Processing {input_type}: {Path(input_path).name}", category="generation", force=True)
    
    # Generate or validate output path
    if output_path is None:
        output_path = generate_output_path(input_path, args.output_format, input_type=input_type)
    elif not Path(output_path).suffix or (args.output_format == "png" and input_type != "image"):
        # No extension or PNG sequence → treat as directory, generate filename
        output_path = generate_output_path(input_path, args.output_format, 
                                         output_dir=output_path, input_type=input_type)
    
    # Show format with auto-detection indicator
    format_prefix = "Auto-detected" if format_auto_detected else "Requested"
    debug.log(f"{format_prefix} output format: {args.output_format}", category="info", force=True, indent_level=1)
    
    # === VIDEO PROCESSING ===
    if input_type == "video":
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        debug.log(f"Video info: {total_frames} frames, {width}x{height}, {fps:.2f} FPS", category="info")
        
       # Skip initial frames
        if args.skip_first_frames > 0:
            debug.log(f"Skipping first {args.skip_first_frames} frames", category="info")
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.skip_first_frames)
        
        # Calculate frames to process (apply load_cap if set)
        frames_to_process = total_frames - args.skip_first_frames
        if args.load_cap > 0:
            frames_to_process = min(frames_to_process, args.load_cap)
        
        # Early exit for empty/exhausted video
        if frames_to_process <= 0:
            debug.log(f"No frames to process after skipping {args.skip_first_frames} of {total_frames}", 
                     level="WARNING", category="file", force=True)
            cap.release()
            return 0
        
        # Streaming mode: process in chunks
        chunk_size = args.chunk_size if args.chunk_size > 0 else frames_to_process
        streaming = args.chunk_size > 0
        total_chunks = (frames_to_process + chunk_size - 1) // chunk_size  # ceiling division
        
        if streaming:
            debug.log(f"Streaming mode: chunks of {chunk_size} frames, overlap={args.temporal_overlap}", 
                     category="info", force=True, indent_level=1)
        
        is_png = args.output_format == "png"
        video_writer = None
        overlap = args.temporal_overlap
        frames_written = 0
        chunk_idx = 0
        base_name = Path(input_path).stem
        
        # Multi-GPU: workers stream their own segments
        if len(device_list) > 1:
            cap.release()  # Workers will reopen
            video_info = {
                'video_path': input_path,
                'start_frame': args.skip_first_frames,
                'frames_to_process': frames_to_process,
            }
            result = _gpu_processing(None, device_list, args, video_info=video_info)
            
            # Save result
            if is_png:
                save_frames_to_image(result, output_path, base_name)
            else:
                video_writer = save_frames_to_video(result, output_path, fps, args=args)
                if video_writer is not None:
                    video_writer.release()
            
            frames_written = result.shape[0]
        
        # Single GPU: stream in main process
        else:
            chunk_count = 0
            for result in _stream_video_chunks(
                cap=cap,
                frames_to_process=frames_to_process,
                chunk_size=chunk_size,
                overlap=overlap,
                args=args,
                device_id=device_list[0],
                debug=debug,
                runner_cache=runner_cache,
                log_progress=streaming,
                total_chunks=total_chunks,
                cleanup_timer_name="chunk_cleanup"
            ):
                chunk_count += 1
                
                # Save output
                if is_png:
                    save_frames_to_image(result, output_path, base_name, start_index=frames_written)
                else:
                    video_writer = save_frames_to_video(result, output_path, fps, writer=video_writer, args=args)
                
                frames_written += result.shape[0]
                del result
            
            chunk_idx = chunk_count
            cap.release()
            if video_writer is not None:
                video_writer.release()
        
        if streaming:
            debug.log("", category="none", force=True)
            if len(device_list) > 1:
                debug.log(f"Streaming complete: {frames_written} frames across {len(device_list)} GPUs", category="success", force=True)
            else:
                debug.log(f"Streaming complete: {frames_written} frames in {chunk_idx} chunks", category="success", force=True)
        
        debug.log(f"Output saved to: {output_path}", category="file", force=True)
        return frames_written
    
    # === IMAGE PROCESSING ===
    frames_tensor, _ = extract_frames_from_image(input_path)
    
    processing_start = time.time()
    # Process frames (multiprocessing only for multi-GPU)
    if len(device_list) > 1:
        result = _gpu_processing(frames_tensor, device_list, args)
    else:
        result = _single_gpu_direct_processing(frames_tensor, args, device_list[0], runner_cache)
    debug.log(f"Processing time: {time.time() - processing_start:.2f}s", category="timing")
    
    # Save single image
    os.makedirs(Path(output_path).parent, exist_ok=True)
    frame_np = (result[0].cpu().numpy() * 255.0).astype(np.uint8)
    _save_image_bgr(frame_np, output_path)
    
    debug.log(f"Output saved to: {output_path}", category="file", force=True)
    return 1


def _read_frames_from_cap(cap: cv2.VideoCapture, max_frames: int) -> Optional[torch.Tensor]:
    """
    Read up to max_frames from an already-open VideoCapture.
    
    Args:
        cap: An already opened cv2.VideoCapture instance
        max_frames: Maximum number of frames to read in this call
    
    Returns:
        Tensor [T, H, W, C] float32 [0,1], or None if no frames available
    """
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame)
    
    if not frames:
        return None
    return torch.from_numpy(np.stack(frames)).to(torch.float32)


def _stream_video_chunks(
    cap: cv2.VideoCapture,
    frames_to_process: int,
    chunk_size: int,
    overlap: int,
    args: argparse.Namespace,
    device_id: str,
    debug: 'Debug',
    runner_cache: Optional[Dict[str, Any]],
    log_progress: bool = False,
    total_chunks: int = 0,
    cleanup_timer_name: Optional[str] = None,
    log_prefix: str = ""
) -> Generator[torch.Tensor, None, None]:
    """
    Generator that streams and processes video chunks.
    
    Handles frame reading, temporal context prepending, processing via
    _process_frames_core, context removal from output, and memory cleanup.
    Caller is responsible for VideoCapture lifecycle and result handling.
    
    Args:
        cap: Open VideoCapture positioned at start frame
        frames_to_process: Total frames to read and process
        chunk_size: Frames per chunk (use frames_to_process for single chunk)
        overlap: Temporal overlap frames between chunks for blending
        args: Processing arguments (copied internally, prepend_frames zeroed after first chunk)
        device_id: GPU device ID for processing
        debug: Debug instance for logging
        runner_cache: Optional model cache dict for reuse across chunks
        log_progress: If True, log chunk progress with separators
        total_chunks: Total chunks for progress display (used if log_progress=True)
        cleanup_timer_name: Optional timer name for memory cleanup logging
        log_prefix: Optional prefix for log messages (e.g., "[GPU 0] " for worker identification)
    
    Yields:
        Processed frames tensor [T, H, W, C] for each chunk, context frames removed
    """
    chunk_args = argparse.Namespace(**vars(args))
    frames_read = 0
    prev_raw_tail = None
    chunk_idx = 0
    streaming = chunk_size < frames_to_process
    
    while frames_read < frames_to_process:
        read_count = min(chunk_size, frames_to_process - frames_read)
        new_frames = _read_frames_from_cap(cap, read_count)
        if new_frames is None:
            break
        frames_read += new_frames.shape[0]
        chunk_idx += 1
        
        # Disable prepend_frames after first chunk
        if chunk_idx > 1:
            chunk_args.prepend_frames = 0
        
        # Prepend context from previous chunk
        if prev_raw_tail is not None and overlap > 0:
            context_count = min(overlap, prev_raw_tail.shape[0])
            frames = torch.cat([prev_raw_tail[-context_count:], new_frames], dim=0)
        else:
            frames = new_frames
            context_count = 0
        
        # Log progress if enabled
        if log_progress and streaming:
            if chunk_idx > 1:
                debug.log("", category="none", force=True)
                debug.log("━" * 60, category="none", force=True)
            debug.log("", category="none", force=True)
            debug.log(f"{log_prefix}Chunk {chunk_idx}/{total_chunks}: {new_frames.shape[0]} new + {context_count} context frames", 
                     category="generation", force=True)
            debug.log("", category="none", force=True)
        
        # Process chunk
        result = _process_frames_core(
            frames_tensor=frames.to(torch.float16),
            args=chunk_args,
            device_id=device_id,
            debug=debug,
            runner_cache=runner_cache
        )
        
        # Remove context frames from output
        if context_count > 0:
            result = result[context_count:]
        
        # Save tail for next chunk context
        prev_raw_tail = new_frames[-overlap:].clone() if overlap > 0 else None
        
        # Cleanup before yield
        del frames
        
        yield result
        
        # Memory cleanup between chunks
        if streaming:
            clear_memory(debug=debug, deep=True, force=True, timer_name=cleanup_timer_name)


def _save_image_bgr(frame_np: np.ndarray, file_path: str) -> None:
    """
    Save a single RGB(A) uint8 frame to disk, converting to BGR(A) for OpenCV.
    
    Args:
        frame_np: Frame as uint8 numpy array [H, W, C] where C is 3 (RGB) or 4 (RGBA)
        file_path: Output file path
    """
    if frame_np.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
    else:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, frame_bgr)


def save_frames_to_video(
    frames_tensor: torch.Tensor, 
    output_path: str, 
    fps: float = 30.0,
    writer: Optional[cv2.VideoWriter] = None,
    args: Optional[argparse.Namespace] = None
) -> Optional[cv2.VideoWriter]:
    """
    Save frames tensor to MP4 video file.
    
    Converts tensor from Float32 [0,1] to uint8 [0,255], RGB to BGR for OpenCV,
    and writes to video file using mp4v codec. Supports streaming mode where
    an existing writer is passed and kept open for subsequent chunks.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_path: Output video file path (directory created if doesn't exist)
        fps: Frames per second for output video (default: 30.0)
        writer: Existing VideoWriter for streaming (if None, creates new one)
    
    Returns:
        VideoWriter if streaming mode (caller must close), None if standalone mode
    
    Raises:
        ValueError: If video writer cannot be initialized
    """
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    T, H, W, C = frames_np.shape

    # Default backend if not specified
    backend = 'opencv' if args is None else getattr(args, 'video_backend', 'opencv')
    save_10bit = False if args is None else getattr(args, 'save_10bit', False)

    class _FFMPEGWriter:
        def __init__(self, path: str, w: int, h: int, fps: float, save_10bit: bool):
            self.path = path
            self.w = w
            self.h = h
            self.fps = fps
            self.save_10bit = save_10bit
            self.proc = None
            self._open()

        def _open(self):
            os.makedirs(Path(self.path).parent, exist_ok=True)
            pix_out = 'yuv420p10le' if self.save_10bit else 'yuv420p'
            # Use x265 for 10-bit output, x264 otherwise
            if self.save_10bit:
                codec = 'libx265'
            else:
                codec = 'libx264'

            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{self.w}x{self.h}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', codec,
                '-pix_fmt', pix_out,
                '-preset', 'medium',
                '-crf', '12',
                self.path
            ]

            # Start ffmpeg process
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def write(self, frame_rgb: np.ndarray):
            # Frame expected as RGB uint8 array HxWx3
            if self.proc is None or self.proc.stdin is None:
                raise ValueError('FFmpeg process not started')
            # Ensure RGB24 (drop alpha if present)
            if frame_rgb.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8)
            self.proc.stdin.write(frame_rgb.tobytes())

        def release(self):
            if self.proc is not None:
                try:
                    if self.proc.stdin:
                        self.proc.stdin.close()
                except Exception:
                    pass
                self.proc.wait()
                self.proc = None

    # Open appropriate writer
    if writer is None:
        debug.log(f"Saving {T} frames to video: {output_path} (backend={backend})", category="file")
        if backend == 'opencv':
            os.makedirs(Path(output_path).parent, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                raise ValueError(f"Cannot create video writer for: {output_path}")
        else:
            writer = _FFMPEGWriter(output_path, W, H, fps, save_10bit)

    # Write frames
    for i, frame in enumerate(frames_np):
        # frame is RGB uint8
        if C == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            frame_rgb = frame

        if backend == 'opencv':
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        else:
            # FFmpeg writer expects RGB bytes
            writer.write(frame_rgb)

        if debug.enabled and (i + 1) % 100 == 0:
            debug.log(f"Written {i + 1}/{T} frames", category="file")

    return writer  # Caller must call .release() for either writer type


def save_frames_to_image(
    frames_tensor: torch.Tensor, 
    output_dir: str, 
    base_name: str,
    start_index: int = 0
) -> int:
    """
    Save frames tensor as sequential PNG image files.
    
    Each frame saved as {base_name}_{index:0Nd}.png with zero-padded indices.
    Converts Float32 [0,1] to uint8 [0,255] and RGB(A) to BGR(A) for OpenCV.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_dir: Directory to save PNG files (created if doesn't exist)
        base_name: Base name for output files (e.g., "frame" → "frame_00000.png")
        start_index: Starting index for filenames (for streaming continuation)
    
    Returns:
        Number of frames saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    total = frames_np.shape[0]
    
    if start_index == 0:
        debug.log(f"Saving {total} frames as PNGs to directory: {output_dir}", category="file")
    digits = 6  # Supports up to 999,999 frames (~11.5 hours at 24fps)

    for idx, frame in enumerate(frames_np):
        filename = f"{base_name}_{start_index + idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        _save_image_bgr(frame, file_path)
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} images", category="file")

    debug.log(f"Saved {total} images to '{output_dir}'", category="success")
    return total


# =============================================================================
# Core Processing Logic
# =============================================================================

def _process_frames_core(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    debug: Debug,
    runner_cache: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Core frame processing logic shared between worker and direct processing.
    
    Executes the complete 4-phase pipeline: encode → upscale → decode → postprocess.
    Supports both cached (direct) and non-cached (worker) execution modes.
    
    Args:
        frames_tensor: Input frames [T, H, W, C], Float16/Float32, range [0,1]
        args: Command-line arguments with all processing settings
        device_id: Device ID for inference ("0", "1", etc.)
        debug: Debug instance for logging
        runner_cache: Optional cache dict for model reuse (direct mode only)
    
    Returns:
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1]
    """    
    # Determine platform and convert device IDs to full names
    platform_type = get_gpu_backend()
    inference_device = _device_id_to_name(device_id, platform_type)
    
    # Parse offload devices (with caching defaults)
    cache_dit = args.cache_dit if runner_cache is not None else False
    cache_vae = args.cache_vae if runner_cache is not None else False
    
    dit_offload = _parse_offload_device(args.dit_offload_device, platform_type, cache_dit)
    vae_offload = _parse_offload_device(args.vae_offload_device, platform_type, cache_vae)
    tensor_offload = _parse_offload_device(args.tensor_offload_device, platform_type, False)
    
    # Setup or reuse generation context
    if runner_cache is not None and 'ctx' in runner_cache:
        ctx = runner_cache['ctx']
        # Clear previous run data but keep device config
        keys_to_keep = {'dit_device', 'vae_device', 'dit_offload_device', 
                       'vae_offload_device', 'tensor_offload_device', 'compute_dtype'}
        for key in list(ctx.keys()):
            if key not in keys_to_keep:
                del ctx[key]
    else:
        ctx = setup_generation_context(
            dit_device=inference_device,
            vae_device=inference_device,
            dit_offload_device=dit_offload,
            vae_offload_device=vae_offload,
            tensor_offload_device=tensor_offload,
            debug=debug
        )
        if runner_cache is not None:
            runner_cache['ctx'] = ctx
    
    # Build torch compile args
    torch_compile_args_dit = None
    torch_compile_args_vae = None
    if args.compile_dit:
        torch_compile_args_dit = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    if args.compile_vae:
        torch_compile_args_vae = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    
    # Prepare runner with caching support
    model_dir = args.model_dir if args.model_dir is not None else f"./models/{SEEDVR2_FOLDER_NAME}"
    
    # Use fixed IDs for CLI caching when enabled
    dit_id = "cli_dit" if cache_dit else None
    vae_id = "cli_vae" if cache_vae else None
    
    runner, cache_context = prepare_runner(
        dit_model=args.dit_model,
        vae_model=DEFAULT_VAE,
        model_dir=model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=cache_dit,
        vae_cache=cache_vae,
        dit_id=dit_id,
        vae_id=vae_id,
        block_swap_config={
            'blocks_to_swap': args.blocks_to_swap,
            'swap_io_components': args.swap_io_components,
            'offload_device': dit_offload,
        },
        encode_tiled=args.vae_encode_tiled,
        encode_tile_size=(args.vae_encode_tile_size, args.vae_encode_tile_size),
        encode_tile_overlap=(args.vae_encode_tile_overlap, args.vae_encode_tile_overlap),
        decode_tiled=args.vae_decode_tiled,
        decode_tile_size=(args.vae_decode_tile_size, args.vae_decode_tile_size),
        decode_tile_overlap=(args.vae_decode_tile_overlap, args.vae_decode_tile_overlap),
        tile_debug=args.tile_debug.lower() if args.tile_debug else "false",
        attention_mode=args.attention_mode,
        torch_compile_args_dit=torch_compile_args_dit,
        torch_compile_args_vae=torch_compile_args_vae
    )
    
    ctx['cache_context'] = cache_context
    if runner_cache is not None:
        runner_cache['runner'] = runner
    
    # Preload text embeddings before Phase 1 to avoid sync stall in Phase 2
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)
    debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Compute generation info and log start (handles prepending internally)
    frames_tensor, gen_info = compute_generation_info(
        ctx=ctx,
        images=frames_tensor,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        batch_size=args.batch_size,
        uniform_batch_size=args.uniform_batch_size,
        seed=args.seed,
        prepend_frames=args.prepend_frames,
        temporal_overlap=args.temporal_overlap,
        debug=debug
    )
    log_generation_start(gen_info, debug)
    
    # Phase 1: Encode
    ctx = encode_all_batches(
        runner, ctx=ctx, images=frames_tensor,
        debug=debug, 
        batch_size=args.batch_size,
        uniform_batch_size=args.uniform_batch_size,
        seed=args.seed,
        progress_callback=None, 
        temporal_overlap=args.temporal_overlap,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        input_noise_scale=args.input_noise_scale,
        color_correction=args.color_correction
    )
    
    # Phase 2: Upscale
    ctx = upscale_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        seed=args.seed,
        latent_noise_scale=args.latent_noise_scale,
        cache_model=cache_dit
    )
    
    # Phase 3: Decode
    ctx = decode_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        cache_model=cache_vae
    )
    
    # Phase 4: Post-process
    ctx = postprocess_all_batches(
        ctx=ctx, debug=debug, progress_callback=None,
        color_correction=args.color_correction,
        prepend_frames=0,  # Worker mode handles this in main process
        temporal_overlap=args.temporal_overlap,
        batch_size=args.batch_size
    )
    
    result_tensor = ctx['final_video']
    
    # Convert to CPU and compatible dtype
    if result_tensor.is_cuda or result_tensor.is_mps:
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
    
    return result_tensor


def _worker_process(
    proc_idx: int, 
    device_id: str, 
    frames_np: Optional[np.ndarray],
    shared_args: Dict[str, Any], 
    return_queue: mp.Queue,
    done_barrier: mp.Barrier,
    video_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Worker process for multi-GPU upscaling.
    
    Supports two modes:
    1. frames_np provided: Process pre-loaded frames (for images)
    2. video_info provided: Stream video segment internally (for videos)
       - Each worker opens the video, seeks to its assigned range, and streams
         with internal chunking and model caching for memory efficiency
    
    Args:
        proc_idx: Worker index for result ordering
        device_id: GPU device ID (used for CUDA_VISIBLE_DEVICES inheritance)
        frames_np: Pre-loaded frames as numpy array, or None for video streaming
        shared_args: Serialized args namespace as dict
        return_queue: Queue for returning results to parent
        done_barrier: Barrier for synchronizing shared memory handoff
        video_info: Optional dict with 'video_path', 'start_frame', 'end_frame'
                   for video streaming mode
    """
    # Create debug instance for this worker
    worker_debug = Debug(enabled=shared_args["debug"])
    
    args = argparse.Namespace(**shared_args)
    
    # Video streaming mode: worker reads and processes its assigned segment
    if video_info is not None:
        cap = cv2.VideoCapture(video_info['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_info['start_frame'])
        
        segment_frames = video_info['end_frame'] - video_info['start_frame']
        chunk_size = args.chunk_size if args.chunk_size > 0 else segment_frames
        
        worker_debug.log(f"GPU {proc_idx}: frames {video_info['start_frame']}-{video_info['end_frame']} "
                        f"({segment_frames} frames, chunks of {chunk_size})",
                        category="generation", force=True)
        
        # Only GPU 0 uses prepend_frames (applies to video start only)
        worker_args = argparse.Namespace(**vars(args))
        if proc_idx != 0:
            worker_args.prepend_frames = 0
        
        # Enable model caching within worker only if requested
        runner_cache = {} if (args.cache_dit or args.cache_vae) else None
        
        total_chunks = (segment_frames + chunk_size - 1) // chunk_size
        results = []
        for result in _stream_video_chunks(
            cap=cap,
            frames_to_process=segment_frames,
            chunk_size=chunk_size,
            overlap=args.temporal_overlap,
            args=worker_args,
            device_id="0",
            debug=worker_debug,
            runner_cache=runner_cache,
            log_progress=total_chunks > 1,
            total_chunks=total_chunks,
            log_prefix=f"[GPU {proc_idx}] "
        ):
            results.append(result.cpu())
        
        cap.release()
        result_tensor = torch.cat(results, dim=0) if results else torch.empty(0, dtype=torch.float32)
    
    # Pre-loaded frames mode (original behavior)
    else:
        frames_tensor = torch.from_numpy(frames_np).to(torch.float16)
        result_tensor = _process_frames_core(
            frames_tensor=frames_tensor,
            args=args,
            device_id="0",
            debug=worker_debug,
            runner_cache=None
        )
    
    # Share tensor memory for efficient cross-process transfer (avoids pickling large arrays)
    return_queue.put((proc_idx, result_tensor.share_memory_()))
    
    # Wait for parent to copy shared tensors before exiting
    # (shared memory requires creating process to stay alive during access)
    done_barrier.wait()


def _single_gpu_direct_processing(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    runner_cache: Optional[Dict[str, Any]]
) -> torch.Tensor:
    """
    Direct single-GPU processing with model caching support.
    
    Uses main process and shared runner cache for efficient multi-file processing.
    """
    return _process_frames_core(
        frames_tensor=frames_tensor,
        args=args,
        device_id=device_id,
        debug=debug,
        runner_cache=runner_cache
    )


def _gpu_processing(
    frames_tensor: Optional[torch.Tensor],
    device_list: List[str], 
    args: argparse.Namespace,
    video_info: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Orchestrate multi-GPU parallel video upscaling with temporal overlap blending.
    
    Supports two modes:
    1. video_info provided: Workers stream their assigned video segments internally
       (each GPU reads and processes its frame range with internal chunking)
    2. frames_tensor provided: Workers process pre-loaded frame chunks
       (non streaming behavior for images or pre-loaded videos)
    
    Args:
        frames_tensor: Input frames [T, H, W, C] or None if using video_info mode
        device_list: List of device IDs as strings (e.g., ["0", "1"])
        args: Parsed command-line arguments containing all processing settings
        video_info: Optional dict with 'video_path', 'start_frame', 'frames_to_process'
                   for streaming mode where workers read video directly
    
    Returns:
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1]
    """
    num_devices = len(device_list)
    overlap = args.temporal_overlap
    
    return_queue = mp.Queue(maxsize=0)
    done_barrier = mp.Barrier(num_devices + 1)
    workers = []
    shared_args = vars(args).copy()
    
    # Video streaming mode: distribute frame ranges to workers
    if video_info is not None:
        total_frames = video_info['frames_to_process']
        start_frame = video_info['start_frame']
        video_path = video_info['video_path']
        
        base_per_gpu = total_frames // num_devices
        remainder = total_frames % num_devices
        
        current_start = start_frame
        for idx, device_id in enumerate(device_list):
            gpu_frames = base_per_gpu + (1 if idx < remainder else 0)
            gpu_end = current_start + gpu_frames
            
            # Add overlap frames for blending (except last GPU)
            if idx < num_devices - 1 and overlap > 0:
                gpu_end = min(gpu_end + overlap, start_frame + total_frames)
            
            worker_video_info = {
                'video_path': video_path,
                'start_frame': current_start,
                'end_frame': gpu_end,
            }
            
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            p = mp.Process(
                target=_worker_process,
                args=(idx, device_id, None, shared_args, return_queue, done_barrier),
                kwargs={'video_info': worker_video_info}
            )
            p.start()
            workers.append(p)
            
            current_start += gpu_frames
    
    # Pre-loaded frames mode (original behavior for images or non-streaming)
    else:
        total_frames = frames_tensor.shape[0]
        
        if overlap > 0 and num_devices > 1:
            chunk_with_overlap = total_frames // num_devices + overlap
            if args.batch_size > 1:
                chunk_with_overlap = ((chunk_with_overlap + args.batch_size - 1) // args.batch_size) * args.batch_size
            base_chunk_size = chunk_with_overlap - overlap

            chunks = []
            for i in range(num_devices):
                start_idx = i * base_chunk_size
                if i == num_devices - 1:
                    end_idx = total_frames
                else:
                    end_idx = min(start_idx + chunk_with_overlap, total_frames)
                chunks.append(frames_tensor[start_idx:end_idx])
        else:
            chunks = torch.chunk(frames_tensor, num_devices, dim=0)

        for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            p = mp.Process(
                target=_worker_process,
                args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue, done_barrier),
            )
            p.start()
            workers.append(p)

    # Collect results before joining to prevent deadlock
    # Tensors arrive via shared memory - copy to numpy while workers still alive
    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, result_tensor = return_queue.get()
        results_np[proc_idx] = result_tensor.numpy()
        collected += 1
    
    # Release workers now that shared tensors are copied
    done_barrier.wait()
    
    # Now safe to join
    for p in workers:
        p.join()

    # Concatenate results with overlap blending using shared function
    if args.temporal_overlap > 0 and num_devices > 1:        
        overlap = args.temporal_overlap
        result_tensor = None
        
        for idx, res_np in enumerate(results_np):
            chunk_tensor = torch.from_numpy(res_np).to(torch.float32)
            
            if idx == 0:
                # First chunk: keep all frames
                result_tensor = chunk_tensor
            else:
                # Subsequent chunks: blend overlapping region with accumulated result
                if chunk_tensor.shape[0] > overlap and result_tensor.shape[0] >= overlap:
                    # Get overlapping regions
                    prev_tail = result_tensor[-overlap:]  # Last N frames from accumulated result
                    cur_head = chunk_tensor[:overlap]      # First N frames from current chunk
                    
                    # Blend using shared function
                    blended = blend_overlapping_frames(prev_tail, cur_head, overlap)
                    
                    # Replace tail of result with blended frames, then append rest of chunk
                    result_tensor = torch.cat([
                        result_tensor[:-overlap],           # Everything except the tail
                        blended,                            # Blended overlapping frames
                        chunk_tensor[overlap:]              # Non-overlapping part of current chunk
                    ], dim=0)
                else:
                    # Edge case: chunk too small, just append non-overlapping part
                    if chunk_tensor.shape[0] > overlap:
                        result_tensor = torch.cat([result_tensor, chunk_tensor[overlap:]], dim=0)
        
        if result_tensor is None:
            result_tensor = torch.from_numpy(results_np[0]).to(torch.float32)
    else:
        # Simple concatenation without overlap
        result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float32)

    # Handle prepend_frames removal (multi-GPU safe - done after all workers complete)
    if args.prepend_frames > 0:
        if args.prepend_frames < result_tensor.shape[0]:
            debug.log(f"Removing {args.prepend_frames} prepended frames from output", category="generation")
            result_tensor = result_tensor[args.prepend_frames:]
        else:
            debug.log(f"prepend_frames ({args.prepend_frames}) >= total frames ({result_tensor.shape[0]}), skipping removal", 
                     level="WARNING", category="generation", force=True)
    
    return result_tensor


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for SeedVR2 CLI.
    
    Configures all available options including model selection, processing parameters,
    memory optimization settings, and output configuration.
    
    Returns:
        Parsed arguments namespace with all CLI parameters
    
    Note:
        - cuda_device argument only available on non-macOS systems
        - Default model directory resolves to "models/SEEDVR2" if not specified
    """
    
    # Get the actual invocation path for usage examples
    invocation = sys.argv[0]
    
    # Multi-line usage examples for --help
    usage_examples = f"""
Examples:

  Basic image upscaling:
    python {invocation} image.jpg

  Basic video upscaling with temporal consistency:
    python {invocation} video.mp4 --resolution 720 --batch_size 33
    
  Streaming mode for long videos:
    python {invocation} long_video.mp4 --resolution 1080 --batch_size 33 --chunk_size 330 --temporal_overlap 3

  Multi-GPU processing with temporal overlap:
    python {invocation} video.mp4 --cuda_device 0,1 --resolution 1080 --batch_size 81 --uniform_batch_size --temporal_overlap 3 --prepend_frames 4 

  Memory-optimized for low VRAM (8GB):
    python {invocation} image.png --dit_model seedvr2_ema_3b-Q8_0.gguf --blocks_to_swap 32 --swap_io_components --dit_offload_device cpu --vae_offload_device cpu
    
  High resolution with VAE tiling:
    python {invocation} video.mp4 --resolution 1440 --batch_size 31 --uniform_batch_size --temporal_overlap 3 --vae_encode_tiled --vae_decode_tiled
    
  Batch directory processing:
    python {invocation} media_folder/ --output processed/ --cuda_device 0 --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu --resolution 1080 --max_resolution 1920

"""
    
    parser = argparse.ArgumentParser(
        description="SeedVR2 Video Upscaler - CLI for high-quality image/video upscaling and batch processing",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output options')
    io_group.add_argument("input", type=str,
                        help="Input: video file (.mp4, .avi, etc.), image file (.png, .jpg, etc.), or directory")
    io_group.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated in 'output/' directory)")
    io_group.add_argument("--output_format", type=str, default=None, choices=["mp4", "png", None],
                        help="Output format: 'mp4' (video) or 'png' (image sequence). Default: auto-detect from input type")
    io_group.add_argument("--video_backend", type=str, default="opencv", choices=["opencv", "ffmpeg"],
                        help="Video backend for encoding: 'opencv' (default) or 'ffmpeg' (requires ffmpeg in PATH).")
    io_group.add_argument("--save_10bit", action="store_true",
                        help="When using --video_backend ffmpeg, save output as 10-bit (yuv420p10le with libx265).")
    io_group.add_argument("--model_dir", type=str, default=None,
                        help=f"Model directory (default: ./models/{SEEDVR2_FOLDER_NAME})")
    
    # Model Selection
    model_group = parser.add_argument_group('Model selection')
    model_group.add_argument("--dit_model", type=str, default=DEFAULT_DIT,
                        choices=get_available_dit_models(),
                        help="DiT model to use. Options: 3B (fp16/fp8/GGUF) or 7B (fp16/fp8/GGUF). Default: 3B FP8")
    
    # Processing Parameters
    process_group = parser.add_argument_group('Processing parameters')
    process_group.add_argument("--resolution", type=int, default=1080,
                        help="Target short-side resolution in pixels (default: 1080)")
    process_group.add_argument("--max_resolution", type=int, default=0,
                        help="Maximum resolution for any edge. Scales down if exceeded. 0 = no limit (default: 0)")
    process_group.add_argument("--batch_size", type=int, default=5,
                        help="Frames per batch (must follow 4n+1: 1, 5, 9, 13, 17, 21,...). "
                         "Ideally matches shot length for best temporal consistency. Higher values improve "
                         "quality and speed but require more VRAM. Default: 5")
    process_group.add_argument("--uniform_batch_size", action="store_true",
                        help="Pad final batch to match batch_size. Prevents temporal artifacts caused by small "
                         "final batches. Add extra compute but recommended for optimal quality.")
    process_group.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    process_group.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip N initial frames (default: 0)")
    process_group.add_argument("--load_cap", type=int, default=0,
                        help="Load maximum N frames from video. 0 = load all (default: 0)")
    process_group.add_argument("--chunk_size", type=int, default=0,
                        help="Frames per chunk for streaming mode. When > 0, processes video in "
                             "memory-bounded chunks of N frames. 0 = load all frames at once (default: 0)")
    process_group.add_argument("--prepend_frames", type=int, default=0,
                        help="Prepend N reversed frames to reduce start artifacts (auto-removed). Default: 0")
    process_group.add_argument("--temporal_overlap", type=int, default=0,
                        help="Frames to overlap between batches/GPUs for smooth blending (default: 0)")
    
    # Quality Control
    quality_group = parser.add_argument_group('Quality control')
    quality_group.add_argument("--color_correction", type=str, default="lab", 
                    choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    help="Color correction method: 'lab' (perceptual color matching, recommended), 'wavelet' (frequency-based), "
                    "'wavelet_adaptive' (wavelet + saturation correction), 'hsv' (hue-conditional), 'adain' (statistical transfer), "
                    "'none' (disabled) (default: lab)")
    quality_group.add_argument("--input_noise_scale", type=float, default=0.0,
                        help="Input noise injection scale (0.0-1.0). Adds variation to input images (default: 0.0)")
    quality_group.add_argument("--latent_noise_scale", type=float, default=0.0,
                        help="Latent noise injection scale (0.0-1.0). Adds variation to latent space (default: 0.0)")
    
    # Device Management
    device_group = parser.add_argument_group('Device management')
    if platform.system() != "Darwin":
        device_group.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device(s): single '0' or multi-GPU '0,1,2'. Default: device 0")
    device_group.add_argument("--dit_offload_device", type=str, default="none",
                        help="DiT offload device when idle: 'none' (keep on GPU), 'cpu' (offload to RAM), or GPU ID. "
                             "Frees VRAM between phases. Required for BlockSwap. Default: none")
    device_group.add_argument("--vae_offload_device", type=str, default="none",
                        help="VAE offload device when idle: 'none', 'cpu', or GPU ID. Frees VRAM between phases. Default: none")
    device_group.add_argument("--tensor_offload_device", type=str, default="cpu",
                        help="Intermediate tensor storage: 'cpu' (recommended), 'none' (keep on GPU), or GPU ID. Default: cpu")
    
    # Memory Optimization (BlockSwap)
    blockswap_group = parser.add_argument_group('Memory optimization (BlockSwap)')
    blockswap_group.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Transformer blocks to swap for VRAM savings. 0-32 (3B) or 0-36 (7B). "
                             "Requires --dit_offload_device. Not available on macOS. Default: 0 (disabled)")
    blockswap_group.add_argument("--swap_io_components", action="store_true",
                        help="Offload DiT I/O layers for extra VRAM savings. Requires --dit_offload_device. "
                             "Not available on macOS")
    
    # VAE Tiling
    vae_group = parser.add_argument_group('VAE tiling (for high resolution upscale)')
    vae_group.add_argument("--vae_encode_tiled", action="store_true",
                        help="Enable VAE encode tiling to reduce VRAM during encoding")
    vae_group.add_argument("--vae_encode_tile_size", type=int, default=1024,
                        help="VAE encode tile size in pixels (default: 1024). Applied to both height and width. Only used if --vae_encode_tiled is set")
    vae_group.add_argument("--vae_encode_tile_overlap", type=int, default=128,
                        help="VAE encode tile overlap in pixels (default: 128). Reduces visible seams between tiles. Only used if --vae_encode_tiled is set")
    vae_group.add_argument("--vae_decode_tiled", action="store_true",
                        help="Enable VAE decode tiling to reduce VRAM during decoding")
    vae_group.add_argument("--vae_decode_tile_size", type=int, default=1024,
                        help="VAE decode tile size in pixels (default: 1024). Applied to both height and width. Only used if --vae_decode_tiled is set")
    vae_group.add_argument("--vae_decode_tile_overlap", type=int, default=128,
                        help="VAE decode tile overlap in pixels (default: 128). Reduces visible seams between tiles. Only used if --vae_decode_tiled is set")
    vae_group.add_argument("--tile_debug", type=str, default="false", choices=["false", "encode", "decode"],
                        help="Visualize tiles: 'false' (default), 'encode', or 'decode'")
    
    # Performance
    perf_group = parser.add_argument_group('Performance optimization')
    perf_group.add_argument("--attention_mode", type=str, default="sdpa",
                        choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                        help="Attention backend: 'sdpa' (default), 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3' (Blackwell GPUs)")
    perf_group.add_argument("--compile_dit", action="store_true", 
                        help="Enable torch.compile for DiT model (20-40%% speedup, requires PyTorch 2.0+ and Triton)")
    perf_group.add_argument("--compile_vae", action="store_true",
                        help="Enable torch.compile for VAE model (15-25%% speedup, requires PyTorch 2.0+ and Triton)")
    perf_group.add_argument("--compile_backend", type=str, default="inductor", choices=["inductor", "cudagraphs"],
                        help="Compilation backend: 'inductor' (full optimization with Triton) or 'cudagraphs' (lightweight, no kernel optimization) (default: inductor)")
    perf_group.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="Optimization level: 'default' (fast compilation), 'reduce-overhead' (lower overhead), 'max-autotune' (best runtime, slow compilation), "
                        "'max-autotune-no-cudagraphs' (like max-autotune without cudagraphs) (default: default)")
    perf_group.add_argument("--compile_fullgraph", action="store_true",
                        help="Compile entire model as single graph (faster but less flexible). May fail with dynamic shapes (default: False)")
    perf_group.add_argument("--compile_dynamic", action="store_true",
                        help="Handle varying input shapes without recompilation. Useful for different resolutions/batch sizes (default: False)")
    perf_group.add_argument("--compile_dynamo_cache_size_limit", type=int, default=64,
                        help="Max cached compiled versions per function. Increase when using many different input shapes. Higher uses more memory (default: 64)")
    perf_group.add_argument("--compile_dynamo_recompile_limit", type=int, default=128,
                        help="Max recompilation attempts before fallback to eager mode. Safety limit to prevent compilation loops (default: 128)")
    
    # Model Caching (for batch processing)
    cache_group = parser.add_argument_group('Model caching (batch processing)')
    cache_group.add_argument("--cache_dit", action="store_true",
                        help="Keep DiT model in memory between generations. Works with single-GPU directory processing "
                             "or multi-GPU streaming (--chunk_size). Requires --dit_offload_device")
    cache_group.add_argument("--cache_vae", action="store_true",
                        help="Keep VAE model in memory between generations. Works with single-GPU directory processing "
                             "or multi-GPU streaming (--chunk_size). Requires --vae_offload_device")
    
    # Debugging
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    
    # Auto-show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """
    Main entry point for SeedVR2 Video Upscaler CLI.
    
    Orchestrates the complete upscaling workflow:
        1. Parse and validate command-line arguments
        2. Extract frames from input video/image(s)
        3. Download required models if not cached
        4. Process frames on single or multiple GPUs
        5. Save results as video or PNG sequence
        6. Report timing and FPS (calculated from total wall-clock time)
    
    Error handling:
        - Validates tile configuration before processing
        - Provides detailed error messages with traceback
        - Ensures proper cleanup on exit (VRAM automatically freed)
    
    Raises:
        SystemExit: On argument validation failure or processing error
    """
    # Parse arguments
    args = parse_arguments()

    # Update debug instance with --debug flag
    debug.enabled = args.debug

    # print header
    debug.print_header(cli=True)
    
    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"{key}: {value}", category="none", indent_level=1)

    if args.vae_encode_tiled and args.vae_encode_tile_overlap >= args.vae_encode_tile_size:
        debug.log(f"VAE encode tile overlap ({args.vae_encode_tile_overlap}) must be smaller than tile size ({args.vae_encode_tile_size})", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    if args.vae_decode_tiled and args.vae_decode_tile_overlap >= args.vae_decode_tile_size:
        debug.log(f"VAE decode tile overlap ({args.vae_decode_tile_overlap}) must be smaller than tile size ({args.vae_decode_tile_size})", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    # Inform about caching defaults
    if args.cache_dit and args.dit_offload_device == "none":
        offload_target = "system memory (CPU)" if get_gpu_backend() != "mps" else "unified memory"
        debug.log(
            f"DiT caching enabled: Using default {offload_target} for offload. "
            "Set --dit_offload_device explicitly to use a different device.",
            category="cache", force=True
        )
    
    if args.cache_vae and args.vae_offload_device == "none":
        offload_target = "system memory (CPU)" if get_gpu_backend() != "mps" else "unified memory"
        debug.log(
            f"VAE caching enabled: Using default {offload_target} for offload. "
            "Set --vae_offload_device explicitly to use a different device.",
            category="cache", force=True
        )

    if args.debug:
        if platform.system() == "Darwin":
            debug.log("You are running on macOS and will use the MPS backend!", category="info", force=True)
        else:
            # Show actual CUDA device visibility
            debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}", category="device")
            if is_cuda_available():
                debug.log(f"torch.cuda.device_count(): {torch.cuda.device_count()}", category="device")
                debug.log(f"Using device index 0 inside script (mapped to selected GPU)", category="device")
    
    try:
        start_time = time.time()
        
        # Parse GPU list
        if platform.system() == "Darwin":
            device_list = ["0"]
        else:
            if args.cuda_device:
                device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()]
            else:
                device_list = ["0"]
        if args.debug:
            debug.log(f"Using devices: {device_list}", category="device")
        
        # Download models once before processing
        if not download_weight(dit_model=args.dit_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug):
            debug.log("Failed to download required models. Check console output above.", level="ERROR", category="download", force=True)
            sys.exit(1)
        
        # Determine input type and process accordingly
        input_type = get_input_type(args.input)

        # Track total frames for FPS calculation (time tracked via start_time)
        total_frames_processed = 0
        
        # Track if output format was user-specified or auto-detected
        format_auto_detected = args.output_format is None
        
        if input_type == 'directory':
            media_files = get_media_files(args.input)
            if not media_files:
                debug.log(f"No video or image files found in directory: {args.input}", 
                        level="ERROR", category="file", force=True)
                sys.exit(1)
            
            debug.log(f"Found {len(media_files)} media files to process", category="file", force=True)
            
            # Multi-GPU caching requires streaming (workers cache within their chunk loops)
            if (args.cache_dit or args.cache_vae) and len(device_list) > 1 and args.chunk_size <= 0:
                debug.log(
                    "Model caching requires streaming mode (--chunk_size > 0) for multi-GPU. "
                    "Disabling caching for this run.",
                    level="WARNING", category="cache", force=True
                )
                args.cache_dit = False
                args.cache_vae = False
            
            # Single-GPU: runner_cache persists across files; multi-GPU: workers cache internally
            runner_cache = {} if (args.cache_dit or args.cache_vae) and len(device_list) == 1 else None
            
            for idx, file_path in enumerate(media_files, 1):
                # Visual separation between files (except before first file)
                if idx > 1:
                    debug.log("", category="none", force=True)
                    debug.log("━" * 60, category="none", force=True)
                    debug.log("", category="none", force=True)
                
                debug.log(f"Processing file {idx}/{len(media_files)}", category="generation", force=True)
                
                # Auto-detect format per file if not user-specified
                if format_auto_detected:
                    file_type = get_input_type(file_path)
                    file_output_format = "mp4" if file_type == "video" else "png"
                else:
                    file_output_format = args.output_format
                
                # Temporarily override args.output_format for this file
                original_format = args.output_format
                args.output_format = file_output_format
                
                # generate_output_path handles None gracefully with "outputs" default
                output_path = generate_output_path(file_path, file_output_format, args.output, 
                                   input_type=get_input_type(file_path), from_directory=True)
                
                # Process with explicit output path and runner cache
                frames = process_single_file(file_path, args, device_list, output_path, 
                                            format_auto_detected=format_auto_detected,
                                            runner_cache=runner_cache)
                total_frames_processed += frames
                
                # Restore original format
                args.output_format = original_format

        elif input_type in ("video", "image"):
            # Auto-detect output format for single file if not specified
            if format_auto_detected:
                args.output_format = "mp4" if input_type == "video" else "png"
            
            # Caching: single-GPU streaming uses runner_cache, multi-GPU streaming workers cache internally
            runner_cache = None
            streaming = args.chunk_size > 0
            
            if args.cache_dit or args.cache_vae:
                if len(device_list) > 1:
                    if not streaming:
                        debug.log(
                            "Model caching requires streaming mode (--chunk_size > 0) for multi-GPU. "
                            "Disabling caching for this run.",
                            level="WARNING", category="cache", force=True
                        )
                        args.cache_dit = False
                        args.cache_vae = False
                elif streaming:
                    runner_cache = {}
                else:
                    debug.log(
                        "Model caching has no benefit for single file processing (only useful for directories or streaming mode). "
                        "Consider removing --cache_dit/--cache_vae for single files.",
                        category="tip", force=True
                    )
            
            frames = process_single_file(args.input, args, device_list, args.output,
                                        format_auto_detected=format_auto_detected,
                                        runner_cache=runner_cache)
            total_frames_processed += frames
        
        else:
            debug.log(f"Unsupported input type: {args.input}", level="ERROR", category="file", force=True)
            sys.exit(1)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        debug.log("", category="none", force=True)
        debug.log(f"All upscaling processes completed successfully in {total_time:.2f}s", category="success", force=True)
        
        # Calculate and display FPS based on overall wall-clock time
        if total_time > 0 and total_frames_processed > 0:
            fps = total_frames_processed / total_time
            debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)
        
    except Exception as e:
        debug.log(f"Error during processing: {e}", level="ERROR", category="generation", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        debug.log(f"Process {os.getpid()} terminating - VRAM will be automatically freed", category="cleanup", force=True)

        # print footer
        debug.print_footer()

if __name__ == "__main__":
    main()