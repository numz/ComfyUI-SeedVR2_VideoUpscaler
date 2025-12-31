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
import tempfile
import threading
import gc
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
import shutil
_psutil_missing_warned = False
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

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
# FFMPEG Class
# =============================================================================

class FFMPEGVideoWriter:
    """
    Video writer using ffmpeg subprocess for encoding with 10-bit support.
    
    Provides cv2.VideoWriter-compatible interface (write, isOpened, release) while
    using ffmpeg for encoding. Enables 10-bit output (yuv420p10le with x265) which
    reduces banding artifacts in gradients compared to 8-bit opencv output.
    
    Args:
        path: Output video file path
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        use_10bit: If True, uses x265 codec with yuv420p10le pixel format.
                   If False, uses x264 with yuv420p (default: False)
    
    Raises:
        RuntimeError: If ffmpeg is not found in system PATH
    
    Note:
        Frames must be passed to write() in BGR format (same as cv2.VideoWriter).
        Internally converts to RGB for ffmpeg rawvideo input.
    """
    
    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        use_10bit: bool = False,
        codec: Optional[str] = None,
        pix_fmt: Optional[str] = None,
        preset: str = "medium",
        crf: Optional[int] = None,
        bitrate: Optional[str] = None,
        input_pix_fmt: str = "rgb24"
    ):
        pix_fmt_effective = pix_fmt or ('yuv420p10le' if use_10bit else 'yuv420p')
        codec_effective = codec or ('libx265' if use_10bit else 'libx264')
        crf_effective = crf if crf is not None else 12
        self._input_dtype = np.uint16 if input_pix_fmt == "rgb48le" else np.uint8
        
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', input_pix_fmt,
            '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            '-c:v', codec_effective, '-pix_fmt', pix_fmt_effective, '-preset', preset
        ]
        if bitrate:
            cmd.extend(['-b:v', bitrate])
        else:
            cmd.extend(['-crf', str(crf_effective)])
        cmd.append(path)
        
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    
    def write(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.proc.stdin.write(frame_rgb.astype(self._input_dtype).tobytes())
    
    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
    
    def release(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
            stderr = self.proc.stderr.read() if self.proc.stderr else b''
            if self.proc.returncode != 0:
                debug.log(f"ffmpeg error: {stderr.decode()}", level="WARNING", category="file")
            self.proc = None


# =============================================================================
# Resilience Helpers
# =============================================================================

def _format_bytes(num_bytes: int) -> str:
    """
    Convert bytes to human-readable string.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _log_ram_usage(debug: Debug, label: str, force: bool = False) -> None:
    """
    Log current process RSS (RAM) and system usage if psutil is available.
    """
    global _psutil_missing_warned  # noqa: PLW0603
    if psutil is None:
        if not _psutil_missing_warned:
            debug.log("[RAM] psutil not installed; install psutil for RAM telemetry", category="memory", force=True)
            _psutil_missing_warned = True
        return
    try:
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        sys_mem = psutil.virtual_memory()
        debug.log(
            f"[RAM] {label}: rss={_format_bytes(rss)} | system_used={_format_bytes(sys_mem.used)}/{_format_bytes(sys_mem.total)} ({sys_mem.percent:.1f}%)",
            category="memory",
            force=force
        )
    except Exception:
        pass


def _start_memory_monitor(pids: List[int], debug: Debug, label: str = "workers", interval: float = 10.0):
    """
    Spawn a background thread that periodically logs RSS for a list of PIDs.
    """
    if psutil is None:
        _log_ram_usage(debug, f"{label} monitor skipped (psutil missing)", force=True)
        return None, None

    stop_event = threading.Event()

    def _monitor():
        while not stop_event.wait(interval):
            try:
                rss_total = 0
                per_proc = []
                for pid in pids:
                    try:
                        p = psutil.Process(pid)
                        rss = p.memory_info().rss
                        rss_total += rss
                        per_proc.append(f"{pid}:{_format_bytes(rss)}")
                    except psutil.NoSuchProcess:
                        continue
                sys_mem = psutil.virtual_memory()
                debug.log(
                    f"[RAM] {label} total={_format_bytes(rss_total)} | system_used={_format_bytes(sys_mem.used)}/{_format_bytes(sys_mem.total)} ({sys_mem.percent:.1f}%) | per_pid={', '.join(per_proc)}",
                    category="memory",
                    force=True
                )
            except Exception:
                continue

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return stop_event, t

def _is_oom_error(exc: BaseException) -> bool:
    """
    Detect CUDA/CPU OOM errors from common exception types/messages.
    """
    msg = str(exc).lower()
    return isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in msg


def _retry_with_cleanup(fn, debug: Debug, description: str = "operation", backoff_base: float = 1.0):
    """
    Retry a callable indefinitely on OOM, with aggressive cleanup and backoff.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_oom_error(exc):
                raise
            attempt += 1
            wait = min(30.0, backoff_base * attempt)
            debug.log(
                f"{description} failed with OOM (attempt {attempt}), cleaning up and retrying in {wait:.1f}s",
                level="WARNING",
                category="memory",
                force=True
            )
            clear_memory(debug=debug, deep=True, force=True, timer_name="oom_retry")
            time.sleep(wait)


def _save_chunk_with_retry(chunk_np: np.ndarray, chunk_path: str, debug: Debug, retries: int = 3) -> None:
    """
    Save numpy chunk to disk with retries and atomic rename to avoid partial files.
    """
    dir_path = os.path.dirname(chunk_path)
    os.makedirs(dir_path, exist_ok=True)
    tmp_path = chunk_path + ".tmp"
    bytes_needed = chunk_np.nbytes

    for attempt in range(1, retries + 1):
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, chunk_np, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, chunk_path)
            return
        except OSError as exc:
            free_bytes = shutil.disk_usage(dir_path).free if os.path.isdir(dir_path) else 0
            debug.log(
                f"Disk write failed for chunk ({bytes_needed/1e6:.1f} MB) to {chunk_path} "
                f"(attempt {attempt}/{retries}). Free space: {free_bytes/1e6:.1f} MB. Error: {exc}",
                level="WARNING",
                category="file",
                force=True
            )
            time.sleep(min(10, attempt * 2))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
    raise OSError(f"Failed to save chunk after {retries} attempts: {chunk_path}")


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
        if args.chunk_size <= 0 and frames_to_process > 300:
            # Auto-enable streaming to avoid unbounded RAM use
            chunk_size = min(300, frames_to_process)
            streaming = True
            debug.log(
                f"--chunk_size not set; auto-enabling streaming with chunk_size={chunk_size} to reduce RAM use",
                category="memory",
                force=True
            )
        else:
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
            spill_dir = args.spill_dir or tempfile.mkdtemp(prefix="seedvr2_spill_")
            debug.log(f"Multi-GPU streaming: spilling chunks to {spill_dir}", category="memory")
            created_temp_spill = args.spill_dir is None
            try:
                result = _gpu_processing(
                    None,
                    device_list,
                    args,
                    video_info=video_info,
                    spill_dir=spill_dir,
                    output_path=output_path,
                    fps=fps,
                    base_name=base_name
                )
                if isinstance(result, dict) and "frames_written" in result:
                    frames_written = result["frames_written"]
                else:
                    # Fallback to in-memory path if spill disabled or failed
                    if is_png:
                        save_frames_to_image(result, output_path, base_name)
                    else:
                        video_writer = save_frames_to_video(result, output_path, fps, 
                            video_backend=args.video_backend, use_10bit=args.use_10bit or args.output_bitdepth > 8,
                            codec=_resolve_codec(args), pix_fmt=_resolve_pix_fmt(args),
                            bitrate=args.video_bitrate, crf=args.video_crf, input_bitdepth=args.output_bitdepth,
                            preset=args.video_preset)
                        if video_writer is not None:
                            video_writer.release()
                    frames_written = result.shape[0]
            finally:
                if created_temp_spill:
                    shutil.rmtree(spill_dir, ignore_errors=True)
        
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
                    video_writer = save_frames_to_video(
                        result, output_path, fps, writer=video_writer,
                        video_backend=args.video_backend, use_10bit=args.use_10bit or args.output_bitdepth > 8,
                        codec=_resolve_codec(args), pix_fmt=_resolve_pix_fmt(args),
                        bitrate=args.video_bitrate, crf=args.video_crf, input_bitdepth=args.output_bitdepth,
                        preset=args.video_preset
                    )
                
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
            debug.log(f"{log_prefix}Chunk {chunk_idx}/{max(1, total_chunks)}: {new_frames.shape[0]} new + {context_count} context frames", 
                     category="generation", force=True)
            debug.log("", category="none", force=True)
        
        # RAM before processing
        _log_ram_usage(debug, f"{log_prefix}Chunk {chunk_idx} pre-process", force=True)
        
        # Process chunk
        result = _retry_with_cleanup(
            lambda: _process_frames_core(
                frames_tensor=frames.to(torch.float16),
                args=chunk_args,
                device_id=device_id,
                debug=debug,
                runner_cache=runner_cache
            ),
            debug=debug,
            description=f"{log_prefix}chunk processing"
        )
        
        _log_ram_usage(debug, f"{log_prefix}Chunk {chunk_idx} post-process (before context trim)", force=True)
        
        # Remove context frames from output
        if context_count > 0:
            result = result[context_count:]
        
        # Save tail for next chunk context
        prev_raw_tail = new_frames[-overlap:].clone() if overlap > 0 else None
        
        # Cleanup before yield
        del frames
        del new_frames
        gc.collect()
        _log_ram_usage(debug, f"{log_prefix}Chunk {chunk_idx} before yield (inputs freed)", force=True)
        
        yield result
        
        # Memory cleanup between chunks
        if streaming:
            clear_memory(debug=debug, deep=True, force=True, timer_name=cleanup_timer_name)
            _log_ram_usage(debug, f"{log_prefix}Chunk {chunk_idx} after cleanup", force=True)


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
    video_backend: str = "opencv",
    use_10bit: bool = False,
    input_bitdepth: int = 8,
    codec: Optional[str] = None,
    pix_fmt: Optional[str] = None,
    bitrate: Optional[str] = None,
    crf: Optional[int] = None,
    preset: str = "medium"
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
    bitdepth = max(1, input_bitdepth or 8)
    if video_backend != "ffmpeg":
        if input_bitdepth > 8:
            debug.log("output_bitdepth > 8 requires --video_backend ffmpeg. Falling back to 8-bit for OpenCV writer.", 
                      level="WARNING", category="file", force=True)
        bitdepth = 8  # OpenCV backend expects 8-bit BGR
    max_val = (1 << bitdepth) - 1 if bitdepth > 1 else 255
    dtype = np.uint8 if bitdepth <= 8 else np.uint16
    frames_np = (frames_tensor.cpu().numpy() * float(max_val)).round().astype(dtype)
    T, H, W, C = frames_np.shape
    input_pix_fmt = "rgb48le" if dtype == np.uint16 else "rgb24"
    
    if writer is None:
        debug.log(f"Saving {T} frames to video: {output_path} (backend={video_backend})", category="file")
        os.makedirs(Path(output_path).parent, exist_ok=True)
        if video_backend == "ffmpeg":
            writer = FFMPEGVideoWriter(
                output_path, W, H, fps,
                use_10bit=use_10bit,
                codec=codec,
                pix_fmt=pix_fmt,
                bitrate=bitrate,
                crf=crf,
                preset=preset,
                input_pix_fmt=input_pix_fmt
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            raise ValueError(f"Cannot create video writer for: {output_path}")
    
    for i, frame in enumerate(frames_np):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        if debug.enabled and (i + 1) % 100 == 0:
            debug.log(f"Written {i + 1}/{T} frames", category="file")
    
    return writer  # Caller always closes


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


def _convert_tensor_for_storage(frames_tensor: torch.Tensor, bitdepth: int) -> torch.Tensor:
    """
    Convert float tensor [0,1] to integer storage tensor for disk spill.
    """
    frames_clamped = torch.clamp(frames_tensor, 0.0, 1.0)
    if bitdepth <= 8:
        return (frames_clamped * 255.0).round().to(torch.uint8)
    max_val = float((1 << bitdepth) - 1)
    return (frames_clamped * max_val).round().to(torch.uint16)


def _load_spilled_chunk(chunk_path: str, bitdepth: int) -> np.ndarray:
    """
    Load a spilled chunk from disk and normalize to float32 [0,1].
    """
    chunk = np.load(chunk_path, mmap_mode="r")
    if np.issubdtype(chunk.dtype, np.integer):
        max_val = 255.0 if chunk.dtype == np.uint8 else float((1 << bitdepth) - 1)
        return (chunk.astype(np.float32) / max_val)
    return chunk.astype(np.float32)


def _resolve_pix_fmt(args: argparse.Namespace) -> Optional[str]:
    if args.video_pix_fmt:
        return args.video_pix_fmt
    if args.output_bitdepth == 8:
        return "yuv420p"
    if args.output_bitdepth == 10:
        return "yuv420p10le"
    return "yuv420p16le"


def _resolve_codec(args: argparse.Namespace) -> Optional[str]:
    if args.video_codec:
        return args.video_codec
    return "libx264" if args.output_bitdepth <= 8 else "libx265"


def _stitch_spilled_chunks(
    worker_chunks: Dict[int, List[str]],
    args: argparse.Namespace,
    fps: float,
    output_path: str,
    base_name: str,
    writer: Optional[cv2.VideoWriter] = None,
    pix_fmt: Optional[str] = None,
    codec: Optional[str] = None
) -> Tuple[int, Optional[cv2.VideoWriter]]:
    """
    Stitch spilled chunks from multiple workers with overlap blending and stream to disk.
    """
    frames_written = 0
    pending_tail: Optional[torch.Tensor] = None
    overlap = args.temporal_overlap
    last_worker_idx = max(worker_chunks.keys())
    pix_fmt = pix_fmt or _resolve_pix_fmt(args)
    codec = codec or _resolve_codec(args)
    frames_to_skip = args.prepend_frames

    def write_out(chunk_tensor: torch.Tensor) -> int:
        nonlocal writer, frames_to_skip
        if chunk_tensor.numel() == 0:
            return 0
        if frames_to_skip > 0:
            if chunk_tensor.shape[0] <= frames_to_skip:
                frames_to_skip -= chunk_tensor.shape[0]
                return 0
            chunk_tensor = chunk_tensor[frames_to_skip:]
            frames_to_skip = 0
        if args.output_format == "png":
            return save_frames_to_image(chunk_tensor, output_path, base_name, start_index=frames_written)
        _log_ram_usage(debug, "Stitch write_out pre-video", force=True)
        writer_local = save_frames_to_video(
            chunk_tensor,
            output_path,
            fps,
            writer=writer,
            video_backend=args.video_backend,
            use_10bit=args.use_10bit or args.output_bitdepth > 8,
            codec=codec,
            pix_fmt=pix_fmt,
            bitrate=args.video_bitrate,
            crf=args.video_crf,
            preset=args.video_preset,
            input_bitdepth=args.output_bitdepth
        )
        _log_ram_usage(debug, "Stitch write_out post-video", force=True)
        if writer is None:
            writer = writer_local
        return chunk_tensor.shape[0]

    for worker_idx in sorted(worker_chunks.keys()):
        chunk_paths = sorted(worker_chunks[worker_idx])
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            chunk_np = _load_spilled_chunk(chunk_path, args.output_bitdepth)
            chunk_tensor = torch.from_numpy(chunk_np)
            is_last_worker = worker_idx == last_worker_idx
            is_last_chunk = chunk_idx == len(chunk_paths) - 1

            if pending_tail is not None:
                if overlap > 0 and chunk_tensor.shape[0] > 0:
                    blend_len = min(overlap, pending_tail.shape[0], chunk_tensor.shape[0])
                    blended = blend_overlapping_frames(
                        pending_tail[-blend_len:],
                        chunk_tensor[:blend_len],
                        blend_len
                    )
                    frames_written += write_out(blended)
                    chunk_tensor = chunk_tensor[blend_len:]
                else:
                    frames_written += write_out(pending_tail)
                pending_tail = None

            if not is_last_worker and overlap > 0 and is_last_chunk:
                if chunk_tensor.shape[0] <= overlap:
                    pending_tail = chunk_tensor
                    chunk_body = chunk_tensor[:0]
                else:
                    pending_tail = chunk_tensor[-overlap:]
                    chunk_body = chunk_tensor[:-overlap]
            else:
                chunk_body = chunk_tensor

        if chunk_body.numel() > 0:
            frames_written += write_out(chunk_body)

            try:
                os.remove(chunk_path)
            except OSError:
                pass
            del chunk_tensor, chunk_np
            gc.collect()

    if pending_tail is not None:
        frames_written += write_out(pending_tail)

    return frames_written, writer


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
    _log_ram_usage(debug, f"Process start device {device_id}", force=True)
    debug.log(f"[SHAPE] input frames {tuple(frames_tensor.shape)}, dtype={frames_tensor.dtype}", category="memory", force=True)
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
    _log_ram_usage(debug, "After text embeddings", force=True)
    
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
    _log_ram_usage(debug, "After compute_generation_info", force=True)
    
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
    _log_ram_usage(debug, "After encode", force=True)
    
    # Phase 2: Upscale
    ctx = upscale_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        seed=args.seed,
        latent_noise_scale=args.latent_noise_scale,
        cache_model=cache_dit
    )
    _log_ram_usage(debug, "After upscale", force=True)
    
    # Phase 3: Decode
    ctx = decode_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        cache_model=cache_vae
    )
    _log_ram_usage(debug, "After decode", force=True)
    
    # Phase 4: Post-process
    ctx = postprocess_all_batches(
        ctx=ctx, debug=debug, progress_callback=None,
        color_correction=args.color_correction,
        prepend_frames=0,  # Worker mode handles this in main process
        temporal_overlap=args.temporal_overlap,
        batch_size=args.batch_size
    )
    _log_ram_usage(debug, "After postprocess", force=True)
    
    result_tensor = ctx['final_video']
    
    # Convert to CPU and compatible dtype
    if result_tensor.is_cuda or result_tensor.is_mps:
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
    
    _log_ram_usage(debug, "After CPU move/finalize", force=True)
    # Aggressively drop intermediate tensors from ctx to avoid accumulation across chunks
    keep_keys = {"cache_context", "dit_device", "vae_device", "dit_offload_device",
                 "vae_offload_device", "tensor_offload_device", "compute_dtype", "text_embeds"}
    for key in list(ctx.keys()):
        if key not in keep_keys:
            try:
                del ctx[key]
            except Exception:
                pass

    # If we are not caching runners, drop everything else as well
    if runner_cache is None:
        try:
            ctx.clear()
        except Exception:
            pass
    else:
        # Even with caching, drop cache_context/text_embeds to avoid CPU bloat between chunks
        for key in ("cache_context", "text_embeds"):
            try:
                if key in ctx:
                    del ctx[key]
            except Exception:
                pass

    clear_memory(debug=debug, deep=True, force=True, timer_name="ctx_cleanup")
    gc.collect()
    _log_ram_usage(debug, "After ctx cleanup", force=True)
    return result_tensor


def _worker_process(
    proc_idx: int, 
    device_id: str, 
    frames_np: Optional[np.ndarray],
    shared_args: Dict[str, Any], 
    return_queue: mp.Queue,
    done_barrier: Optional[mp.Barrier],
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
        
        # Disable runner caching per chunk to avoid CPU accumulation; reload models per chunk
        runner_cache = None
        
        total_chunks = (segment_frames + chunk_size - 1) // chunk_size
        chunk_paths: List[str] = []
        spill_root = shared_args.get("spill_dir")
        if spill_root is None:
            raise RuntimeError("spill_dir not provided for streaming worker")
        worker_spill_dir = os.path.join(spill_root, f"worker_{proc_idx}")
        os.makedirs(worker_spill_dir, exist_ok=True)

        for chunk_idx, result in enumerate(_stream_video_chunks(
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
            ), start=1):
            storage_tensor = _convert_tensor_for_storage(result.cpu(), args.output_bitdepth)
            chunk_path = os.path.join(worker_spill_dir, f"chunk_{chunk_idx:05d}.npy")
            try:
                _log_ram_usage(worker_debug, f"Worker {proc_idx} chunk {chunk_idx} pre-save ({storage_tensor.shape}, {storage_tensor.dtype})", force=True)
                _save_chunk_with_retry(storage_tensor.numpy(), chunk_path, worker_debug)
                chunk_paths.append(chunk_path)
                _log_ram_usage(worker_debug, f"Worker {proc_idx} chunk {chunk_idx} post-save", force=True)
            except Exception as exc:  # noqa: BLE001
                if return_queue is not None:
                    return_queue.put((proc_idx, {"error": str(exc)}))
                return
            finally:
                del result, storage_tensor
                clear_memory(debug=worker_debug, deep=True, force=True, timer_name="worker_chunk_cleanup")
                _log_ram_usage(worker_debug, f"Worker {proc_idx} chunk {chunk_idx} after cleanup", force=True)
                gc.collect()
        
        cap.release()
        if return_queue is not None:
            return_queue.put((proc_idx, chunk_paths))
        return
    
    # Pre-loaded frames mode (original behavior)
    else:
        frames_tensor = torch.from_numpy(frames_np).to(torch.float16)
        result_tensor = _retry_with_cleanup(
            lambda: _process_frames_core(
                frames_tensor=frames_tensor,
                args=args,
                device_id="0",
                debug=worker_debug,
                runner_cache=None
            ),
            debug=worker_debug,
            description="worker processing"
        )
        _log_ram_usage(worker_debug, f"Worker {proc_idx} post-process (pre-return)", force=True)
        clear_memory(debug=worker_debug, deep=True, force=True, timer_name="worker_post_process_cleanup")
    # Share tensor memory for efficient cross-process transfer (avoids pickling large arrays)
    if return_queue is not None:
        return_queue.put((proc_idx, result_tensor.share_memory_()))
    
    # Wait for parent to copy shared tensors before exiting
    # (shared memory requires creating process to stay alive during access)
    if done_barrier is not None:
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
    return _retry_with_cleanup(
        lambda: _process_frames_core(
            frames_tensor=frames_tensor,
            args=args,
            device_id=device_id,
            debug=debug,
            runner_cache=runner_cache
        ),
        debug=debug,
        description="single GPU processing"
    )


def _gpu_processing(
    frames_tensor: Optional[torch.Tensor],
    device_list: List[str], 
    args: argparse.Namespace,
    video_info: Optional[Dict[str, Any]] = None,
    spill_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    fps: float = 30.0,
    base_name: str = ""
) -> Any:
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
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1] or
        dict with {'frames_written': int} when streaming to disk
    """
    num_devices = len(device_list)
    overlap = args.temporal_overlap
    
    return_queue = mp.Queue(maxsize=0)
    done_barrier: Optional[mp.Barrier] = None if video_info is not None and spill_dir else mp.Barrier(num_devices + 1)
    workers = []
    shared_args = vars(args).copy()
    if spill_dir:
        shared_args["spill_dir"] = spill_dir
    
    # Video streaming mode: distribute frame ranges to workers
    recycle_every = max(1, getattr(args, "recycle_workers_every", 1))
    if video_info is not None and spill_dir:
        total_frames = video_info['frames_to_process']
        start_frame = video_info['start_frame']
        video_path = video_info['video_path']
        base_per_gpu = total_frames // num_devices
        remainder = total_frames % num_devices

        cycle_span = (args.chunk_size if args.chunk_size > 0 else max(1, base_per_gpu))
        cycle_span *= recycle_every

        # Build per-device segments
        device_states = []
        seg_start = start_frame
        for idx in range(num_devices):
            gpu_frames = base_per_gpu + (1 if idx < remainder else 0)
            base_end = seg_start + gpu_frames
            final_end = base_end + (overlap if idx < num_devices - 1 else 0)
            device_states.append({
                "idx": idx,
                "device": device_list[idx],
                "cursor": seg_start,
                "final_end": final_end,
                "start_base": seg_start,
                "total_frames": final_end - seg_start
            })
            seg_start = base_end

        frames_written_total = 0
        cycle_index = 0
        writer = None
        pix_fmt = _resolve_pix_fmt(args)
        codec = _resolve_codec(args)

        # Process cycles until all device segments are consumed
        while any(state["cursor"] < state["final_end"] for state in device_states):
            cycle_index += 1
            workers = []
            worker_chunks: Dict[int, List[str]] = {}

            # Spawn workers for devices that still have remaining frames
            for state in device_states:
                if state["cursor"] >= state["final_end"]:
                    continue
                start_cur = state["cursor"]
                end_cur = min(state["final_end"], start_cur + cycle_span)
                state["cursor"] = end_cur

                worker_video_info = {
                    'video_path': video_path,
                    'start_frame': start_cur,
                    'end_frame': end_cur,
                }

                os.environ["CUDA_VISIBLE_DEVICES"] = state["device"]
                p = mp.Process(
                    target=_worker_process,
                    args=(state["idx"], state["device"], None, shared_args, return_queue, done_barrier),
                    kwargs={'video_info': worker_video_info}
                )
                p.start()
                workers.append(p)

            monitor_stop = None
            monitor_thread = None
            if workers:
                monitor_stop, monitor_thread = _start_memory_monitor(
                    [p.pid for p in workers if p.pid],
                    debug,
                    label=f"workers_cycle_{cycle_index}",
                    interval=10.0
                )

            collected = 0
            while collected < len(workers):
                proc_idx, payload = return_queue.get()
                if isinstance(payload, dict) and "error" in payload:
                    raise RuntimeError(f"Worker {proc_idx} failed to spill chunk: {payload['error']}")
                worker_chunks[proc_idx] = payload
                collected += 1

            for p in workers:
                p.join()

            if monitor_stop:
                monitor_stop.set()
                if monitor_thread:
                    monitor_thread.join(timeout=2.0)

            _log_ram_usage(debug, f"Parent pre-stitch cycle {cycle_index}", force=True)
            frames_written, writer = _stitch_spilled_chunks(
                worker_chunks, args, fps, output_path, base_name, writer=writer, pix_fmt=pix_fmt, codec=codec
            )
            frames_written_total += frames_written
            _log_ram_usage(debug, f"Parent post-stitch cycle {cycle_index}", force=True)

        if writer is not None:
            writer.release()
        return {"frames_written": frames_written_total}
    
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
        monitor_stop, monitor_thread = _start_memory_monitor(
            [p.pid for p in workers if p.pid],
            debug,
            label="workers",
            interval=10.0
        )

    if video_info is not None and spill_dir:
        # This path is now handled in the loop above
        return {"frames_written": 0}

    # Collect results before joining to prevent deadlock (pre-loaded frames path)
    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, result_tensor = return_queue.get()
        results_np[proc_idx] = result_tensor.numpy()
        collected += 1
    
    # Release workers now that shared tensors are copied
    if done_barrier is not None:
        done_barrier.wait()
    if monitor_stop:
        monitor_stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=2.0)
    
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
    
  Streaming mode for long videos with 10-bit video output (requires FFMPEG):
    python {invocation} long_video.mp4 --resolution 1080 --batch_size 33 --chunk_size 330 --temporal_overlap 3 --video_backend ffmpeg --10bit

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
        help="Video encoder backend: 'opencv' (default) or 'ffmpeg' (requires ffmpeg in PATH)")
    io_group.add_argument("--10bit", dest="use_10bit", action="store_true",
                        help="Save 10-bit video with x265 codec (reduces banding). Without this flag, "
                         "ffmpeg uses x264 for maximum compatibility. Requires --video_backend ffmpeg")
    io_group.add_argument("--output_bitdepth", type=int, default=8, choices=[8, 10, 12, 16],
                        help="Bit depth for output frames. Influences spill-to-disk dtype and default ffmpeg pixel format. Default: 8")
    io_group.add_argument("--spill_dir", type=str, default=None,
                        help="Directory for spilling streamed chunks in multi-GPU mode. Default: system temp dir")
    io_group.add_argument("--video_codec", type=str, default=None,
                        help="Override video codec for ffmpeg backend (default: libx264 for <=8-bit, libx265 for >8-bit)")
    io_group.add_argument("--video_pix_fmt", type=str, default=None,
                        help="Override ffmpeg pixel format (default derives from output_bitdepth: yuv420p/yuv420p10le/yuv420p16le)")
    io_group.add_argument("--video_crf", type=int, default=None,
                        help="CRF value for ffmpeg (default: 12). Ignored if --video_bitrate is set")
    io_group.add_argument("--video_bitrate", type=str, default=None,
                        help="Bitrate target for ffmpeg, e.g., '20M'. Overrides CRF when provided")
    io_group.add_argument("--video_preset", type=str, default="medium",
                        help="ffmpeg preset (default: medium)")
    io_group.add_argument("--recycle_workers_every", type=int, default=1,
                        help="Chunks processed per worker before recycling (respawn). "
                             "Default: 1 (recycle every chunk). Increase to reuse workers across chunks at the cost of higher peak RAM.")
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
    
    # Validate ffmpeg availability if selected
    if args.video_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
        debug.log("--video_backend ffmpeg requires ffmpeg in PATH. Install ffmpeg or use --video_backend opencv", 
                 level="ERROR", category="setup", force=True)
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
