#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import argparse
import time
import platform
import multiprocessing as mp
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Final
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

if platform.system() != "Darwin":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--cuda_device", type=str, default=None)
    _pre_args, _ = _pre_parser.parse_known_args()
    if _pre_args.cuda_device is not None:
        device_list_env = [x.strip() for x in _pre_args.cuda_device.split(",") if x.strip() != ""]
        import torch as _torch_check
        if _torch_check.cuda.is_available():
            available_count = _torch_check.cuda.device_count()
            invalid_devices = [d for d in device_list_env if not d.isdigit() or int(d) >= available_count]
            if invalid_devices:
                print(f"❌ [ERROR] Invalid CUDA device ID(s): {', '.join(invalid_devices)}. Available: 0-{available_count-1}")
                sys.exit(1)
        else:
            print("❌ [ERROR] CUDA not available.")
            sys.exit(1)
        if len(device_list_env) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

import torch
import numpy as np

from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_dit_models, DEFAULT_DIT, DEFAULT_VAE
from src.utils.constants import SEEDVR2_FOLDER_NAME
from src.core.generation_utils import (
    setup_generation_context,
    prepare_runner,
    compute_generation_info,
    log_generation_start,
    blend_overlapping_frames,
)
from src.core.generation_phases import (
    encode_all_batches,
    upscale_all_batches,
    decode_all_batches,
    postprocess_all_batches,
)
from src.utils.debug import Debug

debug = Debug(enabled=False)

CODEC_PRESETS: Final[Dict[str, Dict[str, Any]]] = {
    "ffv1_yuv16": {
        "codec": "ffv1",
        "pix_fmt": "yuv444p16le",
        "container": "mkv",
        "extra_args": [
            "-context", "1",
            "-coder", "2",
            "-slicecrc", "1",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 16,
    },
    "ffv1_rgb16": {
        "codec": "ffv1",
        "pix_fmt": "gbrp16le",
        "container": "mkv",
        "extra_args": [
            "-context", "1",
            "-coder", "2",
            "-slicecrc", "1",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 16,
    },
    "prores_4444_12": {
        "codec": "prores_ks",
        "pix_fmt": "yuva444p12le",
        "container": "mov",
        "extra_args": [
            "-profile:v", "4",
            "-q:v", "0",
            "-vendor", "apl0",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 12,
    },
    "prores_4444_12_noalpha": {
        "codec": "prores_ks",
        "pix_fmt": "yuv444p12le",
        "container": "mov",
        "extra_args": [
            "-profile:v", "4",
            "-q:v", "0",
            "-vendor", "apl0",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 12,
    },
    "prores_4444xq_12": {
        "codec": "prores_ks",
        "pix_fmt": "yuva444p12le",
        "container": "mov",
        "extra_args": [
            "-profile:v", "5",
            "-q:v", "0",
            "-vendor", "apl0",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 12,
    },
    "prores_4444xq_12_noalpha": {
        "codec": "prores_ks",
        "pix_fmt": "yuv444p12le",
        "container": "mov",
        "extra_args": [
            "-profile:v", "5",
            "-q:v", "0",
            "-vendor", "apl0",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 12,
    },
    "h265_444_12_lossless": {
        "codec": "libx265",
        "pix_fmt": "yuv444p12le",
        "container": "mkv",
        "extra_args": [
            "-preset", "fast",
            "-crf", "0",
            "-x265-params", "profile=main444-12",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 12,
    },
    "h265_444_12_crf10": {
        "codec": "libx265",
        "pix_fmt": "yuv444p12le",
        "container": "mkv",
        "extra_args": [
            "-preset", "fast",
            "-crf", "10",
            "-tune", "psnr",
            "-x265-params", "profile=main444-12:aq-mode=1:rect=1:amp=1",
        ],
        "color": {
            "primaries": "bt709",
            "transfer": "bt709",
            "colorspace": "bt709",
            "range": "tv",
        },
        "max_bit_depth": 12,
    },
}

def _get_platform_type() -> str:
    if platform.system() == "Darwin":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _device_id_to_name(device_id: str, platform_type: Optional[str] = None) -> str:
    if device_id in ("cpu", "none"):
        return device_id
    if platform_type is None:
        platform_type = _get_platform_type()
    if platform_type == "mps":
        return "mps"
    return f"{platform_type}:{device_id}"


def _parse_offload_device(offload_arg: str, platform_type: Optional[str] = None, cache_enabled: bool = False) -> Optional[str]:
    if offload_arg == "none":
        return "cpu" if cache_enabled else None
    if offload_arg == "cpu":
        return "cpu"
    if ":" in offload_arg:
        return offload_arg
    return _device_id_to_name(offload_arg, platform_type)


def verify_bit_depth(video_path: str, expected_depth: int, codec_name: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt,bits_per_raw_sample",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        if not lines:
            debug.log(f"Cannot read pixel format from {video_path}", level="WARNING", category="verification", force=True)
            return False
        pix_fmt = lines[0].strip()
        bits_per_sample: Optional[int] = None
        if len(lines) > 1 and lines[1].strip() and lines[1].strip() != "N/A":
            try:
                bits_per_sample = int(lines[1].strip())
            except ValueError:
                bits_per_sample = None
        if bits_per_sample is None:
            import re
            match = re.search(r"p(\d+)", pix_fmt)
            if match:
                bits_per_sample = int(match.group(1))
        if bits_per_sample is None:
            debug.log(f"Cannot determine bit depth for {codec_name} ({pix_fmt})", level="WARNING", category="verification", force=True)
            return False
        status = "✓" if bits_per_sample >= expected_depth else "✗"
        debug.log(f"{status} {codec_name}: {bits_per_sample}-bit ({pix_fmt}), expected {expected_depth}-bit", category="verification", force=True)
        return bits_per_sample >= expected_depth
    except subprocess.TimeoutExpired:
        debug.log(f"FFprobe timeout for {video_path}", level="WARNING", category="verification", force=True)
        return False
    except Exception as e:
        debug.log(f"Bit depth verification failed: {e}", level="WARNING", category="verification", force=True)
        return False


def get_video_files(directory: str) -> List[str]:
    all_files = [f for f in Path(directory).iterdir() if f.is_file()]
    video_files: List[str] = []
    for file_path in all_files:
        if is_video_file(str(file_path)):
            video_files.append(str(file_path))
    debug.log(f"Found {len(video_files)} videos out of {len(all_files)} files", category="ffmpeg", force=True)
    return sorted(video_files)


def is_video_file(file_path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() == "video"
    except Exception as e:
        debug.log(f"FFprobe validation failed for {file_path}: {e}", level="WARNING", category="ffmpeg", force=True)
        return False


def get_average_fps(video_path: str) -> Tuple[str, float]:
    def _read_rate(field: str) -> Optional[str]:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={field}", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True,
            text=True,
            check=True,
        )
        s = r.stdout.strip()
        return s if s and s not in ("0/0", "N/A") else None
    try:
        for key in ("avg_frame_rate", "r_frame_rate"):
            fps_str = _read_rate(key)
            if fps_str:
                num, den = fps_str.split("/")
                den_i = int(den)
                if den_i != 0:
                    fps = float(num) / den_i
                    return fps_str, fps
        return "30/1", 30.0
    except Exception as e:
        raise RuntimeError(f"Failed to detect FPS for {video_path}: {e}")


def get_video_geometry(video_path: str) -> Tuple[int, int]:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        s = r.stdout.strip()
        if "x" not in s:
            raise RuntimeError(f"Invalid geometry: {s}")
        w_str, h_str = s.split("x")
        return int(w_str), int(h_str)
    except Exception as e:
        raise RuntimeError(f"Failed to get geometry for {video_path}: {e}")


def get_audio_info(video_path: str) -> Optional[Tuple[int, int]]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "csv=p=0:s=,",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        output = result.stdout.strip()
        if not output or output == "N/A" or "," not in output:
            return None
        parts = output.split(",")
        if len(parts) != 2:
            return None
        sample_rate = int(parts[0])
        channels = int(parts[1])
        return (sample_rate, channels)
    except Exception:
        return None


def _container_flags_for_preset(container: str) -> List[str]:
    if container.lower() in ("mov", "mp4"):
        return ["-movflags", "+write_colr+faststart"]
    return []


def _color_metadata_args_from_dict(color_info: Dict[str, str]) -> List[str]:
    args: List[str] = []
    if color_info.get("range"):
        args += ["-color_range", color_info["range"]]
    if color_info.get("colorspace"):
        args += ["-colorspace", color_info["colorspace"]]
    if color_info.get("primaries"):
        args += ["-color_primaries", color_info["primaries"]]
    if color_info.get("transfer"):
        args += ["-color_trc", color_info["transfer"]]
    return args


def _color_metadata_args_for_preset(preset: Dict[str, Any]) -> List[str]:
    if "color" in preset and isinstance(preset["color"], dict):
        return _color_metadata_args_from_dict(preset["color"])
    return []


_ENCODER_PIXFMTS_CACHE: Dict[str, List[str]] = {}

def _encoder_supported_pix_fmts(encoder: str) -> List[str]:
    if encoder in _ENCODER_PIXFMTS_CACHE:
        return _ENCODER_PIXFMTS_CACHE[encoder]
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"encoder={encoder}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = r.stdout.splitlines()
        fmts: List[str] = []
        for ln in lines:
            low = ln.lower().strip()
            if "supported pixel formats" in low or "pixel formats" in low:
                parts = ln.split(":")
                if len(parts) > 1:
                    fmts = [x.strip() for x in parts[1].split() if x.strip()]
                    break
        _ENCODER_PIXFMTS_CACHE[encoder] = fmts
        return fmts
    except Exception:
        return []


def _maybe_fallback_prores_pix_fmt(pix_fmt: str) -> str:
    if not pix_fmt.endswith("12le"):
        return pix_fmt
    fmts = _encoder_supported_pix_fmts("prores_ks")
    if not fmts:
        return pix_fmt
    if pix_fmt in fmts:
        return pix_fmt
    fallback = pix_fmt.replace("12le", "10le")
    return fallback if fallback in fmts else pix_fmt


def read_audio_raw(
    video_path: str,
    fps_rational: str,
    num_frames: int,
    skip_first_frames: int,
    audio_info: Tuple[int, int],
) -> np.ndarray:
    sample_rate, channels = audio_info
    fps_num, fps_den = map(int, fps_rational.split("/"))
    fps_float = fps_num / fps_den if fps_den != 0 else 30.0
    
    start_time = skip_first_frames / fps_float if skip_first_frames > 0 else 0.0
    duration = num_frames / fps_float
    
    cmd = [
        "ffmpeg", "-v", "error", "-hide_banner", "-nostats", "-nostdin",
        "-ss", f"{start_time:.9f}",
        "-t", f"{duration:.9f}",
        "-i", video_path,
        "-map", "0:a:0",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "pipe:1"
    ]
    
    debug.log(f"Reading raw audio: {sample_rate}Hz, {channels}ch, {duration:.3f}s", category="ffmpeg", force=True)
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    
    audio_data = proc.stdout.read()
    proc.wait(timeout=30)
    
    if not audio_data:
        raise RuntimeError("No audio data received from FFmpeg")
    
    audio_array = np.frombuffer(audio_data, dtype=np.float32)
    
    if channels > 1:
        total_samples = len(audio_array)
        if total_samples % channels != 0:
            trim_to = (total_samples // channels) * channels
            audio_array = audio_array[:trim_to]
        audio_array = audio_array.reshape(-1, channels)
    
    debug.log(f"Audio loaded: {audio_array.shape} samples", category="ffmpeg", force=True)
    
    return audio_array


def read_frames_and_audio_raw16(
    video_path: str,
    fps_rational: str,
    skip_first_frames: int = 0,
    load_cap: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[np.ndarray], Tuple[int, int], Optional[Tuple[int, int]]]:
    
    w, h = get_video_geometry(video_path)
    frame_size = w * h * 3 * 2
    vf_filters: List[str] = []
    if skip_first_frames > 0:
        vf_filters.append(f"select=gte(n\\,{skip_first_frames})")
    cmd = ["ffmpeg", "-v", "error", "-hide_banner", "-nostats", "-nostdin", "-threads", "0", "-i", video_path, "-map", "0:v:0"]
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    if load_cap and load_cap > 0:
        cmd += ["-frames:v", str(load_cap)]
    cmd += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"]
    debug.log("Streaming 16-bit video frames via FFmpeg -> Python", category="ffmpeg", force=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frames: List[np.ndarray] = []
    read_count = 0
    try:
        while True:
            buf = b""
            remaining = frame_size
            while remaining > 0:
                chunk = proc.stdout.read(remaining)
                if not chunk:
                    break
                buf += chunk
                remaining -= len(chunk)
            if len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype="<u2").reshape((h, w, 3))
            frames.append((arr.astype(np.float32) / 65535.0))
            read_count += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=10)
    if not frames:
        raise RuntimeError("No frames received from FFmpeg rawvideo pipe")
    frames_tensor = torch.from_numpy(np.stack(frames, axis=0)).to(torch.float32)
    debug.log(f"Loaded {read_count} video frames @ {w}x{h}, 16-bit -> float32", category="ffmpeg", force=True)
    
    audio_info = get_audio_info(video_path)
    audio_array = None
    
    if audio_info is not None:
        try:
            audio_array = read_audio_raw(video_path, fps_rational, read_count, skip_first_frames, audio_info)
        except Exception as e:
            debug.log(f"Audio read failed: {e}", level="WARNING", category="ffmpeg", force=True)
            audio_info = None
    
    return frames_tensor, audio_array, (w, h), audio_info
    
def write_frames_and_audio_raw16_to_ffmpeg(
    frames_tensor: torch.Tensor,
    audio_array: Optional[np.ndarray],
    audio_info: Optional[Tuple[int, int]],
    output_path: str,
    fps_rational: str,
    codec_preset: str,
) -> str:
    if codec_preset not in CODEC_PRESETS:
        raise ValueError(f"Unknown codec preset: {codec_preset}. Available: {list(CODEC_PRESETS.keys())}")
    preset = CODEC_PRESETS[codec_preset]
    h, w = int(frames_tensor.shape[1]), int(frames_tensor.shape[2])
    out_path_obj = Path(output_path)
    if out_path_obj.suffix:
        out_path_obj = out_path_obj.with_suffix("")
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_final = str(out_path_obj.with_suffix(f".{preset['container']}"))
    pix_fmt = preset["pix_fmt"]
    if preset["codec"] == "prores_ks":
        pix_fmt = _maybe_fallback_prores_pix_fmt(pix_fmt)
    color_args = _color_metadata_args_for_preset(preset)
    container_flags = _container_flags_for_preset(preset["container"])
    
    audio_temp_file = None
    if audio_array is not None and audio_info is not None:
        sample_rate, channels = audio_info
        audio_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".raw")
        audio_array.astype(np.float32).tofile(audio_temp_file.name)
        audio_temp_file.close()
        debug.log(f"Audio temp file: {audio_temp_file.name}, {sample_rate}Hz, {channels}ch", category="ffmpeg")
    
    cmd = [
        "ffmpeg", "-v", "error", "-hide_banner", "-nostats", "-nostdin", "-y",
        "-fflags", "+genpts",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-video_size", f"{w}x{h}",
        "-framerate", fps_rational,
        "-i", "pipe:0",
    ]
    
    if audio_temp_file is not None:
        sample_rate, channels = audio_info
        cmd += [
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-i", audio_temp_file.name,
        ]
    
    cmd += ["-map", "0:v:0"]
    
    if audio_temp_file is not None:
        cmd += ["-map", "1:a:0"]
    
    cmd += [
        "-shortest",
        "-vsync", "cfr",
        "-c:v", preset["codec"],
        "-pix_fmt", pix_fmt,
    ]
    
    if audio_temp_file is not None:
        cmd += ["-c:a", "pcm_s24le"]
    
    cmd += list(preset["extra_args"]) + color_args + container_flags + [out_path_final]
    
    debug.log(f"Encoding -> {codec_preset} to {out_path_final}", category="ffmpeg", force=True)
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        arr = frames_tensor.detach().cpu().numpy()
        arr_u16 = np.clip(np.round(arr * 65535.0), 0, 65535).astype("<u2")
        for frame in arr_u16:
            proc.stdin.write(frame.tobytes(order="C"))
    finally:
        try:
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass
    retcode = proc.wait()
    
    if audio_temp_file is not None:
        try:
            os.unlink(audio_temp_file.name)
        except Exception:
            pass
    
    if retcode != 0:
        raise RuntimeError(f"FFmpeg encoding failed with code {retcode}")
    debug.log(f"Created: {out_path_final}", category="success", force=True)
    verify_bit_depth(out_path_final, preset.get("max_bit_depth", 16), codec_preset)
    return out_path_final


def _process_frames_core(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    debug: Debug,
    runner_cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    platform_type = _get_platform_type()
    inference_device = _device_id_to_name(device_id, platform_type)
    cache_dit = args.cache_dit if runner_cache is not None else False
    cache_vae = args.cache_vae if runner_cache is not None else False
    dit_offload = _parse_offload_device(args.dit_offload_device, platform_type, cache_dit)
    vae_offload = _parse_offload_device(args.vae_offload_device, platform_type, cache_vae)
    tensor_offload = _parse_offload_device(args.tensor_offload_device, platform_type, False)

    if runner_cache is not None and "ctx" in runner_cache:
        ctx = runner_cache["ctx"]
        keys_to_keep = {
            "dit_device",
            "vae_device",
            "dit_offload_device",
            "vae_offload_device",
            "tensor_offload_device",
            "compute_dtype",
        }
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
            debug=debug,
        )
        if runner_cache is not None:
            runner_cache["ctx"] = ctx

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

    model_dir = args.model_dir if args.model_dir is not None else f"./models/{SEEDVR2_FOLDER_NAME}"
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
            "blocks_to_swap": args.blocks_to_swap,
            "swap_io_components": args.swap_io_components,
            "offload_device": dit_offload,
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
        torch_compile_args_vae=torch_compile_args_vae,
    )
    ctx["cache_context"] = cache_context
    if runner_cache is not None:
        runner_cache["runner"] = runner

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
        debug=debug,
    )
    log_generation_start(gen_info, debug)

    ctx = encode_all_batches(
        runner,
        ctx=ctx,
        images=frames_tensor,
        debug=debug,
        batch_size=args.batch_size,
        uniform_batch_size=args.uniform_batch_size,
        seed=args.seed,
        progress_callback=None,
        temporal_overlap=args.temporal_overlap,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        input_noise_scale=args.input_noise_scale,
        color_correction=args.color_correction,
    )
    ctx = upscale_all_batches(
        runner,
        ctx=ctx,
        debug=debug,
        progress_callback=None,
        seed=args.seed,
        latent_noise_scale=args.latent_noise_scale,
        cache_model=cache_dit,
    )
    ctx = decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None, cache_model=cache_vae)
    ctx = postprocess_all_batches(
        ctx=ctx,
        debug=debug,
        progress_callback=None,
        color_correction=args.color_correction,
        prepend_frames=0,
        temporal_overlap=args.temporal_overlap,
        batch_size=args.batch_size,
    )
    result_tensor = ctx["final_video"]
    if result_tensor.device.type in ("cuda", "mps"):
        result_tensor = result_tensor.cpu()
    float8_e4 = getattr(torch, "float8_e4m3fn", None)
    float8_e5 = getattr(torch, "float8_e5m2", None)
    float8_types = tuple([t for t in (float8_e4, float8_e5) if t is not None])
    if result_tensor.dtype in ((torch.bfloat16,) + float8_types):
        result_tensor = result_tensor.to(torch.float32)
    return result_tensor


def _worker_process(
    proc_idx: int,
    device_id: str,
    frames_np: np.ndarray,
    shared_args: Dict[str, Any],
    return_queue: mp.Queue,
) -> None:
    if platform.system() != "Darwin":
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    import torch as _torch_check
    worker_debug = Debug(enabled=shared_args["debug"])
    frames_tensor = _torch_check.from_numpy(frames_np)
    args = argparse.Namespace(**shared_args)
    result_tensor = _process_frames_core(
        frames_tensor=frames_tensor,
        args=args,
        device_id="0",
        debug=worker_debug,
        runner_cache=None,
    )
    return_queue.put((proc_idx, result_tensor.numpy()))


def _single_gpu_direct_processing(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    runner_cache: Dict[str, Any],
) -> torch.Tensor:
    return _process_frames_core(
        frames_tensor=frames_tensor,
        args=args,
        device_id=device_id,
        debug=debug,
        runner_cache=runner_cache,
    )


def _gpu_processing(
    frames_tensor: torch.Tensor,
    device_list: List[str],
    args: argparse.Namespace,
) -> torch.Tensor:
    num_devices = len(device_list)
    total_frames = frames_tensor.shape[0]
    if args.temporal_overlap > 0 and num_devices > 1:
        chunk_with_overlap = total_frames // num_devices + args.temporal_overlap
        if args.batch_size > 1:
            chunk_with_overlap = ((chunk_with_overlap + args.batch_size - 1) // args.batch_size) * args.batch_size
        base_chunk_size = max(1, chunk_with_overlap - args.temporal_overlap)
        chunks: List[torch.Tensor] = []
        for i in range(num_devices):
            start_idx = i * base_chunk_size
            if start_idx >= total_frames:
                break
            end_idx = total_frames if i == num_devices - 1 else min(start_idx + chunk_with_overlap, total_frames)
            chunks.append(frames_tensor[start_idx:end_idx])
    else:
        chunks = [c for c in torch.chunk(frames_tensor, num_devices, dim=0) if c.shape[0] > 0]
    num_workers = len(chunks)
    if num_workers == 0:
        return torch.empty((0,) + tuple(frames_tensor.shape[1:]), dtype=torch.float32)

    return_queue: mp.Queue = mp.Queue(maxsize=0)
    workers: List[mp.Process] = []
    shared_args = vars(args).copy()

    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)

    results_np: List[Optional[np.ndarray]] = [None] * num_workers
    collected = 0
    while collected < num_workers:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1

    for p in workers:
        p.join()

    if args.temporal_overlap > 0 and num_workers > 1:
        overlap = args.temporal_overlap
        result_tensor: Optional[torch.Tensor] = None
        for idx, res_np in enumerate(results_np):
            assert res_np is not None
            chunk_tensor = torch.from_numpy(res_np).to(torch.float32)
            if idx == 0:
                result_tensor = chunk_tensor
            else:
                if chunk_tensor.shape[0] > overlap and result_tensor.shape[0] >= overlap:
                    prev_tail = result_tensor[-overlap:]
                    cur_head = chunk_tensor[:overlap]
                    blended = blend_overlapping_frames(prev_tail, cur_head, overlap)
                    result_tensor = torch.cat([result_tensor[:-overlap], blended, chunk_tensor[overlap:]], dim=0)
                else:
                    if chunk_tensor.shape[0] > overlap:
                        result_tensor = torch.cat([result_tensor, chunk_tensor[overlap:]], dim=0)
        if result_tensor is None:
            result_tensor = torch.from_numpy(results_np[0]).to(torch.float32)
    else:
        result_tensor = torch.from_numpy(np.concatenate([x for x in results_np if x is not None], axis=0)).to(torch.float32)

    if args.prepend_frames > 0:
        if args.prepend_frames < result_tensor.shape[0]:
            debug.log(f"Removing {args.prepend_frames} prepended frames", category="generation")
            result_tensor = result_tensor[args.prepend_frames:]
        else:
            debug.log(f"prepend_frames ({args.prepend_frames}) >= total frames ({result_tensor.shape[0]})", level="WARNING", category="generation", force=True)
    return result_tensor


def process_single_video(
    input_path: str,
    args: argparse.Namespace,
    device_list: List[str],
    output_path: Optional[str] = None,
    runner_cache: Optional[Dict[str, Any]] = None,
) -> int:
    input_path_p = Path(input_path)
    debug.log(f"Processing video: {input_path_p.name}", category="generation", force=True)
    fps_str, fps_float = get_average_fps(input_path)
    frames_tensor, audio_array, _wh, audio_info = read_frames_and_audio_raw16(
        video_path=input_path,
        fps_rational=fps_str,
        skip_first_frames=args.skip_first_frames,
        load_cap=args.load_cap,
    )
    input_frame_count = frames_tensor.shape[0]
    processing_start = time.time()

    if runner_cache is not None:
        result = _single_gpu_direct_processing(frames_tensor, args, device_list[0], runner_cache)
        single_gpu = True
    else:
        if len(device_list) == 1:
            result = _single_gpu_direct_processing(frames_tensor, args, device_list[0], runner_cache={})
            single_gpu = True
        else:
            result = _gpu_processing(frames_tensor, device_list, args)
            single_gpu = False

    debug.log(f"Processing time: {time.time() - processing_start:.2f}s", category="timing")

    if single_gpu and args.prepend_frames > 0:
        if args.prepend_frames < result.shape[0]:
            debug.log(f"Removing {args.prepend_frames} prepended frames", category="generation")
            result = result[args.prepend_frames:]
        else:
            debug.log(f"prepend_frames ({args.prepend_frames}) >= total frames ({result.shape[0]})", level="WARNING", category="generation", force=True)

    if output_path is None:
        output_path = f"output/{input_path_p.stem}_upscaled"

    final_video_path = write_frames_and_audio_raw16_to_ffmpeg(
        frames_tensor=result,
        audio_array=audio_array,
        audio_info=audio_info,
        output_path=output_path,
        fps_rational=fps_str,
        codec_preset=args.codec,
    )
    debug.log(f"Output: {final_video_path}", category="file", force=True)
    return input_frame_count


def parse_arguments() -> argparse.Namespace:
    invocation = sys.argv[0]
    usage_examples = f"""
Examples:
  FFV1 16-bit SDR BT.709 (YUV 4:4:4):
    python {invocation} video.mp4 --codec ffv1_yuv16

  FFV1 16-bit SDR BT.709 (RGB 4:4:4):
    python {invocation} video.mp4 --codec ffv1_rgb16

  ProRes 4444 12-bit SDR (sa alpha):
    python {invocation} video.mp4 --codec prores_4444_12

  ProRes 4444 12-bit SDR (bez alpha):
    python {invocation} video.mp4 --codec prores_4444_12_noalpha

  ProRes 4444XQ 12-bit SDR:
    python {invocation} video.mp4 --codec prores_4444xq_12

  HEVC 4:4:4 12-bit SDR lossless:
    python {invocation} video.mp4 --codec h265_444_12_lossless

  HEVC 4:4:4 12-bit SDR CRF 10:
    python {invocation} video.mp4 --codec h265_444_12_crf10
"""
    parser = argparse.ArgumentParser(
        description="SeedVR2 Video Upscaler - SDR BT.709 16-bit pipeline with raw audio",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("input", type=str, help="Input video file or directory")
    io_group.add_argument("--output", type=str, default=None, help="Output base path")
    io_group.add_argument("--codec", type=str, default="ffv1_rgb16", choices=list(CODEC_PRESETS.keys()), help="Video codec preset")
    io_group.add_argument("--model_dir", type=str, default=None, help=f"Model directory (default: ./models/{SEEDVR2_FOLDER_NAME})")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--dit_model", type=str, default=DEFAULT_DIT, choices=get_available_dit_models(), help="DiT model")

    process_group = parser.add_argument_group("Processing")
    process_group.add_argument("--resolution", type=int, default=1080, help="Target resolution")
    process_group.add_argument("--max_resolution", type=int, default=0, help="Max resolution limit, 0=unlimited")
    process_group.add_argument("--batch_size", type=int, default=5, help="Frames per batch (4n+1 pattern)")
    process_group.add_argument("--uniform_batch_size", action="store_true", help="Pad final batch to batch_size")
    process_group.add_argument("--seed", type=int, default=42, help="Random seed")
    process_group.add_argument("--skip_first_frames", type=int, default=0, help="Skip N initial frames")
    process_group.add_argument("--load_cap", type=int, default=0, help="Max frames to load, 0=all")
    process_group.add_argument("--prepend_frames", type=int, default=0, help="Prepend N reversed frames")
    process_group.add_argument("--temporal_overlap", type=int, default=0, help="Overlap frames for blending")

    quality_group = parser.add_argument_group("Quality")
    quality_group.add_argument("--color_correction", type=str, default="lab", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
    quality_group.add_argument("--input_noise_scale", type=float, default=0.0, help="Input noise scale 0.0-1.0")
    quality_group.add_argument("--latent_noise_scale", type=float, default=0.0, help="Latent noise scale 0.0-1.0")

    device_group = parser.add_argument_group("Device")
    if platform.system() != "Darwin":
        device_group.add_argument("--cuda_device", type=str, default=None, help="CUDA device(s): '0' or '0,1,2'")
    device_group.add_argument("--dit_offload_device", type=str, default="none", help="DiT offload: 'none'/'cpu'/GPU_ID")
    device_group.add_argument("--vae_offload_device", type=str, default="none", help="VAE offload: 'none'/'cpu'/GPU_ID")
    device_group.add_argument("--tensor_offload_device", type=str, default="cpu", help="Tensor storage: 'cpu'/'none'/GPU_ID")

    memory_group = parser.add_argument_group("Memory (BlockSwap)")
    memory_group.add_argument("--blocks_to_swap", type=int, default=0, help="Blocks to swap 0-32/36")
    memory_group.add_argument("--swap_io_components", action="store_true", help="Offload DiT I/O layers")

    vae_group = parser.add_argument_group("VAE Tiling")
    vae_group.add_argument("--vae_encode_tiled", action="store_true", help="Enable VAE encode tiling")
    vae_group.add_argument("--vae_encode_tile_size", type=int, default=1024, help="Encode tile size")
    vae_group.add_argument("--vae_encode_tile_overlap", type=int, default=128, help="Encode tile overlap")
    vae_group.add_argument("--vae_decode_tiled", action="store_true", help="Enable VAE decode tiling")
    vae_group.add_argument("--vae_decode_tile_size", type=int, default=1024, help="Decode tile size")
    vae_group.add_argument("--vae_decode_tile_overlap", type=int, default=128, help="Decode tile overlap")
    vae_group.add_argument("--tile_debug", type=str, default="false", choices=["false", "encode", "decode"], help="Tile visualization")

    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--attention_mode", type=str, default="sdpa", choices=["sdpa", "flash_attn"], help="Attention backend")
    perf_group.add_argument("--compile_dit", action="store_true", help="Enable torch.compile for DiT")
    perf_group.add_argument("--compile_vae", action="store_true", help="Enable torch.compile for VAE")
    perf_group.add_argument("--compile_backend", type=str, default="inductor", choices=["inductor", "cudagraphs"], help="Compile backend")
    perf_group.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    perf_group.add_argument("--compile_fullgraph", action="store_true", help="Full graph compilation")
    perf_group.add_argument("--compile_dynamic", action="store_true", help="Dynamic shape support")
    perf_group.add_argument("--compile_dynamo_cache_size_limit", type=int, default=64, help="Dynamo cache limit")
    perf_group.add_argument("--compile_dynamo_recompile_limit", type=int, default=128, help="Recompile limit")

    cache_group = parser.add_argument_group("Caching")
    cache_group.add_argument("--cache_dit", action="store_true", help="Cache DiT (single GPU only)")
    cache_group.add_argument("--cache_vae", action="store_true", help="Cache VAE (single GPU only)")

    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--debug", action="store_true", help="Verbose logging")

    if len(sys.argv) == 1:
        sys.argv.append("--help")
    return parser.parse_args()


def main() -> None:
    debug.print_header(cli=True)
    args = parse_arguments()
    debug.enabled = args.debug
    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"{key}: {value}", category="none", indent_level=1)

    if args.vae_encode_tiled and args.vae_encode_tile_overlap >= args.vae_encode_tile_size:
        debug.log("VAE encode overlap must be < tile size", level="ERROR", category="vae", force=True)
        sys.exit(1)
    if args.vae_decode_tiled and args.vae_decode_tile_overlap >= args.vae_decode_tile_size:
        debug.log("VAE decode overlap must be < tile size", level="ERROR", category="vae", force=True)
        sys.exit(1)

    blockswap_enabled = args.blocks_to_swap > 0 or args.swap_io_components
    if blockswap_enabled and args.dit_offload_device == "none":
        debug.log("BlockSwap requires --dit_offload_device", level="ERROR", category="blockswap", force=True)
        sys.exit(1)
    if args.cache_dit and args.dit_offload_device == "none":
        debug.log("DiT caching: using default CPU offload", category="cache", force=True)
    if args.cache_vae and args.vae_offload_device == "none":
        debug.log("VAE caching: using default CPU offload", category="cache", force=True)

    if args.debug:
        if platform.system() == "Darwin":
            debug.log("Running on macOS with MPS backend", category="info", force=True)
        else:
            debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}", category="device")
            if torch.cuda.is_available():
                debug.log(f"CUDA devices: {torch.cuda.device_count()}", category="device")

    try:
        start_time = time.time()
        if platform.system() == "Darwin":
            device_list = ["0"]
        else:
            if args.cuda_device:
                device_list = [d.strip() for d in str(args.cuda_device).split(",") if d.strip()]
            else:
                device_list = ["0"]

        if args.debug:
            debug.log(f"Using devices: {device_list}", category="device")

        model_dir = args.model_dir if args.model_dir else f"./models/{SEEDVR2_FOLDER_NAME}"
        if not download_weight(
            dit_model=args.dit_model,
            vae_model=DEFAULT_VAE,
            model_dir=model_dir,
            debug=debug,
        ):
            debug.log("Model download failed", level="ERROR", category="download", force=True)
            sys.exit(1)

        total_frames_processed = 0
        input_path = Path(args.input)

        if input_path.is_dir():
            video_files = get_video_files(str(input_path))
            valid_videos = [f for f in video_files if is_video_file(f)]
            if not valid_videos:
                debug.log(f"No valid videos in: {input_path}", level="ERROR", category="file", force=True)
                sys.exit(1)
            debug.log(f"Found {len(valid_videos)} videos", category="file", force=True)

            if (args.cache_dit or args.cache_vae) and len(device_list) > 1:
                debug.log("Caching requires single GPU, disabling", level="WARNING", category="cache", force=True)
                args.cache_dit = False
                args.cache_vae = False
            runner_cache = {} if (args.cache_dit or args.cache_vae) else None

            for idx, video_path in enumerate(valid_videos, 1):
                if idx > 1:
                    debug.log("", category="none", force=True)
                    debug.log("━" * 60, category="none", force=True)
                    debug.log("", category="none", force=True)
                debug.log(f"Processing {idx}/{len(valid_videos)}", category="generation", force=True)
                out_path = None
                if args.output:
                    out_path = str(Path(args.output) / Path(video_path).stem)
                frames = process_single_video(video_path, args, device_list, out_path, runner_cache)
                total_frames_processed += frames

        elif input_path.is_file():
            if not is_video_file(str(input_path)):
                debug.log(f"Not a valid video: {input_path}", level="ERROR", category="file", force=True)
                sys.exit(1)
            if (args.cache_dit or args.cache_vae):
                if len(device_list) > 1:
                    debug.log("Caching requires single GPU, disabling", level="WARNING", category="cache", force=True)
                    args.cache_dit = False
                    args.cache_vae = False
                else:
                    debug.log("Caching has little benefit for a single file", category="tip", force=True)
            frames = process_single_video(str(input_path), args, device_list, args.output, None)
            total_frames_processed += frames

        else:
            debug.log(f"Input not found: {input_path}", level="ERROR", category="file", force=True)
            sys.exit(1)

        total_time = time.time() - start_time
        debug.log("", category="none", force=True)
        debug.log(f"Completed in {total_time:.2f}s", category="success", force=True)
        if total_time > 0 and total_frames_processed > 0:
            fps = total_frames_processed / total_time
            debug.log(f"Average FPS: {fps:.2f}", category="timing", force=True)
    except Exception as e:
        debug.log(f"Error: {e}", level="ERROR", category="generation", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        debug.log(f"Process {os.getpid()} terminating", category="cleanup", force=True)
        debug.print_footer()


if __name__ == "__main__":
    main()    