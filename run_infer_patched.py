#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
SeedVR2 stock-CLI launcher with the Conv2D decomposition patch injected.

Runs the unmodified `inference_cli.py` with the ROCm decode patch installed
via runpy — no source edits, patch lives only for the lifetime of this process.

Usage: identical to inference_cli.py, e.g.
    python run_infer_patched.py video.mp4 --output out/ --resolution 1080 \
        --batch_size 21 \
        --vae_encode_tiled --vae_encode_tile_size 256 --vae_encode_tile_overlap 32 \
        --vae_decode_tiled --vae_decode_tile_size 256 --vae_decode_tile_overlap 32 \
        --attention_mode sdpa

Environment:
    SEEDVR2_CONV2D_PATCH=auto|0|1   auto (default): ROCm build + gfx1100 arch;
                                   1 = force on, 0 = force off
    SEEDVR2_PATCH_TIMING=1          print per-phase patch timing at exit (default off)
"""
import sys
import os

SEEDVR2_ROOT = os.environ.get("SEEDVR2_ROOT", "/workspace/seedvr2")
PATCH_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(SEEDVR2_ROOT)
sys.path.insert(0, SEEDVR2_ROOT)
sys.path.insert(0, PATCH_DIR)

import seedvr2_conv2d_patch  # noqa: E402
seedvr2_conv2d_patch.install()

if os.environ.get("SEEDVR2_PATCH_TIMING", "").strip().lower() in ("1", "true", "on"):
    import atexit

    def _dump_stats():
        st = seedvr2_conv2d_patch.drain_timing()
        if st:
            print(f"[seedvr2_conv2d_patch] timing calls={st['calls']} "
                  f"pack={st['pack_us']/1e6:.3f}s conv={st['conv_us']/1e6:.3f}s "
                  f"sum={st['sum_us']/1e6:.3f}s", flush=True)
    atexit.register(_dump_stats)

sys.argv = ["inference_cli.py"] + sys.argv[1:]
import runpy  # noqa: E402
runpy.run_path(os.path.join(SEEDVR2_ROOT, "inference_cli.py"), run_name="__main__")
