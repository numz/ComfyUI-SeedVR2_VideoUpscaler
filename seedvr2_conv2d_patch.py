#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
SeedVR2 VAE decode — InflatedCausalConv3d → Conv2D temporal decomposition (ROCm patch).

Why
---
On AMD RDNA3 (gfx1100) + MIOpen, 3D convolution (Conv3d) has exactly one solver:
``GemmFwdRest`` (im2col + rocBLAS GEMM). The im2col expansion is a bandwidth wall:
for a [1,256,4,256,256] 3x3x3 conv the column matrix is ~3.4 GiB (26x the input),
and im2col alone accounts for 64.3% of decode GPU time (rocprof, SeedVR2 3B,
121-frame 480p->1080p). On NVIDIA/CUDA the same layers dispatch to optimized
cudnn kernels, so this workaround is ROCm-only and disabled everywhere else.

What it does
------------
The causal 3x3x3 convolution (kernel_depth=3, stride=1, causal = past frames only)
is decomposed exactly into three independent 2D convolutions:

    Y_t = W[0] x X'[t] + W[1] x X'[t+1] + W[2] x X'[t+2]

where X' is the temporally head-extended input (T -> T+2) produced by the
existing causal padding / memory cache machinery. Frames are folded into the
batch dimension ((B, C, T, H, W) -> (B*T, C, H, W)) so each Conv2D call gets a
large effective batch. The decomposition is mathematically exact — no
approximation, no precision loss.

Scope / safety
--------------
- Applies ONLY while the VAE *decode* path is running (encode untouched;
  verified: encode wall-time unchanged).
- Applies ONLY to kernel_size==(3,3,3), stride==(1,1,1), dilation==(1,1,1),
  groups==1, zero padding_mode. Everything else falls back to the original path.
- Hard parameter asserts instead of silent assumptions (fail loudly if a
  different checkpoint changes conv parameters).
- Enabled by default only on ROCm builds **with benchmarked GPU architectures**
  (gfx1100 / RX 7900 XTX); other ROCm parts default to off until benchmarked.
  Override with the SEEDVR2_CONV2D_PATCH=1|0|auto env var. CUDA users are never
  affected.
- T=1 is a strict special case: after head-extension all three taps see the same
  frame, so the sum reduces to (W[0]+W[1]+W[2]) x X0 — the single-frame fast
  path used by community implementations, obtained for free.
- **Inference-only.** SeedVR2 decode runs under torch.no_grad(); training /
  backward behavior is out of scope for this patch and is not tested.

Usage
-----
    import seedvr2_conv2d_patch
    seedvr2_conv2d_patch.install()      # class-level monkey-patch, this process only

or run the stock CLI through the injected launcher (no source edits):

    python run_infer_patched.py <in> --output <out> --resolution 1080 ...

Optional timing (disabled by default; no CUDA events are created when off):
    SEEDVR2_PATCH_TIMING=1 python run_infer_patched.py ...
"""

import os
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Enable logic — ROCm + benchmarked GPU architecture, three-state env
# ---------------------------------------------------------------------------

# GPU architectures this workaround is benchmarked on. It exists because MIOpen
# has no good Conv3D solver on these parts (im2col bandwidth wall). Other ROCm
# GPUs (MI300 / gfx90a / gfx942 / gfx120x, ...) may or may not benefit — we have
# no data for them, so we default them to OFF rather than risk a regression on
# hardware we never measured. `SEEDVR2_CONV2D_PATCH=1` forces it on any ROCm
# architecture.
_SUPPORTED_ARCHS = {"gfx1100"}   # RX 7900 XTX — the benchmarked part


def _rocm_build() -> bool:
    """True when torch is a ROCm (HIP) build."""
    return torch.version.hip is not None


def _visible_archs() -> list:
    """gfx arch names of all visible CUDA/HIP devices (gcnArchName prefix)."""
    archs = []
    if not torch.cuda.is_available():
        return archs
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]
        archs.append(arch or "<unknown>")
    return archs


def patch_enabled() -> bool:
    """
    Three-state env switch:

      SEEDVR2_CONV2D_PATCH=auto (default)  ROCm build AND every visible device
                                           is a supported arch (gfx1100).
      SEEDVR2_CONV2D_PATCH=1               force on (correctness/perf is yours).
      SEEDVR2_CONV2D_PATCH=0               force off (e.g. MIOpen later fixes
                                           Conv3D; the A/B stays trivial).
    """
    env = os.environ.get("SEEDVR2_CONV2D_PATCH", "auto").strip().lower()
    if env in ("1", "true", "on", "yes"):
        # force enable may override the arch allow-list (e.g. unbenchmarked
        # ROCm parts) but NEVER a CUDA build — the workaround targets MIOpen.
        return _rocm_build()
    if env in ("0", "false", "off", "no"):
        return False
    # auto
    if not _rocm_build():
        return False
    archs = _visible_archs()
    return len(archs) > 0 and all(a in _SUPPORTED_ARCHS for a in archs)


_DECODE_ACTIVE = [False]   # set True only while the decode path is running

# Optional event timing (SEEDVR2_PATCH_TIMING=1); when off no CUDA events are
# created and the per-call cost is a single `if _TIMING` branch.
_TIMING = os.environ.get("SEEDVR2_PATCH_TIMING", "").strip().lower() in ("1", "true", "on", "yes")
_TIMERS = {"pack_us": 0.0, "conv_us": 0.0, "sum_us": 0.0, "total_us": 0.0, "calls": 0}
_EVTS = []  # (ev_start, ev_packed, ev_conved, ev_done) tuples, drained at exit

# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------

def make_patched_conv_forward(orig_conv_forward):
    """
    Build the patched ``_conv_forward`` for InflatedCausalConv3d instances.

    ``orig_conv_forward`` is the class's original unbound method; it is used
    for the fallback path (scope guard) so behavior outside the guarded scope
    is bit-identical to stock SeedVR2.
    """
    def patched(self, input, weight, bias, *args, **kwargs):
        # decode-only flag (encode path untouched)
        if not _DECODE_ACTIVE[0]:
            return orig_conv_forward(self, input, weight, bias, *args, **kwargs)

        # ── scope guard: only stride-1 D=3 causal 3x3x3 convs are decomposed ──
        if (self.kernel_size != (3, 3, 3) or input.dim() != 5
                or self.stride != (1, 1, 1) or self.padding_mode != "zeros"):
            return orig_conv_forward(self, input, weight, bias, *args, **kwargs)

        # ── hard asserts: never silently assume equivalence ──
        assert self.dilation == (1, 1, 1), f"dilation={self.dilation}"
        assert self.groups == 1, f"groups={self.groups}"
        assert self.padding[0] == 0, f"temporal pad must be 0 (causal), got {self.padding}"
        assert self.padding[1] == self.padding[2], f"spatial pad must be symmetric, got {self.padding}"

        B, C, Tin, H, W = input.shape
        T = Tin - 2                    # input already head-extended by 2 frames
        assert T >= 1, f"Tin={Tin} unexpected"
        spatial_pad = self.padding[1]  # 0 on the tiled path (input pre-padded), 1 on the basic path
        Bt = B * T
        Cout = weight.shape[0]

        evs = None
        if _TIMING:
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            evs[0].record()

        # pack: one permute + one contiguous copy; the three slices are then
        # contiguous views, so reshape below is zero-copy.
        xp = input.permute(0, 2, 1, 3, 4).contiguous()     # [B, T+2, C, H, W]
        if _TIMING:
            evs[1].record()

        # 3 x batched Conv2D (T folded into batch)
        convs = []
        for d in range(3):
            w2d = weight[:, :, d, :, :]                    # [Cout, Cin, 3, 3] view
            xs = xp[:, d:d + T, :, :, :]                   # [B, T, C, H, W] contiguous view
            xs = xs.reshape(Bt, C, H, W)                   # zero-copy fold
            convs.append(F.conv2d(xs, w2d, None, stride=1, padding=spatial_pad))
        if _TIMING:
            evs[2].record()

        # in-place accumulate + bias + restore 5D (sizes from the conv output)
        acc = convs[0].add_(convs[1]).add_(convs[2])
        if bias is not None:
            acc = acc.add_(bias.reshape(1, -1, 1, 1))
        _, _, OH, OW = acc.shape
        out = acc.reshape(B, T, Cout, OH, OW).permute(0, 2, 1, 3, 4)
        if _TIMING:
            evs[3].record()
            _TIMERS["calls"] += 1
            _EVTS.append((evs[0], evs[1], evs[2], evs[3]))
        return out
    return patched


def drain_timing():
    """Fold all recorded events into _TIMERS (us). Call once at exit."""
    if not _TIMING or not _EVTS:
        return None
    for evs in _EVTS:
        evs[3].synchronize()   # last event on the stream -> all four completed
        _TIMERS["pack_us"] += evs[0].elapsed_time(evs[1]) * 1e3
        _TIMERS["conv_us"] += evs[1].elapsed_time(evs[2]) * 1e3
        _TIMERS["sum_us"] += evs[2].elapsed_time(evs[3]) * 1e3
    _EVTS.clear()
    return dict(_TIMERS)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install(verbose=True):
    """
    Class-level monkey-patch of InflatedCausalConv3d._conv_forward plus a
    decode-only flag on the VAE decode method family. No source edits; the
    patch lives and dies with this process.
    """
    if not patch_enabled():
        if verbose:
            mode = os.environ.get("SEEDVR2_CONV2D_PATCH", "auto").strip().lower()
            if mode in ("0", "false", "off", "no"):
                reason = "disabled by SEEDVR2_CONV2D_PATCH=0"
            elif not _rocm_build():
                reason = "not a ROCm build (torch.version.hip=None)"
            elif not _visible_archs():
                reason = "no visible HIP device"
            else:
                reason = f"GPU arch(s) {_visible_archs()} not benchmarked " \
                         f"(supported: {sorted(_SUPPORTED_ARCHS)})"
            print(f"[seedvr2_conv2d_patch] SKIPPED: {reason}. "
                  "SEEDVR2_CONV2D_PATCH=1 forces ROCm parts only.", flush=True)
        return False

    from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
    from src.models.video_vae_v3.modules import attn_video_vae

    if getattr(InflatedCausalConv3d, "_conv_forward", None).__name__ == "patched":
        return False   # already installed

    _orig_conv = InflatedCausalConv3d._conv_forward
    InflatedCausalConv3d._conv_forward = make_patched_conv_forward(_orig_conv)

    # decode method family sets the flag; encode paths never touch it.
    W = attn_video_vae.VideoAutoencoderKLWrapper
    for meth in ("decode", "_decode", "tiled_decode", "slicing_decode"):
        if not hasattr(W, meth):
            continue
        orig = getattr(W, meth)
        def _wrap(orig):
            def wrapped(self, *a, **k):
                # nested-safe: restore the previous state, not a hard False.
                # decode() may call tiled_decode()/slicing_decode() internally;
                # the inner wrapper must not clear the outer call's flag.
                prev = _DECODE_ACTIVE[0]
                _DECODE_ACTIVE[0] = True
                try:
                    return orig(self, *a, **k)
                finally:
                    _DECODE_ACTIVE[0] = prev
            return wrapped
        setattr(W, meth, _wrap(orig))

    if verbose:
        print("[seedvr2_conv2d_patch] installed: class-level _conv_forward "
              "(D=3 causal -> 3x Conv2D) + decode-only flag", flush=True)
    return True
