#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
Correctness tests for the SeedVR2 ROCm decode patch
(InflatedCausalConv3d -> 3x Conv2D temporal decomposition).

Run (inside the seedvr2 container, cwd = repo root):
    /workspace/venv/bin/python /workspace/setup/pr-package/tests/test_conv2d_decomposition.py

No pytest dependency. Exits non-zero on any failure. Covers:

  [1] Math equivalence, random weights  — patched 3x Conv2D == stock Conv3d, tight fp32 bound
  [2] T=1 reduction                     — patched == (W0+W1+W2) x X0 (single-frame fast path)
  [3] Spatial pre-pad variant           — tiled path (padding zeroed, input pre-padded)
  [4] Scope guard fallback              — D=1 / stride!=1 / flag-off  -> bit-identical to stock
  [5] Hard asserts                      — dilation != 1 raises AssertionError
  [6] Real checkpoint weights           — spot-check decoder D=3 layers from ema_vae_fp16.safetensors
  [7] Decode flag lifecycle             — off before/after decode, restored on exception, nested-safe
  [8] Enable gate                       — ROCm/arch/env three-state matrix (mocked, no GPU needed)
"""
import os
import sys

SEEDVR2_ROOT = os.environ.get("SEEDVR2_ROOT", "/workspace/seedvr2")
PATCH_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patch"))
sys.path.insert(0, SEEDVR2_ROOT)
sys.path.insert(0, PATCH_DIR)

import torch
import torch.nn.functional as F

import seedvr2_conv2d_patch as P
from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

DEV = "cuda"
_PASS = 0
_FAIL = 0

# Inference-only optimization: SeedVR2 decode runs under torch.no_grad(), so the
# tests reproduce that exact mode. Training / backward behavior is out of scope
# for this PR and is deliberately not exercised here.
torch.set_grad_enabled(False)


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def max_abs(a, b):
    return (a - b).abs().max().item()


def make_layer(Cin, Cout, kernel=3, stride=1, padding=(1, 1, 1), dilation=1,
               groups=1, bias=True, dtype=torch.float32, inflation_mode="none"):
    """InflatedCausalConv3d: note the ctor zeroes the temporal pad (causal)."""
    layer = InflatedCausalConv3d(
        Cin, Cout, kernel_size=kernel, stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=bias,
        inflation_mode=inflation_mode, dtype=dtype,
    ).to(DEV)
    with torch.no_grad():
        layer.weight.normal_(0, 0.1)
        if layer.bias is not None:
            layer.bias.normal_(0, 0.1)
    return layer


def head_extend(x):
    """Replicate extend_head: repeat the first frame twice in front (causal)."""
    return torch.cat([x[:, :, :1].repeat(1, 1, 2, 1, 1), x], dim=2)


# ---------------------------------------------------------------------------
#  Setup: class-level patch, decode flag ON (as the wrapped decode path does)
# ---------------------------------------------------------------------------
assert P.patch_enabled(), "patch must be enabled (ROCm build or SEEDVR2_CONV2D_PATCH=1)"
_orig = InflatedCausalConv3d._conv_forward
InflatedCausalConv3d._conv_forward = P.make_patched_conv_forward(_orig)
P._DECODE_ACTIVE[0] = True

torch.manual_seed(1234)

print(f"[env] torch {torch.__version__} hip={torch.version.hip} device={torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# [1] Math equivalence, random weights, fp32, several shapes
# ---------------------------------------------------------------------------
print("\n[1] Math equivalence (random weights, fp32)")
for (B, C, Tin, H, W, Cout) in [
    (1, 32, 3, 32, 32, 64),    # T=1 (Tin=3)
    (1, 32, 4, 32, 32, 64),    # T=2
    (1, 64, 6, 32, 32, 128),   # T=4
    (1, 128, 8, 16, 16, 128),  # T=6
    (2, 64, 5, 24, 24, 64),    # B=2, T=3
]:
    layer = make_layer(C, Cout)
    x = torch.randn(B, C, Tin, H, W, device=DEV, dtype=torch.float32)
    out_p = layer._conv_forward(x, layer.weight, layer.bias)          # patched
    out_r = F.conv3d(x, layer.weight, layer.bias, stride=1, padding=layer.padding)  # stock math
    err = max_abs(out_p, out_r)
    check(f"shape {tuple(x.shape)}->{tuple(out_p.shape)}", out_p.shape == out_r.shape and err < 1e-4,
          f"max_abs={err:.3e}")

# ---------------------------------------------------------------------------
# [2] T=1 reduction to the summed-weight fast path  (W0+W1+W2) x X0
# ---------------------------------------------------------------------------
print("\n[2] T=1 reduces to (W0+W1+W2) x X0 single-frame fast path")
layer = make_layer(64, 96)
# real pipeline: Tin = T+2 with two head-copied frames, so for T=1 all taps see X0
x_raw = torch.randn(1, 64, 1, 32, 32, device=DEV, dtype=torch.float32)
x = head_extend(x_raw)                                            # (1,64,3,32,32), all frames = X0
wsum = layer.weight[:, :, 0] + layer.weight[:, :, 1] + layer.weight[:, :, 2]  # [Cout,Cin,3,3]
ref_sum = F.conv2d(x[:, :, 0], wsum, layer.bias, padding=1)       # single Conv2D, pad=1
out_p = layer._conv_forward(x, layer.weight, layer.bias)
err = max_abs(out_p[:, :, 0], ref_sum)
check("patched == summed-weight Conv2D", out_p.shape == (1, 96, 1, 32, 32) and err < 1e-4,
      f"max_abs={err:.3e}")

# ---------------------------------------------------------------------------
# [3] Spatial pre-pad variant (tiled path: padding zeroed, input pre-padded)
# ---------------------------------------------------------------------------
print("\n[3] Tiled path variant (padding zeroed by ignore_padding, input pre-padded)")
layer = make_layer(128, 128)
x_raw = torch.randn(1, 128, 6, 32, 32, device=DEV, dtype=torch.float32)   # T=4
x = head_extend(x_raw)                                                    # Tin=6
x = F.pad(x, (1, 1, 1, 1))                                                # spatial pre-pad
from src.models.video_vae_v3.modules.causal_inflation_lib import ignore_padding
with ignore_padding(layer):                                               # padding -> (0,0,0)
    out_p = layer._conv_forward(x, layer.weight, layer.bias)
    out_r = F.conv3d(x, layer.weight, layer.bias, stride=1, padding=(0, 0, 0))
err = max_abs(out_p, out_r)
check("pre-padded input, zero padding", out_p.shape == out_r.shape and err < 1e-4,
      f"max_abs={err:.3e}")

# ---------------------------------------------------------------------------
# [4] Scope guard fallback -> bit-identical to stock path
# ---------------------------------------------------------------------------
print("\n[4] Scope guard fallback (D=1 / stride!=1 / flag-off)")
# 4a: D=1 pointwise conv
layer = make_layer(64, 64, kernel=1, padding=(0, 0, 0))
x = torch.randn(1, 64, 5, 32, 32, device=DEV, dtype=torch.float32)
out_p = layer._conv_forward(x, layer.weight, layer.bias)
out_r = F.conv3d(x, layer.weight, layer.bias, stride=1, padding=(0, 0, 0))
check("D=1 kernel falls back", out_p.shape == out_r.shape and max_abs(out_p, out_r) == 0.0)

# 4b: stride != 1 (temporal downsampler)
layer = make_layer(64, 128, stride=(1, 2, 2), padding=(0, 1, 1))
x = torch.randn(1, 64, 5, 32, 32, device=DEV, dtype=torch.float32)
out_p = layer._conv_forward(x, layer.weight, layer.bias)
out_r = F.conv3d(x, layer.weight, layer.bias, stride=(1, 2, 2), padding=(0, 1, 1))
check("stride!=1 falls back", out_p.shape == out_r.shape and max_abs(out_p, out_r) == 0.0)

# 4c: decode flag off (encode path isolation)
layer = make_layer(64, 64)
x = torch.randn(1, 64, 5, 32, 32, device=DEV, dtype=torch.float32)
P._DECODE_ACTIVE[0] = False
out_p = layer._conv_forward(x, layer.weight, layer.bias)
out_r = F.conv3d(x, layer.weight, layer.bias, stride=1, padding=layer.padding)
P._DECODE_ACTIVE[0] = True
check("flag-off falls back (encode untouched)", out_p.shape == out_r.shape and max_abs(out_p, out_r) == 0.0)

# ---------------------------------------------------------------------------
# [5] Hard asserts fire on incompatible params
# ---------------------------------------------------------------------------
print("\n[5] Hard asserts (fail loudly, never silently approximate)")
layer = make_layer(64, 64, dilation=(2, 1, 1))
x = torch.randn(1, 64, 5, 32, 32, device=DEV, dtype=torch.float32)
try:
    layer._conv_forward(x, layer.weight, layer.bias)
    check("dilation=(2,1,1) raises", False, "no exception")
except AssertionError:
    check("dilation=(2,1,1) raises", True, "AssertionError")

layer = make_layer(64, 64, groups=2)
x = torch.randn(2, 64, 5, 32, 32, device=DEV, dtype=torch.float32)
try:
    layer._conv_forward(x, layer.weight, layer.bias)
    check("groups=2 raises", False, "no exception")
except AssertionError:
    check("groups=2 raises", True, "AssertionError")

# ---------------------------------------------------------------------------
# [6] Real checkpoint weights (ema_vae_fp16.safetensors), fp32 + one bf16
# ---------------------------------------------------------------------------
print("\n[6] Real checkpoint weights")
from safetensors.torch import load_file

ckpt = os.environ.get("SEEDVR2_VAE_CKPT", "/workspace/seedvr2/models/SEEDVR2/ema_vae_fp16.safetensors")
assert os.path.exists(ckpt), f"checkpoint not found: {ckpt}"
sd = load_file(ckpt)

picked = []
for k in sd:
    w = sd[k]
    if (k.startswith("decoder.") and "conv" in k and k.endswith(".weight")
            and w.dim() == 5 and w.shape[2] == 3):
        picked.append(k)
    if len(picked) >= 4:
        break

if not picked:
    print("  SKIP  no decoder D=3 layers matched (check key naming)")
else:
    for k in picked:
        w = sd[k].float()
        Cin, Cout = w.shape[1], w.shape[0]
        layer = make_layer(Cin, Cout)
        with torch.no_grad():
            layer.weight.copy_(w)
        x = torch.randn(1, Cin, 5, 16, 16, device=DEV, dtype=torch.float32)  # T=3
        out_p = layer._conv_forward(x, layer.weight, layer.bias)
        out_r = F.conv3d(x, layer.weight, layer.bias, stride=1, padding=layer.padding)
        err = max_abs(out_p, out_r)
        check(f"real {k} [{Cin}->{Cout}]", out_p.shape == out_r.shape and err < 1e-4,
              f"max_abs={err:.3e}")

    # one bf16 run (production dtype), looser bound
    k = picked[0]
    w = sd[k].bfloat16()
    layer = make_layer(w.shape[1], w.shape[0], dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(w)
    x = torch.randn(1, w.shape[1], 5, 16, 16, device=DEV, dtype=torch.bfloat16)
    out_p = layer._conv_forward(x, layer.weight, layer.bias)
    out_r = F.conv3d(x, layer.weight, layer.bias, stride=1, padding=layer.padding)
    err = max_abs(out_p, out_r)
    check(f"real {k} bf16", out_p.shape == out_r.shape and err < 2e-2, f"max_abs={err:.3e}")

# ---------------------------------------------------------------------------
# [7] Decode flag lifecycle — set only during decode, restored on exception
# ---------------------------------------------------------------------------
print("\n[7] Decode flag lifecycle (try/finally restoration)")


def make_wrapper(orig):
    """Same wrapping pattern as install(): save prev state, restore after."""
    def wrapped(self, *a, **k):
        prev = P._DECODE_ACTIVE[0]
        P._DECODE_ACTIVE[0] = True
        try:
            return orig(self, *a, **k)
        finally:
            P._DECODE_ACTIVE[0] = prev
    return wrapped


class _Dummy:
    def decode_ok(self):
        assert P._DECODE_ACTIVE[0], "flag must be True during decode"
        return "ok"

    def decode_boom(self):
        assert P._DECODE_ACTIVE[0], "flag must be True during decode"
        raise RuntimeError("boom")


_Dummy.decode_ok = make_wrapper(_Dummy.decode_ok)
_Dummy.decode_boom = make_wrapper(_Dummy.decode_boom)
d = _Dummy()

P._DECODE_ACTIVE[0] = False   # fresh-process / encode-path state
check("flag False before decode", P._DECODE_ACTIVE[0] is False)
assert d.decode_ok() == "ok"
check("flag False after normal decode", P._DECODE_ACTIVE[0] is False)
try:
    d.decode_boom()
    check("flag restored after exception", False, "no exception raised")
except RuntimeError:
    check("flag restored after exception", P._DECODE_ACTIVE[0] is False)

# nested: decode() -> tiled_decode() must NOT clear the outer call's flag
class _Nested:
    def inner(self):
        assert P._DECODE_ACTIVE[0] is True, "inner sees flag True"
        return "inner"

    def outer(self):
        assert P._DECODE_ACTIVE[0] is True, "outer sees flag True"
        r = self.inner()
        assert P._DECODE_ACTIVE[0] is True, "inner must not clear outer flag"
        return r


_Nested.inner = make_wrapper(_Nested.inner)
_Nested.outer = make_wrapper(_Nested.outer)
n = _Nested()
P._DECODE_ACTIVE[0] = False
assert n.outer() == "inner"
check("nested decode: inner does not clear outer flag", P._DECODE_ACTIVE[0] is False)

# ---------------------------------------------------------------------------
# [8] Enable gate — ROCm / arch / env three-state matrix (mocked, no GPU needed)
# ---------------------------------------------------------------------------
print("\n[8] Enable gate (mocked ROCm/arch)")
def gate_case(name, want, rocm, archs, env):
    old_env = os.environ.get("SEEDVR2_CONV2D_PATCH")
    os.environ["SEEDVR2_CONV2D_PATCH"] = env
    orig_rocm, orig_vis = P._rocm_build, P._visible_archs
    P._rocm_build = lambda: rocm
    P._visible_archs = lambda: archs
    try:
        got = P.patch_enabled()
    finally:
        P._rocm_build, P._visible_archs = orig_rocm, orig_vis
        if old_env is None:
            os.environ.pop("SEEDVR2_CONV2D_PATCH", None)
        else:
            os.environ["SEEDVR2_CONV2D_PATCH"] = old_env
    check(f"gate: {name}", got is want, f"rocm={rocm} archs={archs} env={env} -> {got}")


gate_case("CUDA + auto -> off",         False, rocm=False, archs=[],            env="auto")
gate_case("CUDA + 1 -> off (never CUDA)", False, rocm=False, archs=[],          env="1")
gate_case("ROCm gfx1100 + auto -> on",  True,  rocm=True,  archs=["gfx1100"],   env="auto")
gate_case("ROCm gfx942 + auto -> off",  False, rocm=True,  archs=["gfx942"],    env="auto")
gate_case("ROCm gfx942 + 1 -> on (force)", True, rocm=True, archs=["gfx942"],   env="1")

# ---------------------------------------------------------------------------
print(f"\n==== {_PASS} passed, {_FAIL} failed ====")
sys.exit(1 if _FAIL else 0)
