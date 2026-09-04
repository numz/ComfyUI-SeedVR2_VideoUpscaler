# SeedVR2 ROCm VAE decode patch — causal Conv3D → 3× Conv2D decomposition

**Workaround for the MIOpen gfx1100 Conv3D im2col bandwidth wall, on AMD RDNA3 (RX 7900 XTX).**

This is a self-contained, process-lifetime patch for the **decode** path of the
SeedVR2 3B VAE that decomposes the causal `InflatedCausalConv3d` (3×3×3)
convolution into three mathematically-exact 2D convolutions. On AMD RDNA3
(gfx1100) + MIOpen it accelerates VAE decode **1.66×** and end-to-end video
upscaling from **0.208 → 0.28 fps** (121-frame 480p→1080p), with output
quality unchanged (PSNR 47.4 dB / SSIM 0.9905 vs. baseline).

```
SeedVR2 causal video VAE → InflatedCausalConv3d → MIOpen gfx1100 → GemmFwdRest (im2col+GEMM)
    → Im3d2Col bandwidth wall = 64.3% of decode GPU time
Our fix: D=3 causal Conv3D → exact temporal decomposition → 3 × optimized Conv2D
    → decode 367.4 → 221.2 s · E2E 0.208 → 0.28 fps (no quality loss)
```

- **No source edits** — installed via `install()` (monkey-patch, lives only in
  the process) or via the injected launcher `run_infer_patched.py`.
- **Inference-only, ROCm/gfx1100-only by default** — auto-disabled on CUDA and
  on ROCm GPUs that are not benchmarked (`gfx1100`); three-state
  `SEEDVR2_CONV2D_PATCH=auto|1|0`.
- **Decode-only** — the encode path is untouched (verified: encode wall-time
  unchanged), and non-D=3 / non-stride-1 convs fall back to the stock path
  bit-identically.

---

## Why

On AMD RDNA3 (gfx1100) the MIOpen 3D-convolution stack has exactly one solver
for these shapes: `GemmFwdRest` (im2col + rocBLAS GEMM) — there is no
direct / implicit-GEMM candidate. The im2col expansion is a **bandwidth wall**:

- A `[1,256,4,256,256]` 3×3×3 conv: 128 MiB input → **3.38 GiB** im2col column
  matrix → ~6.9 GiB of traffic per conv.
- `rocprof` on a full decode: **Im3d2Col = 64.3%**, Cijk GEMM = 26.2%,
  elementwise = 9.5% of GPU time.
- The measured 8.3 ms is close to the ~7.7 ms bandwidth-ideal — i.e. the cost
  is im2col traffic, not arithmetic. `MIOPEN_FIND_MODE` does not help.

The same layers on CUDA dispatch to optimized cuDNN kernels; this patch
therefore targets ROCm only. See [`docs/root_cause.md`](docs/root_cause.md).

## Method (exact, no approximation)

A causal 3×3×3 conv (kernel_depth=3, stride=1, past-only) on the temporally
head-extended input `X'` (length `T+2`) satisfies

```
Y_t = W[0]·X'[t] + W[1]·X'[t+1] + W[2]·X'[t+2]
```

Each term is an independent 2D convolution. We fold the temporal axis into the
batch dimension (`(B,C,T,H,W) → (B·T,C,H,W)`), run three batched `F.conv2d`
calls, and accumulate in place. The sum is exact: for every output position the
terms are identical to the stock `F.conv3d` accumulation — only the evaluation
order changes (verified to `max_abs ≈ 1e-5` in fp32, see tests).

**T=1 is a strict special case**: with one frame the head-extension makes all
three taps see the same `X0`, so the sum collapses to `(W[0]+W[1]+W[2])·X0` —
the single-frame summed-weight fast path used by community implementations,
obtained for free. This patch is a strict superset of that idea.

See [`docs/math_equivalence.md`](docs/math_equivalence.md).

## Scope guard & safety

Applied **only** while the decode path runs, and **only** to convs matching:

| condition | behavior |
|---|---|
| `kernel_size == (3,3,3)` · `stride == (1,1,1)` · dim == 5 · `padding_mode == "zeros"` | decompose to 3× Conv2D |
| D=1 pointwise, stride≠1, any other shape | **fallback** to stock `_conv_forward` (bit-identical) |
| `dilation ≠ (1,1,1)` · `groups ≠ 1` · `padding[0] ≠ 0` · asymmetric spatial pad | **hard `assert`** — fail loudly rather than silently approximate |

The hard asserts exist because a different checkpoint could in principle change
conv parameters; we prefer a crash over a silent equivalence violation.

## Results (RX 7900 XTX, ROCm 6.4, 24 GB)

Full 121-frame benchmark (`E2_f121`, batch 21, tile 256/overlap 32, bf16):

| phase | baseline | patched | Δ |
|---|---|---|---|
| encode | 155.64 s | 153.89 s | ≈0 (decode-only ✓) |
| DiT | 46.35 s | 45.49 s | ≈0 |
| **decode** | **367.39 s** | **221.15 s** | **1.66×** |
| **total** | **581.75 s** | **434.63 s** | **0.208 → 0.28 fps** |

Quality (patched vs baseline, per-frame ffmpeg): **PSNR 47.36 dB**
(y 47.09 / u 47.49 / v 48.46, min 46.01) · **SSIM 0.9905**
(Y 0.99028 / U 0.99061 / V 0.99124). Peak VRAM ≤ 7 GiB.

> **Encode and DiT are unchanged; the speedup comes exclusively from VAE
> decode.** The 1.66× decode row is the entire delta — no other parameter of
> the pipeline was touched (batch, tiles, dtype, attention all identical).

> Reference points: the same workload on an RTX 4070 Ti is ≈0.32 fps — the
> AMD/ROCm gap shrinks from ~1.6× to ~1.14× with this patch.

Full numbers, the disabled directions (batch is irrelevant; spatial 32-alignment
does not help; channel-stacking does not converge on MIOpen), and micro
benchmarks: [`bench/benchmark_7900xtx.md`](bench/benchmark_7900xtx.md).

## Correctness tests

`tests/test_conv2d_decomposition.py` — no pytest dependency, runs standalone
under `torch.no_grad()` (matching the real inference path):

```
python tests/test_conv2d_decomposition.py
```

**26 checks, all passing** (RX 7900 XTX, ROCm 6.4):

| layer | checks | what it proves |
|---|---|---|
| Unit correctness | 5 | patched 3×Conv2D == stock Conv3D on random weights, fp32, `max_abs ≤ 6.4e-5` across 5 shapes (T=1/2/3/4/6, B=1/2) |
| T=1 reduction | 1 | degenerate to `(W[0]+W[1]+W[2])·X0` single-frame fast path, `max_abs ≈ 8e-6` |
| Tiled-path variant | 1 | padding-zeroed / pre-padded input path, `max_abs ≈ 7e-5` |
| Fallback / guard | 3 | D=1, stride≠1, flag-off all **bit-identical** (error exactly 0.0) to stock |
| Hard asserts | 2 | `dilation≠1` and `groups≠1` raise `AssertionError` (fail loudly) |
| **Real checkpoint weights** | 5 | `ema_vae_fp16.safetensors`: `conv_in` [16→512], `conv_out` [128→3], `mid_block.resnets.0.conv1/2` [512→512] in fp32, plus one **bf16** production-dtype run |
| Decode flag lifecycle | 4 | flag off before decode, restored after normal decode, **restored after an exception** (try/finally), **nested-safe** — an inner `tiled_decode` inside `decode` does not clear the outer flag |
| Enable gate | 5 | ROCm/arch/env three-state matrix (mocked, no GPU needed): CUDA+auto off, CUDA+`1` off, ROCm gfx1100+auto on, ROCm gfx942+auto off, ROCm gfx942+`1` on (force) |

> **This optimization is intended for inference.** Training/backward behavior
> is not part of this PR and is not tested.

## Community context

Among the SeedVR2 implementations and optimization PRs we examined, we found
**no existing T>1 causal Conv3D→Conv2D decomposition for temporally mixed
kernels**. The closest practice we found is an unmerged PR to the reference
repo (numz/ComfyUI-SeedVR2_VideoUpscaler#421, "Optimize VAE and GGUF"), whose
`check_effective_2d()` fast path fires only for **kT==1** or **tail-inflated**
weights (all temporal taps except the last ≈ 0), folding `B*T` into one
`F.conv2d`. On the real SeedVR2 weights that path cannot apply:

- we numerically inspected all 57 D=3 conv layers of `ema_vae_fp16.safetensors`
  — **0 are tail-inflated**; all are temporally mixed
  (`|W[:,:,:-1]| / |W| ≈ 0.41–0.54`), so `check_effective_2d()` returns `False`
  for every one of them and they all fall back to stock Conv3D.

In other words, the PR-421 fast path covers "effectively-2D" layers (no temporal
mixing) — exactly the layers **outside** our 57 — while this patch covers the
temporally-mixed layers it explicitly skips. `T=1` single-frame summed-weight
implementations, if present, are a special case of this decomposition.

## Usage

```bash
# Option A: inject into the stock CLI (no source edits)
SEEDVR2_CONV2D_PATCH=1 python patch/run_infer_patched.py <in> --output <out> \
    --resolution 1080 --batch_size 21 \
    --vae_encode_tiled --vae_encode_tile_size 256 --vae_encode_tile_overlap 32 \
    --vae_decode_tiled --vae_decode_tile_size 256 --vae_decode_tile_overlap 32 \
    --attention_mode sdpa

# Option B: programmatic (e.g. sitecustomize)
import seedvr2_conv2d_patch
seedvr2_conv2d_patch.install()

# optional timing (disabled by default; no CUDA events are created when off)
SEEDVR2_PATCH_TIMING=1 python patch/run_infer_patched.py ...
```

| env var | effect |
|---|---|
| `SEEDVR2_CONV2D_PATCH=auto` (default) | ROCm build **and** every visible GPU is a benchmarked arch (`gfx1100`) |
| `SEEDVR2_CONV2D_PATCH=1` | force enable (correctness/perf on you) |
| `SEEDVR2_CONV2D_PATCH=0` | force disable (e.g. MIOpen later fixes Conv3D; A/B stays trivial) |
| `SEEDVR2_PATCH_TIMING=1` | dump per-phase patch timing at exit |
| `SEEDVR2_ROOT` | repo root for the launcher (default `/workspace/seedvr2`) |
| `SEEDVR2_VAE_CKPT` | checkpoint path for the real-weight test |

> **Enable scope.** The workaround is benchmarked on RX 7900 XTX / `gfx1100`
> only. On other ROCm parts (MI300, gfx90a, gfx942, gfx120x, …) it defaults to
> **off** — we have no data for them — and can be forced with `=1`. CUDA
> builds are never affected (`torch.version.hip` is `None`).

## When to drop this patch

If MIOpen ever grows a direct/implicit-GEMM Conv3D solver on gfx1100 (or ROCm
adds an alternative), this workaround becomes redundant. The isolated patch +
`SEEDVR2_CONV2D_PATCH=0` makes the A/B comparison trivial: run the same
workload with and without and keep whichever is faster.

## Files

```
patch/seedvr2_conv2d_patch.py    the patch module (install() / make_patched_conv_forward)
patch/run_infer_patched.py       stock-CLI launcher with the patch injected
tests/test_conv2d_decomposition.py  17-check correctness suite
docs/root_cause.md               rocprof evidence chain
docs/math_equivalence.md         exactness proof + scope guard rationale
bench/benchmark_7900xtx.md       micro / e2e benchmarks + disabled directions
```

License: MIT. No model weights or credentials are bundled.
