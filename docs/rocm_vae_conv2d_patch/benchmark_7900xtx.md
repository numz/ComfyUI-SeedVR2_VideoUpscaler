# Benchmarks — RX 7900 XTX (gfx1100), ROCm 6.4, 24 GB

Model: SeedVR2 3B · `numz/ComfyUI-SeedVR2_VideoUpscaler` v2.5.24 · bf16 VAE ·
fp8 DiT · tile 256 / overlap 32 · temporal slicing split 4.

Source runs: `bench/E2_f121.log` (baseline) and `bench/AP_f121.log` (patched),
21-frame runs `E2_f21.log` / `AP_f21.log`.

## End-to-end (121 frames, 480p→1080p)

| phase | baseline | patched | Δ |
|---|---|---|---|
| Phase 1 encode | 155.64 s | 153.89 s | ≈0 (decode-only ✓) |
| Phase 2 DiT | 46.35 s | 45.49 s | ≈0 |
| **Phase 3 decode** | **367.39 s** | **221.15 s** | **1.66×** |
| Phase 4 post | 6.82 s | 8.62 s | — |
| **total** | **581.75 s** | **434.63 s** | — |
| **Average FPS** | **0.208** | **0.28** | **+35%** |

21-frame run: 104.48 → 80.58 s (0.20 → 0.26 fps).

## Quality (patched vs baseline, per-frame ffmpeg)

| metric | value |
|---|---|
| PSNR average | **47.356 dB** (Y 47.089 / U 47.487 / V 48.460; min 46.005 / max 54.434) |
| SSIM All | **0.990495** (Y 0.990281 / U 0.990610 / V 0.991238) |
| NaN / Inf | 0 / 0 |

Visually indistinguishable; no drift / flicker.

## Micro benchmarks (hot shape)

`[1,256,4,256,256]`, 3×3×3, 256 ch:

| path | time | Δ |
|---|---|---|
| stock Conv3d (GemmFwdRest) | 31.2 ms | — |
| 3× Conv2D decomposition (this patch) | 5.9 ms | **5.3×** |

128 ch: 14.6 → 3.7 ms (3.9×).

## Ablation / disabled directions

| experiment | decode total | verdict |
|---|---|---|
| stock (baseline, split=4) | 61.6 s | — |
| **patched (split=4)** | **36.9 s** | **1.67×** (this is the per-clip number; e2e uses 121f) |
| patched, split=24 (batch 4→21) | 35.8 s | batch irrelevant — conv2d is bandwidth-bound too |
| patched, spatial 32-alignment | 54.7 s | alignment hypothesis rejected |
| `no_slicing` / channel-stack / `--compile_vae` | crash / no converge | see root_cause.md |

Peak VRAM during patched decode: ≤ 7.0 GiB (split=4) — well within 24 GB.

> Note on units: the ablation numbers are per 21-frame clip decode (the
> instrumented harness); the 367.4 → 221.2 s row above is the authoritative
> 121-frame e2e measurement.
