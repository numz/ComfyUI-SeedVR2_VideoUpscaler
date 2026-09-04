# Root cause: MIOpen gfx1100 Conv3D = im2col bandwidth wall

## Environment

- GPU: AMD Radeon RX 7900 XTX (gfx1100, RDNA3), 24 GB
- Stack: ROCm 6.4 container, PyTorch 2.8.0+rocm6.4 (HIP), MIOpen
- Model: SeedVR2 3B (numz/ComfyUI-SeedVR2_VideoUpscaler v2.5.24),
  480p→1080p, 121-frame end-to-end run
- Workload config: batch 21, tile 256 / overlap 32, temporal slicing
  split 4, bf16

## Evidence chain

### 1. Where decode time goes (runtime instrumentation)

`decode` dominates the pipeline (≈63% of end-to-end). Instrumenting the VAE
decode path shows **Conv3d = 96%** of decode GPU time (61.0 s of 63.6 s for
21 frames), concentrated in a few shape families:

| shape family | share of conv time |
|---|---|
| `[1,256,4,256,256]` 3×3×3 (upsamplers + resnets) | 46.4% |
| `[1,128,4,256,256]` 3×3×3 (6 resnet convs) | 24.3% |
| `[1,512,4,128,128]` 3×3×3 | remainder |

### 2. Kernel-level attribution (rocprof)

`rocprof` on the decode phase, kernels sorted by accumulated GPU time:

| component | share |
|---|---|
| **Im3d2Col.kd (im2col)** | **64.3%** |
| Cijk GEMM (rocBLAS) | 26.2% |
| elementwise | 9.5% |
| attention | 0.2% |

### 3. The solver: GemmFwdRest (im2col + GEMM) is the only option

`MIOPEN_LOG_LEVEL=6` shows that on gfx1100 the 3D conv shapes in question have
**exactly one MIOpen solver**: `GemmFwdRest` (im2col + rocBLAS GEMM). There is
no direct or implicit-GEMM candidate for 3D convolutions on this target — the
2D conv stack, by contrast, has `Direct` solver candidates (which is why 2D
conv is fast enough to matter).

### 4. The bandwidth wall, quantified

For the hot shape `[1,256,4,256,256]`, kernel 3×3×3, C=256 in/out:

- input: 1·256·4·256·256 · 2 B (bf16) = **128 MiB**
- im2col column matrix: 4 rows × 256·9 columns × 256·256 positions =
  256·2304·65536 · 2 B = **3.38 GiB** (26× the input)
- total traffic ≈ 2 × (3.38 + 0.13) GiB ≈ **6.9 GiB per conv**

At ~960 GB/s (HBM, bf16) the bandwidth-ideal time is ~7.7 ms; the measured
kernel time is 8.3 ms. **The cost is im2col traffic, not arithmetic** — GEMM
itself is only 26%. This is a bandwidth wall, and no solver/algorithm switch
can fix a wall: `MIOPEN_FIND_MODE=FAST/NORMAL/1/2` was verified to have no
effect.

### 5. Why the fix is Conv2D decomposition

The im2col blow-up is inherent to how MIOpen executes 3D convs on this
target. Avoiding Conv3D entirely removes the Im3d2Col stage. The causal
3×3×3 conv can be executed as three 2D convs (see
[`math_equivalence.md`](math_equivalence.md)); microbenchmark on the hot
shape shows 31.2 → 5.9 ms (**5.3×**) for 256-ch and 14.6 → 3.7 ms (3.9×) for
128-ch — i.e. the decomposition lands near the arithmetic-bound floor.

## Disabled directions (measured, do not repeat)

| direction | result | verdict |
|---|---|---|
| larger temporal chunks (batch 4→21) | 36.9 s → 35.8 s | **batch is irrelevant** — total im2col bytes unchanged; conv2d is bandwidth-bound too |
| spatial 32-alignment before conv2d | 54.7 s (slower) | MIOpen did not unlock a Direct solver on aligned shapes |
| `no_slicing` (whole latent, T=6) | >18 min MIOpen autotune | pathological im2col on stock Conv3D |
| channel-stack single conv2d (3C input) | >14 min autotune | mathematically equivalent but no ROCm win |
| `--compile_vae` (inductor) | rc=134 SIGABRT in decode | unusable on ROCm |
| `MIOPEN_FIND_MODE` | no effect | — |

Conclusion: the hot D=3 convs are bound by total im2col bytes; neither batch
size nor spatial alignment changes that. The only effective lever is removing
the 3D→2D im2col amplification, which is exactly what the decomposition does.
