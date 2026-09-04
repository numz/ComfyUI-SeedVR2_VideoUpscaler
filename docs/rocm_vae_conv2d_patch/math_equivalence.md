# Mathematical equivalence of the decomposition

## Setup

`InflatedCausalConv3d` is a `nn.Conv3d` whose temporal padding is **removed in
the constructor** (`self.padding = (0, *self.padding[1:])`); the causal
head-extension is performed by the surrounding machinery
(`extend_head(input, times = padding[0]·2)` inside `basic_forward` /
`slicing_forward`). Therefore the convolution actually executed is

```
kernel  (Kt, Kh, Kw) = (3, 3, 3),  stride = (1, 1, 1),
padding = (0, 1, 1),   dilation = (1, 1, 1),  groups = 1
```

on an input `X'` of length `T+2` (the original `T` frames plus two head-copied
past frames).

## Exactness

Discrete 3D convolution with zero temporal padding is separable in the tap
index `d`:

```
Y[t, oh, ow] = Σ_c Σ_i Σ_j  W[c', c, d, i, j] · X'[t, c, oh+i-1, ow+j-1]   (d = 0..2)
```

Because the sum over `d` is over an index that never mixes across output
positions, it can be rearranged as a sum of three independent 2D convolutions:

```
Y = W[0] ⊛ X'[:, :, 0:T] + W[1] ⊛ X'[:, :, 1:T+1] + W[2] ⊛ X'[:, :, 2:T+2]
```

where each `W[d]` is the 2D slice `weight[:, :, d, :, :]` and `⊛` denotes
`F.conv2d(·, W[d], padding=1)`. The three terms are accumulated in place
(`add_`), then bias is added once. **No term is dropped, scaled, or
approximated** — the only change vs. the stock `F.conv3d` is the evaluation
order of a finite sum, so in exact arithmetic the outputs are identical.

## T=1 special case (community summed-weight path)

For a single frame, the head-extension makes `X'[0] = X'[1] = X'[2] = X0`, so

```
Y = W[0]·X0 + W[1]·X0 + W[2]·X0 = (W[0]+W[1]+W[2]) · X0
```

i.e. one 2D convolution with the summed weights — the fast path used by
community single-frame implementations. The general decomposition above
produces exactly this result automatically; it is a strict superset, valid for
all `T ≥ 1`.

## Empirical verification

`tests/test_conv2d_decomposition.py` checks patched vs. stock `F.conv3d`:

- 5 random shapes, fp32: `max_abs ≤ 6.4e-5` (accumulation-order noise)
- T=1 vs. summed-weight `F.conv2d`: `max_abs ≈ 8e-6`
- tiled path (padding zeroed, input pre-padded): `max_abs ≈ 7e-5`
- real checkpoint weights (`ema_vae_fp16.safetensors`), fp32: `max_abs ≤ 8e-5`
- real weights, bf16 (production dtype): `max_abs ≈ 1.6e-2` — bf16
  accumulation noise, visually irrelevant; the full-pipeline PSNR vs. baseline
  is 47.36 dB / SSIM 0.9905.

## Scope guard

The patch only fires when all of: `kernel_size == (3,3,3)`, `input.dim() == 5`,
`stride == (1,1,1)`, `padding_mode == "zeros"`. Otherwise it calls the original
`_conv_forward` — bit-identical to stock. It then **asserts**
`dilation == (1,1,1)`, `groups == 1`, `padding[0] == 0`,
`padding[1] == padding[2]`: if a future checkpoint changes any of these, the
process fails loudly instead of silently producing wrong output.
