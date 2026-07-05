# Changelog

All notable changes to the ComfyUI-SeedVR2.5 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### SpargeAttn/Sage2 Block-Sparse Attention Integration
- **New attention mode: `sparge_sage2`** - Block-sparse attention optimized for NVIDIA Blackwell (RTX 50xx) GPUs
- **Local vendored implementation** - No global installation required, uses Triton JIT compilation
- Plug-and-play replacement for PyTorch SDPA using `spas_sage2_attn_meansim_topk_cuda`
- Custom block-sparse patterns via `block_sparse_sage2_attn_cuda` with strict mask geometry (128x64 blocks)
- Automatic fallback chain: `sparge_sage2` → `sageattn_3` → `sageattn_2` → `sdpa`

#### Local SpargeAttn Module (`src/optimization/spas_sage_attn/`)
- **Triton JIT compilation** - Kernels compile on first use, optimized for CUDA 12.8+ and 13.x
- Pure Python/Triton implementation - No MSVC/NVCC compilation conflicts
- Files included:
  - `core.py` - Main API functions (`spas_sage2_attn_meansim_topk_cuda`, `block_sparse_sage2_attn_cuda`)
  - `utils.py` - Utility functions for block map computation
  - `quant_per_block.py` - INT8 quantization kernels
  - `autotune.py` - Triton autotuning utilities
- Automatic GPU architecture detection (Blackwell sm100+, Hopper sm90, Ampere sm80+)

#### Blackwell (RTX 50xx) Specific Optimizations
- **`Sage2BlackwellConfig`** class with Blackwell-tuned parameters:
  - Optimized topk sparsity ratios (0.3 fast, 0.5 balanced, 0.7 quality)
  - Block size: 128x64 matching Sage2 kernel expectations
  - Triton kernel parameters tuned for Blackwell L1 cache (128KB) and Tensor Cores
  - FP8/BF16 precision optimization settings
- Automatic Blackwell GPU detection with compute capability checks
- Native FP8 dispatch integration for 4-bit Tensor Core acceleration
- `get_blackwell_config()` function for architecture-specific kernel tuning

#### Verification & Benchmarking Scripts
- `scripts/sage2_verification.py` - Numerical parity verification against SDPA baseline
  - Supports multiple topk sparsity ratios
  - Reports max/mean absolute error, cosine similarity
  - Tests block-sparse mask geometry validation
- `scripts/sage2_benchmark.py` - Comprehensive performance benchmarking
  - Throughput (tokens/second)
  - Peak VRAM memory usage
  - Inference latency with statistical analysis
  - Comparison against SDPA baseline

#### Compatibility Layer Enhancements
- New wrapper functions in `src/optimization/compatibility.py`:
  - `call_sparge_sage2_attn()` - Direct Sage2 attention call
  - `call_block_sparse_sage2_attn()` - Block-sparse with custom masks
  - `call_sparge_sage2_varlen()` - Variable-length sequence support
- Mask geometry validation with `Sage2BlackwellConfig.validate_mask_geometry()`
- SpargeAttn availability detection and version reporting
- **Dual import strategy**: Tries local vendored module first, falls back to global package

### Changed

#### Dependencies
- `torch>=2.3.0` - Minimum PyTorch version for CUDA 12.x compatibility
- `ninja>=1.11` - Required for SpargeAttn Triton kernel compilation

#### Attention Backends
- Updated `FlashAttentionVarlen` class (both dit_3b and dit_7b) to support `sparge_sage2` mode
- Enhanced attention mode validation with SpargeAttn-specific fallback logic
- Updated startup logging to display SpargeAttn/Sage2 availability status

### Technical Details

#### Sage2 API Usage
The Sage2 architecture provides two primary APIs:

1. **Plug-and-Play API** (recommended for most use cases):
   ```python
   from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
   output = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5, is_causal=False)
   ```

2. **Block-Sparse API** (for custom sparsity patterns):
   ```python
   from spas_sage_attn import block_sparse_sage2_attn_cuda
   # mask_id shape: (batch, heads, ceil(seq/128), ceil(seq/64))
   output = block_sparse_sage2_attn_cuda(q, k, v, mask_id)
   ```

#### Blackwell-Specific Tuning
- **Triton Parameters**: `num_warps=8`, `num_stages=4`, `block_m=128`, `block_n=64`
- **Sparsity Thresholds**:
  - `TOPK_FAST = 0.3` - Maximum speed, some accuracy tradeoff
  - `TOPK_BALANCED = 0.5` - Default, balanced speed/accuracy
  - `TOPK_QUALITY = 0.7` - Higher quality, less speedup
- **Precision**: Prefers FP8 on Blackwell, falls back to BF16 for compatibility

#### Block-Sparse Mask Geometry
The block-sparse API requires masks with specific geometry:
- Shape: `(batch_size, num_heads, ceil(seq_len/128), ceil(seq_len/64))`
- Block size: 128 rows × 64 columns
- Non-zero entries indicate which blocks to compute

### Installation

#### Prerequisites
- NVIDIA GPU with CUDA 12.8+ (Blackwell for optimal performance, also supports CUDA 13.x)
- PyTorch 2.3.0 or later
- Triton (included with PyTorch, used for JIT kernel compilation)

#### Local Integration (Recommended - No Build Required)
The SpargeAttn implementation is now vendored locally in `src/optimization/spas_sage_attn/`.
No separate installation is needed - Triton kernels compile JIT on first use.

```bash
# Just ensure Triton is available (usually bundled with PyTorch)
pip install triton

# The local implementation will be used automatically
python -c "from src.optimization.compatibility import SPARGE_SAGE2_AVAILABLE, SPARGE_SAGE2_VERSION; print(f'Available: {SPARGE_SAGE2_AVAILABLE}, Version: {SPARGE_SAGE2_VERSION}')"
```

#### Global Installation (Optional - For Full CUDA Kernel Support)
For maximum performance with precompiled CUDA kernels (if local JIT has issues):
```bash
# Install dependencies
pip install ninja>=1.11

# Build from source:
git clone https://github.com/thu-ml/SpargeAttn
cd SpargeAttn
python setup.py install
```

#### Verification
```bash
# Check availability
python -c "from src.optimization.compatibility import SPARGE_SAGE2_AVAILABLE; print(f'SpargeAttn available: {SPARGE_SAGE2_AVAILABLE}')"

# Run verification tests
python scripts/sage2_verification.py --verbose

# Run benchmarks
python scripts/sage2_benchmark.py --batch-sizes 1,2 --seq-lengths 256,512
```

### Performance Notes

#### Expected Performance (Blackwell GPUs)
Based on Sage2 architecture characteristics:
- **Throughput**: Up to 2x improvement with topk=0.5 sparsity
- **Memory**: 10-30% reduction in peak VRAM usage
- **Latency**: 1.5-2x faster inference for long sequences

#### Fallback Behavior
- If SpargeAttn is unavailable, the system gracefully falls back to SageAttention 3/2 or PyTorch SDPA
- Variable-length sequences automatically fall back to SageAttention 2 (SpargeAttn uses batched attention)
- All attention modes maintain numerical stability with automatic dtype conversion

### Known Limitations
- SpargeAttn Sage2 requires uniform sequence lengths (varlen falls back to SageAttention 2)
- Block-sparse masks must follow strict 128x64 geometry
- Optimal performance requires CUDA 12.8+ and Blackwell architecture

### Migration Guide

#### Enabling SpargeAttn/Sage2
To use the new attention mode, set `attention_mode='sparge_sage2'` in your pipeline configuration:

```python
# In model configuration
attention_mode = 'sparge_sage2'  # New Blackwell-optimized mode

# The system will automatically fall back if SpargeAttn is not available
```

#### Adjusting Sparsity
For custom sparsity levels, pass the `topk` parameter through kwargs:

```python
# Lower topk = more sparsity = faster but less accurate
# Higher topk = less sparsity = slower but more accurate
kwargs['topk'] = 0.5  # Default balanced setting
```
