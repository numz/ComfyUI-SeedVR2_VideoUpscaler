# Blackwell (RTX 50-series) Optimization Guide

This guide covers the NVFP4 and async offloading optimizations for NVIDIA RTX 50-series (Blackwell architecture) GPUs in SeedVR2.

## Overview

NVIDIA Blackwell GPUs (RTX 5070/5080/5090) introduce native FP4 (4-bit floating point) support via 5th generation Tensor Cores. SeedVR2 leverages these capabilities for:

- **2-4x speedup** for linear layers with native FP4 Tensor Cores
- **~75% VRAM reduction** compared to FP16 models
- **Overlapped compute and IO** via async offloading

## Prerequisites

### Hardware Requirements
- NVIDIA RTX 50-series GPU (RTX 5070, 5070 Ti, 5080, 5090)
- Compute capability 10.0+ (SM120 architecture)

### Software Requirements

#### CUDA Version
**Target CUDA 12.8+** (NOT CUDA 13.0)

> ⚠️ **Important**: Testing has shown that CUDA 13.0 is currently SLOWER than CUDA 12.8 for SeedVR2 workloads. Use CUDA 12.8 for optimal performance.

#### PyTorch Version
**PyTorch 2.6+ with CUDA 12.8**

Install the recommended nightly build:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### Driver Requirements
- NVIDIA Driver 565.xx or newer (for CUDA 12.8 support)
- Verify with: `nvidia-smi`

#### Python Requirements
- Python 3.12+ recommended

### Required Packages
```bash
# Core dependencies (standard installation)
pip install safetensors omegaconf einops

# For NVFP4 quantization utilities (optional)
pip install nvidia-modelopt  # Optional: for advanced quantization

# Recommended acceleration packages
pip install flash-attn --no-build-isolation  # Flash Attention 2/3
pip install sageattention  # SageAttention 2
pip install triton  # For torch.compile
```

## Diagnostic Tool

Before running ComfyUI, verify your system configuration:

```bash
python scripts/nvfp4_diagnostic.py
```

This script tests:
1. **Pinned Memory** - Verifies DMA transfers are working
2. **Async Transfers** - Confirms CUDA stream overlap
3. **FP4 Kernels** - Checks native Tensor Core activation
4. **IO vs Compute Analysis** - Identifies bottlenecks

### Expected Output (Fully Optimized)
```
✅ PASS: Python Version
✅ PASS: PyTorch Version
✅ PASS: CUDA Version
✅ PASS: GPU Architecture
✅ PASS: Pinned Memory Transfer
✅ PASS: Async Transfer Overlap
✅ PASS: NVFP4 Kernels
```

## Optimizations Enabled

### 1. Native FP4 Dispatch
SeedVR2 automatically configures PyTorch for optimal FP4 kernel selection:
- TF32 enabled for Tensor Core operations
- cuDNN benchmark mode for kernel auto-tuning
- Blackwell-specific compute paths when available

### 2. Pinned Memory Pool
A reusable pool of pinned (page-locked) memory buffers enables:
- DMA transfers without CPU copies
- Non-blocking async transfers
- Reduced allocation overhead

Pool configuration:
- Default size: 4GB (6GB for Blackwell GPUs)
- Automatic LRU eviction when full
- Hit rate tracking for optimization

### 3. CUDA Stream Management
Dedicated streams for different operations:
- **H2D Stream**: Host-to-Device transfers
- **D2H Stream**: Device-to-Host transfers
- **Compute Stream**: Model inference

This enables overlapping data movement with computation.

### 4. Layer Prefetching
For BlockSwap-style model loading:
- Next layer prefetched while current layer computes
- Minimizes IO stalls during inference
- Automatic synchronization management

## Usage

### Automatic Detection
SeedVR2 automatically detects Blackwell GPUs and enables optimizations:

```
🚀 NVFP4 Blackwell optimization: ✅ (NVIDIA GeForce RTX 5090 - 4-bit Tensor Core acceleration enabled)
   └─ Native FP4 dispatch configured (TF32 enabled, cuDNN benchmark active)
```

### Manual Configuration

#### Enable/Disable Pinned Memory
The async offloader respects system memory constraints:
```python
from src.optimization.nvfp4 import AsyncModelOffloader

offloader = AsyncModelOffloader(
    use_pinned_memory=True,  # Enable pinned memory
    max_pinned_pool_gb=6.0   # Max pool size
)
```

#### Force FP4 Dispatch
```python
from src.optimization.nvfp4 import ensure_native_fp4_dispatch

if ensure_native_fp4_dispatch():
    print("Native FP4 dispatch active")
```

## Troubleshooting

### "NVFP4 same speed as GGUF"
This typically indicates IO-bound inference. Solutions:

1. **Enable async offloading**: Already enabled by default
2. **Check PCIe bandwidth**: Run diagnostic tool to verify
3. **Increase pinned pool**: Set larger `max_pinned_pool_gb`
4. **Reduce model swapping**: Use smaller `blocks_to_swap` in BlockSwap

### "Pinned memory allocation failed"
System may be low on non-pageable memory:
- Close other GPU applications
- Reduce pinned pool size
- Check system RAM availability

### "CUDA out of memory"
Blackwell GPUs have high VRAM, but large models may still exceed:
- Enable BlockSwap with aggressive offloading
- Use tiled VAE encoding/decoding
- Reduce batch size

## Performance Expectations

### RTX 5090 (32GB VRAM)
- DiT 7B: ~2-3x faster than FP16 with NVFP4
- Full video upscaling: ~40-50% faster end-to-end

### RTX 5080 (16GB VRAM)
- DiT 3B: Optimal performance
- DiT 7B: May require BlockSwap

### RTX 5070 Ti (16GB VRAM)
- Similar to RTX 5080
- Async offloading essential for large models

## Changelog

### v2.5.1 - Blackwell Optimization Update

#### New Features
- **NVFP4 Support**: Native 4-bit floating point for Blackwell Tensor Cores
- **Pinned Memory Pool**: Reusable buffer pool with LRU eviction
- **CUDA Stream Manager**: Dedicated streams for H2D/D2H/Compute operations
- **Layer Prefetching**: Overlapped layer loading for BlockSwap
- **Diagnostic Script**: Pre-flight system verification tool

#### Optimizations
- Switched to CUDA Stream handling for layer loading
- Enforced Native FP4 dispatcher for Blackwell GPUs
- Added automatic TF32/cuDNN benchmark configuration
- Implemented async tensor transfers with pinned memory
- Added hit rate tracking for pinned memory pool

#### Fixes
- IO bottleneck causing "same speed as GGUF" issue
- Non-overlapping transfers when loading model layers
- Fallback to software emulation on Blackwell GPUs

#### Requirements
- Target CUDA 12.8+ (CUDA 13.0 known to be slower)
- PyTorch 2.6+ with CUDA 12.8 wheels
- Blackwell GPU (SM120/compute capability 10.0+)

## Technical Details

### E2M1 Format (NVFP4)
NVFP4 uses E2M1 format for weights:
- 1 sign bit
- 2 exponent bits (bias=1)
- 1 mantissa bit

Representable values: `0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0`

### Block-wise Scaling
Each block of 16 weights shares an E4M3 scale factor:
- Preserves accuracy with <1% quality degradation
- Optimal for Tensor Core tile sizes

### Preserved Layers
Critical layers remain in FP16 for quality:
- Bias terms
- Normalization layers (LayerNorm, GroupNorm, RMSNorm)
- Embedding layers
- Output heads

## References

- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [PyTorch FP8 Support](https://pytorch.org/docs/stable/generated/torch.float8_e4m3fn.html)
- [CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
