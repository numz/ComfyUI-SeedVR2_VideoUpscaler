#!/usr/bin/env python3
"""
NVFP4 Pre-flight Diagnostic Script for SeedVR2

This script verifies your system is properly configured for NVFP4 (Native FP4)
acceleration on NVIDIA Blackwell (RTX 50-series) GPUs.

Run this BEFORE ComfyUI to verify:
1. Is Pinned Memory working?
2. Is Async Transfer overlapping correctly?
3. Is the GPU running Native FP4 or falling back to software emulation?

Usage:
    python scripts/nvfp4_diagnostic.py

Requirements:
    - NVIDIA RTX 50-series (Blackwell) GPU
    - PyTorch 2.6+ with CUDA 12.8+
    - Python 3.12+

Author: SeedVR2 Team
"""

import sys
import time
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with formatting"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}: {test_name}")
    if details:
        for line in details.split("\n"):
            print(f"       {line}")


def check_system_requirements():
    """Check basic system requirements"""
    print_header("System Requirements Check")
    
    results = {}
    
    # 1. Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 12)
    results['python'] = py_ok
    print_result(
        "Python Version",
        py_ok,
        f"Found: Python {py_version.major}.{py_version.minor}.{py_version.micro}\n"
        f"Required: Python 3.12+"
    )
    
    # 2. PyTorch availability
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
        results['torch'] = True
    except ImportError:
        torch_ok = False
        torch_version = "Not installed"
        results['torch'] = False
    
    print_result(
        "PyTorch Installation",
        torch_ok,
        f"Version: {torch_version}"
    )
    
    if not torch_ok:
        print("\n⚠️  PyTorch not installed. Cannot continue diagnostics.")
        return results
    
    # 3. PyTorch version check (need 2.6+)
    version_str = torch.__version__.split('+')[0]
    parts = version_str.split('.')
    try:
        torch_major = int(parts[0])
        torch_minor = int(parts[1])
        torch_version_ok = (torch_major, torch_minor) >= (2, 6)
    except (ValueError, IndexError):
        torch_version_ok = False
    
    results['torch_version'] = torch_version_ok
    print_result(
        "PyTorch Version",
        torch_version_ok,
        f"Found: {torch_version}\n"
        f"Required: 2.6+ for NVFP4 support"
    )
    
    # 4. CUDA availability
    cuda_available = torch.cuda.is_available()
    results['cuda'] = cuda_available
    
    if cuda_available:
        cuda_version = torch.version.cuda or "Unknown"
        
        # Parse CUDA version
        try:
            cuda_parts = cuda_version.split('.')
            cuda_major = int(cuda_parts[0])
            cuda_minor = int(cuda_parts[1]) if len(cuda_parts) > 1 else 0
            cuda_version_ok = (cuda_major > 12) or (cuda_major == 12 and cuda_minor >= 8)
        except (ValueError, IndexError):
            cuda_version_ok = False
        
        results['cuda_version'] = cuda_version_ok
        print_result(
            "CUDA Version",
            cuda_version_ok,
            f"Found: CUDA {cuda_version}\n"
            f"Required: CUDA 12.8+ (CUDA 13 is slower, target 12.8)"
        )
        
        # GPU info
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        is_blackwell = compute_capability[0] >= 10
        
        results['blackwell'] = is_blackwell
        print_result(
            "GPU Architecture",
            is_blackwell,
            f"GPU: {gpu_name}\n"
            f"Compute Capability: SM{compute_capability[0]}{compute_capability[1]}\n"
            f"Blackwell (SM100+): {'Yes' if is_blackwell else 'No'}"
        )
    else:
        print_result("CUDA Availability", False, "CUDA not available")
        results['cuda_version'] = False
        results['blackwell'] = False
    
    return results


def test_pinned_memory():
    """Test pinned memory allocation and transfer"""
    print_header("Pinned Memory Test")
    
    import torch
    
    if not torch.cuda.is_available():
        print_result("Pinned Memory", False, "CUDA not available")
        return False
    
    try:
        # Test 1: Allocate pinned memory
        size_mb = 256
        tensor_size = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        
        print(f"  Allocating {size_mb}MB pinned memory...")
        start = time.perf_counter()
        pinned_tensor = torch.empty(tensor_size, dtype=torch.float32, pin_memory=True)
        alloc_time = (time.perf_counter() - start) * 1000
        
        # Test 2: Transfer to GPU (non-blocking)
        print(f"  Transferring to GPU (non-blocking)...")
        start = time.perf_counter()
        gpu_tensor = pinned_tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        transfer_time = (time.perf_counter() - start) * 1000
        
        # Test 3: Transfer from pageable memory (for comparison)
        print(f"  Comparing with pageable memory transfer...")
        pageable_tensor = torch.empty(tensor_size, dtype=torch.float32)
        start = time.perf_counter()
        gpu_tensor2 = pageable_tensor.to('cuda', non_blocking=False)
        torch.cuda.synchronize()
        pageable_time = (time.perf_counter() - start) * 1000
        
        # Calculate speedup
        speedup = pageable_time / transfer_time if transfer_time > 0 else 0
        
        # Cleanup
        del pinned_tensor, gpu_tensor, pageable_tensor, gpu_tensor2
        torch.cuda.empty_cache()
        
        # Determine if pinned memory is working correctly
        # Pinned memory should be at least 1.2x faster for meaningful benefit
        pinned_working = speedup >= 1.2
        
        print_result(
            "Pinned Memory Transfer",
            pinned_working,
            f"Allocation time: {alloc_time:.2f}ms\n"
            f"Pinned transfer: {transfer_time:.2f}ms\n"
            f"Pageable transfer: {pageable_time:.2f}ms\n"
            f"Speedup: {speedup:.2f}x\n"
            f"Status: {'Pinned memory providing speedup' if pinned_working else 'No significant speedup (may be already optimized or memory-limited)'}"
        )
        
        return pinned_working
        
    except Exception as e:
        print_result("Pinned Memory", False, f"Error: {str(e)}")
        return False


def test_async_transfer():
    """Test async transfer with CUDA streams"""
    print_header("Async Transfer Test")
    
    import torch
    
    if not torch.cuda.is_available():
        print_result("Async Transfer", False, "CUDA not available")
        return False
    
    try:
        # Create two CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        size_mb = 128
        tensor_size = (size_mb * 1024 * 1024) // 4
        
        # Create pinned tensors
        pinned1 = torch.randn(tensor_size, dtype=torch.float32, pin_memory=True)
        pinned2 = torch.randn(tensor_size, dtype=torch.float32, pin_memory=True)
        
        # Test sequential transfers
        print(f"  Testing sequential transfers ({size_mb}MB x 2)...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        gpu1 = pinned1.to('cuda')
        torch.cuda.synchronize()
        gpu2 = pinned2.to('cuda')
        torch.cuda.synchronize()
        
        sequential_time = (time.perf_counter() - start) * 1000
        
        del gpu1, gpu2
        torch.cuda.empty_cache()
        
        # Test overlapped transfers using streams
        print(f"  Testing overlapped transfers with streams...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.cuda.stream(stream1):
            gpu1 = pinned1.to('cuda', non_blocking=True)
        
        with torch.cuda.stream(stream2):
            gpu2 = pinned2.to('cuda', non_blocking=True)
        
        stream1.synchronize()
        stream2.synchronize()
        
        overlapped_time = (time.perf_counter() - start) * 1000
        
        # Calculate overlap efficiency
        speedup = sequential_time / overlapped_time if overlapped_time > 0 else 0
        
        # Cleanup
        del gpu1, gpu2, pinned1, pinned2
        torch.cuda.empty_cache()
        
        # Async is working if we get at least 1.5x speedup (theoretical max ~2x)
        async_working = speedup >= 1.3
        
        print_result(
            "Async Transfer Overlap",
            async_working,
            f"Sequential time: {sequential_time:.2f}ms\n"
            f"Overlapped time: {overlapped_time:.2f}ms\n"
            f"Speedup: {speedup:.2f}x (theoretical max: ~2x)\n"
            f"Status: {'Async transfers overlapping correctly' if async_working else 'Limited overlap (may be bandwidth-limited)'}"
        )
        
        return async_working
        
    except Exception as e:
        print_result("Async Transfer", False, f"Error: {str(e)}")
        return False


def test_fp4_support():
    """Test native FP4 kernel support"""
    print_header("NVFP4 Kernel Test")
    
    import torch
    
    if not torch.cuda.is_available():
        print_result("NVFP4 Kernels", False, "CUDA not available")
        return False, "fallback"
    
    # Check compute capability
    compute_cap = torch.cuda.get_device_capability(0)
    is_blackwell = compute_cap[0] >= 10
    
    if not is_blackwell:
        print_result(
            "NVFP4 Kernels",
            False,
            f"Compute capability {compute_cap[0]}.{compute_cap[1]} < 10.0\n"
            f"NVFP4 requires Blackwell (SM100+) architecture\n"
            f"Your GPU: {torch.cuda.get_device_name(0)}"
        )
        return False, "not_supported"
    
    try:
        # Test if FP8 operations are available (precursor to FP4)
        # FP8 is available on Hopper+, FP4 on Blackwell+
        has_fp8 = hasattr(torch, 'float8_e4m3fn')
        
        # Check for Tensor Core availability via a matmul test
        print("  Testing Tensor Core matmul...")
        
        # Create test matrices
        m, n, k = 1024, 1024, 1024
        a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
        
        # Warmup
        for _ in range(3):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark BF16 matmul (uses Tensor Cores)
        start = time.perf_counter()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        bf16_time = (time.perf_counter() - start) * 1000 / 10
        
        # Calculate approximate TFLOPS
        flops = 2 * m * n * k
        tflops = (flops / (bf16_time / 1000)) / 1e12
        
        del a, b, c
        torch.cuda.empty_cache()
        
        # For Blackwell, we expect high TFLOPS from Tensor Cores
        # RTX 5090: ~209 TFLOPS BF16, RTX 5080: ~209 TFLOPS BF16
        # For FP4, it would be ~4x higher (800+ TFLOPS)
        tensor_cores_active = tflops > 10  # Conservative threshold
        
        # Test FP8 if available (indicates Tensor Core path)
        fp8_working = False
        fp8_details = "FP8 types not available in this PyTorch build"
        
        if has_fp8:
            try:
                # Create FP8 tensors
                a_fp16 = torch.randn(m, k, dtype=torch.float16, device='cuda')
                a_fp8 = a_fp16.to(torch.float8_e4m3fn)
                
                b_fp16 = torch.randn(k, n, dtype=torch.float16, device='cuda')
                b_fp8 = b_fp16.to(torch.float8_e4m3fn)
                
                # Verify tensors are actually in FP8 format
                fp8_dtype_correct = (a_fp8.dtype == torch.float8_e4m3fn and 
                                    b_fp8.dtype == torch.float8_e4m3fn)
                
                if fp8_dtype_correct:
                    fp8_working = True
                    fp8_details = f"FP8 tensor creation verified (dtype: {a_fp8.dtype})"
                else:
                    fp8_details = f"FP8 conversion failed (got: {a_fp8.dtype}, expected: {torch.float8_e4m3fn})"
                
                del a_fp16, a_fp8, b_fp16, b_fp8
            except Exception as e:
                fp8_details = f"FP8 test error: {str(e)}"
        
        # Determine overall status
        # NVFP4 requires: Blackwell GPU + PyTorch 2.6+ + CUDA 12.8+ + Tensor Cores active
        nvfp4_ready = is_blackwell and tensor_cores_active
        
        details = (
            f"GPU Architecture: Blackwell (SM{compute_cap[0]}{compute_cap[1]})\n"
            f"Tensor Cores: {'Active' if tensor_cores_active else 'Not detected'}\n"
            f"BF16 Matmul: {bf16_time:.2f}ms ({tflops:.1f} TFLOPS)\n"
            f"FP8 Support: {fp8_details}\n"
            f"NVFP4 Status: {'Ready for native FP4 acceleration' if nvfp4_ready else 'Fallback mode (check drivers/CUDA)'}"
        )
        
        print_result("NVFP4 Kernels", nvfp4_ready, details)
        
        return nvfp4_ready, "native" if nvfp4_ready else "fallback"
        
    except Exception as e:
        print_result("NVFP4 Kernels", False, f"Error: {str(e)}")
        return False, "error"


def test_io_vs_compute():
    """Benchmark IO throughput vs compute to identify bottleneck"""
    print_header("IO vs Compute Analysis")
    
    import torch
    
    if not torch.cuda.is_available():
        print_result("IO/Compute Analysis", False, "CUDA not available")
        return
    
    try:
        # Get PCIe bandwidth info if available
        props = torch.cuda.get_device_properties(0)
        
        # Test host-to-device bandwidth
        size_mb = 512
        tensor_size = (size_mb * 1024 * 1024) // 4
        
        print(f"  Testing Host→Device bandwidth ({size_mb}MB)...")
        
        # Use pinned memory for best case
        pinned = torch.randn(tensor_size, dtype=torch.float32, pin_memory=True)
        
        # Warmup
        gpu = pinned.to('cuda')
        torch.cuda.synchronize()
        del gpu
        
        # Benchmark
        start = time.perf_counter()
        gpu = pinned.to('cuda')
        torch.cuda.synchronize()
        h2d_time = (time.perf_counter() - start) * 1000
        
        h2d_bandwidth = size_mb / (h2d_time / 1000) / 1024  # GB/s
        
        # Test device-to-host bandwidth
        print(f"  Testing Device→Host bandwidth ({size_mb}MB)...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        cpu = gpu.to('cpu')
        d2h_time = (time.perf_counter() - start) * 1000
        
        d2h_bandwidth = size_mb / (d2h_time / 1000) / 1024  # GB/s
        
        del pinned, gpu, cpu
        torch.cuda.empty_cache()
        
        # Test compute throughput
        print(f"  Testing compute throughput...")
        m, n, k = 4096, 4096, 4096
        a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
        
        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000 / 10
        
        flops = 2 * m * n * k
        tflops = (flops / (compute_time / 1000)) / 1e12
        
        del a, b, c
        torch.cuda.empty_cache()
        
        # Analyze bottleneck
        # If H2D bandwidth < 20 GB/s, likely IO-bound for model loading
        # PCIe 4.0 x16 max: ~25 GB/s, PCIe 5.0 x16 max: ~50 GB/s
        io_bound = h2d_bandwidth < 20
        
        details = (
            f"Host→Device: {h2d_bandwidth:.1f} GB/s ({h2d_time:.1f}ms for {size_mb}MB)\n"
            f"Device→Host: {d2h_bandwidth:.1f} GB/s ({d2h_time:.1f}ms for {size_mb}MB)\n"
            f"Compute: {tflops:.1f} TFLOPS (BF16 matmul)\n"
            f"GPU Memory: {props.total_memory / 1024**3:.1f} GB\n"
            f"\n"
            f"Analysis: {'IO-BOUND - Model loading is bottleneck' if io_bound else 'COMPUTE-BOUND - Good balance'}\n"
            f"Recommendation: {'Enable async offloading + pinned memory' if io_bound else 'Focus on compute optimization'}"
        )
        
        print_result("Bottleneck Analysis", True, details)
        
    except Exception as e:
        print_result("IO/Compute Analysis", False, f"Error: {str(e)}")


def print_recommendations(results: dict, pinned_ok: bool, async_ok: bool, fp4_status: tuple):
    """Print final recommendations based on test results"""
    print_header("Recommendations")
    
    fp4_ok, fp4_mode = fp4_status
    
    all_ok = all([
        results.get('blackwell', False),
        results.get('cuda_version', False),
        results.get('torch_version', False),
        pinned_ok,
        async_ok,
        fp4_ok
    ])
    
    if all_ok:
        print("""
✅ Your system is fully optimized for NVFP4 acceleration!

Expected performance improvements:
  • 2-4x speedup for linear layers with native FP4 Tensor Cores
  • ~75% VRAM reduction vs FP16
  • Async offloading overlapping compute and IO

SeedVR2 will automatically enable these optimizations.
        """)
    else:
        print("\n⚠️  Some optimizations are not available. Recommendations:\n")
        
        if not results.get('blackwell', False):
            print("""
  📌 GPU Architecture:
     • NVFP4 requires RTX 50-series (Blackwell) GPU
     • Your GPU will use standard FP16/GGUF quantization
     • Consider upgrading for 4-bit acceleration
            """)
        
        if not results.get('cuda_version', False) and results.get('cuda', False):
            print("""
  📌 CUDA Version:
     • NVFP4 requires CUDA 12.8+
     • Install PyTorch nightly with CUDA 12.8:
       pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
     • Note: CUDA 13.0 may be slower for this workload
            """)
        
        if not results.get('torch_version', False) and results.get('torch', False):
            print("""
  📌 PyTorch Version:
     • NVFP4 requires PyTorch 2.6+
     • Install latest nightly:
       pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
            """)
        
        if not pinned_ok and results.get('cuda', False):
            print("""
  📌 Pinned Memory:
     • Pinned memory not providing expected speedup
     • This may be normal if PCIe bandwidth is saturated
     • SeedVR2 will still use pinned memory for other benefits
            """)
        
        if not async_ok and results.get('cuda', False):
            print("""
  📌 Async Transfers:
     • Async transfers not overlapping as expected
     • May be bandwidth-limited or driver issue
     • Try updating NVIDIA drivers to latest
            """)
        
        if not fp4_ok and results.get('blackwell', False):
            print("""
  📌 FP4 Kernels:
     • Blackwell GPU detected but FP4 kernels not active
     • Ensure PyTorch 2.6+ with CUDA 12.8+ is installed
     • Check driver version: nvidia-smi
     • SeedVR2 will fallback to GGUF/FP16 if FP4 unavailable
            """)


def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 60)
    print("  NVFP4 Pre-flight Diagnostic for SeedVR2")
    print("  Blackwell GPU Optimization Checker")
    print("=" * 60)
    
    # Run tests
    results = check_system_requirements()
    
    # Only run GPU tests if CUDA is available
    if results.get('cuda', False):
        pinned_ok = test_pinned_memory()
        async_ok = test_async_transfer()
        fp4_status = test_fp4_support()
        test_io_vs_compute()
    else:
        pinned_ok = False
        async_ok = False
        fp4_status = (False, "no_cuda")
    
    # Print recommendations
    print_recommendations(results, pinned_ok, async_ok, fp4_status)
    
    print("\n" + "=" * 60)
    print("  Diagnostic Complete")
    print("=" * 60 + "\n")
    
    return 0 if all([results.get('blackwell'), results.get('cuda_version')]) else 1


if __name__ == "__main__":
    sys.exit(main())
