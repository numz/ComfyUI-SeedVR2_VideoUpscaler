#!/usr/bin/env python3
"""
SpargeAttn/Sage2 Benchmark Script

This script benchmarks attention mechanisms to compare performance:
- Throughput (tokens/second)
- Memory efficiency (peak VRAM usage)
- Inference latency (milliseconds)

Optimized for NVIDIA Blackwell (RTX 50xx) GPUs.

Usage:
    python scripts/sage2_benchmark.py [options]
    
Options:
    --batch-sizes     Batch sizes to test (default: 1,2,4)
    --seq-lengths     Sequence lengths to test (default: 256,512,1024)
    --warmup          Warmup iterations (default: 5)
    --iterations      Benchmark iterations (default: 20)
    --device          Device to use (default: cuda)
"""

# Add project root to path for local imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import gc
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    attention_mode: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    
    # Timing metrics
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Throughput (tokens/second)
    tokens_per_second: float
    
    # Memory (peak VRAM in MB)
    peak_memory_mb: float
    
    # Optional sparse parameters
    topk: Optional[float] = None


def check_availability() -> Dict[str, bool]:
    """Check availability of all attention backends."""
    try:
        from src.optimization.compatibility import (
            FLASH_ATTN_2_AVAILABLE,
            FLASH_ATTN_3_AVAILABLE,
            SAGE_ATTN_2_AVAILABLE,
            SAGE_ATTN_3_AVAILABLE,
            SPARGE_SAGE2_AVAILABLE,
            BLACKWELL_GPU_DETECTED,
        )
        return {
            'flash_attn_2': FLASH_ATTN_2_AVAILABLE,
            'flash_attn_3': FLASH_ATTN_3_AVAILABLE,
            'sageattn_2': SAGE_ATTN_2_AVAILABLE,
            'sageattn_3': SAGE_ATTN_3_AVAILABLE,
            'sparge_sage2': SPARGE_SAGE2_AVAILABLE,
            'blackwell_gpu': BLACKWELL_GPU_DETECTED,
            'sdpa': True,  # Always available
        }
    except ImportError:
        return {'sdpa': True}


def create_test_tensors(batch_size: int, num_heads: int, seq_len: int, 
                       head_dim: int, device: str, dtype: torch.dtype):
    """Create random test tensors for benchmarking."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v


def benchmark_sdpa(q, k, v, warmup: int, iterations: int) -> Dict[str, float]:
    """Benchmark PyTorch SDPA."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
    
    # Clear memory tracking
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    
    return {
        'latencies': latencies,
        'peak_memory_mb': peak_memory
    }


def benchmark_sparge_sage2(q, k, v, topk: float, warmup: int, iterations: int) -> Dict[str, float]:
    """Benchmark SpargeAttn/Sage2."""
    from src.optimization.compatibility import call_sparge_sage2_attn
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = call_sparge_sage2_attn(q, k, v, topk=topk, is_causal=False)
        torch.cuda.synchronize()
    
    # Clear memory tracking
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = call_sparge_sage2_attn(q, k, v, topk=topk, is_causal=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return {
        'latencies': latencies,
        'peak_memory_mb': peak_memory
    }


def compute_statistics(latencies: List[float], batch_size: int, 
                       seq_len: int) -> Dict[str, float]:
    """Compute timing statistics from latency measurements."""
    import statistics
    
    mean_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    
    # Tokens per second: batch_size * seq_len * 1000 / mean_latency_ms
    tokens_per_second = (batch_size * seq_len * 1000) / mean_latency
    
    return {
        'mean_latency_ms': mean_latency,
        'std_latency_ms': std_latency,
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'tokens_per_second': tokens_per_second,
    }


def run_benchmark(batch_sizes: List[int], seq_lengths: List[int],
                 num_heads: int = 16, head_dim: int = 64,
                 warmup: int = 5, iterations: int = 20,
                 device: str = 'cuda', dtype: torch.dtype = torch.bfloat16,
                 topk_values: List[float] = [0.3, 0.5, 0.7]) -> List[BenchmarkResult]:
    """
    Run comprehensive benchmarks across different configurations.
    
    Returns list of BenchmarkResult objects.
    """
    results = []
    availability = check_availability()
    
    print("=" * 80)
    print("SpargeAttn/Sage2 Benchmark")
    print("=" * 80)
    print()
    
    # Print availability
    print("Attention Backend Availability:")
    for backend, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {backend}")
    print()
    
    if not availability.get('sparge_sage2', False):
        print("⚠️  SpargeAttn/Sage2 not available. Only benchmarking SDPA baseline.")
        print("   Install with: pip install spas-sage-attn")
        print()
    
    # Check CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, running on CPU (limited accuracy)")
        device = 'cpu'
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]})")
        if capability[0] >= 10:
            print("🚀 Blackwell GPU detected - optimized for RTX 50xx")
        print()
    
    print(f"Configuration:")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")
    print(f"  Dtype: {dtype}")
    print()
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print("-" * 60)
            print(f"Batch: {batch_size}, Seq length: {seq_len}")
            print("-" * 60)
            
            # Create tensors
            q, k, v = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
            
            # Benchmark SDPA baseline
            try:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                sdpa_result = benchmark_sdpa(q, k, v, warmup, iterations)
                stats = compute_statistics(sdpa_result['latencies'], batch_size, seq_len)
                
                result = BenchmarkResult(
                    attention_mode='sdpa',
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mean_latency_ms=stats['mean_latency_ms'],
                    std_latency_ms=stats['std_latency_ms'],
                    min_latency_ms=stats['min_latency_ms'],
                    max_latency_ms=stats['max_latency_ms'],
                    tokens_per_second=stats['tokens_per_second'],
                    peak_memory_mb=sdpa_result['peak_memory_mb'],
                )
                results.append(result)
                
                print(f"  SDPA:        {stats['mean_latency_ms']:.3f}ms ± {stats['std_latency_ms']:.3f}ms | "
                      f"{stats['tokens_per_second']:.0f} tok/s | {sdpa_result['peak_memory_mb']:.1f}MB")
                
                baseline_latency = stats['mean_latency_ms']
                baseline_memory = sdpa_result['peak_memory_mb']
                
            except Exception as e:
                print(f"  SDPA:        ❌ Error: {e}")
                baseline_latency = None
                baseline_memory = None
            
            # Benchmark SpargeAttn/Sage2
            if availability.get('sparge_sage2', False):
                for topk in topk_values:
                    try:
                        gc.collect()
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        
                        sage2_result = benchmark_sparge_sage2(q, k, v, topk, warmup, iterations)
                        stats = compute_statistics(sage2_result['latencies'], batch_size, seq_len)
                        
                        result = BenchmarkResult(
                            attention_mode='sparge_sage2',
                            batch_size=batch_size,
                            seq_len=seq_len,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            mean_latency_ms=stats['mean_latency_ms'],
                            std_latency_ms=stats['std_latency_ms'],
                            min_latency_ms=stats['min_latency_ms'],
                            max_latency_ms=stats['max_latency_ms'],
                            tokens_per_second=stats['tokens_per_second'],
                            peak_memory_mb=sage2_result['peak_memory_mb'],
                            topk=topk,
                        )
                        results.append(result)
                        
                        # Calculate speedup and memory savings
                        speedup = ""
                        mem_saving = ""
                        if baseline_latency:
                            speedup_ratio = baseline_latency / stats['mean_latency_ms']
                            speedup = f" ({speedup_ratio:.2f}x)"
                        if baseline_memory:
                            saving = (baseline_memory - sage2_result['peak_memory_mb']) / baseline_memory * 100
                            mem_saving = f" ({saving:+.1f}%)"
                        
                        print(f"  Sage2 k={topk}: {stats['mean_latency_ms']:.3f}ms ± {stats['std_latency_ms']:.3f}ms{speedup} | "
                              f"{stats['tokens_per_second']:.0f} tok/s | {sage2_result['peak_memory_mb']:.1f}MB{mem_saving}")
                        
                    except Exception as e:
                        print(f"  Sage2 k={topk}: ❌ Error: {e}")
            
            # Cleanup
            del q, k, v
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            print()
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary of benchmark results."""
    if not results:
        print("No benchmark results to summarize.")
        return
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # Group by configuration
    sdpa_results = [r for r in results if r.attention_mode == 'sdpa']
    sage2_results = [r for r in results if r.attention_mode == 'sparge_sage2']
    
    if sdpa_results:
        avg_latency = sum(r.mean_latency_ms for r in sdpa_results) / len(sdpa_results)
        avg_memory = sum(r.peak_memory_mb for r in sdpa_results) / len(sdpa_results)
        avg_throughput = sum(r.tokens_per_second for r in sdpa_results) / len(sdpa_results)
        print(f"SDPA Baseline:")
        print(f"  Average latency:    {avg_latency:.3f} ms")
        print(f"  Average memory:     {avg_memory:.1f} MB")
        print(f"  Average throughput: {avg_throughput:.0f} tokens/s")
        print()
    
    if sage2_results:
        # Group by topk
        for topk in sorted(set(r.topk for r in sage2_results if r.topk)):
            topk_results = [r for r in sage2_results if r.topk == topk]
            avg_latency = sum(r.mean_latency_ms for r in topk_results) / len(topk_results)
            avg_memory = sum(r.peak_memory_mb for r in topk_results) / len(topk_results)
            avg_throughput = sum(r.tokens_per_second for r in topk_results) / len(topk_results)
            
            print(f"Sage2 (topk={topk}):")
            print(f"  Average latency:    {avg_latency:.3f} ms")
            print(f"  Average memory:     {avg_memory:.1f} MB")
            print(f"  Average throughput: {avg_throughput:.0f} tokens/s")
            print()
    
    # Compute overall speedup
    if sdpa_results and sage2_results:
        sdpa_avg = sum(r.mean_latency_ms for r in sdpa_results) / len(sdpa_results)
        sage2_avg = sum(r.mean_latency_ms for r in sage2_results) / len(sage2_results)
        overall_speedup = sdpa_avg / sage2_avg
        
        sdpa_mem = sum(r.peak_memory_mb for r in sdpa_results) / len(sdpa_results)
        sage2_mem = sum(r.peak_memory_mb for r in sage2_results) / len(sage2_results)
        mem_saving = (sdpa_mem - sage2_mem) / sdpa_mem * 100
        
        print("Overall vs SDPA Baseline:")
        print(f"  Speed improvement:  {overall_speedup:.2f}x")
        print(f"  Memory savings:     {mem_saving:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SpargeAttn/Sage2 attention mechanisms")
    parser.add_argument('--batch-sizes', type=str, default='1,2,4',
                       help='Comma-separated batch sizes (default: 1,2,4)')
    parser.add_argument('--seq-lengths', type=str, default='256,512,1024',
                       help='Comma-separated sequence lengths (default: 256,512,1024)')
    parser.add_argument('--heads', type=int, default=16,
                       help='Number of attention heads (default: 16)')
    parser.add_argument('--head-dim', type=int, default=64,
                       help='Head dimension (default: 64)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Warmup iterations (default: 5)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Benchmark iterations (default: 20)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to benchmark on (default: cuda)')
    parser.add_argument('--topk', type=str, default='0.3,0.5,0.7',
                       help='Comma-separated topk sparsity values (default: 0.3,0.5,0.7)')
    
    args = parser.parse_args()
    
    # Parse comma-separated values
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    topk_values = [float(x) for x in args.topk.split(',')]
    
    # Run benchmark
    results = run_benchmark(
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        num_heads=args.heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        topk_values=topk_values,
    )
    
    # Print summary
    print_summary(results)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
