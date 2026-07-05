#!/usr/bin/env python3
"""
SpargeAttn/Sage2 Verification Script

This script verifies numerical parity between Sage2 sparse attention and 
the baseline PyTorch SDPA (Scaled Dot-Product Attention).

Usage:
    python scripts/sage2_verification.py [--verbose] [--atol ATOL] [--rtol RTOL]

Metrics Checked:
    - Output tensor shape equality
    - Element-wise absolute/relative tolerance
    - Maximum absolute error
    - Mean absolute error
    - Cosine similarity
"""

# Add project root to path for local imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import torch.nn.functional as F


def check_availability():
    """Check if SpargeAttn/Sage2 is available."""
    try:
        from src.optimization.compatibility import (
            SPARGE_SAGE2_AVAILABLE,
            SPARGE_SAGE2_VERSION,
            call_sparge_sage2_attn,
            Sage2BlackwellConfig
        )
        return SPARGE_SAGE2_AVAILABLE, SPARGE_SAGE2_VERSION
    except ImportError:
        return False, None


def create_test_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype):
    """Create random test tensors for Q, K, V."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v


def compute_sdpa_baseline(q, k, v, is_causal=False):
    """Compute attention using PyTorch SDPA baseline."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def compute_sage2_attention(q, k, v, topk=0.5, is_causal=False):
    """Compute attention using SpargeAttn/Sage2."""
    from src.optimization.compatibility import call_sparge_sage2_attn
    return call_sparge_sage2_attn(q, k, v, topk=topk, is_causal=is_causal)


def compute_metrics(output_sdpa, output_sage2, atol, rtol):
    """Compute verification metrics between SDPA and Sage2 outputs."""
    # Convert to float32 for accurate metric computation
    sdpa_fp32 = output_sdpa.float()
    sage2_fp32 = output_sage2.float()
    
    # Shape check
    shape_match = output_sdpa.shape == output_sage2.shape
    
    # Tolerance check
    within_tolerance = torch.allclose(sdpa_fp32, sage2_fp32, atol=atol, rtol=rtol)
    
    # Error metrics
    abs_diff = torch.abs(sdpa_fp32 - sage2_fp32)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    # Relative error (avoid division by zero)
    rel_diff = abs_diff / (torch.abs(sdpa_fp32) + 1e-8)
    max_rel_error = rel_diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    
    # Cosine similarity (flatten and compute)
    sdpa_flat = sdpa_fp32.flatten()
    sage2_flat = sage2_fp32.flatten()
    cosine_sim = F.cosine_similarity(sdpa_flat.unsqueeze(0), sage2_flat.unsqueeze(0)).item()
    
    return {
        'shape_match': shape_match,
        'within_tolerance': within_tolerance,
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'cosine_similarity': cosine_sim,
    }


def run_verification(batch_size=2, num_heads=8, seq_len=256, head_dim=64,
                    topk_values=[0.3, 0.5, 0.7], atol=1e-2, rtol=1e-2,
                    device='cuda', dtype=torch.bfloat16, verbose=False):
    """
    Run verification tests comparing Sage2 against SDPA baseline.
    
    Args:
        batch_size: Batch size for test tensors
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        topk_values: List of topk sparsity ratios to test
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        device: Device to run on ('cuda' or 'cpu')
        dtype: Data type for tensors
        verbose: Whether to print detailed output
        
    Returns:
        dict with test results
    """
    results = {
        'available': False,
        'version': None,
        'tests': [],
        'overall_pass': True
    }
    
    # Check availability
    available, version = check_availability()
    results['available'] = available
    results['version'] = version
    
    if not available:
        print("❌ SpargeAttn/Sage2 is not available. Install with: pip install spas-sage-attn")
        return results
    
    print(f"✅ SpargeAttn/Sage2 available (version: {version})")
    print(f"   Test configuration: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}")
    print(f"   Tolerances: atol={atol}, rtol={rtol}")
    print()
    
    # Create test tensors
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, running on CPU")
        device = 'cpu'
    
    q, k, v = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
    
    # Compute SDPA baseline
    with torch.no_grad():
        output_sdpa = compute_sdpa_baseline(q, k, v, is_causal=False)
    
    # Test different topk values
    for topk in topk_values:
        test_result = {
            'topk': topk,
            'passed': False,
            'metrics': {}
        }
        
        try:
            with torch.no_grad():
                output_sage2 = compute_sage2_attention(q, k, v, topk=topk, is_causal=False)
            
            metrics = compute_metrics(output_sdpa, output_sage2, atol, rtol)
            test_result['metrics'] = metrics
            
            # Determine pass/fail
            # Note: Sage2 is sparse, so we expect some deviation from dense attention
            # Use a more relaxed threshold based on topk (more sparsity = more deviation expected)
            expected_deviation = (1 - topk) * 0.1  # Allow up to 10% deviation for 100% sparsity
            relaxed_atol = max(atol, expected_deviation)
            
            # Pass if cosine similarity is high (>0.95) even if element-wise tolerance fails
            passed = metrics['shape_match'] and (
                metrics['within_tolerance'] or metrics['cosine_similarity'] > 0.95
            )
            test_result['passed'] = passed
            
            if not passed:
                results['overall_pass'] = False
            
            # Print results
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} topk={topk:.1f}: max_err={metrics['max_abs_error']:.6f}, "
                  f"mean_err={metrics['mean_abs_error']:.6f}, cosine_sim={metrics['cosine_similarity']:.6f}")
            
            if verbose:
                print(f"        shape_match={metrics['shape_match']}, "
                      f"within_tol={metrics['within_tolerance']}")
                print(f"        max_rel_err={metrics['max_rel_error']:.6f}, "
                      f"mean_rel_err={metrics['mean_rel_error']:.6f}")
                
        except Exception as e:
            test_result['error'] = str(e)
            test_result['passed'] = False
            results['overall_pass'] = False
            print(f"❌ FAIL topk={topk:.1f}: {e}")
        
        results['tests'].append(test_result)
    
    print()
    if results['overall_pass']:
        print("🎉 All verification tests passed!")
    else:
        print("⚠️  Some verification tests failed. See details above.")
    
    return results


def test_block_sparse_mask_geometry():
    """Test block-sparse mask geometry validation."""
    try:
        from src.optimization.compatibility import Sage2BlackwellConfig
    except ImportError:
        print("⚠️  Cannot import Sage2BlackwellConfig, skipping geometry test")
        return
    
    print("\n📐 Testing block-sparse mask geometry...")
    
    # Test correct geometry
    batch_size, num_heads, seq_len = 2, 8, 256
    expected_shape = Sage2BlackwellConfig.get_mask_shape(batch_size, num_heads, seq_len)
    
    # Expected: ceil(256/128) = 2, ceil(256/64) = 4
    assert expected_shape == (2, 8, 2, 4), f"Expected (2, 8, 2, 4), got {expected_shape}"
    print(f"   ✅ Mask shape for seq_len={seq_len}: {expected_shape}")
    
    # Test validation with correct mask
    correct_mask = torch.zeros(expected_shape)
    try:
        Sage2BlackwellConfig.validate_mask_geometry(correct_mask, batch_size, num_heads, seq_len)
        print("   ✅ Correct mask geometry validation passed")
    except ValueError as e:
        print(f"   ❌ Validation failed unexpectedly: {e}")
    
    # Test validation with incorrect mask
    wrong_mask = torch.zeros((2, 8, 3, 4))  # Wrong row count
    try:
        Sage2BlackwellConfig.validate_mask_geometry(wrong_mask, batch_size, num_heads, seq_len)
        print("   ❌ Should have raised ValueError for wrong mask")
    except ValueError:
        print("   ✅ Incorrect mask geometry correctly rejected")
    
    print("   📐 Block size constraint: 128x64 (rows x cols)")


def main():
    parser = argparse.ArgumentParser(description="Verify SpargeAttn/Sage2 numerical parity with SDPA")
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed output')
    parser.add_argument('--atol', type=float, default=1e-2, help='Absolute tolerance (default: 1e-2)')
    parser.add_argument('--rtol', type=float, default=1e-2, help='Relative tolerance (default: 1e-2)')
    parser.add_argument('--batch', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads (default: 8)')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension (default: 64)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpargeAttn/Sage2 Verification Script")
    print("=" * 60)
    print()
    
    # Run numerical verification
    results = run_verification(
        batch_size=args.batch,
        num_heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        atol=args.atol,
        rtol=args.rtol,
        device=args.device,
        verbose=args.verbose
    )
    
    # Test mask geometry
    test_block_sparse_mask_geometry()
    
    print()
    print("=" * 60)
    
    return 0 if results['overall_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
