"""
Compatibility module for SeedVR2
Contains FP8/FP16 compatibility layers and wrappers for different model architectures

Extracted from: seedvr2.py (lines 1045-1630)
"""

# Compatibility shims - Must run before any torch/diffusers import
import sys
import types
import importlib.machinery


def ensure_triton_compat():
    """Create minimal triton.ops stubs only if missing, to allow bitsandbytes import."""
    if 'triton.ops.matmul_perf_model' in sys.modules:
        return
    
    try:
        from triton.ops.matmul_perf_model import early_config_prune  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    
    if 'triton.ops' not in sys.modules:
        sys.modules['triton.ops'] = types.ModuleType('triton.ops')
    
    matmul_perf = types.ModuleType('triton.ops.matmul_perf_model')
    matmul_perf.early_config_prune = lambda configs, *a, **kw: configs
    matmul_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    
    sys.modules['triton.ops'].matmul_perf_model = matmul_perf
    sys.modules['triton.ops.matmul_perf_model'] = matmul_perf


def ensure_flash_attn_safe():
    """
    Pre-test flash_attn package; stub if DLL is broken.
    Prevents diffusers from crashing when flash_attn has broken DLLs.
    """
    if 'flash_attn' in sys.modules:
        return  # Already loaded
    
    try:
        import flash_attn
    except (ImportError, OSError):
        # DLL broken or not installed - create stub with proper __spec__
        stub = types.ModuleType('flash_attn')
        stub.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__loader__ = None
        # Provide attributes that diffusers/transformers import
        stub.flash_attn_func = None
        stub.flash_attn_varlen_func = None
        sys.modules['flash_attn'] = stub


def ensure_xformers_flash_compat():
    """
    Pre-test xformers._C_flashattention; stub if DLL is broken.
    Prevents xformers.ops.fmha.flash from crashing on import.
    """
    if 'xformers._C_flashattention' in sys.modules:
        return  # Already loaded
    
    try:
        from xformers import _C_flashattention  # noqa: F401
    except (ImportError, OSError):
        # DLL broken or not installed - create stub with proper __spec__
        class _FailingStub(types.ModuleType):
            """Stub that lets xformers gracefully disable its flash backend."""
            def __getattr__(self, name):
                raise ImportError("_C_flashattention unavailable")
        
        stub = _FailingStub('xformers._C_flashattention')
        stub.__spec__ = importlib.machinery.ModuleSpec('xformers._C_flashattention', None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__loader__ = None
        sys.modules['xformers._C_flashattention'] = stub


def ensure_bitsandbytes_safe():
    """
    Pre-test bitsandbytes; stub if broken to prevent import conflicts.
    
    On some systems (e.g., ROCm without proper binaries), bitsandbytes registers
    PyTorch kernels during import then fails. If another node already triggered 
    this partial load, re-importing causes kernel registration conflicts.
    
    This shim catches such failures and stubs the module so diffusers can load
    gracefully without bitsandbytes quantization support.
    """
    if 'bitsandbytes' in sys.modules:
        return  # Already loaded or stubbed
    
    try:
        import bitsandbytes
        # Success - bitsandbytes works, other nodes can use it
    except (ImportError, OSError, RuntimeError, ValueError):
        # Installation broken, not present, or version detection failed - create stub
        stub = types.ModuleType('bitsandbytes')
        stub.__spec__ = importlib.machinery.ModuleSpec('bitsandbytes', None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__version__ = "0.0.0"
        sys.modules['bitsandbytes'] = stub


# Run all shims immediately on import, before torch/diffusers
ensure_triton_compat()
ensure_flash_attn_safe()
ensure_xformers_flash_compat()
ensure_bitsandbytes_safe()


import torch
import os
import math
import logging

# Configure logging for Blackwell optimization verification
logger = logging.getLogger("SeedVR2.Blackwell")

# Global frame counter for per-frame verification logging
_sparge_sage2_frame_counter = 0

# Windows/Performance optimization: control logging and cache behavior via environment variables
# Set SEEDVR2_VERBOSE_LOGGING=1 to enable detailed per-call logging (for debugging)
# Set SEEDVR2_CACHE_CLEARING=1 to enable torch.cuda.empty_cache() between phases
ENABLE_VERBOSE_LOGGING = os.environ.get('SEEDVR2_VERBOSE_LOGGING', '0') == '1'
ENABLE_CACHE_CLEARING = os.environ.get('SEEDVR2_CACHE_CLEARING', '0') == '1'


def reset_sparge_sage2_verification(clear_cache: bool = None):
    """
    Reset the sparge_sage2 frame counter and optionally clear CUDA cache.
    Call this before starting a new DiT phase to ensure clean kernel execution.
    
    Args:
        clear_cache: Override cache clearing behavior. If None, uses ENABLE_CACHE_CLEARING.
                     Cache clearing is disabled by default for performance.
    """
    global _sparge_sage2_frame_counter
    _sparge_sage2_frame_counter = 0
    
    # Determine if we should clear cache
    should_clear = clear_cache if clear_cache is not None else ENABLE_CACHE_CLEARING
    
    # Clear CUDA cache only if explicitly requested (disabled by default for performance)
    if should_clear and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if ENABLE_VERBOSE_LOGGING:
            print(">> MEMORY ISOLATION: CUDA cache cleared before DiT phase", flush=True)
            logger.info("CUDA cache cleared before DiT phase for memory isolation")


# Flash/Sage Attention & Triton Compatibility Layer

# 1. Flash Attention 3 (Hopper+, faster, no dropout/window support)
flash_attn_3_varlen_func = None
FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn_interface
    flash_attn_3_varlen_func = flash_attn_interface.flash_attn_varlen_func
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

# 2. Flash Attention 2 (wider compatibility, supports dropout/window)
flash_attn_2_varlen_func = None
FLASH_ATTN_2_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen
    import flash_attn_2_cuda  # noqa: F401
    flash_attn_2_varlen_func = _fa2_varlen
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

FLASH_ATTN_AVAILABLE = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE

# 3. SageAttention 2 (varlen support)
sageattn_varlen = None
SAGE_ATTN_2_AVAILABLE = False
try:
    from sageattention import sageattn_varlen as _sa2_varlen
    sageattn_varlen = _sa2_varlen
    SAGE_ATTN_2_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass

# 4. SageAttention 3 / Blackwell (RTX 50xx only, batched attention)
sageattn_blackwell = None
SAGE_ATTN_3_AVAILABLE = False
try:
    from sageattn3 import sageattn3_blackwell as _sa3_blackwell
    sageattn_blackwell = _sa3_blackwell
    SAGE_ATTN_3_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    try:
        from sageattention import sageattn_blackwell as _sa3_blackwell
        sageattn_blackwell = _sa3_blackwell
        SAGE_ATTN_3_AVAILABLE = True
    except (ImportError, AttributeError, OSError):
        pass

SAGE_ATTN_AVAILABLE = SAGE_ATTN_2_AVAILABLE or SAGE_ATTN_3_AVAILABLE

# 5. SpargeAttn / Sage2 (Block-sparse attention for Blackwell optimization)
# Provides spas_sage2_attn_meansim_topk_cuda for plug-and-play SDPA replacement
# and block_sparse_sage2_attn_cuda for custom block-sparse patterns
# Uses local vendored implementation with Triton JIT compilation (no global install required)
spas_sage2_attn_meansim_topk = None
block_sparse_sage2_attn = None
SPARGE_SAGE2_AVAILABLE = False
SPARGE_SAGE2_VERSION = None
_SPARGE_IMPORT_ERROR = None  # Store error for diagnostics

# Try local vendored implementation first (Triton JIT, no compilation needed)
try:
    from .spas_sage_attn import spas_sage2_attn_meansim_topk_cuda as _spas_sage2
    from .spas_sage_attn import block_sparse_sage2_attn_cuda as _block_sparse_sage2
    from .spas_sage_attn import SPARGE_LOCAL_AVAILABLE, SPARGE_LOCAL_VERSION
    if SPARGE_LOCAL_AVAILABLE:
        spas_sage2_attn_meansim_topk = _spas_sage2
        block_sparse_sage2_attn = _block_sparse_sage2
        SPARGE_SAGE2_AVAILABLE = True
        SPARGE_SAGE2_VERSION = SPARGE_LOCAL_VERSION
    else:
        _SPARGE_IMPORT_ERROR = "Local SpargeAttn module loaded but SPARGE_LOCAL_AVAILABLE is False (Triton may not be available)"
except (ImportError, AttributeError, OSError) as e:
    _SPARGE_IMPORT_ERROR = f"Local import failed: {type(e).__name__}: {e}"
    # Print diagnostic for debugging
    print(f"[SpargeAttn Debug] {_SPARGE_IMPORT_ERROR}")
    # Fall back to globally installed package if local fails
    try:
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda as _spas_sage2
        from spas_sage_attn import block_sparse_sage2_attn_cuda as _block_sparse_sage2
        spas_sage2_attn_meansim_topk = _spas_sage2
        block_sparse_sage2_attn = _block_sparse_sage2
        SPARGE_SAGE2_AVAILABLE = True
        _SPARGE_IMPORT_ERROR = None  # Clear error on success
        try:
            import spas_sage_attn
            SPARGE_SAGE2_VERSION = getattr(spas_sage_attn, '__version__', 'unknown')
        except (ImportError, AttributeError):
            SPARGE_SAGE2_VERSION = 'unknown'
    except (ImportError, AttributeError, OSError) as e2:
        _SPARGE_IMPORT_ERROR = f"Both local and global imports failed. Global error: {type(e2).__name__}: {e2}"
        print(f"[SpargeAttn Debug] {_SPARGE_IMPORT_ERROR}")


# Blackwell-specific Sage2 configuration for RTX 50xx GPUs
class Sage2BlackwellConfig:
    """
    Configuration for Sage2 sparse attention optimized for NVIDIA Blackwell (RTX 50xx) GPUs.
    
    Blackwell-specific optimizations:
    - Enhanced L1 cache utilization (128KB vs 64KB on Ada)
    - FP8/BF16 Tensor Core throughput optimization
    - Tuned block sizes for 5th gen Tensor Cores
    - Optimized sparsity thresholds for compute/accuracy tradeoff
    
    Block-sparse mask geometry:
    - mask_id shape: (batch_size, num_heads, ceil(seq_len/128), ceil(seq_len/64))
    - Block size: 128x64 (rows x cols)
    """
    
    # Default topk sparsity ratio (0.5 = 50% of attention weights kept)
    # Lower values = more sparsity = faster but potentially less accurate
    DEFAULT_TOPK = 0.5
    
    # Blackwell-optimized topk for different use cases
    TOPK_FAST = 0.3       # Maximum speed, some accuracy tradeoff
    TOPK_BALANCED = 0.5   # Balanced speed/accuracy (default)
    TOPK_QUALITY = 0.7    # Higher quality, less speedup
    
    # Block-sparse mask dimensions (must match Sage2 kernel expectations)
    BLOCK_ROWS = 128      # Query block size
    BLOCK_COLS = 64       # Key/Value block size
    
    # Triton kernel parameters tuned for Blackwell architecture
    # These leverage the increased SM count and L1 cache
    TRITON_NUM_WARPS = 8          # Optimal for Blackwell SM architecture
    TRITON_NUM_STAGES = 4         # Memory pipeline stages
    TRITON_BLOCK_M = 128          # Matches block-sparse row size
    TRITON_BLOCK_N = 64           # Matches block-sparse col size
    
    # Precision settings for Blackwell
    PREFER_FP8 = True             # Use FP8 when hardware supports it
    FALLBACK_DTYPE = torch.bfloat16  # Fallback for non-FP8 operations
    
    @classmethod
    def get_mask_shape(cls, batch_size: int, num_heads: int, seq_len: int):
        """
        Calculate the required mask_id shape for block-sparse attention.
        
        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            seq_len: Sequence length
            
        Returns:
            Tuple of (batch_size, num_heads, ceil(seq_len/128), ceil(seq_len/64))
        """
        rows = math.ceil(seq_len / cls.BLOCK_ROWS)
        cols = math.ceil(seq_len / cls.BLOCK_COLS)
        return (batch_size, num_heads, rows, cols)
    
    @classmethod
    def validate_mask_geometry(cls, mask_id: torch.Tensor, batch_size: int, 
                                num_heads: int, seq_len: int) -> bool:
        """
        Validate that mask_id has correct shape for block-sparse Sage2 attention.
        
        Args:
            mask_id: The block-sparse mask tensor
            batch_size: Expected batch size
            num_heads: Expected number of heads
            seq_len: Sequence length
            
        Returns:
            True if mask geometry is valid
            
        Raises:
            ValueError: If mask geometry is invalid
        """
        expected_shape = cls.get_mask_shape(batch_size, num_heads, seq_len)
        if mask_id.shape != expected_shape:
            raise ValueError(
                f"Invalid mask_id shape for block-sparse Sage2 attention.\n"
                f"Expected: {expected_shape} (batch, heads, ceil(seq/{cls.BLOCK_ROWS}), ceil(seq/{cls.BLOCK_COLS}))\n"
                f"Got: {mask_id.shape}\n"
                f"Block size constraint: {cls.BLOCK_ROWS}x{cls.BLOCK_COLS}"
            )
        return True


def validate_attention_mode(requested_mode: str, debug=None) -> str:
    """
    Validate attention mode availability with automatic fallback.
    
    Args:
        requested_mode: 'sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3', or 'sparge_sage2'
        debug: Optional debug instance for logging
        
    Returns:
        Validated mode that is available
    """
    # Flash Attention 3
    if requested_mode == 'flash_attn_3':
        if FLASH_ATTN_3_AVAILABLE:
            return requested_mode
        if FLASH_ATTN_2_AVAILABLE:
            if debug:
                debug.log(
                    "Flash Attention 3 not available (requires Hopper+ GPU and flash-attn with FA3 support).\n"
                    "Falling back to Flash Attention 2.",
                    level="WARNING", category="setup", force=True
                )
            return 'flash_attn_2'
        error_msg = (
            "Cannot use 'flash_attn_3' attention mode: Flash Attention is not installed.\n"
            "\n"
            "Flash Attention 3 provides maximum speedup on Hopper+ GPUs through optimized CUDA kernels.\n"
            "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            "\n"
            "To fix this issue:\n"
            "  1. Install Flash Attention: pip install flash-attn\n"
            "  2. OR change attention_mode to 'sdpa' (default, always available)\n"
            "\n"
            "For more info: https://github.com/Dao-AILab/flash-attention"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        return 'sdpa'
    
    # Flash Attention 2
    if requested_mode == 'flash_attn_2':
        if FLASH_ATTN_2_AVAILABLE:
            return requested_mode
        error_msg = (
            "Cannot use 'flash_attn_2' attention mode: Flash Attention 2 is not installed.\n"
            "\n"
            "Flash Attention 2 provides speedup on Ampere+ GPUs through optimized CUDA kernels.\n"
            "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            "\n"
            "To fix this issue:\n"
            "  1. Install Flash Attention: pip install flash-attn\n"
            "  2. OR change attention_mode to 'sdpa' (default, always available)\n"
            "\n"
            "For more info: https://github.com/Dao-AILab/flash-attention"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        return 'sdpa'
    
    # SageAttention 3 (Blackwell)
    if requested_mode == 'sageattn_3':
        if SAGE_ATTN_3_AVAILABLE:
            return requested_mode
        if SAGE_ATTN_2_AVAILABLE:
            if debug:
                debug.log(
                    "SageAttention 3 (Blackwell) not available (requires RTX 50xx GPU and sageattn3 package).\n"
                    "Falling back to SageAttention 2.",
                    level="WARNING", category="setup", force=True
                )
            return 'sageattn_2'
        error_msg = (
            "Cannot use 'sageattn_3' attention mode: SageAttention is not installed.\n"
            "\n"
            "SageAttention 3 provides maximum speedup on Blackwell (RTX 50xx) GPUs.\n"
            "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            "\n"
            "To fix this issue:\n"
            "  1. Install SageAttention: pip install sageattention\n"
            "  2. For SA3 Blackwell support: pip install sageattn3\n"
            "  3. OR change attention_mode to 'flash_attn_2' or 'sdpa'\n"
            "\n"
            "For more info: https://github.com/thu-ml/SageAttention"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        return 'sdpa'
    
    # SageAttention 2
    if requested_mode == 'sageattn_2':
        if SAGE_ATTN_2_AVAILABLE:
            return requested_mode
        error_msg = (
            "Cannot use 'sageattn_2' attention mode: SageAttention is not installed.\n"
            "\n"
            "SageAttention provides speedup on NVIDIA GPUs through optimized CUDA kernels.\n"
            "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            "\n"
            "To fix this issue:\n"
            "  1. Install SageAttention: pip install sageattention\n"
            "  2. OR change attention_mode to 'flash_attn_2' or 'sdpa'\n"
            "\n"
            "For more info: https://github.com/thu-ml/SageAttention"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        return 'sdpa'
    
    # SpargeAttn / Sage2 (Block-sparse attention for Blackwell)
    if requested_mode == 'sparge_sage2':
        if SPARGE_SAGE2_AVAILABLE:
            return requested_mode
        # Fallback chain: sageattn_3 -> sageattn_2 -> sdpa
        if SAGE_ATTN_3_AVAILABLE:
            if debug:
                debug.log(
                    "SpargeAttn/Sage2 not available (requires spas_sage_attn package).\n"
                    "Falling back to SageAttention 3 (Blackwell).",
                    level="WARNING", category="setup", force=True
                )
            return 'sageattn_3'
        if SAGE_ATTN_2_AVAILABLE:
            if debug:
                debug.log(
                    "SpargeAttn/Sage2 not available (requires spas_sage_attn package).\n"
                    "Falling back to SageAttention 2.",
                    level="WARNING", category="setup", force=True
                )
            return 'sageattn_2'
        error_msg = (
            "Cannot use 'sparge_sage2' attention mode: SpargeAttn is not installed.\n"
            "\n"
            "SpargeAttn/Sage2 provides block-sparse attention optimized for Blackwell (RTX 50xx) GPUs.\n"
            "It uses the spas_sage2_attn_meansim_topk_cuda API for plug-and-play SDPA replacement.\n"
            "Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            "\n"
            "To fix this issue:\n"
            "  1. Install SpargeAttn: pip install spas-sage-attn\n"
            "  2. For CUDA 12.8+ support, ensure proper toolchain: pip install ninja>=1.11\n"
            "  3. OR use 'sageattn_3', 'flash_attn_2', or 'sdpa' attention modes\n"
            "\n"
            "For more info: https://github.com/thu-ml/SpargeAttn"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        return 'sdpa'
    
    return requested_mode


@torch._dynamo.disable
def call_flash_attn_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    """
    Wrapper for Flash Attention 2 flash_attn_varlen_func that handles tensor-to-scalar conversion.
    
    Flash Attention 2 supports dropout_p and window_size parameters.
    Works on Ampere+ GPUs (RTX 30xx, 40xx, A100, etc.).
    
    This function is excluded from torch.compile because:
    1. flash_attn is a C++ extension that can't be compiled anyway
    2. It requires Python int scalars for max_seqlen parameters
    3. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (total_seq, heads, head_dim)
        k: Key tensor (total_seq, heads, head_dim)
        v: Value tensor (total_seq, heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length (can be tensor or int)
        max_seqlen_k: Maximum key sequence length (can be tensor or int)
        **kwargs: Additional arguments (dropout_p, softmax_scale, causal, window_size, deterministic)
        
    Returns:
        Attention output tensor (total_seq, heads, head_dim)
    """
    if not FLASH_ATTN_2_AVAILABLE:
        raise ImportError("Flash Attention 2 is not available")
    
    # Convert tensor max_seqlen to Python int if needed
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    return flash_attn_2_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        **kwargs
    )


@torch._dynamo.disable
def call_flash_attn_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    """
    Wrapper for Flash Attention 3 flash_attn_varlen_func that handles tensor-to-scalar conversion.
    
    Flash Attention 3 is faster than FA2 but does NOT support dropout_p and window_size.
    Works on Hopper+ GPUs (H100, etc.) - requires flash_attn_interface package.
    
    This function is excluded from torch.compile because:
    1. flash_attn is a C++ extension that can't be compiled anyway
    2. It requires Python int scalars for max_seqlen parameters
    3. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (total_seq, heads, head_dim)
        k: Key tensor (total_seq, heads, head_dim)
        v: Value tensor (total_seq, heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length (can be tensor or int)
        max_seqlen_k: Maximum key sequence length (can be tensor or int)
        **kwargs: Additional arguments (softmax_scale, causal, deterministic)
                  Note: dropout_p and window_size are ignored (not supported by FA3)
        
    Returns:
        Attention output tensor (total_seq, heads, head_dim)
    """
    if not FLASH_ATTN_3_AVAILABLE:
        raise ImportError("Flash Attention 3 is not available")
    
    # Convert tensor max_seqlen to Python int if needed
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    # FA3 doesn't support dropout_p and window_size - filter them out
    fa3_kwargs = {key: val for key, val in kwargs.items() if key not in ('dropout_p', 'window_size')}
    
    # FA3 returns a tuple (output, softmax_lse), we only need output
    return flash_attn_3_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=None,
        seqused_k=None,
        **fa3_kwargs
    )[0]


@torch._dynamo.disable
def call_sage_attn_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    """
    Wrapper for SageAttention 2 sageattn_varlen that handles tensor-to-scalar conversion.
    
    SageAttention 2 provides optimized attention for NVIDIA GPUs with native varlen support.
    Works on most modern NVIDIA GPUs.
    
    This function is excluded from torch.compile because:
    1. SageAttention is a C++ extension that can't be compiled anyway
    2. It requires Python int scalars for max_seqlen parameters
    3. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (total_seq, heads, head_dim)
        k: Key tensor (total_seq, heads, head_dim)
        v: Value tensor (total_seq, heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length (can be tensor or int)
        max_seqlen_k: Maximum key sequence length (can be tensor or int)
        **kwargs: Additional arguments (causal supported, others ignored)
        
    Returns:
        Attention output tensor (total_seq, heads, head_dim)
    """
    if not SAGE_ATTN_2_AVAILABLE:
        raise ImportError("SageAttention 2 is not available")
    
    # Convert tensor max_seqlen to Python int if needed
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    # SageAttention requires half precision (fp16/bf16)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    
    is_causal = kwargs.get('causal', False)
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    out = sageattn_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        is_causal, sm_scale
    )
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_sage_attn_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    """
    Wrapper for SageAttention 3 (Blackwell) that converts varlen format to batched format.
    
    SageAttention 3 / Blackwell provides maximum performance on RTX 50xx series GPUs.
    However, it only supports batched attention (uniform sequence lengths), not varlen.
    
    This wrapper detects uniform-length batches and reshapes accordingly.
    For variable-length sequences, it automatically falls back to SageAttention 2.
    
    This function is excluded from torch.compile because:
    1. SageAttention is a C++ extension that can't be compiled anyway
    2. It requires Python int scalars for max_seqlen parameters
    3. The varlen-to-batched conversion involves dynamic shapes
    4. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (total_seq, heads, head_dim)
        k: Key tensor (total_seq, heads, head_dim)
        v: Value tensor (total_seq, heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length (can be tensor or int)
        max_seqlen_k: Maximum key sequence length (can be tensor or int)
        **kwargs: Additional arguments (passed to SA2 fallback if needed)
        
    Returns:
        Attention output tensor (total_seq, heads, head_dim)
    """
    if not SAGE_ATTN_3_AVAILABLE:
        raise ImportError("SageAttention 3 (Blackwell) is not available")
    
    # Convert tensor max_seqlen to Python int if needed
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    # Check if all sequences have uniform length (required for SA3 batched API)
    # SA3/Blackwell uses batched attention, not varlen, so we need uniform lengths
    seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    
    uniform_q = (seq_lens_q == seq_lens_q[0]).all()
    uniform_k = (seq_lens_k == seq_lens_k[0]).all()
    
    if not (uniform_q and uniform_k):
        # Fall back to SA2 for variable-length sequences
        # This is expected behavior - SA3 Blackwell doesn't support varlen natively
        if SAGE_ATTN_2_AVAILABLE:
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        raise RuntimeError(
            "SageAttention 3 (Blackwell) requires uniform sequence lengths, "
            "and SageAttention 2 is not available as fallback. "
            "Please install sageattention package or use flash_attn/sdpa instead."
        )
    
    # Extract batch dimensions
    batch_size = len(cu_seqlens_q) - 1
    seq_len_q = int(seq_lens_q[0].item())
    seq_len_k = int(seq_lens_k[0].item())
    heads = q.shape[1]
    dim = q.shape[2]
    
    # SageAttention requires half precision (fp16/bf16)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    
    # Reshape varlen (total_seq, heads, dim) -> batched (batch, seq, heads, dim)
    q_batched = q.view(batch_size, seq_len_q, heads, dim)
    k_batched = k.view(batch_size, seq_len_k, heads, dim)
    v_batched = v.view(batch_size, seq_len_k, heads, dim)
    
    # SA3/Blackwell expects (batch, heads, seq, dim) layout
    q_batched = q_batched.transpose(1, 2)  # (batch, heads, seq, dim)
    k_batched = k_batched.transpose(1, 2)
    v_batched = v_batched.transpose(1, 2)
    
    # Call SA3 Blackwell
    out = sageattn_blackwell(q_batched, k_batched, v_batched, per_block_mean=False)
    
    # Reshape back to varlen format (total_seq, heads, dim)
    out = out.transpose(1, 2).reshape(-1, heads, dim).contiguous()
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_sparge_sage2_attn(q, k, v, topk=None, is_causal=False, **kwargs):
    """
    Wrapper for SpargeAttn Sage2 spas_sage2_attn_meansim_topk_cuda.
    
    This is the plug-and-play replacement for torch.nn.functional.scaled_dot_product_attention.
    Optimized for NVIDIA Blackwell (RTX 50xx) GPUs with block-sparse attention patterns.
    
    Uses mean-similarity based top-k selection to determine which attention weights to keep,
    providing a balance between computational efficiency and output quality.
    
    This function is excluded from torch.compile because:
    1. SpargeAttn is a CUDA extension that can't be compiled by Dynamo
    2. It uses custom Triton kernels that need direct execution
    3. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim) or (batch, seq_len, heads, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim) or (batch, seq_len, heads, head_dim)
        v: Value tensor (batch, heads, seq_len, head_dim) or (batch, seq_len, heads, head_dim)
        topk: Sparsity ratio (0.0-1.0). Default uses Sage2BlackwellConfig.DEFAULT_TOPK (0.5)
              Lower values = more sparsity = faster but potentially less accurate
        is_causal: Whether to apply causal masking (default: False)
        **kwargs: Additional arguments (ignored for API compatibility)
        
    Returns:
        Attention output tensor (same shape as input)
    """
    if not SPARGE_SAGE2_AVAILABLE:
        raise ImportError(
            "SpargeAttn/Sage2 is not available. "
            "Install with: pip install spas-sage-attn"
        )
    
    # Use Blackwell-optimized default if not specified
    if topk is None:
        topk = Sage2BlackwellConfig.DEFAULT_TOPK
    
    # SpargeAttn requires half precision (fp16/bf16)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        # Prefer bf16 for Blackwell's enhanced Tensor Core support
        target_dtype = Sage2BlackwellConfig.FALLBACK_DTYPE
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)
    
    # Call SpargeAttn Sage2 API
    out = spas_sage2_attn_meansim_topk(q, k, v, topk=topk, is_causal=is_causal)
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_block_sparse_sage2_attn(q, k, v, mask_id, **kwargs):
    """
    Wrapper for SpargeAttn Sage2 block_sparse_sage2_attn_cuda with custom block-sparse patterns.
    
    This function provides fine-grained control over sparsity patterns using a custom mask.
    Optimized for NVIDIA Blackwell (RTX 50xx) GPUs.
    
    IMPORTANT: Mask geometry must follow strict block size constraints:
    - mask_id shape: (batch_size, num_heads, ceil(seq_len/128), ceil(seq_len/64))
    - Block size: 128x64 (rows x cols)
    
    This function is excluded from torch.compile because:
    1. SpargeAttn is a CUDA extension that can't be compiled by Dynamo
    2. It uses custom Triton kernels that need direct execution
    3. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        v: Value tensor (batch, heads, seq_len, head_dim)
        mask_id: Block-sparse mask tensor with shape 
                 (batch_size, num_heads, ceil(seq_len/128), ceil(seq_len/64))
        **kwargs: Additional arguments (ignored for API compatibility)
        
    Returns:
        Attention output tensor (same shape as input)
        
    Raises:
        ValueError: If mask_id has incorrect geometry
    """
    if not SPARGE_SAGE2_AVAILABLE:
        raise ImportError(
            "SpargeAttn/Sage2 is not available. "
            "Install with: pip install spas-sage-attn"
        )
    
    # Validate mask geometry
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    seq_len = q.shape[2]
    Sage2BlackwellConfig.validate_mask_geometry(mask_id, batch_size, num_heads, seq_len)
    
    # SpargeAttn requires half precision (fp16/bf16)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        # Prefer bf16 for Blackwell's enhanced Tensor Core support
        target_dtype = Sage2BlackwellConfig.FALLBACK_DTYPE
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)
    
    # Call SpargeAttn Sage2 block-sparse API
    out = block_sparse_sage2_attn(q, k, v, mask_id)
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


@torch._dynamo.disable
def call_sparge_sage2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    """
    Wrapper for SpargeAttn Sage2 that handles variable-length sequences.
    
    Since SpargeAttn Sage2 uses batched attention (not varlen natively), this wrapper
    converts varlen format to batched format for uniform sequence lengths, or falls back
    to SageAttention 2 for true variable-length sequences.
    
    This function is excluded from torch.compile because:
    1. SpargeAttn is a CUDA extension that can't be compiled by Dynamo
    2. It requires Python int scalars for max_seqlen parameters
    3. The varlen-to-batched conversion involves dynamic shapes
    4. Disabling compilation here keeps the rest of the model compilable
    
    Args:
        q: Query tensor (total_seq, heads, head_dim)
        k: Key tensor (total_seq, heads, head_dim)
        v: Value tensor (total_seq, heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length (can be tensor or int)
        max_seqlen_k: Maximum key sequence length (can be tensor or int)
        **kwargs: Additional arguments (topk for sparsity ratio, etc.)
        
    Returns:
        Attention output tensor (total_seq, heads, head_dim)
    """
    if not SPARGE_SAGE2_AVAILABLE:
        raise ImportError("SpargeAttn/Sage2 is not available")
    
    # Convert tensor max_seqlen to Python int if needed
    if torch.is_tensor(max_seqlen_q):
        max_seqlen_q = int(max_seqlen_q.item())
    if torch.is_tensor(max_seqlen_k):
        max_seqlen_k = int(max_seqlen_k.item())
    
    # Check if all sequences have uniform length (required for batched Sage2)
    seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    
    uniform_q = (seq_lens_q == seq_lens_q[0]).all()
    uniform_k = (seq_lens_k == seq_lens_k[0]).all()
    
    if not (uniform_q and uniform_k):
        # Fall back to SA2 for variable-length sequences
        # SpargeAttn Sage2 doesn't support varlen natively
        if SAGE_ATTN_2_AVAILABLE:
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        raise RuntimeError(
            "SpargeAttn/Sage2 requires uniform sequence lengths, "
            "and SageAttention 2 is not available as fallback. "
            "Install with: pip install sageattention, or use flash_attn/sdpa instead."
        )
    
    # Extract batch dimensions
    batch_size = len(cu_seqlens_q) - 1
    seq_len_q = int(seq_lens_q[0].item())
    seq_len_k = int(seq_lens_k[0].item())
    heads = q.shape[1]
    dim = q.shape[2]
    
    # Get sparsity parameters with explicit mapping
    # Performance Mode Mapping: Fast=0.3, Balanced=0.5, High Quality=0.7
    topk = kwargs.get('topk', Sage2BlackwellConfig.DEFAULT_TOPK)
    is_causal = kwargs.get('causal', False)
    
    # STRICT VERIFICATION: Assert sparsity_threshold is a valid float
    assert isinstance(topk, (int, float)), f"sparsity_threshold must be a float, got {type(topk)}"
    assert 0.0 < topk <= 1.0, f"sparsity_threshold must be in (0.0, 1.0], got {topk}"
    
    # Update frame counter (always, for tracking)
    global _sparge_sage2_frame_counter
    _sparge_sage2_frame_counter += 1
    
    # Logging only on first call OR if verbose logging is explicitly enabled
    # This eliminates Python-side overhead during normal execution
    if _sparge_sage2_frame_counter == 1:
        # Map topk value to performance mode name for logging
        if abs(topk - 0.3) < 0.01:
            perf_mode = "Fast"
        elif abs(topk - 0.5) < 0.01:
            perf_mode = "Balanced"
        elif abs(topk - 0.7) < 0.01:
            perf_mode = "High Quality"
        else:
            perf_mode = f"Custom({topk})"
        
        blackwell_msg = (f"!!! BLACKWELL EXECUTION: Mode={perf_mode}, Threshold={topk}, "
                         f"Warps={Sage2BlackwellConfig.TRITON_NUM_WARPS}, Stages={Sage2BlackwellConfig.TRITON_NUM_STAGES}, "
                         f"BlockM={Sage2BlackwellConfig.TRITON_BLOCK_M}, BlockN={Sage2BlackwellConfig.TRITON_BLOCK_N}")
        print(blackwell_msg, flush=True)
        logger.info(blackwell_msg)
    elif ENABLE_VERBOSE_LOGGING and _sparge_sage2_frame_counter % 100 == 0:
        # Verbose logging every 100th call (only when SEEDVR2_VERBOSE_LOGGING=1)
        print(f">> KERNEL #{_sparge_sage2_frame_counter}: topk={topk}", flush=True)
    
    # SpargeAttn requires half precision (fp16/bf16)
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    
    if q.dtype not in half_dtypes:
        q = q.to(Sage2BlackwellConfig.FALLBACK_DTYPE)
        k = k.to(Sage2BlackwellConfig.FALLBACK_DTYPE)
        v = v.to(Sage2BlackwellConfig.FALLBACK_DTYPE)
    
    # Reshape varlen (total_seq, heads, dim) -> batched (batch, heads, seq, dim)
    q_batched = q.view(batch_size, seq_len_q, heads, dim).transpose(1, 2)
    k_batched = k.view(batch_size, seq_len_k, heads, dim).transpose(1, 2)
    v_batched = v.view(batch_size, seq_len_k, heads, dim).transpose(1, 2)
    
    # Call SpargeAttn Sage2 - STRICT: raise error if kernel fails
    try:
        out = spas_sage2_attn_meansim_topk(q_batched, k_batched, v_batched, topk=topk, is_causal=is_causal)
    except Exception as e:
        raise RuntimeError(
            f"SpargeAttn Sage2 kernel FAILED with Blackwell parameters "
            f"(num_warps={Sage2BlackwellConfig.TRITON_NUM_WARPS}, topk={topk}). "
            f"Error: {e}. No silent fallback - fix the kernel or use a different attention_mode."
        ) from e
    
    # Reshape back to varlen format (total_seq, heads, dim)
    out = out.transpose(1, 2).reshape(-1, heads, dim).contiguous()
    
    return out.to(out_dtype) if out.dtype != out_dtype else out


# 2. Triton - Required for torch.compile with inductor backend
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# 3. GGUF - Required for quantized model loading
try:
    import gguf
    from gguf import GGMLQuantizationType
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None
    GGMLQuantizationType = None


# 4. NVFP4 - Native 4-bit floating point for Blackwell (RTX 50-series)
# Deferred import to avoid circular dependency - just set flags here
NVFP4_AVAILABLE = False
BLACKWELL_GPU_DETECTED = False

def _check_nvfp4_support():
    """Check if NVFP4 is supported (Blackwell GPU + PyTorch 2.6+ + CUDA 12.8+)"""
    global NVFP4_AVAILABLE, BLACKWELL_GPU_DETECTED
    
    if not torch.cuda.is_available():
        return False, False
    
    try:
        # Check for Blackwell GPU (compute capability 10.0+)
        capability = torch.cuda.get_device_capability(0)
        is_blackwell = capability[0] >= 10
        BLACKWELL_GPU_DETECTED = is_blackwell
        
        if not is_blackwell:
            return False, False
        
        # Check PyTorch version (need 2.6+)
        version_str = torch.__version__.split('+')[0]
        parts = version_str.split('.')
        torch_version = tuple(int(p) for p in parts[:2])
        if torch_version < (2, 6):
            return False, True  # Blackwell detected but PyTorch too old
        
        # Check CUDA version (need 12.8+)
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return False, True
        
        cuda_parts = cuda_version.split('.')
        cuda_major = int(cuda_parts[0])
        cuda_minor = int(cuda_parts[1]) if len(cuda_parts) > 1 else 0
        
        if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 8):
            return False, True  # Blackwell detected but CUDA too old
        
        NVFP4_AVAILABLE = True
        return True, True
        
    except Exception:
        return False, False

# Run NVFP4 check at module load
_nvfp4_result = _check_nvfp4_support()
NVFP4_AVAILABLE = _nvfp4_result[0]
BLACKWELL_GPU_DETECTED = _nvfp4_result[1]


def validate_gguf_availability(operation: str = "load GGUF model", debug=None) -> None:
    """
    Validate GGUF availability and raise error if not installed.
    
    Args:
        operation: Description of the operation requiring GGUF
        debug: Optional debug instance for logging
        
    Raises:
        RuntimeError: If GGUF is not available
    """
    if not GGUF_AVAILABLE:
        error_msg = (
            f"Cannot {operation}: GGUF library is not installed.\n"
            f"\n"
            f"GGUF provides quantized model support for memory-efficient loading.\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install GGUF: pip install gguf\n"
            f"  2. OR use a non-quantized model format (.safetensors)\n"
            f"\n"
            f"For more info: https://github.com/ggerganov/ggml"
        )
        if debug:
            debug.log(error_msg, level="ERROR", category="setup", force=True)
        raise RuntimeError(f"GGUF library required to {operation}")


# 4. NVIDIA Conv3d Memory Bug - Workaround for PyTorch >= 2.9 + cuDNN >= 91002
def _check_conv3d_memory_bug():
    """
    Check if Conv3d memory bug workaround needed.
    Bug: PyTorch 2.9+ with cuDNN >= 91002 uses 3x memory for Conv3d 
    with fp16/bfloat16 due to buggy dispatch layer.
    """
    try:
        # Exclude AMD ROCm/HIP builds (they use MIOpen, not cuDNN)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return False
        
        # Must have CUDA available
        if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
            return False
        
        # Must have cuDNN actually available (not just the attribute)
        if not (hasattr(torch.backends.cudnn, 'is_available') and 
                torch.backends.cudnn.is_available()):
            return False
        
        # Check device capability (NVIDIA GPUs)
        if torch.cuda.get_device_capability()[0] < 3:
            return False
        
        # Parse torch version
        version_str = torch.__version__.split('+')[0]
        parts = version_str.split('.')
        torch_version = tuple(int(p) for p in parts[:2])
        
        # Bug affects PyTorch 2.9 and later versions
        if torch_version < (2, 9):
            return False
        
        if not hasattr(torch.backends.cudnn, 'version'):
            return False
            
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None or cudnn_version < 91002:
            return False
        
        return True
    except:
        return False

NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = _check_conv3d_memory_bug()


# Log all optimization status once globally (cross-process) using environment variable
if not os.environ.get("SEEDVR2_OPTIMIZATIONS_LOGGED"):
    os.environ["SEEDVR2_OPTIMIZATIONS_LOGGED"] = "1"
    
    # Build status strings
    sage_status = "✅" if SAGE_ATTN_AVAILABLE else "❌"
    flash_status = "✅" if FLASH_ATTN_AVAILABLE else "❌"
    triton_status = "✅" if TRITON_AVAILABLE else "❌"
    nvfp4_status = "✅" if NVFP4_AVAILABLE else "❌"
    sparge_status = "✅" if SPARGE_SAGE2_AVAILABLE else "❌"
    
    # Count available optimizations
    available = [SAGE_ATTN_AVAILABLE, FLASH_ATTN_AVAILABLE, TRITON_AVAILABLE]
    num_available = sum(available)
    
    if num_available == 3:
        print(f"⚡ SeedVR2 optimizations check: SageAttention {sage_status} | Flash Attention {flash_status} | Triton {triton_status}")
    elif num_available == 0:
        print(f"⚠️  SeedVR2 optimizations check: SageAttention {sage_status} | Flash Attention {flash_status} | Triton {triton_status}")
        print("💡 For best performance: pip install sageattention flash-attn triton")
    else:
        icon = "⚡" if num_available >= 2 else "⚠️ "
        print(f"{icon} SeedVR2 optimizations check: SageAttention {sage_status} | Flash Attention {flash_status} | Triton {triton_status}")
        
        # Build install suggestions for missing packages
        missing = []
        if not SAGE_ATTN_AVAILABLE:
            missing.append("sageattention")
        if not FLASH_ATTN_AVAILABLE:
            missing.append("flash-attn")
        if not TRITON_AVAILABLE:
            missing.append("triton")
        if missing:
            print(f"💡 Optional: pip install {' '.join(missing)}")
    
    # SpargeAttn/Sage2 status (Blackwell block-sparse optimization)
    if SPARGE_SAGE2_AVAILABLE:
        version_info = f" v{SPARGE_SAGE2_VERSION}" if SPARGE_SAGE2_VERSION and SPARGE_SAGE2_VERSION != 'unknown' else ""
        print(f"🔥 SpargeAttn/Sage2 block-sparse attention: {sparge_status}{version_info} (optimized for RTX 50xx)")
    elif BLACKWELL_GPU_DETECTED:
        print(f"🔷 SpargeAttn/Sage2: {sparge_status} (install with: pip install spas-sage-attn for Blackwell optimization)")
    
    # NVFP4/Blackwell status (RTX 50-series optimizations)
    if BLACKWELL_GPU_DETECTED:
        if NVFP4_AVAILABLE:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Blackwell GPU"
            print(f"🚀 NVFP4 Blackwell optimization: {nvfp4_status} ({gpu_name} - 4-bit Tensor Core acceleration enabled)")
            
            # Enable native FP4 dispatch for Blackwell
            try:
                from .nvfp4 import ensure_native_fp4_dispatch
                if ensure_native_fp4_dispatch():
                    print("   └─ Native FP4 dispatch configured (TF32 enabled, cuDNN benchmark active)")
            except ImportError:
                pass
        else:
            # Blackwell GPU detected but NVFP4 not available (needs PyTorch 2.6+ with CUDA 12.8+)
            print(f"🔷 Blackwell GPU detected but NVFP4 unavailable (requires PyTorch 2.6+ with CUDA 12.8+)")
    
    # Conv3d workaround status (if applicable)
    if NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND:
        torch_ver = torch.__version__.split('+')[0]
        cudnn_ver = torch.backends.cudnn.version()
        print(f"🔧 Conv3d workaround active: PyTorch {torch_ver}, cuDNN {cudnn_ver} (fixing VAE 3x memory bug)")


# Bfloat16 CUBLAS support
def _probe_bfloat16_support() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda:0')
        _ = torch.matmul(a, a)
        del a
        return True
    except RuntimeError as e:
        if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
            return False
        raise

BFLOAT16_SUPPORTED = _probe_bfloat16_support()
COMPUTE_DTYPE = torch.bfloat16 if BFLOAT16_SUPPORTED else torch.float16


def call_rope_with_stability(method, *args, **kwargs):
    """
    Call RoPE method with stability fixes:
    1. Clear cache if available
    2. Disable autocast to prevent numerical issues (CUDA only)
    This prevents artifacts in FP8/mixed precision models.
    """
    if hasattr(method, 'cache_clear'):
        method.cache_clear()
    
    # Only use CUDA autocast context on CUDA devices
    # MPS has no CUDA autocast to disable
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=False):
            return method(*args, **kwargs)
    else:
        return method(*args, **kwargs)
    
    
class CompatibleDiT(torch.nn.Module):
    """
    Wrapper for DiT models with automatic compatibility management + advanced optimizations
    
    Precision Handling:
    - FP8: Keeps native FP8 parameters (memory efficient), converts inputs/outputs to compute_dtype for arithmetic
    - FP16/BFloat16/Float32: Uses native precision throughout
    - GGUF: On-the-fly dequantization to compute_dtype
    - MPS: Forces all parameters to compute_dtype (unified memory requires dtype consistency)
    - RoPE: Converted from FP8 to compute_dtype for numerical consistency
    
    Optimizations:
    - RoPE Stabilization: Error handling for numerical stability in mixed precision
    - MPS Compatibility: Unified dtype conversion for Apple Silicon backends
    """
    
    def __init__(self, dit_model, debug: 'Debug', compute_dtype: torch.dtype = torch.bfloat16, skip_conversion: bool = False):
        super().__init__()
        self.dit_model = dit_model
        self.debug = debug
        self.compute_dtype = compute_dtype
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16

        # Only convert if not already done (e.g., when reusing cached weights)
        if not skip_conversion and self.is_fp8_model:
            # FP8 models need RoPE frequency conversion to compute dtype
            model_variant = self._get_model_variant()
            self.debug.log(f"Detected NaDiT {model_variant} FP8 - Converting RoPE freqs for FP8 compatibility", 
                        category="precision")
            self.debug.start_timer("_convert_rope_freqs")
            self._convert_rope_freqs(target_dtype=self.compute_dtype)
            self.debug.end_timer("_convert_rope_freqs", "RoPE freqs conversion")
        
        # MPS requires unified dtype for all parameters/buffers (no autocast fallback)
        # Apply to ALL model types (FP8, FP16, GGUF) when dtype differs from compute_dtype
        if not skip_conversion and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if self.model_dtype != self.compute_dtype:
                self.debug.log(f"Converting NaDiT parameters/buffers to {self.compute_dtype} for MPS backend", category="setup", force=True)
                self.debug.start_timer("_force_nadit_precision")
                self._force_nadit_precision(target_dtype=self.compute_dtype)
                self.debug.end_timer("_force_nadit_precision", "NaDiT parameters/buffers conversion")
            
        # Apply RoPE stabilization for numerical stability
        self.debug.log(f"Stabilizing RoPE computations for numerical stability", category="setup")
        self.debug.start_timer("_stabilize_rope_computations")
        self._stabilize_rope_computations()
        self.debug.end_timer("_stabilize_rope_computations", "RoPE stabilization")
    
    def _detect_model_dtype(self) -> torch.dtype:
        """Detect main model dtype"""
        try:
            return next(self.dit_model.parameters()).dtype
        except:
            return torch.bfloat16
    
    def _get_model_variant(self) -> str:
        """Detect model variant from module path"""
        model_module = str(self.dit_model.__class__.__module__).lower()
        if 'dit_7b' in model_module:
            return "7B"
        elif 'dit_3b' in model_module:
            return "3B"
        else:
            return "Unknown"
        
    def _convert_rope_freqs(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Convert RoPE frequency buffers from FP8 to target dtype for compatibility.
        
        Args:
            target_dtype: Target dtype for RoPE freqs (default: bfloat16 for stability)
        """
        converted = 0
        for module in self.dit_model.modules():
            if 'RotaryEmbedding' in type(module).__name__:
                if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
                    if module.rope.freqs.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        if module.rope.freqs.device.type == "mps":
                            module.rope.freqs.data = module.rope.freqs.to("cpu").to(target_dtype).to("mps")
                        else:
                            module.rope.freqs.data = module.rope.freqs.to(target_dtype)
                        converted += 1
        self.debug.log(f"Converted {converted} RoPE frequency buffers from FP8 to {target_dtype} for compatibility", category="success")
                        
    def _force_nadit_precision(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Force ALL NaDiT parameters to target dtype to avoid promotion errors (MPS requirement).
        
        Args:
            target_dtype: Target dtype for all parameters (default: bfloat16 for MPS compatibility)
        """
        converted_count = 0
        original_dtype = None
        
        # Convert ALL parameters to target dtype
        for name, param in self.dit_model.named_parameters():
            if original_dtype is None:
                original_dtype = param.dtype
            if param.dtype != target_dtype:
                if param.device.type == "mps":
                    temp_cpu = param.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    param.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    param.data = param.data.to(target_dtype)
                converted_count += 1
                
        # Also convert buffers (skip GGUF quantized buffers - they have tensor_type attribute)
        for name, buffer in self.dit_model.named_buffers():
            # Skip GGUF quantized buffers - these must stay in packed format for on-the-fly dequantization
            if hasattr(buffer, 'tensor_type'):
                continue
            if buffer.dtype != target_dtype:
                if buffer.device.type == "mps":
                    temp_cpu = buffer.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    buffer.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    buffer.data = buffer.data.to(target_dtype)
                converted_count += 1
        
        self.debug.log(f"Converted {converted_count} NaDiT parameters/buffers to {target_dtype} for MPS", category="success")
        
        # Update detected dtype
        self.model_dtype = target_dtype
        self.is_fp8_model = (target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2))

    def _stabilize_rope_computations(self):
        """
        Add error handling to RoPE computations to prevent artifacts.
        
        Wraps the get_axial_freqs method of RoPE modules with a try-except handler.
        During normal operation, uses the original cached method for performance.
        Only on exceptions (e.g., numerical instability, NaN propagation) does it
        intervene by clearing the cache and retrying the computation through
        call_rope_with_stability.
        
        This prevents artifacts in FP8, mixed precision, and edge cases while
        maintaining optimal performance for normal operations.
        """
        if not hasattr(self.dit_model, 'blocks'):
            return
        
        rope_count = 0
        
        # Wrap RoPE modules to handle numerical instability
        for name, module in self.dit_model.named_modules():
            if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
                # Check if already wrapped
                if hasattr(module, '_rope_wrapped'):
                    continue
                    
                original_method = module.get_axial_freqs
                
                # Mark as wrapped and store original
                module._rope_wrapped = 'stability'
                module._original_get_axial_freqs = original_method
                
                # Error handler that prevents NaN propagation
                def stable_rope_computation(self, *args, **kwargs):
                    try:
                        return original_method(*args, **kwargs)
                    except Exception:
                        return call_rope_with_stability(original_method, *args, **kwargs)
                
                module.get_axial_freqs = types.MethodType(stable_rope_computation, module)
                rope_count += 1
        
        if rope_count > 0:
            self.debug.log(f"Stabilized {rope_count} RoPE modules", category="success")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with minimal dtype conversion overhead
        
        Conversion strategy:
            - FP16/BFloat16/Float32 models: Use native precision (no conversion needed)
            - FP8 models: Convert FP8 tensors to compute_dtype for arithmetic operations
            (FP8 parameters stay in FP8 for memory efficiency, only converted for computation)
        """
        
        # Only convert if we have an FP8 model for arithmetic operations 
        if self.is_fp8_model:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            target_dtype = self.compute_dtype
            
            # Convert args
            converted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype in fp8_dtypes:
                    converted_args.append(arg.to(target_dtype))
                else:
                    converted_args.append(arg)
            
            # Convert kwargs
            converted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.dtype in fp8_dtypes:
                    converted_kwargs[key] = value.to(target_dtype)
                else:
                    converted_kwargs[key] = value
            
            args = tuple(converted_args)
            kwargs = converted_kwargs
        
        # Execute forward pass
        try:
            return self.dit_model(*args, **kwargs)
        except Exception as e:
            self.debug.log(f"Forward pass error: {e}", level="ERROR", category="generation", force=True)
            if self.is_fp8_model:
                self.debug.log(f"FP8 model - converted FP8 tensors to {self.compute_dtype}", category="info", force=True)
            else:
                self.debug.log(f"{self.model_dtype} model - no conversion applied", category="info", force=True)
            raise
    
    def __getattr__(self, name):
        """Redirect all other attributes to original model"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            return super().__getattr__(name)
        return getattr(self.dit_model, name)
    
    def __setattr__(self, name, value):
        """Redirect assignments to original model except for our attributes"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'dit_model'):
                setattr(self.dit_model, name, value)
            else:
                super().__setattr__(name, value)
                