"""
Fast GGUF dequantization functions
Adapted from ComfyUI-GGUF
Optimized for SeedVR2 with proper debug logging and error handling
"""
import torch
import traceback
from typing import Optional, Tuple, List
from ..utils.constants import QK_K, K_SCALE_SIZE, suppress_tensor_warnings
from ..optimization.compatibility import GGUF_AVAILABLE, validate_gguf_availability

# Import GGUF library
if GGUF_AVAILABLE:
    import gguf
    TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
else:
    gguf = None
    TORCH_COMPATIBLE_QTYPES = (None,)


def is_torch_compatible(tensor: torch.Tensor) -> bool:
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES


def is_quantized(tensor: torch.Tensor) -> bool:
    return not is_torch_compatible(tensor)


@torch._dynamo.disable
def dequantize_tensor(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None, 
                     dequant_dtype: Optional[torch.dtype] = None, 
                     debug: Optional['Debug'] = None) -> torch.Tensor:
    """
    Fast dequantization using optimized PyTorch operations
    Returns regular PyTorch tensors to avoid infinite loops
    
    Args:
        tensor: GGUF tensor to dequantize
        dtype: Target dtype for final result
        dequant_dtype: Intermediate dtype for dequantization
        debug: Optional Debug instance for logging
    """
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    
    # Suppress tensor copy warning - we intentionally convert GGUFTensor to regular tensor
    suppress_tensor_warnings()

    if qtype in TORCH_COMPATIBLE_QTYPES:
        result = tensor.to(dtype)
        # Ensure we return a regular tensor, not GGUFTensor
        if hasattr(result, 'tensor_type'):
            result = torch.tensor(result.data, dtype=dtype, device=result.device, requires_grad=False)
        return result
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        # Use tensor.data like ComfyUI-GGUF does
        result = dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype, debug=debug)
        final_result = result.to(dtype)
        
        # Ensure we return a regular tensor, not GGUFTensor
        if hasattr(final_result, 'tensor_type'):
            final_result = torch.tensor(final_result.data, dtype=dtype, device=final_result.device, requires_grad=False)
        
        return final_result
    else:
        raise NotImplementedError(f"No dequantization for {qtype}")


@torch._dynamo.disable
def dequantize(data: torch.Tensor, qtype: 'gguf.GGMLQuantizationType', 
              oshape: Tuple[int, ...], dtype: Optional[torch.dtype] = None, 
              debug: Optional['Debug'] = None) -> torch.Tensor:
    """
    Dequantize tensor back to usable shape/dtype using fast operations
    
    Args:
        data: Quantized data to dequantize
        qtype: GGUF quantization type
        oshape: Original shape to restore
        dtype: Target dtype (default: torch.float16)
        debug: Optional Debug instance for logging
        
    Returns:
        Dequantized tensor in original shape
        
    Raises:
        ValueError: If quantization type is not supported
        RuntimeError: If dequantization fails
    """
    if not GGUF_AVAILABLE:
        validate_gguf_availability("dequantize GGUF tensor", debug)
        
    if dtype is None:
        dtype = torch.float16
        
    if qtype not in dequantize_functions:
        raise ValueError(f"Unsupported quantization type: {qtype}")
    try:
        if debug:
            debug.start_timer(f"dequant_{qtype.name if hasattr(qtype, 'name') else qtype}")
        
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        dequantize_blocks = dequantize_functions[qtype]

        # Ensure data is contiguous and properly formatted
        data = data.contiguous()
        
        # More robust reshaping to handle edge cases
        rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

        # Calculate number of blocks
        n_blocks = rows.numel() // type_size
        if rows.numel() % type_size != 0:
            error_msg = f"Data size {rows.numel()} not divisible by type_size {type_size}. This usually indicates corrupted GGUF data."
            if debug:
                debug.log(error_msg, level="ERROR", category="precision", force=True)
            raise ValueError(error_msg)
        blocks = rows.reshape((n_blocks, type_size))
        
        # Call the dequantization function
        blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
        
        result = blocks.reshape(oshape)
        
        # Ensure the result tensor is properly formatted
        result = result.contiguous()
        result.requires_grad_(False)

        if debug:
            debug.end_timer(f"dequant_{qtype.name if hasattr(qtype, 'name') else qtype}", 
                          f"Dequantized {qtype} tensor")
        
        return result
    except Exception as e:
        if debug:
            debug.log(f"Error in dequantize: {e}", level="ERROR", category="precision", force=True)
            debug.log(f"Data shape: {data.shape if 'data' in locals() else 'unknown'}", level="ERROR", category="precision", force=True)
            debug.log(f"qtype: {qtype}, oshape: {oshape}", level="ERROR", category="precision", force=True)
            debug.log(f"Traceback: {traceback.format_exc()}", level="ERROR", category="precision", force=True)
        else:
            traceback.print_exc()
        raise


def to_uint32(x: torch.Tensor) -> torch.Tensor:
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks: torch.Tensor, *args: int) -> List[torch.Tensor]:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def get_scale_min(scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q4_K(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q4_K dequantization"""
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q8_0(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Fast Q8_0 dequantization"""
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)


def dequantize_blocks_BF16(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """BF16 dequantization"""
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def dequantize_blocks_Q5_1(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q5_1 dequantization"""
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = (ql | (qh << 4))
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q5_0 dequantization"""
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)


def dequantize_blocks_Q4_1(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q4_1 dequantization"""
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q4_0 dequantization"""
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)


def dequantize_blocks_Q6_K(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q6_K dequantization"""
    n_blocks = blocks.shape[0]
    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q5_K dequantization"""
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))
    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q3_K dequantization"""
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks: torch.Tensor, block_size: int, type_size: int, 
                           dtype: Optional[torch.dtype] = None, debug: Optional['Debug'] = None) -> torch.Tensor:
    """Q2_K dequantization"""
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


# Main dequantization function lookup
if GGUF_AVAILABLE:
    dequantize_functions = {
        gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
        gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
        gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
        gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
        gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
        gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
        gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
        gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
        gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
        gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
        gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    }
else:
    dequantize_functions = {}