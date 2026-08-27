# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""Narrow MPS fallbacks for causal-VAE large-tensor MPSGraph limitations."""

from typing import Sequence

import torch
from torch import Tensor


_MPS_INT_MAX_ELEMENTS = 2_147_483_647


def _requires_mps_cpu_fallback(tensors: Sequence[Tensor], element_count: int) -> bool:
    """Return whether an MPS operation would exceed the conservative int32 limit."""
    return bool(
        tensors
        and element_count >= _MPS_INT_MAX_ELEMENTS
        and all(tensor.device == tensors[0].device for tensor in tensors)
        and tensors[0].device.type == "mps"
    )


def _tile_on_cpu(tensor: Tensor, tile_repeat: Sequence[int]) -> Tensor:
    """Materialize a tile on CPU and return it to the source tensor's device."""
    return torch.tile(tensor.cpu(), tile_repeat).to(tensor)


def repeat_first_frame(tensor: Tensor, times: int) -> Tensor:
    """Repeat the first temporal frame, handling oversized MPS parent views."""
    if tensor.ndim < 3:
        raise ValueError(f"expected at least three dimensions, got {tensor.ndim}")
    if times < 0:
        raise ValueError(f"times must be non-negative, got {times}")

    first_frame = tensor[:, :, :1]
    tile_repeat = [1] * tensor.ndim
    tile_repeat[2] = times
    # ``first_frame`` is a view. MPS ``tile`` can still observe its much larger
    # parent tensor, so guard both the parent input and the tiled output.
    operation_elements = max(tensor.numel(), first_frame.numel() * times)
    if _requires_mps_cpu_fallback((tensor,), operation_elements):
        return _tile_on_cpu(first_frame, tile_repeat)
    return torch.tile(first_frame, tile_repeat)


def mps_safe_cat(tensors: Sequence[Tensor], dim: int) -> Tensor:
    """Concatenate normally, with CPU aggregation only for oversized MPS output."""
    values = tuple(tensors)
    if not _requires_mps_cpu_fallback(values, sum(value.numel() for value in values)):
        return torch.cat(values, dim=dim)

    return torch.cat(tuple(value.cpu() for value in values), dim=dim).to(values[0])
