# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# // Modifications Copyright (c) 2025-2026 SeedVR2 Contributors
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

"""
Optimization utilities for SeedVR2 Video Upscaler.

Exports:
    IS_ROCM: Boolean indicating if running on AMD ROCm/HIP backend
    portable_repeat_interleave: Cross-platform repeat_interleave that works on ROCm
"""

from .compatibility import IS_ROCM, portable_repeat_interleave

__all__ = [
    "IS_ROCM",
    "portable_repeat_interleave",
]
