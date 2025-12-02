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

from enum import Enum
from typing import Dict, Literal, NamedTuple, Optional
import torch

_receptive_field_t = Literal["half", "full"]
_inflation_mode_t = Literal["none", "tail", "replicate"]
_memory_device_t = Optional[Literal["cpu", "same"]]
_gradient_checkpointing_t = Optional[Literal["half", "full"]]
_selective_checkpointing_t = Optional[Literal["coarse", "fine"]]

class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self) -> torch.FloatTensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean**2 + self.var - 1.0 - self.logvar,
            dim=list(range(1, self.mean.ndim)),
        )

class MemoryState(Enum):
    """
    State[Disabled]:        No memory bank will be enabled.
    State[Initializing]:    The model is handling the first clip, need to reset the memory bank.
    State[Active]:          There has been some data in the memory bank.
    State[Unset]:           Error state, indicating users didn't pass correct memory state in.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3


class QuantizerOutput(NamedTuple):
    latent: torch.Tensor
    extra_loss: torch.Tensor
    statistics: Dict[str, torch.Tensor]


class CausalAutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor


class DecoderOutput:
    """Output of decoding method - matches diffusers.models.autoencoders.vae.DecoderOutput"""
    def __init__(self, sample: torch.Tensor, commit_loss: Optional[torch.Tensor] = None):
        self.sample = sample
        self.commit_loss = commit_loss


class DiagonalGaussianDistribution:
    """Matches diffusers.models.autoencoders.vae.DiagonalGaussianDistribution exactly."""
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.deterministic:
            return self.mode()
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3],
        )
