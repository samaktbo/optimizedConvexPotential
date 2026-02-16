from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import torch


@dataclass(frozen=True)
class PotentialOutputSpec:
    """Shape conventions for potentials.

    We treat logits as a batch of vectors: z in R^{B x K}.
    - phi(z) returns a per-example scalar: shape [B]
    - grad(z) returns per-example gradient: shape [B, K]
    """

    batch_dim: Final[int] = 0
    class_dim: Final[int] = -1


class Potential(ABC):
    """Convex potential φ with (optionally) closed-form gradient.

    Contract:
    - Input `z` is a floating tensor of shape [B, K].
    - `phi(z)` returns shape [B].
    - `grad(z)` returns shape [B, K], the gradient ∇φ(z).
    """

    spec: PotentialOutputSpec = PotentialOutputSpec()

    @abstractmethod
    def phi(self, z: torch.Tensor) -> torch.Tensor:
        """Compute φ(z) for each example. Returns shape [B]."""

    @abstractmethod
    def grad(self, z: torch.Tensor) -> torch.Tensor:
        """Compute ∇φ(z) for each example. Returns shape [B, K]."""

    def _check_logits(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError(f"Expected logits z with shape [B, K], got shape {tuple(z.shape)}")
        if not z.is_floating_point():
            raise TypeError(f"Expected floating logits tensor, got dtype={z.dtype}")

