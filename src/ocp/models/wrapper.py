from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ocp.losses.potential_nll import potential_nll
from ocp.potentials.base import Potential


@dataclass(frozen=True)
class ForwardOutput:
    logits: torch.Tensor  # [B, K]


class PotentialModel(nn.Module):
    """Thin wrapper: backbone produces logits; potential defines loss/probs."""

    def __init__(self, backbone: nn.Module, potential: Potential):
        super().__init__()
        self.backbone = backbone
        self.potential = potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits z = F(x), shape [B, K]."""
        return self.backbone(x)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute mean(φ(z) - z_y) by default."""
        return potential_nll(self.potential, logits, targets, reduction=reduction)

    @torch.no_grad()
    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Return p = ∇φ(z), shape [B, K]."""
        return self.potential.grad(logits)

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Return argmax_k z_k, shape [B]."""
        return logits.argmax(dim=-1)

