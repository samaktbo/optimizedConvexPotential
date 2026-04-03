from __future__ import annotations

import torch

from .base import Potential


class LogSumExpPotential(Potential):
    r"""φ(z) = λ log \sum_k exp(z_k / λ), λ > 0 (temperature-scaled log-sum-exp).

    For λ=1 this reduces to log \sum_k exp(z_k), the standard softmax / cross-entropy potential.

    - phi(z) uses a numerically-stable logsumexp.
    - grad(z) computes softmax(z/λ) in closed form (no autograd.grad).
    """

    def __init__(self, lam: float = 1.0):
        if lam <= 0:
            raise ValueError(f"lam must be > 0, got {lam}")
        self.lam = float(lam)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)
        z32 = z.float()
        return self.lam * torch.logsumexp(z32 / self.lam, dim=self.spec.class_dim)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)

        # Stable softmax: exp(z/λ - max) / sum exp(z/λ - max)
        # Compute in float32 for stability; cast back to original dtype.
        orig_dtype = z.dtype
        z32 = z.float() / self.lam
        zmax = z32.max(dim=self.spec.class_dim, keepdim=True).values
        exp = torch.exp(z32 - zmax)
        denom = exp.sum(dim=self.spec.class_dim, keepdim=True)
        p = exp / denom
        return p.to(dtype=orig_dtype)

