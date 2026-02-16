from __future__ import annotations

import torch

from .base import Potential


class LogSumExpPotential(Potential):
    r"""φ(z) = log \sum_k exp(z_k).

    This is the standard softmax / cross-entropy potential.

    - phi(z) uses a numerically-stable logsumexp.
    - grad(z) computes softmax(z) in closed form (no autograd.grad).
    """

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)
        return torch.logsumexp(z, dim=self.spec.class_dim)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)

        # Stable softmax: exp(z - max) / sum exp(z - max)
        # Compute in float32 for stability; cast back to original dtype.
        orig_dtype = z.dtype
        z32 = z.float()
        zmax = z32.max(dim=self.spec.class_dim, keepdim=True).values
        exp = torch.exp(z32 - zmax)
        denom = exp.sum(dim=self.spec.class_dim, keepdim=True)
        p = exp / denom
        return p.to(dtype=orig_dtype)

