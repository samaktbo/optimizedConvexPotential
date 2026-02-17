from __future__ import annotations

import torch

from .base import Potential


def _proj_simplex(v: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Project onto the probability simplex along `dim`.

    Computes: argmin_p 0.5||p - v||^2 s.t. p >= 0, sum(p)=1.

    This is the standard O(K log K) algorithm using sorting (Duchi et al., 2008),
    implemented in a batched way for tensors.
    """
    if v.numel() == 0:
        return v

    # Sort descending.
    v_sorted, _ = torch.sort(v, dim=dim, descending=True)
    cssv = v_sorted.cumsum(dim=dim) - 1.0

    k = v.shape[dim]

    # 1..K (broadcastable across all dims except `dim`).
    ind = torch.arange(1, k + 1, device=v.device, dtype=v.dtype)
    shape = [1] * v.ndim
    shape[dim] = k
    ind = ind.view(*shape)

    # Find rho = max { j : v_sorted[j] - cssv[j]/j > 0 }.
    cond = v_sorted - (cssv / ind) > 0
    rho = cond.sum(dim=dim, keepdim=True).clamp_min(1)
    rho_idx = rho - 1

    theta = cssv.gather(dim=dim, index=rho_idx) / rho.to(dtype=v.dtype)
    return torch.clamp(v - theta, min=0.0)


class MoreauMaxPotential(Potential):
    r"""Moreau envelope of the max function.

    Let f(z) = max_k z_k. For λ > 0, define the Moreau envelope:

        φ_λ(z) = min_u [ max_k u_k + (1/(2λ)) ||u - z||^2 ].

    Closed forms used here:
    - ∇φ_λ(z) = proj_Δ(z / λ), the Euclidean projection onto the probability simplex.
      (Equivalently, sparsemax applied to z/λ.)
    - φ_λ(z) = <z, p> - (λ/2) ||p||^2, where p = proj_Δ(z/λ).
    """

    def __init__(self, lam: float = 1.0):
        if lam <= 0:
            raise ValueError(f"lam must be > 0, got {lam}")
        self.lam = float(lam)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)

        orig_dtype = z.dtype
        v = (z / self.lam).float()
        p = _proj_simplex(v, dim=self.spec.class_dim)
        return p.to(dtype=orig_dtype)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)

        z32 = z.float()
        p32 = self.grad(z).float()
        return (z32 * p32).sum(dim=self.spec.class_dim) - 0.5 * self.lam * (p32 * p32).sum(
            dim=self.spec.class_dim
        )

