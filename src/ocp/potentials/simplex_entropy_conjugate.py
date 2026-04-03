from __future__ import annotations

import torch

from .base import Potential


def _lambda_star_sorted(z: torch.Tensor, *, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Closed-form maximizer on simplex via sorting-based active-set.

    Solves, per example:
        argmax_{lambda in Delta^K} <lambda, z> - h(lambda)
    with
        h(lambda) = (1 / A^2) * sum_i (A lambda_i + B) log(A lambda_i + B),
        A=(K-2)/(K-1), B=1/(K(K-1)).

    The KKT solution has form:
        lambda_i = max(0, c * exp(A z_i) - B) / A
    where c is determined by an active-set size rho selected by sorting.
    """
    if z.numel() == 0:
        return z

    if dim != -1 and dim != z.ndim - 1:
        raise ValueError("Only class dimension at the last axis is supported.")

    k = z.shape[-1]
    if k < 3:
        raise ValueError(f"Expected K >= 3 for this potential, got K={k}")

    A = (k - 2) / (k - 1)
    B = 1 / (k * (k - 1))

    A_on_B = A / B

    z32 = z.float()
    az = A * z32
    az_shift = az - az.max(dim=-1, keepdim=True).values
    u = torch.exp(az_shift)

    u_sorted, _ = torch.sort(u, dim=-1, descending=True)
    prefix = torch.cumsum(u_sorted, dim=-1)

    denom = A_on_B + torch.arange(1, k + 1, device=z.device, dtype=z32.dtype).view(1, k)
    rhs = prefix / denom
    cond = u_sorted > rhs

    rho = cond.sum(dim=dim, keepdim=True).clamp_min(1)
    rho_idx = (rho - 1).view(-1, 1)

    s_rho = prefix.gather(dim=-1, index=rho_idx)
    rho_f = rho.to(dtype=z32.dtype).view(-1, 1)
    c = (A + rho_f * B) / s_rho

    lam = torch.clamp(c * u - B, min=0.0) / A
    #lam = lam / lam.sum(dim=-1, keepdim=True).clamp_min(eps)
    return lam


class SimplexEntropyConjugatePotential(Potential):
    r"""Convex potential with simplex-constrained conjugate form.

    For z in R^K:
        phi(z) = sup_{lambda in Delta^K} <lambda, z> - h(lambda),
    where
        h(lambda) = (1 / A^2) * sum_i (A lambda_i + B) log(A lambda_i + B),
        A=(K-2)/(K-1), B=1/(K(K-1)).

    The gradient is the maximizer lambda*(z), computed in closed form
    with a sorting-based active-set rule.
    """

    def __init__(self, *, lam: float = 1.0, eps: float = 1e-12):
        if lam <= 0:
            raise ValueError(f"lam must be > 0, got {lam}")
        self.lam = float(lam)
        self.eps = float(eps)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)
        lam32 = _lambda_star_sorted(
            z / self.lam, dim=self.spec.class_dim, eps=self.eps
        )
        return lam32.to(dtype=z.dtype)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        self._check_logits(z)

        k = z.shape[self.spec.class_dim]
        if k < 3:
            raise ValueError(f"Expected K >= 3 for this potential, got K={k}")

        A = (k - 2.0) / (k - 1.0)
        B = 1.0 / (k * (k - 1.0))

        z32 = z.float() / self.lam
        lam32 = _lambda_star_sorted(z32, dim=self.spec.class_dim, eps=self.eps)
        s = torch.clamp(A * lam32 + B, min=self.eps)
        h = (s * torch.log(s)).sum(dim=self.spec.class_dim) / (A * A)
        inner = (z32 * lam32).sum(dim=self.spec.class_dim) - h
        return self.lam * inner
