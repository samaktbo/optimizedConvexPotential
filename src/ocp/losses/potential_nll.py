from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ocp.potentials.base import Potential


def potential_nll(
    potential: Potential,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the potential-based negative log-likelihood: φ(z) - z_y.

    Args:
        potential: Convex potential φ.
        logits: Tensor of shape [B, K].
        targets: Long tensor of shape [B] with class indices in [0, K-1].
        reduction: "mean" or "sum" or "none".

    Returns:
        Scalar tensor if reduction != "none", else shape [B].
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [B, K], got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets with shape [B], got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch size mismatch: logits.shape[0]={logits.shape[0]} vs targets.shape[0]={targets.shape[0]}"
        )
    if targets.dtype != torch.long:
        targets = targets.long()

    # Per-example potential value: [B]
    phi = potential.phi(logits)
    if phi.ndim != 1 or phi.shape[0] != logits.shape[0]:
        raise ValueError(f"potential.phi(logits) must return shape [B], got {tuple(phi.shape)}")

    # Gather correct-class logits: [B]
    z_y = logits.gather(dim=1, index=targets.view(-1, 1)).squeeze(1)
    per_example = phi - z_y

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    if reduction == "mean":
        return per_example.mean()
    raise ValueError(f"Unknown reduction={reduction!r} (expected 'mean', 'sum', or 'none')")


def _self_check_against_cross_entropy(
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> None:
    """Sanity check: LogSumExpPotential NLL equals torch cross-entropy.

    Not used by training code; handy for debugging.
    """
    from ocp.potentials.logsumexp import LogSumExpPotential

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    B, K = 32, 10
    logits = torch.randn(B, K, generator=g, device=device, dtype=dtype)
    targets = torch.randint(low=0, high=K, size=(B,), generator=g, device=device, dtype=torch.long)

    pot = LogSumExpPotential()
    a = potential_nll(pot, logits, targets, reduction="mean")
    b = F.cross_entropy(logits, targets, reduction="mean")
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def _self_check_simplex_entropy_conjugate(
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> None:
    """Sanity checks for simplex-entropy conjugate potential.

    Checks:
    - grad shape and simplex constraints
    - autograd path through potential_nll
    """
    from ocp.potentials.simplex_entropy_conjugate import SimplexEntropyConjugatePotential

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    B, K = 16, 5  # K >= 3 required by the potential.
    logits = torch.randn(B, K, generator=g, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(low=0, high=K, size=(B,), generator=g, device=device, dtype=torch.long)

    pot = SimplexEntropyConjugatePotential()
    p = pot.grad(logits.detach())
    if p.shape != logits.shape:
        raise ValueError(f"Expected grad shape {tuple(logits.shape)}, got {tuple(p.shape)}")
    if (p < -1e-6).any():
        raise ValueError("Simplex gradient has negative entries.")
    torch.testing.assert_close(
        p.sum(dim=1),
        torch.ones(B, device=device, dtype=p.dtype),
        rtol=1e-5,
        atol=1e-5,
    )

    loss = potential_nll(pot, logits, targets, reduction="mean")
    loss.backward()
    if logits.grad is None:
        raise RuntimeError("Expected logits.grad to be populated after backward().")
    if not torch.isfinite(logits.grad).all():
        raise RuntimeError("Found non-finite gradients in logits.grad.")

