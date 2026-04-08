from __future__ import annotations

import os
import random
from typing import Any, Dict, Literal, Optional

import torch
from torch import nn

from ocp.models.wrapper import PotentialModel
from ocp.potentials.base import Potential
from ocp.potentials.logsumexp import LogSumExpPotential
from ocp.potentials.moreau_max import MoreauMaxPotential
from ocp.potentials.simplex_entropy_conjugate import SimplexEntropyConjugatePotential
from ocp.train.loop import eval_one_epoch, train_one_epoch

PotentialKind = Literal["logsumexp", "moreau_max", "simplex_entropy"]
OptimizerName = Literal["adamw", "sgd"]


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass


def build_resnet18_for_cifar(*, num_classes: int) -> nn.Module:
    """ResNet-18 with CIFAR-friendly stem (3x3 conv, no initial maxpool)."""
    try:
        import torchvision
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This experiment requires `torchvision` for ResNet-18. Install it, then re-run."
        ) from e

    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_optimizer(
    *,
    name: OptimizerName,
    params,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer={name!r}")


def build_potential(*, kind: PotentialKind, lam: float) -> Potential:
    if kind == "logsumexp":
        return LogSumExpPotential(lam=lam)
    if kind == "moreau_max":
        return MoreauMaxPotential(lam=lam)
    if kind == "simplex_entropy":
        return SimplexEntropyConjugatePotential(lam=lam)
    raise ValueError(f"Unknown potential kind={kind!r}")


def train_one_potential(
    *,
    kind: PotentialKind,
    lam: float,
    lr: float,
    num_classes: int,
    epochs: int,
    train_loader,
    eval_loader,
    optimizer: OptimizerName,
    weight_decay: float,
    momentum: float,
    amp: bool,
    grad_clip_norm: Optional[float],
    device: torch.device,
    seed: int,
    data_dict: Dict[str, Any],
    tag: str,
    extra: Optional[Dict[str, Any]] = None,
    eval_prefix: str = "test",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one potential on CIFAR-style loaders; evaluate on ``eval_loader`` each epoch.

    Metrics use ``train_{loss,acc}`` plus ``{eval_prefix}_{loss,acc}`` and
    ``final_{eval_prefix}_acc``. Re-seeds so runs are reproducible for a fixed ``seed``.
    """
    set_training_seed(seed)
    potential = build_potential(kind=kind, lam=lam)
    backbone = build_resnet18_for_cifar(num_classes=num_classes)
    model = PotentialModel(backbone, potential).to(device)

    opt = make_optimizer(
        name=optimizer,
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    ev_loss_key = f"{eval_prefix}_loss"
    ev_acc_key = f"{eval_prefix}_acc"
    final_ev_acc_key = f"final_{eval_prefix}_acc"

    metrics: Dict[str, Any] = {
        "tag": tag,
        "seed": seed,
        "device": str(device),
        "data": data_dict,
        "optimizer": {"name": optimizer, "lr": lr, "weight_decay": weight_decay, "momentum": momentum},
        "epochs": epochs,
        "amp": bool(amp),
        "grad_clip_norm": grad_clip_norm,
        "train_loss": [],
        "train_acc": [],
        ev_loss_key: [],
        ev_acc_key: [],
    }
    if extra:
        metrics.update(extra)

    loss_fn = lambda logits, targets: model.loss(logits, targets, reduction="mean")

    tr0 = eval_one_epoch(model=model, loader=train_loader, loss_fn=loss_fn, device=device)
    te0 = eval_one_epoch(model=model, loader=eval_loader, loss_fn=loss_fn, device=device)
    metrics["train_loss"].append(tr0.loss)
    metrics["train_acc"].append(tr0.acc)
    metrics[ev_loss_key].append(te0.loss)
    metrics[ev_acc_key].append(te0.acc)
    if verbose:
        print(
            f"[{tag}] epoch {0:03d}/{epochs} | "
            f"train loss {tr0.loss:.4f} acc {tr0.acc:.4f} | "
            f"{eval_prefix} loss {te0.loss:.4f} acc {te0.acc:.4f}"
        )

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=opt,
            loss_fn=loss_fn,
            device=device,
            amp=amp,
            grad_clip_norm=grad_clip_norm,
        )
        te = eval_one_epoch(model=model, loader=eval_loader, loss_fn=loss_fn, device=device)

        metrics["train_loss"].append(tr.loss)
        metrics["train_acc"].append(tr.acc)
        metrics[ev_loss_key].append(te.loss)
        metrics[ev_acc_key].append(te.acc)

        if verbose:
            print(
                f"[{tag}] epoch {ep:03d}/{epochs} | "
                f"train loss {tr.loss:.4f} acc {tr.acc:.4f} | "
                f"{eval_prefix} loss {te.loss:.4f} acc {te.acc:.4f}"
            )

    metrics[final_ev_acc_key] = float(metrics[ev_acc_key][-1]) if metrics[ev_acc_key] else None
    return metrics
