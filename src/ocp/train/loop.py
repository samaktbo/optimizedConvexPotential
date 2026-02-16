from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    n: int


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    amp: bool = False,
    grad_clip_norm: Optional[float] = None,
) -> EpochMetrics:
    model.train()

    use_amp = amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        bsz = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bsz
        total_correct += int((logits.detach().argmax(dim=-1) == y).sum().item())
        total_n += int(bsz)

    return EpochMetrics(loss=total_loss / max(1, total_n), acc=total_correct / max(1, total_n), n=total_n)


@torch.no_grad()
def eval_one_epoch(
    *,
    model: nn.Module,
    loader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> EpochMetrics:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        bsz = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bsz
        total_correct += int((logits.detach().argmax(dim=-1) == y).sum().item())
        total_n += int(bsz)

    return EpochMetrics(loss=total_loss / max(1, total_n), acc=total_correct / max(1, total_n), n=total_n)

