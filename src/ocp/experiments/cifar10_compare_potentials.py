from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Literal, Optional

import torch
from torch import nn

from ocp.data.cifar10 import Cifar10DataConfig, build_cifar10_loaders
from ocp.models.wrapper import PotentialModel
from ocp.potentials.logsumexp import LogSumExpPotential
from ocp.potentials.moreau_max import MoreauMaxPotential
from ocp.train.loop import eval_one_epoch, train_one_epoch


Which = Literal["logsumexp", "moreau", "both"]
OptimizerName = Literal["adamw", "sgd"]


def _set_seed(seed: int) -> None:
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


def _build_resnet18_for_cifar(*, num_classes: int) -> nn.Module:
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


def _make_optimizer(
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


def run_experiment(
    *,
    which: Which,
    num_classes: int,
    lam: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    data_dir: str,
    optimizer: OptimizerName,
    lr: float,
    weight_decay: float,
    momentum: float,
    amp: bool,
    grad_clip_norm: Optional[float],
    out_jsonl: Optional[str],
    device: Optional[str],
) -> List[Dict]:
    _set_seed(seed)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    data_cfg = Cifar10DataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=True,
        seed=seed,
        num_classes=num_classes,
        augment=augment,
    )
    train_loader, test_loader = build_cifar10_loaders(data_cfg)

    def _run_one(potential, *, tag: str, extra: Dict) -> Dict:
        backbone = _build_resnet18_for_cifar(num_classes=num_classes)
        model = PotentialModel(backbone, potential).to(device_t)

        opt = _make_optimizer(
            name=optimizer,
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        metrics = {
            "tag": tag,
            "seed": seed,
            "device": str(device_t),
            "data": asdict(data_cfg),
            "optimizer": {"name": optimizer, "lr": lr, "weight_decay": weight_decay, "momentum": momentum},
            "epochs": epochs,
            "amp": bool(amp),
            "grad_clip_norm": grad_clip_norm,
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            **extra,
        }

        loss_fn = lambda logits, targets: model.loss(logits, targets, reduction="mean")

        for ep in range(1, epochs + 1):
            tr = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=opt,
                loss_fn=loss_fn,
                device=device_t,
                amp=amp,
                grad_clip_norm=grad_clip_norm,
            )
            te = eval_one_epoch(model=model, loader=test_loader, loss_fn=loss_fn, device=device_t)

            metrics["train_loss"].append(tr.loss)
            metrics["train_acc"].append(tr.acc)
            metrics["test_loss"].append(te.loss)
            metrics["test_acc"].append(te.acc)

            print(
                f"[{tag}] epoch {ep:03d}/{epochs} | "
                f"train loss {tr.loss:.4f} acc {tr.acc:.4f} | "
                f"test loss {te.loss:.4f} acc {te.acc:.4f}"
            )

        metrics["final_test_acc"] = float(metrics["test_acc"][-1]) if metrics["test_acc"] else None
        return metrics

    results: List[Dict] = []
    if which in ("logsumexp", "both"):
        results.append(
            _run_one(
                LogSumExpPotential(),
                tag="logsumexp",
                extra={"potential": "logsumexp"},
            )
        )
    if which in ("moreau", "both"):
        results.append(
            _run_one(
                MoreauMaxPotential(lam=lam),
                tag=f"moreau_lam{lam:g}",
                extra={"potential": "moreau_max", "lam": float(lam)},
            )
        )

    if out_jsonl is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)) or ".", exist_ok=True)
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} result(s) to {out_jsonl}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare LogSumExp vs Moreau-max potentials on CIFAR-10.")
    p.add_argument("--which", type=str, default="both", choices=["logsumexp", "moreau", "both"])
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--lam", type=float, default=2.0, help="Moreau λ (only used for --which moreau/both).")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--augment", type=int, default=1, help="1 to enable train augmentation, 0 to disable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9, help="Only used for SGD.")
    p.add_argument("--amp", type=int, default=1, help="1 to enable AMP on CUDA, 0 to disable.")
    p.add_argument("--grad_clip_norm", type=float, default=0.0, help="0 disables grad clipping.")
    p.add_argument("--out_jsonl", type=str, default=None)
    p.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    run_experiment(
        which=args.which,
        num_classes=args.num_classes,
        lam=args.lam,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=bool(args.augment),
        seed=args.seed,
        data_dir=args.data_dir,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        amp=bool(args.amp),
        grad_clip_norm=(None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)),
        out_jsonl=args.out_jsonl,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

