from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Literal, Optional

import torch

from ocp.data.cifar10 import Cifar10DataConfig, build_cifar10_loaders
from ocp.experiments.cifar10_potential_training import (
    PotentialKind,
    set_training_seed,
    train_one_potential,
)

Which = Literal["logsumexp", "moreau", "simplex_entropy", "both"]
OptimizerName = Literal["adamw", "sgd"]


def run_experiment(
    *,
    which: Which,
    num_classes: int,
    lam: float,
    moreau_lams: Optional[List[float]],
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
    set_training_seed(seed)

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
    data_dict = asdict(data_cfg)

    def _run_one(
        *,
        kind: PotentialKind,
        lam_i: float,
        tag: str,
        extra: Dict,
    ) -> Dict:
        return train_one_potential(
            kind=kind,
            lam=lam_i,
            lr=lr,
            num_classes=num_classes,
            epochs=epochs,
            train_loader=train_loader,
            eval_loader=test_loader,
            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            amp=amp,
            grad_clip_norm=grad_clip_norm,
            device=device_t,
            seed=seed,
            data_dict=data_dict,
            tag=tag,
            extra=extra,
            eval_prefix="test",
            verbose=True,
        )

    results: List[Dict] = []
    if which in ("logsumexp", "both"):
        results.append(
            _run_one(
                kind="logsumexp",
                lam_i=lam,
                tag="logsumexp",
                extra={"potential": "logsumexp", "lam": float(lam)},
            )
        )
    if which in ("moreau", "both"):
        lams = [lam] if (not moreau_lams) else list(moreau_lams)
        for lam_i in lams:
            results.append(
                _run_one(
                    kind="moreau_max",
                    lam_i=lam_i,
                    tag=f"moreau_lam{lam_i:g}",
                    extra={"potential": "moreau_max", "lam": float(lam_i)},
                )
            )
    if which in ("simplex_entropy", "both"):
        results.append(
            _run_one(
                kind="simplex_entropy",
                lam_i=lam,
                tag="simplex_entropy",
                extra={"potential": "simplex_entropy_conjugate", "lam": float(lam)},
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
    p = argparse.ArgumentParser(
        description="Compare LogSumExp, Moreau-max, and simplex-entropy conjugate potentials on CIFAR-10."
    )
    p.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["logsumexp", "moreau", "simplex_entropy", "both"],
    )
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument(
        "--lam",
        type=float,
        default=2.0,
        help=(
            "λ for temperature-scaled log-sum-exp, simplex-entropy conjugate, and Moreau-max "
            "(Moreau only if --moreau_lams is not provided)."
        ),
    )
    p.add_argument(
        "--moreau_lams",
        type=float,
        nargs="*",
        default=None,
        help="If provided, run one Moreau experiment per λ value (e.g. --moreau_lams 1 2).",
    )
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
        moreau_lams=args.moreau_lams,
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
