from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from itertools import product
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from ocp.data.cifar10 import (
    Cifar10DataConfig,
    build_cifar10_loaders,
    build_cifar10_train_val_test_loaders,
)
from ocp.experiments.cifar10_potential_training import PotentialKind, train_one_potential

CliPotential = Literal["logsumexp", "moreau", "simplex_entropy"]
OptimizerName = Literal["adamw", "sgd"]


def _kind_from_cli(which: CliPotential) -> PotentialKind:
    if which == "moreau":
        return "moreau_max"
    return which  # logsumexp | simplex_entropy


def _grid_search(
    *,
    kind: PotentialKind,
    lam_grid: List[float],
    lr_grid: List[float],
    num_classes: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    data_dir: str,
    optimizer: OptimizerName,
    weight_decay: float,
    momentum: float,
    amp: bool,
    grad_clip_norm: Optional[float],
    val_fraction: float,
    device: Optional[str],
    out_jsonl: Optional[str],
    verbose_trial: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
    train_loader, val_loader, _test_holdout = build_cifar10_train_val_test_loaders(
        data_cfg, val_fraction=val_fraction
    )
    data_dict = asdict(data_cfg)

    best: Optional[Dict[str, Any]] = None
    all_metrics: List[Dict[str, Any]] = []

    jsonl_f = None
    if out_jsonl is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)) or ".", exist_ok=True)
        jsonl_f = open(out_jsonl, "a", encoding="utf-8")

    try:
        for trial_idx, (lam, lr) in enumerate(product(lam_grid, lr_grid)):
            tag = f"trial{trial_idx}_lam{lam:g}_lr{lr:g}"
            metrics = train_one_potential(
                kind=kind,
                lam=float(lam),
                lr=float(lr),
                num_classes=num_classes,
                epochs=epochs,
                train_loader=train_loader,
                eval_loader=val_loader,
                optimizer=optimizer,
                weight_decay=weight_decay,
                momentum=momentum,
                amp=amp,
                grad_clip_norm=grad_clip_norm,
                device=device_t,
                seed=seed,
                data_dict=data_dict,
                tag=tag,
                extra={
                    "tune_trial": trial_idx,
                    "potential": kind if kind != "moreau_max" else "moreau_max",
                    "lam": float(lam),
                    "objective": "max_final_val_acc",
                    "val_fraction": float(val_fraction),
                },
                eval_prefix="val",
                verbose=verbose_trial,
            )
            all_metrics.append(metrics)
            if not verbose_trial:
                print(
                    f"[{tag}] final_val_acc={metrics.get('final_val_acc')} "
                    f"lam={lam:g} lr={lr:g}"
                )
            if jsonl_f is not None:
                jsonl_f.write(json.dumps(metrics) + "\n")
            score = metrics.get("final_val_acc")
            if score is None:
                continue
            if best is None or float(score) > float(best.get("final_val_acc", 0.0)):
                best = metrics
    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    if best is None:
        raise RuntimeError("No trial produced a valid final_val_acc; check grids and data config.")

    return best, all_metrics


def _final_train_test(
    *,
    kind: PotentialKind,
    lam: float,
    lr: float,
    num_classes: int,
    epochs: int,
    epochs_final: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    data_dir: str,
    optimizer: OptimizerName,
    weight_decay: float,
    momentum: float,
    amp: bool,
    grad_clip_norm: Optional[float],
    device: Optional[str],
    tag: str,
    verbose: bool,
) -> Dict[str, Any]:
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

    return train_one_potential(
        kind=kind,
        lam=float(lam),
        lr=float(lr),
        num_classes=num_classes,
        epochs=epochs_final,
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
        extra={
            "phase": "final_full_train_test",
            "potential": kind if kind != "moreau_max" else "moreau_max",
            "lam": float(lam),
            "epochs_tune_reference": int(epochs),
        },
        eval_prefix="test",
        verbose=verbose,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Grid search λ and learning rate for one convex potential on CIFAR-10 using a "
            "held-out validation slice of the training set (maximize final validation accuracy)."
        )
    )
    p.add_argument(
        "--potential",
        type=str,
        required=True,
        choices=["logsumexp", "moreau", "simplex_entropy"],
    )
    p.add_argument("--lam_grid", type=float, nargs="+", required=True, help="λ values to try.")
    p.add_argument("--lr_grid", type=float, nargs="+", required=True, help="Learning rates to try.")
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of the training split held out for validation (default: 0.1).",
    )
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--epochs_final",
        type=int,
        default=None,
        help="Epochs for --final_test retrain (default: same as --epochs).",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--augment", type=int, default=1, help="1 to enable train augmentation, 0 to disable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9, help="Only used for SGD.")
    p.add_argument("--amp", type=int, default=1, help="1 to enable AMP on CUDA, 0 to disable.")
    p.add_argument("--grad_clip_norm", type=float, default=0.0, help="0 disables grad clipping.")
    p.add_argument("--out_jsonl", type=str, default=None, help="Append one JSON line per trial.")
    p.add_argument(
        "--quiet_trials",
        type=int,
        default=0,
        help="1 to print only a short summary per trial instead of full per-epoch logs.",
    )
    p.add_argument(
        "--final_test",
        type=int,
        default=0,
        help="1 to retrain on the full training set with the best (λ, lr) and evaluate on test once.",
    )
    p.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    kind = _kind_from_cli(args.potential)
    grad_clip = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)
    verbose_trials = args.quiet_trials == 0

    best, _all = _grid_search(
        kind=kind,
        lam_grid=list(args.lam_grid),
        lr_grid=list(args.lr_grid),
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=bool(args.augment),
        seed=args.seed,
        data_dir=args.data_dir,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        amp=bool(args.amp),
        grad_clip_norm=grad_clip,
        val_fraction=float(args.val_fraction),
        device=args.device,
        out_jsonl=args.out_jsonl,
        verbose_trial=verbose_trials,
    )

    best_lam = float(best["lam"])
    best_lr = float(best["optimizer"]["lr"])
    print(
        f"Best by final_val_acc: lam={best_lam:g} lr={best_lr:g} "
        f"final_val_acc={best.get('final_val_acc')}"
    )

    if args.final_test:
        ef = args.epochs_final if args.epochs_final is not None else args.epochs
        final_m = _final_train_test(
            kind=kind,
            lam=best_lam,
            lr=best_lr,
            num_classes=args.num_classes,
            epochs=args.epochs,
            epochs_final=ef,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=bool(args.augment),
            seed=args.seed,
            data_dir=args.data_dir,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            amp=bool(args.amp),
            grad_clip_norm=grad_clip,
            device=args.device,
            tag="final_full_train",
            verbose=not bool(args.quiet_trials),
        )
        print(
            f"Final test (full train): final_test_acc={final_m.get('final_test_acc')} "
            f"lam={best_lam:g} lr={best_lr:g}"
        )
        if args.out_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
            with open(args.out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_m) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())