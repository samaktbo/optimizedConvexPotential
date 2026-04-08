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
from ocp.experiments.hyperparam_search import (
    build_rung_epochs,
    run_successive_halving,
    sample_candidates,
)

CliPotential = Literal["logsumexp", "moreau", "simplex_entropy"]
OptimizerName = Literal["adamw", "sgd"]


def _kind_from_cli(which: CliPotential) -> PotentialKind:
    if which == "moreau":
        return "moreau_max"
    return which  # logsumexp | simplex_entropy


def _append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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


def _random_halving_search(
    *,
    kind: PotentialKind,
    num_trials: int,
    lam_range: Tuple[float, float],
    lr_range: Tuple[float, float],
    keep_ratio: float,
    min_rung_epochs: int,
    rung_eta: int,
    search_seed: int,
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    rung_epochs = build_rung_epochs(max_epochs=epochs, min_epochs=min_rung_epochs, eta=rung_eta)
    candidates = sample_candidates(
        num_trials=num_trials,
        lam_range=lam_range,
        lr_range=lr_range,
        seed=search_seed,
    )

    def _evaluate_trial(candidate, rung_idx: int, rung_epochs_i: int) -> Dict[str, Any]:
        tag = (
            f"trial{candidate.trial_id}_r{rung_idx}_e{rung_epochs_i}"
            f"_lam{candidate.lam:g}_lr{candidate.lr:g}"
        )
        metrics = train_one_potential(
            kind=kind,
            lam=float(candidate.lam),
            lr=float(candidate.lr),
            num_classes=num_classes,
            epochs=int(rung_epochs_i),
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
                "search": "random_halving",
                "search_seed": int(search_seed),
                "tune_trial": int(candidate.trial_id),
                "rung_idx": int(rung_idx),
                "rung_epochs": int(rung_epochs_i),
                "potential": kind if kind != "moreau_max" else "moreau_max",
                "lam": float(candidate.lam),
                "objective": "max_final_val_acc",
                "val_fraction": float(val_fraction),
                "search_space": {
                    "lam_range": [float(lam_range[0]), float(lam_range[1])],
                    "lr_range": [float(lr_range[0]), float(lr_range[1])],
                    "num_trials": int(num_trials),
                    "keep_ratio": float(keep_ratio),
                    "rung_epochs": [int(e) for e in rung_epochs],
                },
            },
            eval_prefix="val",
            verbose=verbose_trial,
        )
        _append_jsonl(out_jsonl, metrics)
        if not verbose_trial:
            print(
                f"[{tag}] final_val_acc={metrics.get('final_val_acc')} "
                f"lam={candidate.lam:g} lr={candidate.lr:g}"
            )
        return metrics

    search_out = run_successive_halving(
        candidates=candidates,
        rung_epochs=rung_epochs,
        keep_ratio=keep_ratio,
        evaluate_trial=_evaluate_trial,
    )
    return search_out["best_metrics"], search_out


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
            "Tune λ and learning rate for one convex potential on CIFAR-10 using a held-out "
            "validation slice of the training set (maximize final validation accuracy)."
        )
    )
    p.add_argument(
        "--potential",
        type=str,
        required=True,
        choices=["logsumexp", "moreau", "simplex_entropy"],
    )
    p.add_argument(
        "--search",
        type=str,
        default="random_halving",
        choices=["random_halving", "grid"],
        help="Hyperparameter search strategy.",
    )
    p.add_argument("--lam_grid", type=float, nargs="+", default=None, help="Grid mode only: λ values.")
    p.add_argument("--lr_grid", type=float, nargs="+", default=None, help="Grid mode only: learning rates.")
    p.add_argument("--num_trials", type=int, default=12, help="Random-halving mode: sampled candidates.")
    p.add_argument(
        "--lam_range",
        type=float,
        nargs=2,
        default=[1e-2, 1e1],
        metavar=("LOW", "HIGH"),
        help="Random-halving mode: λ log-uniform range [low high].",
    )
    p.add_argument(
        "--lr_range",
        type=float,
        nargs=2,
        default=[1e-5, 1e-1],
        metavar=("LOW", "HIGH"),
        help="Random-halving mode: learning-rate log-uniform range [low high].",
    )
    p.add_argument(
        "--keep_ratio",
        type=float,
        default=0.5,
        help="Random-halving mode: fraction of candidates kept per rung.",
    )
    p.add_argument(
        "--min_rung_epochs",
        type=int,
        default=2,
        help="Random-halving mode: first rung epoch budget (geometric schedule up to --epochs).",
    )
    p.add_argument(
        "--rung_eta",
        type=int,
        default=2,
        help="Random-halving mode: epoch multiplier between rungs.",
    )
    p.add_argument(
        "--search_seed",
        type=int,
        default=None,
        help="Random-halving mode: seed for hyperparameter sampling (default: --seed).",
    )
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
    search_seed = args.seed if args.search_seed is None else int(args.search_seed)

    search_summary: Optional[Dict[str, Any]] = None
    if args.search == "grid":
        if not args.lam_grid or not args.lr_grid:
            raise ValueError("--lam_grid and --lr_grid are required when --search grid is selected.")
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
    else:
        best, search_summary = _random_halving_search(
            kind=kind,
            num_trials=int(args.num_trials),
            lam_range=(float(args.lam_range[0]), float(args.lam_range[1])),
            lr_range=(float(args.lr_range[0]), float(args.lr_range[1])),
            keep_ratio=float(args.keep_ratio),
            min_rung_epochs=int(args.min_rung_epochs),
            rung_eta=int(args.rung_eta),
            search_seed=search_seed,
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
        best_lam = float(search_summary["best_lam"])
        best_lr = float(search_summary["best_lr"])

    print(
        f"Best by final_val_acc: lam={best_lam:g} lr={best_lr:g} "
        f"final_val_acc={best.get('final_val_acc')}"
    )
    if search_summary is not None:
        print(
            f"Random-halving summary: best_trial_id={search_summary['best_trial_id']} "
            f"rungs={len(search_summary['rung_history'])} trials={args.num_trials}"
        )
        _append_jsonl(
            args.out_jsonl,
            {
                "phase": "search_summary",
                "search": "random_halving",
                "potential": kind,
                "search_seed": int(search_seed),
                "num_trials": int(args.num_trials),
                "keep_ratio": float(args.keep_ratio),
                "rung_history": search_summary["rung_history"],
                "best_trial_id": int(search_summary["best_trial_id"]),
                "best_lam": float(best_lam),
                "best_lr": float(best_lr),
                "best_final_val_acc": float(search_summary["best_score"]),
            },
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
        _append_jsonl(args.out_jsonl, final_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())