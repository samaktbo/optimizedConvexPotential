from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
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


def _append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _kind_from_cli(which: CliPotential) -> PotentialKind:
    if which == "moreau":
        return "moreau_max"
    return which


def _tune_one_potential(
    *,
    kind: PotentialKind,
    potential_name: str,
    train_loader,
    val_loader,
    data_dict: Dict[str, Any],
    num_classes: int,
    epochs: int,
    optimizer: OptimizerName,
    weight_decay: float,
    momentum: float,
    amp: bool,
    grad_clip_norm: Optional[float],
    device: torch.device,
    seed: int,
    num_trials: int,
    lam_range: Tuple[float, float],
    lr_range: Tuple[float, float],
    keep_ratio: float,
    min_rung_epochs: int,
    rung_eta: int,
    search_seed: int,
    out_jsonl: Optional[str],
    verbose_trial: bool,
) -> Dict[str, Any]:
    rung_epochs = build_rung_epochs(max_epochs=epochs, min_epochs=min_rung_epochs, eta=rung_eta)
    candidates = sample_candidates(
        num_trials=num_trials,
        lam_range=lam_range,
        lr_range=lr_range,
        seed=search_seed,
    )

    def _evaluate_trial(candidate, rung_idx: int, rung_epochs_i: int) -> Dict[str, Any]:
        tag = (
            f"{potential_name}_trial{candidate.trial_id}_r{rung_idx}_e{rung_epochs_i}"
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
            device=device,
            seed=seed,
            data_dict=data_dict,
            tag=tag,
            extra={
                "phase": "tune_val",
                "search": "random_halving",
                "search_seed": int(search_seed),
                "potential": potential_name,
                "tune_trial": int(candidate.trial_id),
                "rung_idx": int(rung_idx),
                "rung_epochs": int(rung_epochs_i),
                "lam": float(candidate.lam),
                "objective": "max_final_val_acc",
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
                f"[{potential_name}] trial={candidate.trial_id} rung={rung_idx} "
                f"final_val_acc={metrics.get('final_val_acc')} "
                f"lam={candidate.lam:g} lr={candidate.lr:g}"
            )
        return metrics

    search_out = run_successive_halving(
        candidates=candidates,
        rung_epochs=rung_epochs,
        keep_ratio=keep_ratio,
        evaluate_trial=_evaluate_trial,
    )

    best_lam = float(search_out["best_lam"])
    best_lr = float(search_out["best_lr"])
    best_metrics = search_out["best_metrics"]

    summary = {
        "phase": "tune_summary",
        "search": "random_halving",
        "potential": potential_name,
        "search_seed": int(search_seed),
        "num_trials": int(num_trials),
        "keep_ratio": float(keep_ratio),
        "rung_history": search_out["rung_history"],
        "best_trial_id": int(search_out["best_trial_id"]),
        "best_lam": best_lam,
        "best_lr": best_lr,
        "best_final_val_acc": float(search_out["best_score"]),
    }
    _append_jsonl(out_jsonl, summary)

    print(
        f"[{potential_name}] best_val: trial={search_out['best_trial_id']} "
        f"lam={best_lam:g} lr={best_lr:g} final_val_acc={best_metrics.get('final_val_acc')}"
    )
    return {
        "best_lam": best_lam,
        "best_lr": best_lr,
        "best_metrics": best_metrics,
        "search_summary": summary,
    }


def _final_train_test(
    *,
    kind: PotentialKind,
    potential_name: str,
    lam: float,
    lr: float,
    num_classes: int,
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
    device: torch.device,
) -> Dict[str, Any]:
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

    final_metrics = train_one_potential(
        kind=kind,
        lam=float(lam),
        lr=float(lr),
        num_classes=num_classes,
        epochs=int(epochs_final),
        train_loader=train_loader,
        eval_loader=test_loader,
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
        amp=amp,
        grad_clip_norm=grad_clip_norm,
        device=device,
        seed=seed,
        data_dict=data_dict,
        tag=f"{potential_name}_final_full_train",
        extra={
            "phase": "final_full_train_test",
            "potential": potential_name,
            "lam": float(lam),
        },
        eval_prefix="test",
        verbose=True,
    )
    return final_metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Tune lam/lr for each convex potential with random-halving on validation, then "
            "retrain each winner on full train and compare on test."
        )
    )
    p.add_argument(
        "--potentials",
        type=str,
        nargs="+",
        default=["logsumexp", "moreau", "simplex_entropy"],
        choices=["logsumexp", "moreau", "simplex_entropy"],
        help="Potentials to tune and compare.",
    )
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8, help="Max epochs per tuning trial.")
    p.add_argument(
        "--epochs_final",
        type=int,
        default=None,
        help="Epochs for final full-train runs (default: same as --epochs).",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--augment", type=int, default=1, help="1 enables train augmentation.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--search_seed", type=int, default=None, help="Default: same as --seed.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9, help="Only used for SGD.")
    p.add_argument("--amp", type=int, default=1, help="1 enables AMP on CUDA.")
    p.add_argument("--grad_clip_norm", type=float, default=0.0, help="0 disables grad clipping.")
    p.add_argument("--num_trials", type=int, default=12, help="Candidates sampled per potential.")
    p.add_argument(
        "--lam_range",
        type=float,
        nargs=2,
        default=[1e-2, 1e1],
        metavar=("LOW", "HIGH"),
        help="Log-uniform lam range [low high].",
    )
    p.add_argument(
        "--lr_range",
        type=float,
        nargs=2,
        default=[1e-5, 1e-1],
        metavar=("LOW", "HIGH"),
        help="Log-uniform learning rate range [low high].",
    )
    p.add_argument("--keep_ratio", type=float, default=0.5, help="Fraction kept after each rung.")
    p.add_argument("--min_rung_epochs", type=int, default=2)
    p.add_argument("--rung_eta", type=int, default=2)
    p.add_argument("--quiet_trials", type=int, default=0, help="1 prints compact logs for trials.")
    p.add_argument("--out_jsonl", type=str, default=None, help="Append per-trial and summary logs.")
    p.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    device_t = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    grad_clip = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)
    verbose_trial = args.quiet_trials == 0
    search_seed_base = args.seed if args.search_seed is None else int(args.search_seed)
    epochs_final = args.epochs if args.epochs_final is None else int(args.epochs_final)

    tune_cfg = Cifar10DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        seed=args.seed,
        num_classes=args.num_classes,
        augment=bool(args.augment),
    )
    train_loader, val_loader, _ = build_cifar10_train_val_test_loaders(
        tune_cfg, val_fraction=float(args.val_fraction)
    )
    data_dict = asdict(tune_cfg)

    tuned: List[Dict[str, Any]] = []
    for i, p_name in enumerate(list(args.potentials)):
        kind = _kind_from_cli(p_name)
        search_seed = search_seed_base + i
        print(f"\n=== Tuning potential={p_name} (kind={kind}) ===")
        t = _tune_one_potential(
            kind=kind,
            potential_name=("moreau_max" if kind == "moreau_max" else kind),
            train_loader=train_loader,
            val_loader=val_loader,
            data_dict=data_dict,
            num_classes=args.num_classes,
            epochs=int(args.epochs),
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            amp=bool(args.amp),
            grad_clip_norm=grad_clip,
            device=device_t,
            seed=args.seed,
            num_trials=int(args.num_trials),
            lam_range=(float(args.lam_range[0]), float(args.lam_range[1])),
            lr_range=(float(args.lr_range[0]), float(args.lr_range[1])),
            keep_ratio=float(args.keep_ratio),
            min_rung_epochs=int(args.min_rung_epochs),
            rung_eta=int(args.rung_eta),
            search_seed=int(search_seed),
            out_jsonl=args.out_jsonl,
            verbose_trial=verbose_trial,
        )
        tuned.append(
            {
                "potential_cli": p_name,
                "kind": kind,
                "potential_name": ("moreau_max" if kind == "moreau_max" else kind),
                **t,
            }
        )

    final_rows: List[Dict[str, Any]] = []
    for t in tuned:
        print(
            f"\n=== Final full-train test potential={t['potential_name']} "
            f"lam={t['best_lam']:g} lr={t['best_lr']:g} ==="
        )
        final_metrics = _final_train_test(
            kind=t["kind"],
            potential_name=t["potential_name"],
            lam=float(t["best_lam"]),
            lr=float(t["best_lr"]),
            num_classes=args.num_classes,
            epochs_final=epochs_final,
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
            device=device_t,
        )
        _append_jsonl(args.out_jsonl, final_metrics)
        final_rows.append(
            {
                "potential": t["potential_name"],
                "best_lam": float(t["best_lam"]),
                "best_lr": float(t["best_lr"]),
                "final_val_acc": float(t["best_metrics"].get("final_val_acc")),
                "final_test_acc": float(final_metrics.get("final_test_acc")),
                "test_metrics": final_metrics,
            }
        )

    final_rows.sort(key=lambda r: (-r["final_test_acc"], -r["final_val_acc"], r["potential"]))
    winner = final_rows[0]
    print("\n=== Final ranking by final_test_acc ===")
    for j, row in enumerate(final_rows, start=1):
        print(
            f"{j}. potential={row['potential']} test_acc={row['final_test_acc']:.6f} "
            f"val_acc={row['final_val_acc']:.6f} "
            f"lam={row['best_lam']:g} lr={row['best_lr']:g}"
        )
    print(
        f"\nWinner: potential={winner['potential']} "
        f"final_test_acc={winner['final_test_acc']:.6f}"
    )

    _append_jsonl(
        args.out_jsonl,
        {
            "phase": "final_ranking",
            "objective": "max_final_test_acc",
            "rows": final_rows,
            "winner": winner,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
