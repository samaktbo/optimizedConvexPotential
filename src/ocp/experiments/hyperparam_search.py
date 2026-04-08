from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SearchCandidate:
    trial_id: int
    lam: float
    lr: float


def build_rung_epochs(*, max_epochs: int, min_epochs: int = 2, eta: int = 2) -> List[int]:
    """Build a geometric rung schedule ending at ``max_epochs``.

    Example: min=2, eta=2, max=8 -> [2, 4, 8]
    """
    if max_epochs <= 0:
        raise ValueError(f"max_epochs must be > 0; got {max_epochs}.")
    if min_epochs <= 0:
        raise ValueError(f"min_epochs must be > 0; got {min_epochs}.")
    if eta < 2:
        raise ValueError(f"eta must be >= 2; got {eta}.")

    first = min(min_epochs, max_epochs)
    out = [int(first)]
    while out[-1] < max_epochs:
        nxt = out[-1] * eta
        if nxt >= max_epochs:
            break
        out.append(int(nxt))
    if out[-1] != max_epochs:
        out.append(int(max_epochs))
    return out


def sample_log_uniform(
    *,
    rng: random.Random,
    low: float,
    high: float,
) -> float:
    if low <= 0 or high <= 0:
        raise ValueError(f"log-uniform bounds must be positive; got low={low}, high={high}.")
    if low > high:
        raise ValueError(f"low must be <= high; got low={low}, high={high}.")
    if low == high:
        return float(low)
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_candidates(
    *,
    num_trials: int,
    lam_range: Tuple[float, float],
    lr_range: Tuple[float, float],
    seed: int,
) -> List[SearchCandidate]:
    if num_trials <= 0:
        raise ValueError(f"num_trials must be > 0; got {num_trials}.")

    rng = random.Random(seed)
    return [
        SearchCandidate(
            trial_id=i,
            lam=sample_log_uniform(rng=rng, low=float(lam_range[0]), high=float(lam_range[1])),
            lr=sample_log_uniform(rng=rng, low=float(lr_range[0]), high=float(lr_range[1])),
        )
        for i in range(num_trials)
    ]


TrialEvaluator = Callable[[SearchCandidate, int, int], Dict[str, Any]]


def _extract_val_loss(metrics: Dict[str, Any]) -> float:
    if "final_val_loss" in metrics:
        return float(metrics["final_val_loss"])
    val_loss_seq = metrics.get("val_loss")
    if isinstance(val_loss_seq, list) and val_loss_seq:
        return float(val_loss_seq[-1])
    return float("inf")


def run_successive_halving(
    *,
    candidates: Sequence[SearchCandidate],
    rung_epochs: Sequence[int],
    keep_ratio: float,
    evaluate_trial: TrialEvaluator,
) -> Dict[str, Any]:
    """Run successive halving over candidates.

    ``evaluate_trial`` is called as:
      evaluate_trial(candidate, rung_index, epochs_for_this_rung) -> metrics dict
    and should provide at least ``final_val_acc``.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty.")
    if not rung_epochs:
        raise ValueError("rung_epochs must be non-empty.")
    if not (0.0 < keep_ratio < 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1); got {keep_ratio}.")

    active: List[SearchCandidate] = list(candidates)
    rung_history: List[Dict[str, Any]] = []
    final_rung_best: Optional[Dict[str, Any]] = None
    all_trial_records: List[Dict[str, Any]] = []

    for rung_idx, epochs in enumerate(rung_epochs):
        rung_records: List[Dict[str, Any]] = []
        for c in active:
            metrics = evaluate_trial(c, rung_idx, int(epochs))
            val_acc = float(metrics.get("final_val_acc", float("-inf")))
            val_loss = _extract_val_loss(metrics)

            record = {
                "trial_id": int(c.trial_id),
                "lam": float(c.lam),
                "lr": float(c.lr),
                "rung_idx": int(rung_idx),
                "epochs": int(epochs),
                "objective": "max_final_val_acc",
                "score": float(val_acc),
                "tie_breaker_val_loss": float(val_loss),
                "metrics": metrics,
            }
            rung_records.append(record)
            all_trial_records.append(record)

        # Higher score first; then lower val loss; then smaller trial id.
        rung_records.sort(key=lambda r: (-r["score"], r["tie_breaker_val_loss"], r["trial_id"]))
        if rung_idx == len(rung_epochs) - 1:
            final_rung_best = rung_records[0]
        keep_n = 1 if rung_idx == len(rung_epochs) - 1 else max(1, int(math.ceil(len(rung_records) * keep_ratio)))
        survivor_ids = {int(r["trial_id"]) for r in rung_records[:keep_n]}
        active = [c for c in active if c.trial_id in survivor_ids]

        rung_history.append(
            {
                "rung_idx": int(rung_idx),
                "epochs": int(epochs),
                "num_candidates_in": int(len(rung_records)),
                "num_candidates_kept": int(keep_n),
                "ranking": [
                    {
                        "trial_id": int(r["trial_id"]),
                        "lam": float(r["lam"]),
                        "lr": float(r["lr"]),
                        "score": float(r["score"]),
                        "tie_breaker_val_loss": float(r["tie_breaker_val_loss"]),
                    }
                    for r in rung_records
                ],
            }
        )

    if final_rung_best is None:
        raise RuntimeError("No trial was evaluated.")

    return {
        "best_trial_id": int(final_rung_best["trial_id"]),
        "best_lam": float(final_rung_best["lam"]),
        "best_lr": float(final_rung_best["lr"]),
        "best_score": float(final_rung_best["score"]),
        "best_metrics": final_rung_best["metrics"],
        "rung_history": rung_history,
        "trial_records": all_trial_records,
    }
