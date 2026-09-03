"""Tuned-threshold decision rule (spec §5.2 baseline rule; used by B0/B1
via a scalar risk score r[t], and by P1 on F[t,6] in Phase 2).

Zero learned parameters — pure post-hoc search over a scalar threshold,
tuned to maximise an objective (normalised challenge utility, once the
Phase-1 harness exists) evaluated only on the validation split.

Guard #4 (spec §3.6): "Threshold tau for baselines tuned on validation
only. Assert test data is never passed to the tuning routine." Enforced
here as a runtime assertion on `split_name`, not as a docstring promise —
`tune_threshold` refuses to run against anything but "train" or "val".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class ThresholdSplitLeakageError(RuntimeError):
    pass


@dataclass
class ThresholdResult:
    threshold: float
    objective_value: float
    split_name: str
    all_thresholds: np.ndarray
    all_objective_values: np.ndarray


def tune_threshold(
    candidate_thresholds: np.ndarray,
    objective_fn: Callable[[float], float],
    split_name: str,
) -> ThresholdResult:
    """Search `candidate_thresholds` for the value maximising `objective_fn`.

    `objective_fn(tau)` should evaluate whatever alarm rule + metric is
    being tuned (e.g. "apply r[t] > tau on the validation cohort, return
    normalised utility") — it is the caller's responsibility to make sure
    the data closed over by `objective_fn` is actually the validation
    split; `split_name` is an explicit, checked declaration of that fact,
    not an inference from the closure.
    """
    if split_name != "val":
        raise ThresholdSplitLeakageError(
            f"tune_threshold refuses to run with split_name={split_name!r}. "
            "Threshold tau must be tuned on the validation split only "
            "(spec §3.6 guard #4) — pass split_name='val'."
        )

    candidate_thresholds = np.asarray(candidate_thresholds, dtype=np.float64)
    values = np.array([objective_fn(float(tau)) for tau in candidate_thresholds])
    best_idx = int(np.argmax(values))

    return ThresholdResult(
        threshold=float(candidate_thresholds[best_idx]),
        objective_value=float(values[best_idx]),
        split_name=split_name,
        all_thresholds=candidate_thresholds,
        all_objective_values=values,
    )


def apply_threshold(scores: np.ndarray, tau: float) -> np.ndarray:
    """scores: (T,) per-hour scalar risk r[t]. Returns a binary (T,) array,
    1 from the first hour scores > tau onward (once fired, stays alarmed —
    see harness.py for how this maps to an alarm_hour)."""
    hits = np.flatnonzero(scores > tau)
    out = np.zeros_like(scores, dtype=np.float64)
    if hits.size:
        out[hits[0]:] = 1.0
    return out


def first_alarm_hour(scores: np.ndarray, tau: float) -> int | None:
    hits = np.flatnonzero(scores > tau)
    return int(hits[0]) if hits.size else None
