# turns model scores into binary alarms, and searches for the threshold that maximises
# the objective function (normalised utility) on the validation split.

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


def tune_threshold( # tries each candidate cutoff, calls objective_fn, returns the best one
    candidate_thresholds: np.ndarray,
    objective_fn: Callable[[float], float],
    split_name: str,
) -> ThresholdResult:
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


def apply_threshold(scores: np.ndarray, tau: float) -> np.ndarray: # finds the first hour where scores > tau, returns a binary array of alarms from that hour onward
    hits = np.flatnonzero(scores > tau)
    out = np.zeros_like(scores, dtype=np.float64)
    if hits.size:
        out[hits[0]:] = 1.0
    return out


def first_alarm_hour(scores: np.ndarray, tau: float) -> int | None: # returns the first hour where scores > tau, or None if never
    hits = np.flatnonzero(scores > tau)
    return int(hits[0]) if hits.size else None
