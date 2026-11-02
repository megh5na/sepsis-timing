# estimates confidence intervals on per-patient metrics (e.g. utility) by resampling

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    alpha: float


def bootstrap_ci(
    per_patient_values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    values = np.asarray(per_patient_values, dtype=np.float64)
    n = len(values)
    rng = np.random.default_rng(seed)

    point = float(statistic(values))
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic(values[idx])

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return BootstrapCI(point_estimate=point, ci_low=lo, ci_high=hi, n_boot=n_boot, alpha=alpha)


def paired_bootstrap_diff_ci(
    per_patient_values_a: np.ndarray,
    per_patient_values_b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    a = np.asarray(per_patient_values_a, dtype=np.float64)
    b = np.asarray(per_patient_values_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires the same patients in both arrays")
    n = len(a)
    rng = np.random.default_rng(seed)

    point = float(statistic(b) - statistic(a))
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = statistic(b[idx]) - statistic(a[idx])

    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return BootstrapCI(point_estimate=point, ci_low=lo, ci_high=hi, n_boot=n_boot, alpha=alpha)
