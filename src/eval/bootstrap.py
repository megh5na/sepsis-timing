"""Paired bootstrap over patients (spec §7.3, §13 "every result carries a
confidence interval").

Phase 1 uses this for single-condition CIs on B0/B1 (stage 6 gate) and for
the threshold-headroom numbers (stage 7). The paired-difference form
(comparing two conditions evaluated on the same patients, e.g. P1 vs P2) is
Phase 2's primary analysis — this module is written generically enough to
serve both, but Phase 1 only calls the single-sample path.
"""
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
    """Resample patients with replacement n_boot times, recompute
    `statistic` on each resample, and take the (alpha/2, 1-alpha/2)
    percentiles as the CI. `per_patient_values` must already be one scalar
    per patient (e.g. per-patient utility) — resampling patients, not
    hours, is what makes this a *paired* bootstrap in the spec's sense
    (each patient's full trajectory is resampled as a unit)."""
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
    """CI on the difference (B - A) between two conditions evaluated on the
    SAME patients in the same order (e.g. P1 vs P2 with shared weights,
    Phase 2's primary comparison). Not called anywhere in Phase 1 — B0 and
    B1 use different decision rules over the same encoder but are not the
    kind of shared-weights paired comparison this is built for; Phase 1
    reports independent bootstrap_ci per condition instead. Included now so
    Phase 2 has a tested implementation to build on."""
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
