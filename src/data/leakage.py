"""Leakage guards as runtime assertions (spec §3.6), not comments.

Four guards, matching the spec's numbering:

  1. No patient_id in more than one split          -> src/data/splits.py
     (assert_no_patient_in_multiple_splits, assert_splits_partition_manifest)
  2. Normalisation stats fitted on train only,
     used on other data only after fit()           -> src/data/stats.py
     (TrainStats.is_fitted / NotFittedError)
  3. No future rows visible at any hour             -> this module
     (assert_causal_in_time)
  4. Threshold tau tuned on validation only,
     test data never passed to the tuning routine  -> src/decision/threshold.py
     (tune_threshold's split_name assertion)

This module holds guard #3: a generic causality check usable against any
per-timestep feature-construction function (preprocessing) or sequence
model (the encoder, once it exists) — perturb a future timestep of the
input and assert that outputs at or before the perturbed timestep are
unchanged.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def assert_causal_in_time(
    fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_trials: int = 5,
    seed: int = 0,
    atol: float = 1e-8,
) -> None:
    """fn maps (T, C_in) -> (T, C_out) (or any array whose first axis is
    time). For several random perturbation points, replace X from that
    point onward with noise and assert fn's output strictly before that
    point is unchanged. Raises AssertionError with the offending timestep
    on the first violation found.

    This directly executes the spec's guard #3 ("no forward-looking
    operation ... exists in the pipeline") against the real function, not
    just against a description of it.
    """
    T = X.shape[0]
    if T < 2:
        return  # nothing to perturb

    rng = np.random.default_rng(seed)
    baseline = np.asarray(fn(X))

    trial_points = rng.integers(1, T, size=min(n_trials, T - 1))
    for cut in trial_points:
        X_perturbed = X.copy()
        # Overwrite everything from `cut` onward with values far outside the
        # normal range, so any leak is very likely to show up numerically.
        X_perturbed[cut:] = rng.normal(loc=1e6, scale=1e3, size=X_perturbed[cut:].shape)

        out_perturbed = np.asarray(fn(X_perturbed))

        before = baseline[:cut]
        before_perturbed = out_perturbed[:cut]
        if not np.allclose(before, before_perturbed, atol=atol, equal_nan=True):
            bad_t = int(np.argmax(~np.isclose(before, before_perturbed, atol=atol, equal_nan=True).any(axis=tuple(range(1, before.ndim)))))
            raise AssertionError(
                f"causality violation: perturbing input from t={cut} onward changed "
                f"output at t={bad_t} (< cut). A forward-looking operation "
                f"(e.g. backward-fill, centred rolling window) exists in `fn`."
            )
