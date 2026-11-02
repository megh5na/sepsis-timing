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
