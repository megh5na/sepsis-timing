"""Training-split-only normalisation statistics (spec §3.1, §3.6 guard #2).

`TrainStats` holds per-channel median (for imputing never-yet-observed
values), mean, and std (for z-scoring), computed once from the training
split. It is a "fitted" object: reading it before `fit()` has been called
raises, which is the runtime form of leakage guard #2 ("Assert the stats
object is fitted before any validation or test data is touched").
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


class NotFittedError(RuntimeError):
    pass


@dataclass
class TrainStats:
    n_channels: int
    median_: np.ndarray | None = field(default=None, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)
    _fitted: bool = False

    def fit(self, X_list: list[np.ndarray], channel_names: list[str] | None = None) -> "TrainStats":
        """X_list: list of (T_i, n_channels) arrays from TRAINING patients only.

        Median is computed from every genuinely observed value across all
        training patients and all hours (used to impute channels with no
        observation yet, spec §3.1 block 1). Mean/std are computed the same
        way, over observed values only, and used for z-scoring.

        A channel with literally zero observations across the entire
        training split is a real, disclosed data property here, not a
        hypothetical edge case: EtCO2 is 0% observed across all of Hospital
        A (confirmed against the full downloaded Hospital A archive during
        development — see README "Design decisions"), so the within-hospital
        Phase 1 split hits this for real. Such a channel falls back to
        median=0, mean=0, std=1 — after z-scoring it is uniformly 0 for
        every patient, contributing no VALUES signal, while its MASK stays
        0 and DELTA stays at maximum staleness everywhere (computed
        upstream in preprocess.py, unaffected by this fallback) — so the
        model still sees the correct "never observed" signal through
        MASK/DELTA, it just isn't handed a fabricated VALUES number. This
        is a printed warning, not a silent patch.
        """
        stacked = np.concatenate(X_list, axis=0)  # (sum T_i, n_channels)
        if stacked.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {stacked.shape[1]}")

        with warnings.catch_warnings():
            # An all-NaN channel (see docstring) legitimately triggers numpy's
            # "All-NaN slice" / "Mean of empty slice" warnings here; the NaN
            # result is handled explicitly right below, not silently ignored.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            median = np.nanmedian(stacked, axis=0)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)

        never_observed = np.isnan(median) | np.isnan(mean)
        if np.any(never_observed):
            bad = np.flatnonzero(never_observed)
            names = [channel_names[i] if channel_names else str(i) for i in bad]
            print(
                f"[TrainStats.fit] WARNING: channel(s) {names} have ZERO observations "
                f"in this training split. Falling back to median=0, mean=0, std=1 "
                f"(uniformly 0 after z-scoring; MASK/DELTA still correctly signal "
                f"'never observed' for these channels). This is a real property of "
                f"the split, not a bug — see README."
            )
            median = np.where(never_observed, 0.0, median)
            mean = np.where(never_observed, 0.0, mean)
            std = np.where(never_observed, 1.0, std)

        std = np.where(std < 1e-6, 1.0, std)  # avoid divide-by-zero for near-constant channels

        self.median_ = median
        self.mean_ = mean
        self.std_ = std
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(
                "TrainStats.fit() must be called on the training split before "
                "these statistics are used on any other data (spec §3.6 guard #2)."
            )

    def impute_and_zscore(self, values_filled: np.ndarray) -> np.ndarray:
        """Apply this stats object's median-impute + z-score to a forward-
        filled VALUES block (see preprocess.py block 1). Raises
        NotFittedError if called before fit()."""
        self._require_fitted()
        out = np.where(np.isnan(values_filled), self.median_[None, :], values_filled)
        return (out - self.mean_[None, :]) / self.std_[None, :]

    @property
    def median(self) -> np.ndarray:
        self._require_fitted()
        return self.median_

    @property
    def mean(self) -> np.ndarray:
        self._require_fitted()
        return self.mean_

    @property
    def std(self) -> np.ndarray:
        self._require_fitted()
        return self.std_
