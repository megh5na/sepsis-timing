"""Feature construction (spec §3.1), multi-horizon targets (§3.2), and
batching (§3.4).

Citation (spec §13): the MASK/DELTA representation below is the GRU-D
lineage of missingness-aware recurrent features —
    Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).
    "Recurrent Neural Networks for Multivariate Time Series with Missing
    Values." Scientific Reports 8, 6085.
Per spec §1.3, this is implemented and used here, not claimed as a
contribution of this project.

§3.1 — VALUES / MASK / DELTA
    From X (T, 40) raw, produce three (T, 40) blocks:
      VALUES : forward-filled; leading not-yet-observed entries are left as
               NaN here and filled with the *training-split* median by
               `stats.TrainStats.impute_and_zscore` — never with any other
               statistic, and never from validation/test data (guard #2).
      MASK   : 1 iff genuinely observed at this hour in the raw data, 0 if
               forward-filled or imputed.
      DELTA  : hours since last genuine observation, clipped at 24, scaled
               to [0, 1]; 1.0 for channels never yet observed.
    Demographics (Age, Gender, Unit1, Unit2, HospAdmTime) are forced to
    MASK=1, DELTA=0 throughout (static/broadcast, per spec). ICULOS gets the
    standard treatment (it is "already time-varying", per spec).
    Z = concat([VALUES_imputed_zscored, MASK, DELTA], axis=-1), shape (T, 120).

§3.2 — multi-horizon targets
    y[t, k]    = 1 if (is_septic and t_sepsis <= t + k) else 0,   k = 1..K
    mask[t, k] = 0 if (not is_septic and t + k > T) else 1
    Built here as a pure data artifact (no model/head/loss code touches it
    in Phase 1); it exists so Phase 2's heads and loss have a stable,
    already-tested interface to consume ("head-agnostic" per the scope
    control block). Not used by B0/B1 in Phase 1 — B1 trains directly
    against per-hour SepsisLabel (spec §4.2).

§3.4 — batching
    `collate_patients` pads a list of variable-length (T_i, C) sequences to
    the batch max length and returns a `length_mask` (B, T_max) so loss
    code can zero out padded positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.parse import FEATURE_COLUMNS
from src.data.stats import TrainStats

DEMOGRAPHIC_COLUMNS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
DEMOGRAPHIC_INDICES = [FEATURE_COLUMNS.index(c) for c in DEMOGRAPHIC_COLUMNS]
N_CHANNELS = len(FEATURE_COLUMNS)  # 40
N_FEATURES_Z = 3 * N_CHANNELS  # 120

DELTA_CLIP_HOURS = 24.0


def _forward_fill(X: np.ndarray) -> np.ndarray:
    """Causal forward-fill along axis 0. Leading NaNs (no observation yet)
    are left as NaN. No backward-looking operation anywhere (guard #3)."""
    T, C = X.shape
    out = X.copy()
    last = np.full(C, np.nan)
    for t in range(T):
        row = out[t]
        observed = ~np.isnan(row)
        last = np.where(observed, row, last)
        out[t] = last
    return out


def _compute_mask(X_raw: np.ndarray) -> np.ndarray:
    return (~np.isnan(X_raw)).astype(np.float64)


def _compute_delta(mask: np.ndarray) -> np.ndarray:
    """Hours since last genuine observation, causal, clipped to
    DELTA_CLIP_HOURS then scaled to [0, 1]. 1.0 for a channel never yet
    observed up to and including hour t."""
    T, C = mask.shape
    delta_hours = np.zeros((T, C))
    since = np.full(C, np.inf)  # np.inf == "never observed yet"
    for t in range(T):
        observed_now = mask[t] == 1
        since = np.where(observed_now, 0.0, since + 1.0)
        # first hour ever seen for a channel: since==0 for that channel this
        # step (correct: 0 hours have elapsed since it was just observed)
        delta_hours[t] = since
    delta_hours = np.minimum(delta_hours, DELTA_CLIP_HOURS)
    delta_hours = np.where(np.isinf(delta_hours), DELTA_CLIP_HOURS, delta_hours)
    return delta_hours / DELTA_CLIP_HOURS


def compute_values_mask_delta(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X_raw: (T, 40) with the column order in src.data.parse.FEATURE_COLUMNS.
    Returns (VALUES, MASK, DELTA), each (T, 40). VALUES may still contain
    NaN for leading not-yet-observed entries — see module docstring."""
    if X_raw.shape[1] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels, got {X_raw.shape[1]}")

    mask = _compute_mask(X_raw)
    values = _forward_fill(X_raw)
    delta = _compute_delta(mask)

    mask[:, DEMOGRAPHIC_INDICES] = 1.0
    delta[:, DEMOGRAPHIC_INDICES] = 0.0

    return values, mask, delta


def build_Z(X_raw: np.ndarray, stats: TrainStats) -> np.ndarray:
    """Full (T, 120) feature tensor: [VALUES_imputed_zscored | MASK | DELTA].
    Raises NotFittedError (via stats.impute_and_zscore) if `stats` was not
    fit on the training split first."""
    values, mask, delta = compute_values_mask_delta(X_raw)
    values_z = stats.impute_and_zscore(values)
    return np.concatenate([values_z, mask, delta], axis=-1)


# ---------------------------------------------------------------------------
# §3.2 multi-horizon targets
# ---------------------------------------------------------------------------

def build_multi_horizon_targets(T: int, t_sepsis: int | None, K: int) -> tuple[np.ndarray, np.ndarray]:
    """y, mask each (T, K). y[t, k] uses k in {1..K} (1-indexed horizon,
    0-indexed as y[:, k-1])."""
    is_septic = t_sepsis is not None
    t = np.arange(T)[:, None]          # (T, 1)
    k = np.arange(1, K + 1)[None, :]   # (1, K)

    if is_septic:
        y = (t_sepsis <= t + k).astype(np.float64)
        mask = np.ones((T, K), dtype=np.float64)
    else:
        y = np.zeros((T, K), dtype=np.float64)
        mask = (~(t + k > T)).astype(np.float64)

    return y, mask


# ---------------------------------------------------------------------------
# §3.4 batching
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    Z: np.ndarray            # (B, T_max, 120)
    length_mask: np.ndarray  # (B, T_max), 1 for real timesteps, 0 for padding
    lengths: np.ndarray      # (B,)


def collate_patients(Z_list: list[np.ndarray]) -> Batch:
    """Pad a list of (T_i, C) arrays to (B, T_max, C) with zeros, and build
    the length_mask used to zero out loss contributions from padded
    positions (spec §3.4)."""
    B = len(Z_list)
    lengths = np.array([z.shape[0] for z in Z_list])
    T_max = int(lengths.max())
    C = Z_list[0].shape[1]

    Z = np.zeros((B, T_max, C), dtype=np.float64)
    length_mask = np.zeros((B, T_max), dtype=np.float64)
    for i, z in enumerate(Z_list):
        t = z.shape[0]
        Z[i, :t] = z
        length_mask[i, :t] = 1.0

    return Batch(Z=Z, length_mask=length_mask, lengths=lengths)
