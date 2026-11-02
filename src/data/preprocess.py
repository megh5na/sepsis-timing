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
    if X_raw.shape[1] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels, got {X_raw.shape[1]}")

    mask = _compute_mask(X_raw)
    values = _forward_fill(X_raw)
    delta = _compute_delta(mask)

    mask[:, DEMOGRAPHIC_INDICES] = 1.0
    delta[:, DEMOGRAPHIC_INDICES] = 0.0

    return values, mask, delta


def build_Z(X_raw: np.ndarray, stats: TrainStats, use_mask_delta: bool = True) -> np.ndarray:
    values, mask, delta = compute_values_mask_delta(X_raw)
    values_z = stats.impute_and_zscore(values)
    if not use_mask_delta:
        return values_z
    return np.concatenate([values_z, mask, delta], axis=-1)


# ---------------------------------------------------------------------------
# §3.2 multi-horizon targets
# ---------------------------------------------------------------------------

def build_multi_horizon_targets(T: int, t_sepsis: int | None, K: int) -> tuple[np.ndarray, np.ndarray]:
    is_septic = t_sepsis is not None
    t = np.arange(T)[:, None]          # (T, 1)
    k = np.arange(1, K + 1)[None, :]   # (1, K)

    if is_septic:
        y = (t_sepsis <= t + k).astype(np.float64)
        mask = np.ones((T, K), dtype=np.float64)
    else:
        y = np.zeros((T, K), dtype=np.float64)
        # Hours are 0-indexed here (rows 0..T-1), so the first unobserved hour is
        # index T. The spec's `t + k > T` is written for 1-indexed hours, which is
        # `t + k >= T` in 0-indexed terms.
        mask = (~(t + k >= T)).astype(np.float64)

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
