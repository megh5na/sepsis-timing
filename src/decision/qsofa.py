# for B0, the qSOFA baseline, we implement a 2-criterion proxy (Resp >= 22 AND SBP <= 100) since the dataset lacks a mental-status/GCS column. The alarm fires at the first hour both criteria are met. This uses causally forward-filled VALUES in raw units, not z-scored features.

from __future__ import annotations

import numpy as np

from src.data.parse import FEATURE_COLUMNS
from src.data.preprocess import compute_values_mask_delta

RESP_IDX = FEATURE_COLUMNS.index("Resp")
SBP_IDX = FEATURE_COLUMNS.index("SBP")

RESP_THRESHOLD = 22.0   # breaths/min, qSOFA criterion: >=22
SBP_THRESHOLD = 100.0   # mmHg, qSOFA criterion: <=100


def qsofa_positive_mask(X_raw: np.ndarray) -> np.ndarray:
    values, _, _ = compute_values_mask_delta(X_raw)
    resp = values[:, RESP_IDX]
    sbp = values[:, SBP_IDX]
    # A leading stretch with no observation yet is NaN (not yet
    # forward-fillable); NaN comparisons are False, so qSOFA correctly
    # cannot fire before both vitals have been observed at least once.
    return (resp >= RESP_THRESHOLD) & (sbp <= SBP_THRESHOLD)


def qsofa_alarm_hour(X_raw: np.ndarray) -> int | None:
    positive = qsofa_positive_mask(X_raw)
    hits = np.flatnonzero(positive)
    return int(hits[0]) if hits.size else None
