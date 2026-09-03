"""B0: rule-based qSOFA baseline (spec §6, condition B0 — "non-ML floor").

DEVIATION FROM TRUE qSOFA, DISCLOSED: full qSOFA is 3 criteria (respiratory
rate >= 22, systolic BP <= 100 mmHg, altered mentation / GCS < 15), positive
screen at score >= 2. This dataset has no mental-status/GCS column (spec
§2.2's 41 columns do not include one), so the third criterion cannot be
computed. This module implements a 2-criterion proxy: alarm fires at the
first hour where BOTH available criteria are met simultaneously (Resp >= 22
AND SBP <= 100), i.e. the only way to reach a qSOFA-equivalent score of 2
using what this dataset actually contains. This is a real, disclosed
limitation of B0 as a "non-ML floor", not a hidden approximation.

Uses the causally forward-filled VALUES block (src/data/preprocess.py) in
raw clinical units — not the z-scored features B1 consumes — since qSOFA is
a fixed-unit clinical threshold rule, not a learned model.
"""
from __future__ import annotations

import numpy as np

from src.data.parse import FEATURE_COLUMNS
from src.data.preprocess import compute_values_mask_delta

RESP_IDX = FEATURE_COLUMNS.index("Resp")
SBP_IDX = FEATURE_COLUMNS.index("SBP")

RESP_THRESHOLD = 22.0   # breaths/min, qSOFA criterion: >=22
SBP_THRESHOLD = 100.0   # mmHg, qSOFA criterion: <=100


def qsofa_positive_mask(X_raw: np.ndarray) -> np.ndarray:
    """X_raw: (T, 40) raw patient array. Returns (T,) boolean: True at
    hours where the 2-criterion qSOFA proxy is positive."""
    values, _, _ = compute_values_mask_delta(X_raw)
    resp = values[:, RESP_IDX]
    sbp = values[:, SBP_IDX]
    # A leading stretch with no observation yet is NaN (not yet
    # forward-fillable); NaN comparisons are False, so qSOFA correctly
    # cannot fire before both vitals have been observed at least once.
    return (resp >= RESP_THRESHOLD) & (sbp <= SBP_THRESHOLD)


def qsofa_alarm_hour(X_raw: np.ndarray) -> int | None:
    """First hour the 2-criterion qSOFA proxy is positive, or None if it
    never is. This is the B0 predict_fn: patient -> alarm_hour | None."""
    positive = qsofa_positive_mask(X_raw)
    hits = np.flatnonzero(positive)
    return int(hits[0]) if hits.size else None
