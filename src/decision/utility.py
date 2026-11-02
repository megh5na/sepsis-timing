# in short, this module is a direct translation of the official scorer's per-hour utility function into a standalone function U() that can be used in our codebase without relying on the vendored script.
# d = hour - t_sepsis
from __future__ import annotations

import numpy as np

from third_party.physionet.evaluate_sepsis_score import compute_prediction_utility

# --- constants, copied verbatim from third_party/physionet/evaluate_sepsis_score.py ---
DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3
MAX_U_TP = 1
MIN_U_FN = -2
U_FP = -0.05
U_TN = 0

# Slopes/intercepts for the piecewise-linear pieces, same derivation as the
# vendored compute_prediction_utility.
_M1 = float(MAX_U_TP) / float(DT_OPTIMAL - DT_EARLY)
_B1 = -_M1 * DT_EARLY
_M2 = float(-MAX_U_TP) / float(DT_LATE - DT_OPTIMAL)
_B2 = -_M2 * DT_LATE
_M3 = float(MIN_U_FN) / float(DT_LATE - DT_OPTIMAL)
_B3 = -_M3 * DT_OPTIMAL


def U(prediction: int, hour: int, t_sepsis: float | None) -> float:
    is_septic = t_sepsis is not None
    t_sepsis_val = float(t_sepsis) if is_septic else float("inf")
    t = hour

    if t > t_sepsis_val + DT_LATE:
        return 0.0

    if is_septic and prediction:
        if t <= t_sepsis_val + DT_OPTIMAL:
            return max(_M1 * (t - t_sepsis_val) + _B1, U_FP)
        else:  # t <= t_sepsis_val + DT_LATE, checked above
            return _M2 * (t - t_sepsis_val) + _B2
    elif not is_septic and prediction:
        return float(U_FP)
    elif is_septic and not prediction:
        if t <= t_sepsis_val + DT_OPTIMAL:
            return 0.0
        else:
            return _M3 * (t - t_sepsis_val) + _B3
    else:  # not is_septic and not prediction
        return float(U_TN)


def total_utility_for_patient(predictions: np.ndarray, t_sepsis: float | None, T: int | None = None) -> float:
    predictions = np.asarray(predictions)
    T = T if T is not None else len(predictions)
    return float(sum(U(int(predictions[t]), t, t_sepsis) for t in range(T)))


def reference_total_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    return compute_prediction_utility(
        list(labels), list(predictions), DT_EARLY, DT_OPTIMAL, DT_LATE, MAX_U_TP, MIN_U_FN, U_FP, U_TN
    )
