"""The single-hour utility function U(prediction, hour, t_sepsis) (spec
§5.1), extracted from the official scorer — not from memory, not from a
paper's description of it.

Source of truth: third_party/physionet/evaluate_sepsis_score.py,
`compute_prediction_utility` (vendored unmodified; see
third_party/physionet/PROVENANCE.md). That function computes a *per-patient
total* utility by looping over hours and, at each hour, applying a
piecewise-linear function of (prediction, t - t_sepsis) that depends on
whether the patient is septic. This module inlines exactly that per-hour
piecewise logic as a standalone function U(), so it can be called directly
by the decision layer (Phase 2) and by threshold tuning / evaluation
(Phase 1) without re-deriving it.

Constants (also copied verbatim from the vendored script, not recalled):
    dt_early = -12, dt_optimal = -6, dt_late = 3
    max_u_tp = 1, min_u_fn = -2, u_fp = -0.05, u_tn = 0

Verification: `tests/test_utility.py` checks, for many random synthetic
patients, that summing U() over every hour of a patient's stay reproduces
`compute_prediction_utility`'s output on the *same* vendored function,
exactly. The PhysioNet challenge repositories do not ship a separate
"example data" bundle for this script (checked directly against
physionetchallenges/evaluation-2019 and the *-example-2019 repos — see
third_party/physionet/PROVENANCE.md) — the vendored script itself is the
reference, and this equivalence check against it, run over many random
cases rather than one fixed example, is the validation the spec calls for.
"""
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
    """Utility of a single binary prediction (0/1) at a given hour of a
    patient's stay, given that patient's t_sepsis (None if non-septic).

    Mirrors, hour-by-hour, the body of the vendored
    `compute_prediction_utility`'s loop. Summing U() over t=0..T-1 for a
    patient reproduces that function's per-patient total utility exactly
    (tests/test_utility.py checks this on random cases).
    """
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
    """Σ_t U(predictions[t], t, t_sepsis) over the full stay. Equivalent to
    calling the vendored compute_prediction_utility with the corresponding
    labels array, but takes t_sepsis directly rather than a labels vector
    (so it can be called with a hypothetical/counterfactual t_sepsis, which
    the decision layer needs)."""
    predictions = np.asarray(predictions)
    T = T if T is not None else len(predictions)
    return float(sum(U(int(predictions[t]), t, t_sepsis) for t in range(T)))


def reference_total_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Pass-through to the vendored official function, for direct
    side-by-side comparison in tests."""
    return compute_prediction_utility(
        list(labels), list(predictions), DT_EARLY, DT_OPTIMAL, DT_LATE, MAX_U_TP, MIN_U_FN, U_FP, U_TN
    )
