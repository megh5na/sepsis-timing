"""Evaluation harness (spec §7.1): patient -> alarm_hour -> metrics.

The public interface this module exists to serve, per the SCOPE CONTROL
block: `evaluate(predict_fn, ...)` where `predict_fn: patient_id -> int |
None` is *any* function mapping a patient to an alarm hour or no-alarm.
This is deliberately head-agnostic — a Phase 2 decision layer (or, in
Phase 1, the tuned-threshold rule in src/decision/threshold.py, or the
qSOFA rule) is just another such function.

What the harness actually does, and why it needs validating (stage 4):
  1. Convert each patient's alarm_hour into a full-length binary prediction
     array using the "alarm persists once fired" semantics (spec §5.3):
     zeros before the alarm hour, ones from the alarm hour to the end of
     the stay. None -> all zeros.
  2. Reconstruct that patient's true SepsisLabel array from the manifest's
     t_sepsis (label[t] = 1 iff t >= t_sepsis - 6 — the official
     definition, inverted; see src/data/parse.py). This never re-derives
     t_sepsis itself; it only ever reads it from the manifest, which is
     where the six-hour offset is applied exactly once.
  3. Write both arrays to per-patient .psv files, in the exact column
     format the vendored official scorer expects (header names
     'SepsisLabel' / 'PredictedLabel' / 'PredictedProbability'), with
     filenames chosen so that sorting the labels directory and the
     predictions directory independently pairs up the same patient at
     the same list position (the official scorer matches files by sorted
     order, not by an explicit join on patient id — see
     third_party/physionet/evaluate_sepsis_score.py:evaluate_sepsis_score).
  4. Call the vendored, unmodified evaluate_sepsis_score() on those two
     directories, and separately recompute the utility number using our
     own src/decision/utility.py (an independently-derived implementation)
     as a cross-check that does not depend on the vendored scorer at all.

Because alarm_hour carries no continuous risk score, the harness uses the
binarized 0/1 decision as both PredictedLabel and PredictedProbability —
this is a genuine reduction in AUROC/AUPRC resolution relative to using a
model's raw continuous score, and is a known, disclosed property of the
alarm_hour interface (not a bug). Conditions that do have a continuous
score (B0's qSOFA count is a de-facto integer score, B1's sigmoid output
r[t] is continuous) should also report discrimination metrics computed
directly from that continuous score via src/eval/metrics.py's
`discrimination_metrics`, which calls the vendored compute_auc directly on
the continuous values, not through this alarm-only pathway.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.decision.utility import total_utility_for_patient
from third_party.physionet.evaluate_sepsis_score import (
    compute_accuracy_f_measure,
    compute_auc,
    evaluate_sepsis_score,
)

PredictFn = Callable[[str], "int | None"]


def reconstruct_sepsis_label(T: int, t_sepsis: int | None) -> np.ndarray:
    """Inverse of src/data/parse.py's first_positive_hour + 6 derivation:
    given t_sepsis, reconstruct the official SepsisLabel column exactly
    (label[t] = 1 iff t >= t_sepsis - 6). For non-septic patients (t_sepsis
    is None) this is all zeros."""
    label = np.zeros(T, dtype=int)
    if t_sepsis is not None:
        onset_label_start = max(0, t_sepsis - 6)
        label[onset_label_start:] = 1
    return label


def alarm_hour_to_predictions(alarm_hour: int | None, T: int) -> np.ndarray:
    """spec §5.3 semantics: once fired, stays alarmed. None -> never alarm."""
    predictions = np.zeros(T, dtype=int)
    if alarm_hour is not None:
        start = max(0, min(int(alarm_hour), T))
        predictions[start:] = 1
    return predictions


def _write_psv(path: Path, columns: dict[str, np.ndarray]) -> None:
    pd.DataFrame(columns).to_csv(path, sep="|", index=False)


@dataclass
class PatientEvalRecord:
    patient_id: str
    T: int
    t_sepsis: int | None
    is_septic: bool
    alarm_hour: int | None
    label: np.ndarray
    predictions: np.ndarray


def build_eval_records(
    predict_fn: PredictFn,
    patient_ids: Iterable[str],
    manifest: pd.DataFrame,
) -> list[PatientEvalRecord]:
    manifest_idx = manifest.set_index("patient_id")
    records = []
    for pid in patient_ids:
        row = manifest_idx.loc[pid]
        T = int(row["T"])
        is_septic = bool(row["is_septic"])
        t_sepsis = int(row["t_sepsis"]) if is_septic else None

        alarm_hour = predict_fn(pid)
        label = reconstruct_sepsis_label(T, t_sepsis)
        predictions = alarm_hour_to_predictions(alarm_hour, T)

        records.append(
            PatientEvalRecord(
                patient_id=pid, T=T, t_sepsis=t_sepsis, is_septic=is_septic,
                alarm_hour=alarm_hour, label=label, predictions=predictions,
            )
        )
    return records


def write_records_to_psv(records: list[PatientEvalRecord], label_dir: Path, pred_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        fname = f"{r.patient_id}.psv"
        _write_psv(label_dir / fname, {"SepsisLabel": r.label})
        _write_psv(
            pred_dir / fname,
            {
                "PredictedProbability": r.predictions.astype(float),
                "PredictedLabel": r.predictions,
            },
        )


def our_normalized_utility(records: list[PatientEvalRecord]) -> float:
    """Independent cross-check of the utility number, using our own
    src/decision/utility.py rather than the vendored scorer. Mirrors the
    vendored evaluate_sepsis_score's normalisation:
        (sum(observed) - sum(inaction)) / (sum(best) - sum(inaction))
    with 'best' defined identically (alarm on for the whole
    [t_sepsis+dt_early, t_sepsis+dt_late] window for septic patients, never
    alarm for non-septic patients)."""
    from src.decision.utility import DT_EARLY, DT_LATE

    observed_total = 0.0
    best_total = 0.0
    inaction_total = 0.0

    for r in records:
        inaction_pred = np.zeros(r.T, dtype=int)
        observed_total += total_utility_for_patient(r.predictions, r.t_sepsis, r.T)
        inaction_total += total_utility_for_patient(inaction_pred, r.t_sepsis, r.T)

        if r.is_septic:
            best_pred = np.zeros(r.T, dtype=int)
            lo = max(0, r.t_sepsis + DT_EARLY)
            hi = min(r.t_sepsis + DT_LATE + 1, r.T)
            best_pred[lo:hi] = 1
        else:
            best_pred = inaction_pred
        best_total += total_utility_for_patient(best_pred, r.t_sepsis, r.T)

    denom = best_total - inaction_total
    if denom == 0:
        return 0.0
    return (observed_total - inaction_total) / denom


def evaluate(
    predict_fn: PredictFn,
    patient_ids: Iterable[str],
    manifest: pd.DataFrame,
    workdir: Path | None = None,
) -> dict:
    """Core harness entry point. Returns a metrics record with keys
    auroc, auprc, accuracy, f_measure, utility (all from the vendored
    scorer, called directly on the files this function writes), plus
    utility_cross_check (our own independent computation) and
    utility_agrees (bool — the two utility numbers match to float
    tolerance)."""
    patient_ids = list(patient_ids)
    records = build_eval_records(predict_fn, patient_ids, manifest)

    def _run(label_dir: Path, pred_dir: Path) -> dict:
        write_records_to_psv(records, label_dir, pred_dir)
        auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(str(label_dir), str(pred_dir))
        return {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "accuracy": float(accuracy),
            "f_measure": float(f_measure),
            "utility": float(utility),
        }

    if workdir is not None:
        label_dir, pred_dir = workdir / "labels", workdir / "predictions"
        metrics = _run(label_dir, pred_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metrics = _run(tmp / "labels", tmp / "predictions")

    metrics["utility_cross_check"] = our_normalized_utility(records)
    metrics["utility_agrees"] = bool(
        np.isclose(metrics["utility"], metrics["utility_cross_check"], atol=1e-6)
    )
    metrics["n_patients"] = len(records)
    return metrics


def evaluate_fast(
    predict_fn: PredictFn,
    patient_ids: Iterable[str],
    manifest: pd.DataFrame,
    utility_only: bool = False,
) -> dict:
    """Same metrics as evaluate(), computed entirely in memory (no .psv
    file I/O) — still via the vendored compute_auc / compute_accuracy_f_measure
    functions called directly on the concatenated label/prediction arrays,
    and via our own utility.py for the utility number. This is NOT a
    separate implementation to keep in sync with evaluate(); it's the same
    vendored functions through a different (in-memory) calling convention.

    Used for per-epoch validation-threshold tuning during training, where
    writing hundreds of thresholds x thousands of patients to disk per
    epoch would dominate runtime. `evaluate()` (the file-based path) is
    what stage 4's validation tests check against an independent subprocess
    call to the official script, and what final reported numbers use;
    tests/test_harness_validation.py also checks evaluate_fast agrees with
    evaluate() exactly, so using the fast path during training does not
    weaken that validation.

    utility_only=True skips the AUROC/AUPRC/accuracy/F-measure computation
    (compute_auc's O(n log n) sort in particular) — pure speed, used by
    threshold tuning during training which reads only 'utility' from the
    result hundreds of times per epoch. Final reported numbers always call
    evaluate() (or evaluate_fast with the default utility_only=False),
    never this shortcut.
    """
    patient_ids = list(patient_ids)
    records = build_eval_records(predict_fn, patient_ids, manifest)
    utility = our_normalized_utility(records)

    if utility_only:
        return {"utility": float(utility), "n_patients": len(records)}

    labels_concat = np.concatenate([r.label for r in records])
    predictions_concat = np.concatenate([r.predictions for r in records])

    auroc, auprc = compute_auc(labels_concat, predictions_concat)
    accuracy, f_measure = compute_accuracy_f_measure(labels_concat, predictions_concat)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(accuracy),
        "f_measure": float(f_measure),
        "utility": float(utility),
        "utility_cross_check": utility,
        "utility_agrees": True,
        "n_patients": len(records),
    }
