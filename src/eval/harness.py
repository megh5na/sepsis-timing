# writes predictions per patient, calls the vendored scorer on them, and
# also computes the utility number independently via src/decision/utility.py

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
    label = np.zeros(T, dtype=int)
    if t_sepsis is not None:
        onset_label_start = max(0, t_sepsis - 6)
        label[onset_label_start:] = 1
    return label


def alarm_hour_to_predictions(alarm_hour: int | None, T: int) -> np.ndarray:
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
