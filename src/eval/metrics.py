# besides score, adds extra continuous metrics (auroc/auprc) and alarm-fatigue metrics (alarms per patient-day, median lead time)

from __future__ import annotations

import numpy as np
import pandas as pd

from third_party.physionet.evaluate_sepsis_score import compute_auc


def discrimination_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    auroc, auprc = compute_auc(labels, scores)
    return {"auroc": float(auroc), "auprc": float(auprc)}


def alarms_per_patient_day(alarm_hours: list[int | None], stay_lengths: list[int]) -> float:
    total_hours = sum(stay_lengths)
    if total_hours == 0:
        return float("nan")
    n_alarms = sum(1 for a in alarm_hours if a is not None)
    total_days = total_hours / 24.0
    return n_alarms / total_days


def median_lead_time(
    alarm_hours: list[int | None], t_sepsis_values: list[int | None]
) -> float:
    leads = []
    for alarm, t_sepsis in zip(alarm_hours, t_sepsis_values):
        if t_sepsis is None or alarm is None:
            continue
        lead = t_sepsis - alarm
        if lead >= 0:
            leads.append(lead)
    if not leads:
        return float("nan")
    return float(np.median(leads))


def full_metrics_report(
    harness_metrics: dict,
    alarm_hours: list[int | None],
    t_sepsis_values: list[int | None],
    stay_lengths: list[int],
    labels_concat: np.ndarray | None = None,
    scores_concat: np.ndarray | None = None,
) -> dict:
    report = dict(harness_metrics)
    report["alarms_per_patient_day"] = alarms_per_patient_day(alarm_hours, stay_lengths)
    report["median_lead_time_hours"] = median_lead_time(alarm_hours, t_sepsis_values)

    if labels_concat is not None and scores_concat is not None:
        disc = discrimination_metrics(labels_concat, scores_concat)
        report["auroc_continuous"] = disc["auroc"]
        report["auprc_continuous"] = disc["auprc"]

    return report
