"""Metrics beyond the core harness output (spec §7.2).

Implemented in Phase 1 (apply to any condition, including B0/B1):
  - discrimination_metrics: AUROC/AUPRC computed directly from a
    continuous per-hour risk score (not the binarized alarm_hour
    decision harness.evaluate() uses) — calls the vendored compute_auc
    directly, so it uses the exact same convention as the official
    scorer rather than sklearn's.
  - alarms_per_patient_day: alarm-fatigue proxy.
  - median_lead_time: is the warning actionable, for patients who did get
    a (correct, before-onset) alarm.

Not implemented in Phase 1 (structurally require the multi-horizon head,
which is Phase 2 — see scope control):
  - calibration / reliability curves + ECE of F[t,k]
  - coherence-violation rate of F[t,k]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from third_party.physionet.evaluate_sepsis_score import compute_auc


def discrimination_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """labels, scores: concatenated across all patients and all hours.
    scores is a continuous per-hour risk value (not binarized)."""
    auroc, auprc = compute_auc(labels, scores)
    return {"auroc": float(auroc), "auprc": float(auprc)}


def alarms_per_patient_day(alarm_hours: list[int | None], stay_lengths: list[int]) -> float:
    """Alarm-fatigue proxy: total alarms issued / total patient-days
    observed. Each patient contributes at most one alarm under the
    'alarm persists once fired' semantics, so this is just
    n_alarmed_patients / total_patient_hours * 24."""
    total_hours = sum(stay_lengths)
    if total_hours == 0:
        return float("nan")
    n_alarms = sum(1 for a in alarm_hours if a is not None)
    total_days = total_hours / 24.0
    return n_alarms / total_days


def median_lead_time(
    alarm_hours: list[int | None], t_sepsis_values: list[int | None]
) -> float:
    """Median (t_sepsis - alarm_hour) in hours, over septic patients whose
    alarm fired at or before their own onset (a lead time is only
    meaningful for alarms that actually preceded onset; an alarm that
    fires after onset has zero or negative 'lead')."""
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
    """Combine harness.evaluate()'s output with the extra Phase-1 metrics
    above into one record. Pass labels_concat/scores_concat (continuous
    risk score, not binarized) to get 'true' discrimination metrics for a
    condition that has one — otherwise falls back to the harness's own
    (binarized) auroc/auprc."""
    report = dict(harness_metrics)
    report["alarms_per_patient_day"] = alarms_per_patient_day(alarm_hours, stay_lengths)
    report["median_lead_time_hours"] = median_lead_time(alarm_hours, t_sepsis_values)

    if labels_concat is not None and scores_concat is not None:
        disc = discrimination_metrics(labels_concat, scores_concat)
        report["auroc_continuous"] = disc["auroc"]
        report["auprc_continuous"] = disc["auprc"]

    return report
