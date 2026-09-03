"""Full Phase-1 condition x seed grid (spec repo layout: experiments/run_grid.py).

Phase 1 grid: {B0, B1} x seeds x within-hospital split (the only regime run
in Phase 1 — spec §11.1). B0 has no seed-dependence (rule-based) but is
still evaluated once per nominal "seed" so results/analyse.py can treat
every condition uniformly in the results table.

Cross-hospital / pooled splits, the K / hidden-size / mask-delta ablations
(spec §6.1), and A1/P1/P2 are all Phase 2 (spec §11.1, §10.2 stage 11) —
not run here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.splits import within_hospital_split
from src.data.stats import TrainStats
from src.decision.qsofa import qsofa_alarm_hour
from src.decision.threshold import first_alarm_hour
from src.data.parse import load_patient_tensor
from src.eval.bootstrap import bootstrap_ci
from src.eval.harness import evaluate
from src.eval.metrics import full_metrics_report
from src.experiments.train import B1Model, fit_train_stats, score_all_patients, train_b1

SEEDS = [0, 1, 2]


def run_b0(manifest: pd.DataFrame, processed_root: Path, split_ids: list[str], out_dir: Path) -> dict:
    def predict_fn(pid):
        X_raw = load_patient_tensor(processed_root, pid)
        return qsofa_alarm_hour(X_raw)

    workdir = out_dir / "B0" / "eval"
    metrics = evaluate(predict_fn, split_ids, manifest, workdir=workdir)

    manifest_idx = manifest.set_index("patient_id")
    alarm_hours = [predict_fn(pid) for pid in split_ids]
    t_sepsis_values = [
        int(manifest_idx.loc[pid, "t_sepsis"]) if manifest_idx.loc[pid, "is_septic"] else None
        for pid in split_ids
    ]
    stay_lengths = [int(manifest_idx.loc[pid, "T"]) for pid in split_ids]

    return full_metrics_report(metrics, alarm_hours, t_sepsis_values, stay_lengths)


def run_b1_seed(manifest: pd.DataFrame, processed_root: Path, seed: int, out_dir: Path,
                 stats: TrainStats, test_ids: list[str]) -> tuple[dict, dict[str, np.ndarray]]:
    """`stats` (fit once on the training split, spec §3.6 guard #2) and
    `test_ids` are passed in rather than recomputed here — fitting stats
    means a full pass over every training patient's raw tensor, and doing
    that three times per seed (train_b1 internally, here again, and once
    more for the bootstrap CI in main()) would triple the most expensive
    repeated step in this script for no reason. Returns (report, scores)
    so main() can reuse `scores` for the bootstrap CI instead of
    re-scoring every test patient a second time."""
    train_result = train_b1(manifest, processed_root, seed=seed, out_dir=out_dir / "B1" / "checkpoints")

    model = B1Model()
    model.load_state_dict(torch.load(out_dir / "B1" / "checkpoints" / f"b1_seed{seed}.pt"))
    scores = score_all_patients(model, test_ids, manifest, processed_root, stats)
    tau = train_result["tau"]

    def predict_fn(pid):
        return first_alarm_hour(scores[pid], tau)

    workdir = out_dir / "B1" / f"eval_seed{seed}"
    metrics = evaluate(predict_fn, test_ids, manifest, workdir=workdir)

    manifest_idx = manifest.set_index("patient_id")
    alarm_hours = [predict_fn(pid) for pid in test_ids]
    t_sepsis_values = [
        int(manifest_idx.loc[pid, "t_sepsis"]) if manifest_idx.loc[pid, "is_septic"] else None
        for pid in test_ids
    ]
    stay_lengths = [int(manifest_idx.loc[pid, "T"]) for pid in test_ids]
    labels_concat = np.concatenate([
        (np.arange(int(manifest_idx.loc[pid, "T"])) >= max(0, (t_sepsis_values[i] or 10**9) - 6)).astype(int)
        if manifest_idx.loc[pid, "is_septic"] else np.zeros(int(manifest_idx.loc[pid, "T"]), dtype=int)
        for i, pid in enumerate(test_ids)
    ])
    scores_concat = np.concatenate([scores[pid] for pid in test_ids])

    report = full_metrics_report(metrics, alarm_hours, t_sepsis_values, stay_lengths,
                                  labels_concat=labels_concat, scores_concat=scores_concat)
    report["seed"] = seed
    report["tau"] = tau
    report["best_epoch"] = train_result["best_epoch"]
    return report, scores


def per_patient_utility_components_for_ci(manifest, processed_root, predict_fn, patient_ids) -> np.ndarray:
    """(n_patients, 3) array of [observed, inaction, best] per patient, for
    bootstrap_ci with `normalized_utility_statistic` below.

    The reported metric is normalised utility = (sum(observed) -
    sum(inaction)) / (sum(best) - sum(inaction)) — a ratio of sums across
    the cohort, not a mean of independently-normalised per-patient values.
    Bootstrapping the mean of raw per-patient utility would give a CI on a
    *different* statistic. Instead we keep the three per-patient
    ingredients and let bootstrap_ci's `statistic` callable recompute the
    same ratio-of-sums on each resample, which is the correct paired
    bootstrap for this metric (spec §7.3)."""
    from src.decision.utility import DT_EARLY, DT_LATE, total_utility_for_patient

    manifest_idx = manifest.set_index("patient_id")
    rows = []
    for pid in patient_ids:
        row = manifest_idx.loc[pid]
        T = int(row["T"])
        is_septic = bool(row["is_septic"])
        t_sepsis = int(row["t_sepsis"]) if is_septic else None

        alarm = predict_fn(pid)
        pred = np.zeros(T, dtype=int)
        if alarm is not None:
            pred[max(0, min(alarm, T)):] = 1
        observed = total_utility_for_patient(pred, t_sepsis, T)

        inaction_pred = np.zeros(T, dtype=int)
        inaction = total_utility_for_patient(inaction_pred, t_sepsis, T)

        if is_septic:
            best_pred = np.zeros(T, dtype=int)
            lo = max(0, t_sepsis + DT_EARLY)
            hi = min(t_sepsis + DT_LATE + 1, T)
            best_pred[lo:hi] = 1
            best = total_utility_for_patient(best_pred, t_sepsis, T)
        else:
            best = inaction

        rows.append([observed, inaction, best])
    return np.array(rows)


def normalized_utility_statistic(components: np.ndarray) -> float:
    """components: (n, 3) array of [observed, inaction, best]. Ratio of
    sums, matching the official scorer's normalisation exactly."""
    observed_sum = components[:, 0].sum()
    inaction_sum = components[:, 1].sum()
    best_sum = components[:, 2].sum()
    denom = best_sum - inaction_sum
    if denom == 0:
        return 0.0
    return float((observed_sum - inaction_sum) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results/grid"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.processed / "manifest.parquet")
    split = within_hospital_split(manifest, hospital="A", seed=0)
    test_ids = split.patients("test")
    stats = fit_train_stats(split.patients("train"), args.processed)  # fit once, reused across seeds below

    all_results = []

    print("=== B0 (qSOFA proxy) ===")
    b0_result = run_b0(manifest, args.processed, test_ids, args.out)
    b0_result["condition"] = "B0"
    b0_result["seed"] = "n/a"
    components = per_patient_utility_components_for_ci(
        manifest, args.processed,
        lambda pid: qsofa_alarm_hour(load_patient_tensor(args.processed, pid)),
        test_ids,
    )
    ci = bootstrap_ci(components, statistic=normalized_utility_statistic)
    b0_result["utility_ci_low"] = ci.ci_low
    b0_result["utility_ci_high"] = ci.ci_high
    all_results.append(b0_result)

    for seed in SEEDS:
        print(f"=== B1 seed {seed} ===")
        b1_result, scores = run_b1_seed(manifest, args.processed, seed, args.out, stats, test_ids)
        b1_result["condition"] = "B1"

        tau = b1_result["tau"]
        components = per_patient_utility_components_for_ci(
            manifest, args.processed, lambda pid: first_alarm_hour(scores[pid], tau), test_ids,
        )
        ci = bootstrap_ci(components, statistic=normalized_utility_statistic)
        b1_result["utility_ci_low"] = ci.ci_low
        b1_result["utility_ci_high"] = ci.ci_high

        all_results.append(b1_result)

    df = pd.DataFrame(all_results)
    df.to_csv(args.out / "results_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
