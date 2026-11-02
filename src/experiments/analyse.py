from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.splits import within_hospital_split
from src.data.stats import TrainStats
from src.decision.threshold import apply_threshold, first_alarm_hour
from src.decision.utility import DT_EARLY, total_utility_for_patient
from src.eval.harness import evaluate
from src.experiments.train import B1Model, fit_train_stats, score_all_patients


def achieved_utility(scores: dict[str, np.ndarray], tau: float, manifest: pd.DataFrame,
                      patient_ids: list[str], workdir: Path) -> float:
    def predict_fn(pid):
        return first_alarm_hour(scores[pid], tau)

    metrics = evaluate(predict_fn, patient_ids, manifest, workdir=workdir)
    return metrics["utility"]


def per_patient_optimal_threshold_utility(
    scores: dict[str, np.ndarray], manifest: pd.DataFrame, patient_ids: list[str],
    candidate_taus: np.ndarray,
) -> tuple[float, float, float]:
    manifest_idx = manifest.set_index("patient_id")
    observed_total = 0.0
    best_total = 0.0
    inaction_total = 0.0

    for pid in patient_ids:
        row = manifest_idx.loc[pid]
        T = int(row["T"])
        is_septic = bool(row["is_septic"])
        t_sepsis = int(row["t_sepsis"]) if is_septic else None
        r = scores[pid]

        best_u_this_patient = -float("inf")
        for tau in candidate_taus:
            pred = apply_threshold(r, tau).astype(int)
            u = total_utility_for_patient(pred, t_sepsis, T)
            if u > best_u_this_patient:
                best_u_this_patient = u
        # tau=1.0 (never alarm) is always a valid candidate outcome too
        never = np.zeros(T, dtype=int)
        best_u_this_patient = max(best_u_this_patient, total_utility_for_patient(never, t_sepsis, T))
        observed_total += best_u_this_patient

        inaction_pred = np.zeros(T, dtype=int)
        inaction_total += total_utility_for_patient(inaction_pred, t_sepsis, T)

        if is_septic:
            best_pred = np.zeros(T, dtype=int)
            lo = max(0, t_sepsis + DT_EARLY)
            from src.decision.utility import DT_LATE
            hi = min(t_sepsis + DT_LATE + 1, T)
            best_pred[lo:hi] = 1
        else:
            best_pred = inaction_pred
        best_total += total_utility_for_patient(best_pred, t_sepsis, T)

    denom = best_total - inaction_total
    normalized = (observed_total - inaction_total) / denom if denom != 0 else 0.0
    return normalized, observed_total, denom


def oracle_alarm_time_utility(manifest: pd.DataFrame, patient_ids: list[str]) -> tuple[float, float, float]:
    from src.decision.utility import DT_LATE

    manifest_idx = manifest.set_index("patient_id")
    observed_total = 0.0
    best_total = 0.0
    inaction_total = 0.0

    for pid in patient_ids:
        row = manifest_idx.loc[pid]
        T = int(row["T"])
        is_septic = bool(row["is_septic"])
        t_sepsis = int(row["t_sepsis"]) if is_septic else None

        inaction_pred = np.zeros(T, dtype=int)
        inaction_total += total_utility_for_patient(inaction_pred, t_sepsis, T)

        if is_septic:
            oracle_pred = np.zeros(T, dtype=int)
            oracle_alarm_hour = max(0, t_sepsis + DT_EARLY)
            oracle_pred[oracle_alarm_hour:] = 1
            best_pred = np.zeros(T, dtype=int)
            hi = min(t_sepsis + DT_LATE + 1, T)
            best_pred[oracle_alarm_hour:hi] = 1
        else:
            oracle_pred = inaction_pred
            best_pred = inaction_pred

        observed_total += total_utility_for_patient(oracle_pred, t_sepsis, T)
        best_total += total_utility_for_patient(best_pred, t_sepsis, T)

    denom = best_total - inaction_total
    normalized = (observed_total - inaction_total) / denom if denom != 0 else 0.0
    return normalized, observed_total, denom


def run_headroom_analysis(
    checkpoint_path: Path, tau: float, manifest: pd.DataFrame, processed_root: Path,
    out_dir: Path, n_threshold_candidates: int = 100,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    split = within_hospital_split(manifest, hospital="A", seed=0)
    train_ids = split.patients("train")
    val_ids = split.patients("val")

    stats = fit_train_stats(train_ids, processed_root)

    model = B1Model()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    scores = score_all_patients(model, val_ids, manifest, processed_root, stats)

    achieved = achieved_utility(scores, tau, manifest, val_ids, out_dir / "achieved")
    candidate_taus = np.linspace(0.01, 0.99, n_threshold_candidates)
    threshold_ceiling, _, _ = per_patient_optimal_threshold_utility(scores, manifest, val_ids, candidate_taus)
    oracle_ceiling, _, _ = oracle_alarm_time_utility(manifest, val_ids)

    result = {
        "achieved": achieved,
        "threshold_ceiling": threshold_ceiling,
        "oracle_ceiling": oracle_ceiling,
        "gap_achieved_to_threshold_ceiling": threshold_ceiling - achieved,
        "gap_threshold_ceiling_to_oracle": oracle_ceiling - threshold_ceiling,
        "n_val_patients": len(val_ids),
        "global_tau": tau,
    }

    with open(out_dir / "headroom_results.json", "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["achieved\n(global tau)", "threshold-ceiling\n(per-patient tau)", "oracle-ceiling\n(knows t_sepsis)"]
    values = [achieved, threshold_ceiling, oracle_ceiling]
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Normalised challenge utility")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage 7: threshold headroom analysis (B1, validation set)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "headroom_bar_chart.png", dpi=150)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True, help="the _meta.json written by train.py, for tau")
    ap.add_argument("--out", type=Path, default=Path("results/headroom"))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.processed / "manifest.parquet")
    with open(args.meta) as f:
        meta = json.load(f)

    run_headroom_analysis(args.checkpoint, meta["tau"], manifest, args.processed, args.out)


if __name__ == "__main__":
    main()
