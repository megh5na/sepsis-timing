"""Stage 2 EDA (spec §10.1 stage 2): missingness, onset-time distribution,
does measurement frequency rise before onset, hospital A vs B differences.

Run as a script; writes figures and a small text summary to results/eda/.
notebooks/01_eda.ipynb is a thin wrapper that calls `run_all` and displays
the same figures inline — the analysis itself lives here so it's testable
and re-runnable outside a notebook kernel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.parse import FEATURE_COLUMNS, load_patient_tensor


def missingness_by_channel(manifest: pd.DataFrame, processed_root: Path,
                            hospital: str | None = None) -> pd.DataFrame:
    """Fraction of (patient, hour) cells that are NaN, per channel.
    hospital=None pools A+B; hospital='A'/'B' restricts to one site."""
    sub = manifest if hospital is None else manifest[manifest["hospital"] == hospital]
    n_missing = np.zeros(len(FEATURE_COLUMNS))
    n_total = np.zeros(len(FEATURE_COLUMNS))
    for pid in sub["patient_id"]:
        X = load_patient_tensor(processed_root, pid)
        n_missing += np.isnan(X).sum(axis=0)
        n_total += X.shape[0]
    frac = n_missing / n_total
    return pd.DataFrame({"channel": FEATURE_COLUMNS, "frac_missing": frac}).sort_values(
        "frac_missing", ascending=False
    )


def missingness_by_channel_and_hospital(manifest: pd.DataFrame, processed_root: Path) -> pd.DataFrame:
    """Wide table: one row per channel, one column per hospital's
    frac_missing, plus the absolute difference — surfaces site-specific
    data-collection artifacts (e.g. a channel entirely absent at one
    hospital) that a pooled-only missingness table would hide. This is
    exactly the kind of hospital A/B difference spec stage 2 asks for, and
    it directly affects stage 3: src/data/stats.py.TrainStats.fit() must
    (and does) handle a channel with zero training-split observations
    rather than crashing on it."""
    a = missingness_by_channel(manifest, processed_root, hospital="A").set_index("channel")
    b = missingness_by_channel(manifest, processed_root, hospital="B").set_index("channel")
    out = pd.DataFrame({"frac_missing_A": a["frac_missing"], "frac_missing_B": b["frac_missing"]})
    out["abs_diff"] = (out["frac_missing_A"] - out["frac_missing_B"]).abs()
    out["never_observed_at_A"] = out["frac_missing_A"] >= 0.9999
    out["never_observed_at_B"] = out["frac_missing_B"] >= 0.9999
    return out.sort_values("abs_diff", ascending=False)


def onset_time_distribution(manifest: pd.DataFrame) -> np.ndarray:
    septic = manifest[manifest["is_septic"]]
    return septic["t_sepsis"].to_numpy()


def measurement_frequency_before_onset(
    manifest: pd.DataFrame, processed_root: Path, window: int = 24
) -> pd.DataFrame:
    """For septic patients, does the fraction of genuinely-observed (non-NaN)
    cells rise in the `window` hours immediately before t_sepsis, relative
    to earlier in the stay? This is the mechanism check the spec asks for:
    if measurement frequency does NOT rise before onset, the mask/delta
    ablation (Phase 2) has no signal to exploit.

    Returns a DataFrame indexed by hours-relative-to-onset (-window..-1)
    with the mean observed-fraction across all channels and septic
    patients at that relative hour, plus a 'baseline' fraction computed
    from all hours more than `window` before onset, for comparison.
    """
    septic = manifest[manifest["is_septic"]]
    rows_by_offset: dict[int, list[float]] = {o: [] for o in range(-window, 0)}
    baseline_vals: list[float] = []

    for _, row in septic.iterrows():
        X = load_patient_tensor(processed_root, row["patient_id"])
        observed = (~np.isnan(X)).mean(axis=1)  # (T,) fraction of channels observed each hour
        t_sepsis = int(row["t_sepsis"])
        T = X.shape[0]

        for offset in range(-window, 0):
            t = t_sepsis + offset
            if 0 <= t < T:
                rows_by_offset[offset].append(observed[t])

        baseline_end = t_sepsis - window
        if baseline_end > 0:
            baseline_vals.extend(observed[:baseline_end].tolist())

    out = pd.DataFrame(
        {
            "hours_before_onset": list(range(-window, 0)),
            "mean_frac_observed": [
                float(np.mean(rows_by_offset[o])) if rows_by_offset[o] else np.nan
                for o in range(-window, 0)
            ],
            "n_patients": [len(rows_by_offset[o]) for o in range(-window, 0)],
        }
    )
    out.attrs["baseline_mean_frac_observed"] = float(np.mean(baseline_vals)) if baseline_vals else float("nan")
    return out


def hospital_comparison(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, sub in manifest.groupby("hospital"):
        rows.append(
            {
                "hospital": h,
                "n_patients": len(sub),
                "septic_frac": sub["is_septic"].mean(),
                "median_T": sub["T"].median(),
                "mean_T": sub["T"].mean(),
            }
        )
    return pd.DataFrame(rows)


def run_all(processed_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(processed_root / "manifest.parquet")

    # 1. Missingness by channel
    miss = missingness_by_channel(manifest, processed_root)
    miss.to_csv(out_dir / "missingness_by_channel.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(miss["channel"], miss["frac_missing"])
    ax.set_xlabel("Fraction of (patient, hour) cells missing")
    ax.set_title("Missingness by channel")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "missingness_by_channel.png", dpi=150)
    plt.close(fig)

    # 1b. Missingness by channel, split by hospital — surfaces site-specific
    # collection artifacts a pooled table would hide (e.g. a channel with
    # zero observations at one hospital; see src/data/stats.py docstring).
    miss_by_hospital = missingness_by_channel_and_hospital(manifest, processed_root)
    miss_by_hospital.to_csv(out_dir / "missingness_by_channel_and_hospital.csv")
    never_observed = miss_by_hospital[
        miss_by_hospital["never_observed_at_A"] | miss_by_hospital["never_observed_at_B"]
    ]
    if len(never_observed):
        print("Channels never observed at at least one hospital:")
        print(never_observed[["frac_missing_A", "frac_missing_B"]].to_string())

    # 2. Onset-time distribution
    onset = onset_time_distribution(manifest)
    np.save(out_dir / "onset_time_distribution.npy", onset)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(onset, bins=50)
    ax.set_xlabel("t_sepsis (hour of ICU stay)")
    ax.set_ylabel("Number of septic patients")
    ax.set_title("Onset-time distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "onset_time_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Measurement frequency before onset (mask/delta mechanism check)
    freq = measurement_frequency_before_onset(manifest, processed_root)
    freq.to_csv(out_dir / "measurement_frequency_before_onset.csv", index=False)
    baseline = freq.attrs["baseline_mean_frac_observed"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq["hours_before_onset"], freq["mean_frac_observed"], marker="o")
    ax.axhline(baseline, color="gray", linestyle="--", label=f"baseline ({baseline:.3f})")
    ax.set_xlabel("Hours before onset")
    ax.set_ylabel("Mean fraction of channels genuinely observed")
    ax.set_title("Does measurement frequency rise before onset?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "measurement_frequency_before_onset.png", dpi=150)
    plt.close(fig)

    rising = freq["mean_frac_observed"].iloc[-6:].mean() > baseline
    with open(out_dir / "mechanism_check.txt", "w") as f:
        f.write(
            f"Mean fraction observed, last 6h before onset: "
            f"{freq['mean_frac_observed'].iloc[-6:].mean():.4f}\n"
            f"Baseline (>{freq.shape[0]}h before onset): {baseline:.4f}\n"
            f"Measurement frequency rises before onset: {rising}\n"
        )
    print(f"Measurement frequency rises before onset: {rising}")

    # 4. Hospital A vs B
    hosp = hospital_comparison(manifest)
    hosp.to_csv(out_dir / "hospital_comparison.csv", index=False)
    print(hosp.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results/eda"))
    args = ap.parse_args()
    run_all(args.processed, args.out)


if __name__ == "__main__":
    main()
