"""Parse raw .psv files into per-patient tensors and a manifest table.

Implements spec §2.2-2.4, with special attention to §2.3 (the six-hour
label offset — see CRITICAL note below).

For each patient file:
    X            (T, 40) float array, NaN preserved, column order = the
                 41-column header minus SepsisLabel (same order as file)
    t_sepsis     int hour index, or None if non-septic
    is_septic    bool
    T            int, number of hours in the stay

CRITICAL — six-hour offset (spec §2.3):
    SepsisLabel is NOT the onset indicator. Per the official PhysioNet
    definition, and independently confirmed by reading the official scoring
    script (third_party/physionet/evaluate_sepsis_score.py,
    compute_prediction_utility: `t_sepsis = np.argmax(labels) - dt_optimal`
    with dt_optimal = -6), SepsisLabel is 1 for all hours t >= t_sepsis - 6.
    Therefore:

        first_positive_hour = index of first row where SepsisLabel == 1
        t_sepsis             = first_positive_hour + 6

    This file computes t_sepsis exactly this way and nowhere else. Every
    downstream module (targets, decision layer, utility) must consume
    t_sepsis from the manifest/parsed record — never re-derive it from
    SepsisLabel directly — so this offset is applied in exactly one place.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "SepsisLabel"

# Canonical 40 feature columns in the order they appear in the raw files
# (spec §2.2). Asserted against each file's header at parse time.
FEATURE_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
]
assert len(FEATURE_COLUMNS) == 40


@dataclass
class ParsedPatient:
    patient_id: str
    hospital: str
    T: int
    X: np.ndarray  # (T, 40)
    t_sepsis: int | None
    is_septic: bool


def first_positive_hour(sepsis_label: np.ndarray) -> int | None:
    """0-indexed row of the first SepsisLabel == 1, or None if never positive."""
    hits = np.flatnonzero(sepsis_label == 1)
    if hits.size == 0:
        return None
    return int(hits[0])


def parse_patient_file(path: Path, hospital: str) -> ParsedPatient:
    df = pd.read_csv(path, sep="|")
    if list(df.columns[:-1]) != FEATURE_COLUMNS or df.columns[-1] != LABEL_COL:
        raise ValueError(
            f"{path}: unexpected column layout.\n"
            f"  got:      {list(df.columns)}\n"
            f"  expected: {FEATURE_COLUMNS + [LABEL_COL]}"
        )

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    label = df[LABEL_COL].to_numpy()

    fph = first_positive_hour(label)
    is_septic = fph is not None
    t_sepsis = fph + 6 if is_septic else None

    patient_id = path.stem  # e.g. "p000001"
    return ParsedPatient(
        patient_id=patient_id,
        hospital=hospital,
        T=X.shape[0],
        X=X,
        t_sepsis=t_sepsis,
        is_septic=is_septic,
    )


def parse_all(raw_root: Path, out_root: Path) -> pd.DataFrame:
    """Parse every patient file under raw_root/{training_setA,training_setB}/.

    Writes one .npz per patient to out_root/tensors/, and returns the
    manifest DataFrame (also written to out_root/manifest.parquet and
    out_root/manifest.csv).
    """
    tensor_dir = out_root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    hospital_dirs = {"A": raw_root / "training_setA", "B": raw_root / "training_setB"}
    rows = []
    for hospital, hdir in hospital_dirs.items():
        if not hdir.exists():
            raise FileNotFoundError(f"missing raw directory: {hdir}")
        files = sorted(hdir.glob("p*.psv"))
        if not files:
            raise RuntimeError(f"no .psv files found under {hdir}")
        for f in files:
            p = parse_patient_file(f, hospital)
            np.savez_compressed(tensor_dir / f"{p.patient_id}.npz", X=p.X)
            rows.append(
                {
                    "patient_id": p.patient_id,
                    "hospital": p.hospital,
                    "T": p.T,
                    "t_sepsis": p.t_sepsis if p.t_sepsis is not None else -1,
                    "is_septic": p.is_septic,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["hospital", "patient_id"]).reset_index(drop=True)
    # t_sepsis stored as -1 (not NaN) for non-septic so the column stays integer;
    # consumers must check is_septic before reading t_sepsis.
    manifest["t_sepsis"] = manifest["t_sepsis"].astype(int)
    manifest.to_parquet(out_root / "manifest.parquet", index=False)
    manifest.to_csv(out_root / "manifest.csv", index=False)
    return manifest


def load_patient_tensor(out_root: Path, patient_id: str) -> np.ndarray:
    with np.load(out_root / "tensors" / f"{patient_id}.npz") as d:
        return d["X"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    manifest = parse_all(args.raw, args.out)
    n = len(manifest)
    n_septic = int(manifest["is_septic"].sum())
    print(f"Parsed {n} patients ({manifest.groupby('hospital').size().to_dict()})")
    print(f"Septic: {n_septic} ({100 * n_septic / n:.2f}%)")


if __name__ == "__main__":
    main()
