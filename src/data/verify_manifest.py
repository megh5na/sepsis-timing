"""Stage 1 gate check (spec §10.1): does the manifest reproduce the
published patient and prevalence counts?

Reference values and their source (checked at implementation time against
the official PhysioNet challenge page and the Reyna et al. 2019 paper, not
recalled from memory without a citation):

  - Hospital A: 20,336 patients   (physionet.org/content/challenge-2019/1.0.0/)
  - Hospital B: 20,000 patients   (physionet.org/content/challenge-2019/1.0.0/)
  - Total:      40,336 patients
  - Septic (overall): ~2,932 patients (~7.3%)
      Reyna, M.A. et al. (2019), "Early Prediction of Sepsis From Clinical
      Data: The PhysioNet/Computing in Cardiology Challenge 2019."
      Widely reported as ~7.2-7.3% overall prevalence; we check within a
      tolerance band rather than an exact count since slightly different
      sources round differently.

This script only *compares* — it never feeds these numbers into
preprocessing, training, or evaluation logic. If the parsed manifest
disagrees, that's a signal our parser or the six-hour offset logic is
wrong, not a signal to change the reference values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_N_A = 20336
EXPECTED_N_B = 20000
EXPECTED_SEPTIC_FRAC_LOW = 0.05
EXPECTED_SEPTIC_FRAC_HIGH = 0.10


def verify(manifest: pd.DataFrame) -> bool:
    ok = True
    counts = manifest.groupby("hospital").size().to_dict()
    n_a = counts.get("A", 0)
    n_b = counts.get("B", 0)
    n_total = len(manifest)
    n_septic = int(manifest["is_septic"].sum())
    septic_frac = n_septic / n_total if n_total else float("nan")

    def check(label, got, expected, tol=0):
        nonlocal ok
        passed = abs(got - expected) <= tol
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}: got={got}, expected={expected}" + (f" (+/-{tol})" if tol else ""))

    check("Hospital A patient count", n_a, EXPECTED_N_A)
    check("Hospital B patient count", n_b, EXPECTED_N_B)
    check("Total patient count", n_total, EXPECTED_N_A + EXPECTED_N_B)

    frac_ok = EXPECTED_SEPTIC_FRAC_LOW <= septic_frac <= EXPECTED_SEPTIC_FRAC_HIGH
    ok = ok and frac_ok
    status = "PASS" if frac_ok else "FAIL"
    print(
        f"[{status}] Overall septic prevalence: got={septic_frac:.4f} "
        f"({n_septic}/{n_total}), expected in "
        f"[{EXPECTED_SEPTIC_FRAC_LOW}, {EXPECTED_SEPTIC_FRAC_HIGH}] "
        f"(~0.073 per Reyna et al. 2019)"
    )

    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    ok = verify(manifest)
    print("\nSTAGE 1 GATE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
