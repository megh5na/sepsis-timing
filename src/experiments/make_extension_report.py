from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path("results/phase2/extension")
OUT = Path("docs/EXTENSION.md")


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def ci(df):
    df = df.copy()
    df["95% CI"] = [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(df.pop("ci_low"), df.pop("ci_high"))]
    df["difference B - A"] = df["difference B - A"].map(lambda v: f"{v:+.3f}")
    return df[["B", "A", "difference B - A", "95% CI", "excludes 0", "per-seed differences"]]


def main():
    s = json.loads((EXT / "summary.json").read_text())
    t8 = pd.read_csv("results/phase2/tables/table8_p2_implied_cutoff.csv")
    t8 = t8[t8["condition"] == "P"]
    lam = pd.DataFrame({"seed": ["0", "1", "2"], "lambda* (validation)": [s["lambda_star_per_seed"][str(i)] for i in range(3)],
                        "tau for P1 (validation)": [s["tau_p1_per_seed"][str(i)] for i in range(3)],
                        "mean F[t,12] on non-septic hours": t8["mean F[t,12] on non-septic hours"].to_numpy()})
    keep = ["rule", "utility mean", "utility sd", "per-seed", "95% CI", "stays alarmed", "sensitivity", "false-alarm share of non-septic", "median lead (h)"]
    A = pd.read_csv(EXT / "hospitalA_test_summary.csv")
    B = pd.read_csv(EXT / "hospitalB_summary_blank_etco2.csv")[keep]
    B2 = pd.read_csv(EXT / "hospitalB_summary_etco2_left_in.csv")[keep]
    text = f"""# Post hoc extension: a one-parameter P2, and Hospital B

This extends the locked, negative Phase 2 result (`docs/REPORT.md`). It does not change H1. Everything below is
inference on saved checkpoints; nothing was retrained. Generated from `results/phase2/extension/` by
`src/experiments/make_extension_report.py`; the analysis is `src/experiments/extension_cost_scale.py`.

## What was done

The parameter-free P2 behaves like a fixed cutoff of about 5% on F[t,12] (REPORT.md section 4.3), while P1's tau is tuned per model.
P2 was given one parameter, lambda, that multiplies the 0.05-per-hour false-alarm cost on the no-onset branch
(`fa_scale` in `src/decision/stopping.py`; 1.0 is the primary P2). Lambda is chosen per seed on the **Hospital A validation split**
from the grid 0.5 * 2^(i/4), i = 0..28, declared before any result was seen, exactly as P1's tau is. Then:

1. Hospital A test (already used for the primary result, so this is a second look, not a confirmation).
2. **Hospital B, all 20,000 stays**, which no model, threshold or lambda had seen. Checkpoints trained on A, A's training statistics, tau and lambda from A
   validation. End-tidal CO2 is never observed at A, so the model learned it as always missing; it is blanked at B too (decided before running).
   A run with it left in is reported as a sensitivity check.

{md(lam, "{:.3f}")}

Lambda grows with the model's baseline risk on non-septic patients (last column), which is the adaptivity the parameter-free P2 lacked.

## Hospital A test (second look)

{md(A[keep[:1] + ["utility mean", "utility sd", "per-seed", "stays alarmed", "sensitivity", "false-alarm share of non-septic", "median lead (h)"]])}

{md(ci(pd.read_csv(EXT / "hospitalA_test_bootstrap.csv")))}

## Hospital B, untouched (primary check; CO2 blanked)

{md(B)}

{md(ci(pd.read_csv(EXT / "hospitalB_bootstrap_blank_etco2.csv")))}

## Hospital B sensitivity: CO2 left in

{md(B2)}

{md(ci(pd.read_csv(EXT / "hospitalB_bootstrap_etco2_left_in.csv")))}

## How to read it

* On the validation split P1 is still slightly ahead of the one-parameter P2 (both tuned there), so (P1's tau grid is also much finer than the 29-value lambda grid) the Hospital A test gain of the
  tuned P2 over P1 is not expected to be a real advantage; it is a second look at a test set already used.
* On Hospital B, a fresh hospital, the one-parameter P2 ties P1 and is clearly better than the parameter-free P2. Neither is better than P1.
* Utilities are lower at B than at A for every learned rule (domain shift), and the plain GRU with a sigmoid output (B1) is about as good as P1.
* The sensitivity run (CO2 left in) shows that handling this channel matters: all learned rules drop, and B1 is then the best.
"""
    OUT.write_text(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
