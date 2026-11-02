from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision.stopping import event_alarm_hour, p2_alarm_hour
from src.eval import phase2 as m
from src.eval.calibrate import apply_platt, fit_platt
from src.eval.fast_utility import UtilityTable, alarms_from_hours
from src.experiments.run_phase2 import RUNS, get_bundle, load_cfg

OUT = Path("results/phase2/exploratory")
EVAL = Path("results/phase2/eval")
READINGS = {"persistent": p2_alarm_hour, "event-level": event_alarm_hour}
CALS = ("none", "shared Platt", "per-horizon Platt")


def alarms_for(F_all, split, reading_fn):
    hours = [reading_fn(F_all[split.offsets[i]:split.offsets[i + 1]].astype(np.float64)) for i in range(len(split))]
    return alarms_from_hours(hours, split.lengths)


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    bundle, _ = get_bundle(cfg)
    val, test = bundle.val, bundle.test
    tab_v, tab_t = UtilityTable.from_split(val), UtilityTable.from_split(test)
    K = cfg["conditions"]["P"]["K"]
    yv, mkv = m.targets_flat(val.lengths, val.t_sepsis, K)
    selv = m.selected_hours_mask(val.lengths, val.t_sepsis)
    yt, mkt = m.targets_flat(test.lengths, test.t_sepsis, K)
    selt = m.selected_hours_mask(test.lengths, test.t_sepsis)

    rows, test_alarms, cal_rows, params_out = [], {}, [], {}
    for s in cfg["seeds"]:
        Fv = np.load(RUNS / "P" / f"seed{s}" / "preds_val.npz")["F"].astype(np.float64)
        Ft = np.load(RUNS / "P" / f"seed{s}" / "preds_test.npz")["F"].astype(np.float64)
        ev = np.load(EVAL / f"P_seed{s}.npz")
        wv = mkv * selv[:, None]
        fits = {"none": None,
                "shared Platt": fit_platt(Fv, yv, wv, per_horizon=False),
                "per-horizon Platt": fit_platt(Fv, yv, wv, per_horizon=True)}
        params_out[s] = {k: (None if v is None else {"a": v["a"].tolist(), "b": v["b"].tolist()}) for k, v in fits.items()}
        # P1 reference (tau from validation, as in the locked analysis)
        rows.append({"seed": s, "variant": "P1 (locked)", "val utility": tab_v.normalised_for(ev["p1_val"]),
                     "test utility": tab_t.normalised_for(ev["p1_test"])})
        test_alarms[("P1 (locked)", s)] = ev["p1_test"]
        for cal in CALS:
            Fv_c = Fv if fits[cal] is None else apply_platt(Fv, fits[cal])
            Ft_c = Ft if fits[cal] is None else apply_platt(Ft, fits[cal])
            rel_t = m.reliability(Ft_c, yt, mkt * selt[:, None])
            cal_rows.append({"seed": s, "calibration": cal, "ECE (training hours)": rel_t["ece"].mean(),
                             "tail ECE F>=0.1": np.nanmean(rel_t["ece_tail"]),
                             "mean F[t,12] on non-septic hours": Ft_c[np.repeat(~test.is_septic, test.lengths), K - 1].mean()})
            for reading, fn in READINGS.items():
                name = f"{reading} + {cal}"
                av, at = alarms_for(Fv_c, val, fn), alarms_for(Ft_c, test, fn)
                op = m.operating_point(at, test.lengths, test.t_sepsis)
                rows.append({"seed": s, "variant": name, "val utility": tab_v.normalised_for(av),
                             "test utility": tab_t.normalised_for(at), "alarmed share (test)": op["frac_alarmed"],
                             "non-septic false-alarm share (test)": op["false_alarm_frac_nonseptic"],
                             "sensitivity (test)": op["sensitivity"], "median lead (h, test)": op["median_lead_time_h"]})
                test_alarms[(name, s)] = at
        print(f"seed {s} done")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "variants_per_seed.csv", index=False)
    S = (R.groupby("variant").agg(**{"val utility (mean)": ("val utility", "mean"), "val utility (sd)": ("val utility", "std"),
                                      "test utility (mean)": ("test utility", "mean"), "test utility (sd)": ("test utility", "std"),
                                      "alarmed share (test)": ("alarmed share (test)", "mean"),
                                      "non-septic false-alarm share (test)": ("non-septic false-alarm share (test)", "mean"),
                                      "sensitivity (test)": ("sensitivity (test)", "mean"),
                                      "median lead (h, test)": ("median lead (h, test)", "mean")})
         .reset_index())
    S.to_csv(OUT / "variants_summary.csv", index=False)
    variants = [v for v in S["variant"] if v != "P1 (locked)"]
    chosen = max(variants, key=lambda v: float(S.loc[S["variant"] == v, "val utility (mean)"].iloc[0]))

    def comp(name):
        return np.stack([tab_t.components(test_alarms[(name, s)]) for s in cfg["seeds"]])

    primary_p2 = np.stack([tab_t.components(np.load(EVAL / f"P_seed{s}.npz")["p2_test"]) for s in cfg["seeds"]])
    comps = []
    for name in variants:
        for ref_name, ref in (("P1 (locked)", comp("P1 (locked)")), ("primary P2 (locked)", primary_p2)):
            r = m.paired_bootstrap(ref, comp(name), n_boot=2000)
            comps.append({"variant (B)": name, "versus (A)": ref_name, "difference B - A": r["diff"],
                          "95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", "excludes 0": "yes" if r["ci_excludes_zero"] else "no",
                          "per-seed differences": ", ".join(f"{x:+.3f}" for x in r["per_seed_diff"]),
                          "chosen on validation": "yes" if name == chosen else ""})
    C = pd.DataFrame(comps)
    C.to_csv(OUT / "variants_paired_bootstrap.csv", index=False)
    CAL = pd.DataFrame(cal_rows).groupby("calibration").mean(numeric_only=True).drop(columns="seed").reset_index()
    CAL.to_csv(OUT / "calibration_effect.csv", index=False)
    (OUT / "calibration_params.json").write_text(json.dumps(params_out, indent=2))
    (OUT / "chosen_variant.json").write_text(json.dumps({"chosen_on_validation": chosen,
                                                          "rule": "highest seed-averaged validation utility among the six variants"}, indent=2))
    for name, df, fmt in (("variants_summary", S, "{:.3f}"), ("variants_paired_bootstrap", C, "{:.3f}"), ("calibration_effect", CAL, "{:.4f}")):
        (OUT / f"{name}.md").write_text(md(df, fmt) + "\n")
    print("\nCHOSEN ON VALIDATION:", chosen)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(S.round(3).to_string(index=False)); print(); print(C.drop(columns=["per-seed differences"]).to_string(index=False)); print(); print(CAL.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
