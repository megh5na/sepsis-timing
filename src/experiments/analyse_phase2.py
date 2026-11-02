from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.decision.stopping import P1_HORIZON, margin, onset_distribution
from src.eval import phase2 as m
from src.eval.fast_utility import UtilityTable, hours_from_alarms
from src.eval.harness import evaluate
from src.experiments.evaluate_runs import EVAL
from src.experiments.run_phase2 import RUNS, get_bundle, load_cfg
from src.experiments.train_mh import patient_scores

OUT = Path("results/phase2")
TAB, FIG = OUT / "tables", OUT / "figures"
TEAL, AMBER, ROSE, INK, GREY = "#0E6B6E", "#C9862E", "#A4453F", "#1C2521", "#9AA7A1"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


class Ctx:
    def __init__(self):
        self.cfg = load_cfg()
        self.bundle, self.manifest = get_bundle(self.cfg)
        self.test = self.bundle.test
        self.tab = UtilityTable.from_split(self.test)
        self.labels = m.sepsis_labels(self.test.lengths, self.test.t_sepsis)
        self.base = np.load(EVAL / "baselines.npz")
        self.seeds = self.cfg["seeds"]
        self.F, self.ev, self.meta = {}, {}, {}
        for cond in self.cfg["conditions"]:
            for s in self.seeds:
                run = RUNS / cond / f"seed{s}"
                if (run / "meta.json").exists() and (EVAL / f"{cond}_seed{s}.npz").exists():
                    self.F[(cond, s)] = np.load(run / "preds_test.npz")["F"]
                    self.ev[(cond, s)] = dict(np.load(EVAL / f"{cond}_seed{s}.npz"))
                    self.meta[(cond, s)] = json.loads((run / "meta.json").read_text())
        self.phase1 = pd.read_csv("results/grid/results_table.csv")

    def have(self, cond):
        return all((cond, s) in self.ev for s in self.seeds)

    def alarms(self, rule) -> np.ndarray:
        if rule == "B0":
            return np.stack([self.base["B0"]] * len(self.seeds))
        if rule == "B1":
            return np.stack([self.base[f"B1_seed{s}"] for s in self.seeds])
        cond, kind = rule
        return np.stack([self.ev[(cond, s)]["p1_test" if kind == "P1" else "p2_test"] for s in self.seeds])

    def comp(self, rule) -> np.ndarray:
        return np.stack([self.tab.components(a) for a in self.alarms(rule)])


def label(rule):
    if rule in ("B0", "B1"):
        return {"B0": "B0 qSOFA proxy", "B1": "B1 GRU + sigmoid"}[rule]
    cond, kind = rule
    if cond == "P":
        return kind
    if cond == "A1":
        return "A1 (decision layer)" if kind == "P2" else "A1 + threshold"
    return f"{cond} {kind}"


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, floatfmt="{:.3f}"):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(md_table(df, floatfmt) + "\n")


def pm(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


# ------------------------------------------------------------------ official-scorer cross-check
def official_check(ctx: Ctx, rules) -> dict:
    ids, res = ctx.test.ids, {}
    for rule in rules:
        A = ctx.alarms(rule)
        for si, s in enumerate(ctx.seeds):
            hours = dict(zip(ids, hours_from_alarms(A[si], ctx.test.lengths)))
            with tempfile.TemporaryDirectory() as tmp:
                off = evaluate(lambda pid: hours[pid], ids, ctx.manifest, workdir=Path(tmp))["utility"]
            ours = ctx.tab.normalised_for(A[si])
            res[f"{label(rule)} seed{s}"] = {"official": off, "table": ours, "abs_diff": abs(off - ours)}
            print(f"  official check {label(rule):20s} seed{s}: {off:.6f} vs {ours:.6f}")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "official_scorer_check.json").write_text(json.dumps(res, indent=2))
    assert max(v["abs_diff"] for v in res.values()) < 1e-9, "table utility disagrees with the official scorer"
    return res


# ------------------------------------------------------------------ Stage 10 primary table
def primary_table(ctx: Ctx) -> tuple[pd.DataFrame, dict]:
    rules = ["B0", "B1", ("P", "P1"), ("P", "P2")]
    rows, extra = [], {}
    p1_t = ctx.phase1.set_index(ctx.phase1["condition"] + ctx.phase1["seed"].astype(str))
    for rule in rules:
        comp = ctx.comp(rule)
        u = [UtilityTable.normalised(comp[i]) for i in range(len(ctx.seeds))]
        ci = m.bootstrap_mean_utility(comp)
        ops = [m.operating_point(a, ctx.test.lengths, ctx.test.t_sepsis) for a in ctx.alarms(rule)]
        if rule == "B0":
            auroc = auprc = float("nan")
        elif rule == "B1":
            auroc = float(np.mean([p1_t.loc[f"B1{float(s)}"]["auroc_continuous"] for s in ctx.seeds]))
            auprc = float(np.mean([p1_t.loc[f"B1{float(s)}"]["auprc_continuous"] for s in ctx.seeds]))
        elif rule[1] == "P1":
            ap = [m.auc_pair(ctx.labels, ctx.F[("P", s)][:, P1_HORIZON - 1]) for s in ctx.seeds]
            auroc, auprc = np.mean([a for a, _ in ap]), np.mean([b for _, b in ap])
        else:
            ap = [m.auc_pair(ctx.labels, ctx.ev[("P", s)]["margin_test"]) for s in ctx.seeds]
            auroc, auprc = np.mean([a for a, _ in ap]), np.mean([b for _, b in ap])
        rows.append({"condition": label(rule), "utility (mean ± sd over 3 seeds)": pm(u),
                     "utility 95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                     "per-seed utility": ", ".join(f"{x:.3f}" for x in u),
                     "AUROC": auroc, "AUPRC": auprc,
                     "sensitivity": float(np.mean([o["sensitivity"] for o in ops])),
                     "alarms/patient-day": float(np.mean([o["alarms_per_patient_day"] for o in ops])),
                     "median lead (h)": float(np.nanmean([o["median_lead_time_h"] for o in ops]))})
        extra[label(rule)] = {"utility_per_seed": u, "utility_mean": float(np.mean(u)), "ci": ci}
    return pd.DataFrame(rows), extra


def matched_table(ctx: Ctx) -> pd.DataFrame:
    taus = np.linspace(0.0005, 0.9995, 1999)
    rows = []
    for s in ctx.seeds:
        scores = patient_scores(ctx.F[("P", s)], ctx.test, P1_HORIZON - 1)
        curve = m.p1_curve(scores, ctx.test.lengths, ctx.test.t_sepsis, taus)
        p2 = m.operating_point(ctx.ev[("P", s)]["p2_test"], ctx.test.lengths, ctx.test.t_sepsis)
        p1 = m.operating_point(ctx.ev[("P", s)]["p1_test"], ctx.test.lengths, ctx.test.t_sepsis)
        mc = m.matched_comparison(p2, curve)
        rows.append({"seed": s, "P1 sensitivity": p1["sensitivity"], "P2 sensitivity": p2["sensitivity"],
                     "P1 alarms/day": p1["alarms_per_patient_day"], "P2 alarms/day": p2["alarms_per_patient_day"],
                     "P1 alarms/day at P2's sensitivity": mc["p1_alarms_per_day_at_p2_sensitivity"],
                     "P1 lead (h)": p1["median_lead_time_h"], "P2 lead (h)": p2["median_lead_time_h"],
                     "P1 lead (h) at P2's alarm rate": mc["p1_lead_time_at_p2_alarm_rate"],
                     "P1 sensitivity at P2's alarm rate": mc["p1_sensitivity_at_p2_alarm_rate"]})
    df = pd.DataFrame(rows)
    mean = df.mean(numeric_only=True); mean["seed"] = "mean"
    return pd.concat([df, mean.to_frame().T], ignore_index=True)


# ------------------------------------------------------------------ Stage 12 comparisons
def comparisons(ctx: Ctx) -> pd.DataFrame:
    pairs = [("P1 -> P2 (primary, same weights)", ("P", "P1"), ("P", "P2")),
             ("B1 -> P1 (output representation)", "B1", ("P", "P1")),
             ("B1 -> P2", "B1", ("P", "P2")),
             ("B0 -> B1", "B0", "B1")]
    if ctx.have("A1"):
        pairs += [("P2 -> A1 decision layer (H3)", ("P", "P2"), ("A1", "P2")),
                  ("P1 -> A1 + threshold (H3)", ("P", "P1"), ("A1", "P1"))]
    for cond in ("K6", "K18", "NOMD", "H32", "H128"):
        if ctx.have(cond):
            pairs += [(f"P2 -> {cond} P2 (ablation)", ("P", "P2"), (cond, "P2")),
                      (f"P1 -> {cond} P1 (ablation)", ("P", "P1"), (cond, "P1")),
                      (f"{cond}: P1 -> P2", (cond, "P1"), (cond, "P2"))]
    rows = []
    for name, a, b in pairs:
        r = m.paired_bootstrap(ctx.comp(a), ctx.comp(b))
        rows.append({"comparison (B - A)": name, "difference": r["diff"],
                     "95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                     "excludes 0": "yes" if r["ci_excludes_zero"] else "no",
                     "standardised diff": r["standardised_diff"], "Cohen d_z": r["cohens_dz"],
                     "per-seed differences": ", ".join(f"{x:+.3f}" for x in r["per_seed_diff"])})
    return pd.DataFrame(rows)


def all_conditions_table(ctx: Ctx) -> pd.DataFrame:
    rules = ["B0", "B1", ("P", "P1"), ("P", "P2")]
    if ctx.have("A1"):
        rules += [("A1", "P1"), ("A1", "P2")]
    for cond in ("K6", "K18", "NOMD", "H32", "H128"):
        if ctx.have(cond):
            rules += [(cond, "P1"), (cond, "P2")]
    rows = []
    for rule in rules:
        comp = ctx.comp(rule)
        u = [UtilityTable.normalised(comp[i]) for i in range(len(ctx.seeds))]
        ci = m.bootstrap_mean_utility(comp, n_boot=1000)
        rows.append({"condition": label(rule), "utility mean ± sd": pm(u), "95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                     "seed 0": u[0], "seed 1": u[1], "seed 2": u[2]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration + coherence
def calibration(ctx: Ctx) -> tuple[pd.DataFrame, dict]:
    K = 12
    y, mk = m.targets_flat(ctx.test.lengths, ctx.test.t_sepsis, K)
    sel = m.selected_hours_mask(ctx.test.lengths, ctx.test.t_sepsis)
    rows, curves = [], {}
    for cond in [c for c in ("P", "A1") if ctx.have(c)]:
        Fs = np.concatenate([ctx.F[(cond, s)] for s in ctx.seeds])
        rel_sel = m.reliability(Fs, np.tile(y, (3, 1)), np.tile(mk * sel[:, None], (3, 1)))
        rel_all = m.reliability(Fs, np.tile(y, (3, 1)), np.tile(mk, (3, 1)))
        curves[cond] = rel_sel
        for k in range(K):
            rows.append({"condition": cond, "horizon k": k + 1, "ECE (training hours)": rel_sel["ece"][k],
                         "ECE (all hours)": rel_all["ece"][k], "tail ECE, F>=0.1 (training hours)": rel_sel["ece_tail"][k],
                         "n terms (training hours)": int(rel_sel["n"][k]), "n terms F>=0.1": int(rel_sel["n_tail"][k])})
    df = pd.DataFrame(rows)
    mean = df.groupby("condition")[["ECE (training hours)", "ECE (all hours)", "tail ECE, F>=0.1 (training hours)"]].mean().reset_index()
    mean["horizon k"] = "mean"
    return pd.concat([df, mean], ignore_index=True), curves


def coherence_table(ctx: Ctx) -> pd.DataFrame:
    sel = m.selected_hours_mask(ctx.test.lengths, ctx.test.t_sepsis)
    rows = []
    for cond in [c for c in ("P", "A1") if ctx.have(c)]:
        for s in ctx.seeds:
            F = ctx.F[(cond, s)].astype(np.float64)
            drop = (F[:, :-1] - F[:, 1:])
            viol = drop > 0
            rows.append({"condition": cond, "seed": s,
                         "violation rate (all test hours)": m.coherence_violation_rate(F),
                         "violation rate (training hours)": m.coherence_violation_rate(F[sel]),
                         "mean size of a violation": float(drop[viol].mean()) if viol.any() else 0.0,
                         "largest violation": float(drop.max()) if viol.any() else 0.0,
                         "rows with any violation": float(viol.any(axis=1).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ error analysis
def disagreement(ctx: Ctx):
    test, tab = ctx.test, ctx.tab
    rows = []
    for s in ctx.seeds:
        F = ctx.F[("P", s)].astype(np.float64)
        ev = ctx.ev[("P", s)]
        tau = float(ev["tau"])
        a1, a2 = ev["p1_test"], ev["p2_test"]
        o1, o2 = tab.observed(a1), tab.observed(a2)
        for i in np.flatnonzero(a1 != a2):
            T = int(test.lengths[i]); off = test.offsets[i]
            Fi = F[off:off + T]
            p, p_none = onset_distribution(Fi)
            def at(t):
                if t >= T:
                    return {"F6": np.nan, "FK": np.nan, "E_onset": np.nan}
                mass = p[t].sum()
                return {"F6": Fi[t, 5], "FK": Fi[t, -1],
                        "E_onset": float((p[t] * np.arange(1, p.shape[1] + 1)).sum() / mass) if mass > 1e-9 else np.nan}
            if a1[i] < T and a2[i] < T:
                cat = "both alarm, P2 earlier" if a2[i] < a1[i] else "both alarm, P2 later"
            else:
                cat = "only P2 alarms" if a1[i] >= T else "only P1 alarms"
            r = {"seed": s, "patient": test.ids[i], "septic": bool(test.is_septic[i]),
                 "t_sepsis": int(test.t_sepsis[i]), "T": T, "p1_alarm": int(a1[i]) if a1[i] < T else None,
                 "p2_alarm": int(a2[i]) if a2[i] < T else None, "category": cat, "tau": tau,
                 "utility_p2_minus_p1": float(o2[i] - o1[i])}
            for tag, a in (("p1", a1[i]), ("p2", a2[i])):
                for k_, v in at(int(a)).items():
                    r[f"{k_}_at_{tag}_alarm"] = v
            rows.append(r)
    D = pd.DataFrame(rows)
    summary = []
    for cat, g in D.groupby("category"):
        summary.append({"category": cat, "patient-seeds": len(g), "septic share": g["septic"].mean(),
                        "mean utility P2 - P1": g["utility_p2_minus_p1"].mean(),
                        "sum utility P2 - P1": g["utility_p2_minus_p1"].sum(),
                        "median alarm shift P1 - P2 (h)": float(np.nanmedian((g["p1_alarm"].astype(float) - g["p2_alarm"].astype(float)))),
                        "median F[t,6] at P2 alarm": g["F6_at_p2_alarm"].median(),
                        "median F[t,6] at P1 alarm": g["F6_at_p1_alarm"].median(),
                        "median expected onset ahead at P2 alarm (h)": g["E_onset_at_p2_alarm"].median(),
                        "median expected onset ahead at P1 alarm (h)": g["E_onset_at_p1_alarm"].median()})
    n_total = len(ctx.seeds) * len(test)
    return D, pd.DataFrame(summary), {"disagree_frac": len(D) / n_total, "n_disagree_patient_seeds": len(D)}


def mechanism_table(ctx: Ctx) -> pd.DataFrame:
    T, ts, septic = ctx.test.lengths, ctx.test.t_sepsis, ctx.test.is_septic
    nonsep_rows = np.repeat(~septic, T)
    rows = []
    for cond in [c for c in ctx.cfg["conditions"] if ctx.have(c)]:
        K = ctx.cfg["conditions"][cond]["K"]
        for s in ctx.seeds:
            F, ev = ctx.F[(cond, s)].astype(np.float64), ctx.ev[(cond, s)]
            fired = ev["margin_test"] >= -1e-12
            fa = lambda a: float(((a < T) & ~septic).sum() / (~septic).sum())
            rows.append({"condition": cond, "seed": s, "tau for P1 (validation)": float(ev["tau"]),
                         f"smallest F[t,{K}] at which P2 fires": float(F[fired, K - 1].min()) if fired.any() else np.nan,
                         f"median F[t,{K}] when P2 fires": float(np.median(F[fired, K - 1])) if fired.any() else np.nan,
                         f"mean F[t,{K}] on non-septic hours": float(F[nonsep_rows, K - 1].mean()),
                         "P1 false-alarm share of non-septic": fa(ev["p1_test"]),
                         "P2 false-alarm share of non-septic": fa(ev["p2_test"]),
                         "P2 alarm never later than P1": bool(np.all(ev["p2_test"] <= ev["p1_test"]))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_primary(ctx, extra):
    names = ["B0 qSOFA proxy", "B1 GRU + sigmoid", "P1", "P2"]
    cols = [GREY, TEAL, AMBER, ROSE]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, n in enumerate(names):
        e = extra[n]
        ax.bar(i, e["utility_mean"], color=cols[i], width=0.62, yerr=[[e["utility_mean"] - e["ci"]["ci_low"]], [e["ci"]["ci_high"] - e["utility_mean"]]],
               capsize=4, ecolor=INK)
        ax.scatter([i] * len(e["utility_per_seed"]), e["utility_per_seed"], color="white", edgecolor=INK, s=18, zorder=3)
        ax.text(i, e["ci"]["ci_high"] + 0.012, f"{e['utility_mean']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(["B0\nqSOFA", "B1\nGRU+sigmoid", "P1\nthreshold", "P2\nexpected utility"])
    ax.set_ylabel("Normalised challenge utility (test)")
    ax.set_title("Primary comparison, 3 seeds (dots), 95% patient-bootstrap CI", fontsize=9)
    fig.savefig(FIG / "fig_primary_utility.png"); plt.close(fig)


def fig_matched(ctx):
    taus = np.linspace(0.0005, 0.9995, 1999)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for s, c in zip(ctx.seeds, (TEAL, AMBER, ROSE)):
        scores = patient_scores(ctx.F[("P", s)], ctx.test, P1_HORIZON - 1)
        cu = m.p1_curve(scores, ctx.test.lengths, ctx.test.t_sepsis, taus)
        p2 = m.operating_point(ctx.ev[("P", s)]["p2_test"], ctx.test.lengths, ctx.test.t_sepsis)
        p1 = m.operating_point(ctx.ev[("P", s)]["p1_test"], ctx.test.lengths, ctx.test.t_sepsis)
        axes[0].plot(cu["alarms_per_patient_day"], cu["sensitivity"], color=c, lw=1.2, label=f"P1 sweep, seed {s}")
        axes[0].scatter([p1["alarms_per_patient_day"]], [p1["sensitivity"]], marker="o", color=c, edgecolor=INK, zorder=3)
        axes[0].scatter([p2["alarms_per_patient_day"]], [p2["sensitivity"]], marker="*", s=130, color=c, edgecolor=INK, zorder=3)
        axes[1].plot(cu["alarms_per_patient_day"], cu["median_lead_time_h"], color=c, lw=1.2)
        axes[1].scatter([p1["alarms_per_patient_day"]], [p1["median_lead_time_h"]], marker="o", color=c, edgecolor=INK, zorder=3)
        axes[1].scatter([p2["alarms_per_patient_day"]], [p2["median_lead_time_h"]], marker="*", s=130, color=c, edgecolor=INK, zorder=3)
    axes[0].set_xlabel("Alarms per patient-day"); axes[0].set_ylabel("Sensitivity (alarm in [onset-12, onset+3])")
    axes[1].set_xlabel("Alarms per patient-day"); axes[1].set_ylabel("Median lead time (h)")
    for a in axes: a.set_xlim(0, 0.6)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("P1 swept over tau (lines); circle = P1 at its tuned tau, star = P2", fontsize=9)
    fig.savefig(FIG / "fig_matched_operating_points.png"); plt.close(fig)


def fig_calibration(curves):
    conds = list(curves)
    fig, axes = plt.subplots(1, len(conds), figsize=(4.4 * len(conds), 3.8), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        rel = curves[cond]
        ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
        for k, c in zip((0, 2, 5, 8, 11), ("#7FB3B5", "#4F9C9F", TEAL, AMBER, ROSE)):
            ok = rel["count"][k] >= 30
            ax.plot(rel["mean_pred"][k][ok], rel["frac_pos"][k][ok], marker="o", ms=3, lw=1.1, color=c, label=f"k={k + 1}")
        ax.set_xlabel("Predicted F[t,k]"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"{'P (monotone head)' if cond == 'P' else 'A1 (independent head)'}, test, training hours", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)


def fig_ablation(ctx):
    rules = [(c, k) for c in ("K6", "K18", "NOMD", "H32", "H128") if ctx.have(c) for k in ("P1", "P2")]
    if not rules:
        return
    base = {k: np.mean([UtilityTable.normalised(c) for c in ctx.comp(("P", k))]) for k in ("P1", "P2")}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = ["P (K=12, h=64, mask+delta)"] + [f"{c}" for c, _ in rules[::2]]
    x = np.arange(len(names)); w = 0.36
    for j, (k, col) in enumerate((("P1", AMBER), ("P2", ROSE))):
        vals = [base[k]] + [np.mean([UtilityTable.normalised(c) for c in ctx.comp((cond, k))]) for cond, kk in rules if kk == k]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=k)
        for xi, v in zip(x, vals): ax.text(xi + (j - 0.5) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(["main"] + names[1:]); ax.set_ylabel("Normalised utility (test, mean of 3 seeds)")
    ax.legend(frameon=False); ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    fig.savefig(FIG / "fig_ablations.png"); plt.close(fig)


def fig_examples(ctx, D):
    s = ctx.seeds[0]
    d = D[(D["seed"] == s) & D["septic"] & D["p1_alarm"].notna() & D["p2_alarm"].notna()].copy()
    if d.empty:
        return
    d["abs"] = (d["p1_alarm"] - d["p2_alarm"]).abs()
    pick = pd.concat([d[d["category"] == "both alarm, P2 earlier"].nlargest(2, "abs"),
                      d[d["category"] == "both alarm, P2 later"].nlargest(2, "abs")])
    if pick.empty:
        return
    F = ctx.F[("P", s)].astype(np.float64); tau = float(ctx.ev[("P", s)]["tau"])
    fig, axes = plt.subplots(len(pick), 1, figsize=(6.6, 2.1 * len(pick)), sharex=False, squeeze=False)
    for ax, (_, r) in zip(axes[:, 0], pick.iterrows()):
        i = ctx.test.ids.index(r["patient"]); T = int(ctx.test.lengths[i]); off = ctx.test.offsets[i]
        Fi = F[off:off + T]; t = np.arange(T)
        ax.plot(t, Fi[:, 5], color=TEAL, lw=1.3, label="F[t,6] (P1 score)")
        ax.axhline(tau, color=TEAL, ls=":", lw=1, label=f"tau = {tau:.3f}")
        ax.plot(t, Fi[:, -1], color=GREY, lw=1, label="F[t,12]")
        ax.axvline(r["t_sepsis"], color=INK, lw=1.4, label="onset")
        ax.axvline(r["p1_alarm"], color=AMBER, lw=1.6, label="P1 alarm")
        ax.axvline(r["p2_alarm"], color=ROSE, lw=1.6, ls="--", label="P2 alarm")
        ax.set_ylabel("probability"); ax.set_title(f"{r['patient']}: {r['category']}, utility P2-P1 = {r['utility_p2_minus_p1']:+.2f}", fontsize=8)
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=3, loc="upper left")
    axes[-1, 0].set_xlabel("hour of stay")
    fig.tight_layout(); fig.savefig(FIG / "fig_disagreement_examples.png"); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-official", action="store_true")
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    assert ctx.have("P"), "train condition P (all seeds) and run evaluate_runs first"

    # shared weights: both rules carry the same checkpoint hash, and it matches training
    for s in ctx.seeds:
        ev, meta = ctx.ev[("P", s)], ctx.meta[("P", s)]
        assert str(ev["weights_sha256_p1"]) == str(ev["weights_sha256_p2"]) == meta["weights_sha256"]

    summary = {"n_test_patients": len(ctx.test), "n_septic_test": int(ctx.test.is_septic.sum()), "seeds": ctx.seeds}
    if not args.skip_official:
        rules = ["B0", "B1", ("P", "P1"), ("P", "P2")] + ([("A1", "P1"), ("A1", "P2")] if ctx.have("A1") else [])
        official_check(ctx, rules)

    prim, extra = primary_table(ctx); save_table(prim, "table1_primary")
    mt = matched_table(ctx); save_table(mt, "table2_matched_operating_points")
    allc = all_conditions_table(ctx); save_table(allc, "table3_all_conditions")
    cmp_ = comparisons(ctx); save_table(cmp_, "table4_paired_bootstrap")
    cal, curves = calibration(ctx); save_table(cal, "table5_calibration_ece", "{:.4f}")
    coh = coherence_table(ctx); save_table(coh, "table6_coherence_violations", "{:.4f}")
    mech = mechanism_table(ctx); save_table(mech, "table8_p2_implied_cutoff", "{:.3f}")
    D, dsum, dstat = disagreement(ctx); save_table(dsum, "table7_disagreement_summary")
    D.to_csv(TAB / "disagreement_patients.csv", index=False)
    summary.update({"primary": extra, "disagreement": dstat})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig_primary(ctx, extra); fig_matched(ctx); fig_calibration(curves); fig_ablation(ctx); fig_examples(ctx, D)
    for name, df in (("PRIMARY", prim), ("MATCHED", mt), ("PAIRED BOOTSTRAP", cmp_), ("DISAGREEMENT", dsum)):
        print(f"\n=== {name} ===\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
