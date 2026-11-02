from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.cache import SplitTensors
from src.data.parse import FEATURE_COLUMNS, load_patient_tensor
from src.data.preprocess import compute_values_mask_delta
from src.decision.qsofa import qsofa_alarm_hour
from src.decision.stopping import P1_HORIZON, TIE_TOL, hourly_gain_table, onset_distribution, p2_alarm_hour
from src.eval import phase2 as m
from src.eval.fast_utility import UtilityTable, alarms_from_hours
from src.experiments.evaluate_runs import EVAL
from src.experiments.run_phase2 import PROCESSED, RUNS, get_bundle, load_cfg
from src.experiments.train import B1Model
from src.experiments.train_mh import patient_scores, score_split, weights_sha256
from src.models.model import SepsisModel
from src.data.cache import make_batch

OUT = Path("results/phase2/extension")
LAMBDAS = 0.5 * 2.0 ** (np.arange(29) / 4.0)
ETCO2 = FEATURE_COLUMNS.index("EtCO2")


# ---------------------------------------------------------------- vectorised scaled P2
def margin_parts(F_all: np.ndarray):
    p, p_none = onset_distribution(F_all.astype(np.float64))
    K = p.shape[-1]
    gain, none_gain = hourly_gain_table(K)
    cumA = np.cumsum(p @ gain.T, axis=-1)
    cumN = (p_none * none_gain)[:, None] * np.arange(1, K + 1)[None, :]
    return cumA, cumN


def scaled_alarms(parts, lam: float, offsets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    cumA, cumN = parts
    mg = (cumA + lam * cumN).min(axis=-1)
    hit = np.flatnonzero(mg >= -TIE_TOL)
    out = lengths.astype(int).copy()
    if hit.size:
        pid = np.searchsorted(offsets, hit, side="right") - 1
        first_pid, first_pos = np.unique(pid, return_index=True)
        out[first_pid] = hit[first_pos] - offsets[first_pid]
    return out


# ---------------------------------------------------------------- Hospital B tensors
def _raw(args):
    pid, blank = args
    X = load_patient_tensor(PROCESSED, pid).copy()
    if blank:
        X[:, ETCO2] = np.nan
    values, mask, delta = compute_values_mask_delta(X)
    return pid, X, values, mask.astype(np.float32), delta.astype(np.float32)


def build_hospital_b(stats, manifest: pd.DataFrame, blank_etco2: bool) -> tuple[SplitTensors, dict]:
    sub = manifest[manifest["hospital"] == "B"].reset_index(drop=True)
    with Pool(6) as pool:
        res = pool.map(_raw, [(p, blank_etco2) for p in sub["patient_id"]], chunksize=64)
    zs, lengths = [], []
    for _, X, values, mask, delta in res:
        zs.append(np.concatenate([stats.impute_and_zscore(values).astype(np.float32), mask, delta], axis=-1))
        lengths.append(values.shape[0])
    lengths = np.array(lengths)
    split = SplitTensors(name="hospital_B", ids=list(sub["patient_id"]), lengths=lengths,
                         offsets=np.concatenate([[0], np.cumsum(lengths)]), Z=np.concatenate(zs, axis=0),
                         t_sepsis=np.where(sub["is_septic"], sub["t_sepsis"], -1).astype(int),
                         is_septic=sub["is_septic"].to_numpy().astype(bool))
    assert np.array_equal(lengths, sub["T"].to_numpy())
    raw = {"X": [r[1] for r in res]}
    return split, raw


# ---------------------------------------------------------------- helpers
def comps(tab: UtilityTable, per_seed_alarms: list[np.ndarray]) -> np.ndarray:
    return np.stack([tab.components(a) for a in per_seed_alarms])


def boot_row(a_name, b_name, ca, cb):
    r = m.paired_bootstrap(ca, cb, n_boot=2000)
    return {"B": b_name, "A": a_name, "difference B - A": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "excludes 0": "yes" if r["ci_excludes_zero"] else "no",
            "per-seed differences": ", ".join(f"{x:+.3f}" for x in r["per_seed_diff"])}


def op_row(name, per_seed_alarms, split):
    ops = [m.operating_point(a, split.lengths, split.t_sepsis) for a in per_seed_alarms]
    return {"rule": name, "stays alarmed": np.mean([o["frac_alarmed"] for o in ops]),
            "sensitivity": np.mean([o["sensitivity"] for o in ops]),
            "false-alarm share of non-septic": np.mean([o["false_alarm_frac_nonseptic"] for o in ops]),
            "median lead (h)": np.mean([o["median_lead_time_h"] for o in ops])}


def load_model(seed: int, cfg: dict):
    run = RUNS / "P" / f"seed{seed}"
    meta = json.loads((run / "meta.json").read_text())
    c = cfg["conditions"]["P"]
    model = SepsisModel(c["head"], c["K"], hidden_size=c["hidden"], input_size=120, seed=seed)
    model.load_state_dict(torch.load(run / "model.pt"))
    assert weights_sha256(model) == meta["weights_sha256"], "checkpoint differs from the weights used in training"
    return model, meta


def b1_scores(split: SplitTensors, seed: int):
    ckpt = Path("results/grid/B1/checkpoints")
    meta = json.loads((ckpt / f"b1_seed{seed}_meta.json").read_text())
    model = B1Model(); model.load_state_dict(torch.load(ckpt / f"b1_seed{seed}.pt")); model.eval()
    scores = [None] * len(split)
    order = np.argsort(split.lengths)
    with torch.no_grad():
        for s in range(0, len(order), 256):
            idx = [int(i) for i in order[s:s + 256]]
            r = model(make_batch(split, idx, K=None)["Z"])
            for j, i in enumerate(idx):
                scores[i] = r[j, :int(split.lengths[i])].numpy().astype(np.float64)
    return scores, meta["tau"]


def rules_on(split, F, tau, lam, tab):
    scores = patient_scores(F, split, P1_HORIZON - 1)
    p1 = tab.alarms_at_threshold(scores, tau)
    p2 = alarms_from_hours([p2_alarm_hour(F[split.offsets[i]:split.offsets[i + 1]].astype(np.float64)) for i in range(len(split))], split.lengths)
    parts = margin_parts(F)
    p2l = scaled_alarms(parts, lam, split.offsets, split.lengths)
    assert np.array_equal(scaled_alarms(parts, 1.0, split.offsets, split.lengths), p2), "vectorised P2 disagrees with the reference rule"
    return p1, p2, p2l


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    bundle, manifest = get_bundle(cfg)
    val, test = bundle.val, bundle.test
    tab_v, tab_t = UtilityTable.from_split(val), UtilityTable.from_split(test)
    seeds = cfg["seeds"]

    # ---- 1. tune lam on Hospital A validation, per seed (like P1's tau)
    lam_rows, lam_star, taus = [], {}, {}
    for s in seeds:
        meta = json.loads((RUNS / "P" / f"seed{s}" / "meta.json").read_text())
        taus[s] = meta["tau_p1"]
        Fv = np.load(RUNS / "P" / f"seed{s}" / "preds_val.npz")["F"]
        parts = margin_parts(Fv)
        utils = np.array([tab_v.normalised_for(scaled_alarms(parts, l, val.offsets, val.lengths)) for l in LAMBDAS])
        best = utils.max()
        cand = LAMBDAS[utils >= best - 1e-12]
        lam_star[s] = float(cand[np.argmin(np.abs(np.log(cand)))])          # ties: closest to lam = 1
        for l, u in zip(LAMBDAS, utils):
            lam_rows.append({"seed": s, "lambda": l, "val utility": u})
        print(f"seed {s}: lam* = {lam_star[s]:.3f}  val utility {best:.4f}  (lam=1: {utils[np.argmin(np.abs(LAMBDAS - 1))]:.4f}; P1 val {meta['val_p1_utility_at_tau']:.4f})")
    pd.DataFrame(lam_rows).to_csv(OUT / "lambda_selection.csv", index=False)

    # ---- 2. Hospital A test (second look)
    res_A = {"P1": [], "P2": [], "P2 (lam*)": []}
    for s in seeds:
        Ft = np.load(RUNS / "P" / f"seed{s}" / "preds_test.npz")["F"]
        p1, p2, p2l = rules_on(test, Ft, taus[s], lam_star[s], tab_t)
        saved = np.load(EVAL / f"P_seed{s}.npz")
        assert np.array_equal(p1, saved["p1_test"]) and np.array_equal(p2, saved["p2_test"]), "does not reproduce the locked P1/P2 alarms"
        res_A["P1"].append(p1); res_A["P2"].append(p2); res_A["P2 (lam*)"].append(p2l)
    summ_A, ca = [], {k: comps(tab_t, v) for k, v in res_A.items()}
    for k, v in res_A.items():
        u = [tab_t.normalised_for(a) for a in v]
        summ_A.append({"rule": k, "utility mean": np.mean(u), "utility sd": np.std(u, ddof=1), "per-seed": ", ".join(f"{x:.3f}" for x in u),
                       **{kk: vv for kk, vv in op_row(k, v, test).items() if kk != "rule"}})
    pd.DataFrame(summ_A).to_csv(OUT / "hospitalA_test_summary.csv", index=False)
    pd.DataFrame([boot_row("P1", "P2 (lam*)", ca["P1"], ca["P2 (lam*)"]), boot_row("P2", "P2 (lam*)", ca["P2"], ca["P2 (lam*)"]),
                  boot_row("P1", "P2", ca["P1"], ca["P2"])]).to_csv(OUT / "hospitalA_test_bootstrap.csv", index=False)

    # ---- 3. Hospital B
    summ_B_all, boot_B_all = {}, {}
    for blank in (True, False):
        tag = "blank_etco2" if blank else "etco2_left_in"
        splitB, raw = build_hospital_b(bundle.stats, manifest, blank)
        tabB = UtilityTable.from_split(splitB)
        res = {"P1": [], "P2": [], "P2 (lam*)": [], "B1": []}
        for s in seeds:
            model, meta = load_model(s, cfg)
            FB = score_split(model, splitB, True)
            p1, p2, p2l = rules_on(splitB, FB, taus[s], lam_star[s], tabB)
            res["P1"].append(p1); res["P2"].append(p2); res["P2 (lam*)"].append(p2l)
            sc, tau_b1 = b1_scores(splitB, s)
            res["B1"].append(tabB.alarms_at_threshold(sc, tau_b1))
            print(f"[{tag}] seed {s}: P1 {tabB.normalised_for(p1):.3f}  P2 {tabB.normalised_for(p2):.3f}  P2 lam* {tabB.normalised_for(p2l):.3f}  B1 {tabB.normalised_for(res['B1'][-1]):.3f}")
        if blank:
            res["B0"] = [alarms_from_hours([qsofa_alarm_hour(x) for x in raw["X"]], splitB.lengths)] * 3
        cb = {k: comps(tabB, v) for k, v in res.items()}
        rows = []
        for k, v in res.items():
            u = [tabB.normalised_for(a) for a in v]
            rows.append({"rule": k, "utility mean": np.mean(u), "utility sd": np.std(u, ddof=1), "per-seed": ", ".join(f"{x:.3f}" for x in u),
                         **{kk: vv for kk, vv in op_row(k, v, splitB).items() if kk != "rule"}})
            ci = m.bootstrap_mean_utility(cb[k]); rows[-1]["95% CI"] = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        pd.DataFrame(rows).to_csv(OUT / f"hospitalB_summary_{tag}.csv", index=False)
        boots = [boot_row("P1", "P2 (lam*)", cb["P1"], cb["P2 (lam*)"]), boot_row("P1", "P2", cb["P1"], cb["P2"]),
                 boot_row("P2", "P2 (lam*)", cb["P2"], cb["P2 (lam*)"]), boot_row("B1", "P1", cb["B1"], cb["P1"])]
        if blank:
            boots.append(boot_row("B0", "B1", cb["B0"], cb["B1"]))
        pd.DataFrame(boots).to_csv(OUT / f"hospitalB_bootstrap_{tag}.csv", index=False)
        summ_B_all[tag], boot_B_all[tag] = rows, boots
        meta_B = {"n_stays": len(splitB), "n_septic": int(splitB.is_septic.sum()), "hours": int(splitB.lengths.sum())}

    (OUT / "summary.json").write_text(json.dumps({
        "lambda_grid": LAMBDAS.tolist(), "lambda_star_per_seed": lam_star, "tau_p1_per_seed": taus,
        "hospital_B": meta_B}, indent=2))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\nHOSPITAL A TEST\n", pd.read_csv(OUT / "hospitalA_test_summary.csv").round(3).to_string(index=False))
    print(pd.read_csv(OUT / "hospitalA_test_bootstrap.csv").round(3).to_string(index=False))
    for tag in ("blank_etco2", "etco2_left_in"):
        print(f"\nHOSPITAL B ({tag}), {meta_B}\n", pd.read_csv(OUT / f"hospitalB_summary_{tag}.csv").round(3).to_string(index=False))
        print(pd.read_csv(OUT / f"hospitalB_bootstrap_{tag}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
