from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.cache import make_batch
from src.data.parse import load_patient_tensor
from src.decision.qsofa import qsofa_alarm_hour
from src.decision.stopping import P1_HORIZON, margin, p2_alarm_hour
from src.eval.fast_utility import UtilityTable, alarms_from_hours
from src.experiments.run_phase2 import PROCESSED, RUNS, get_bundle, load_cfg
from src.experiments.train_mh import patient_scores, score_split, weights_sha256
from src.experiments.train import B1Model
from src.models.model import SepsisModel

EVAL = Path("results/phase2/eval")


def derive_rules(F: np.ndarray, split, tau: float, repair: bool) -> dict:
    scores = patient_scores(F, split, P1_HORIZON - 1)
    tab = UtilityTable.from_split(split)
    p1 = tab.alarms_at_threshold(scores, tau)
    hours, margins = [], []
    for i in range(len(split)):
        Fi = F[split.offsets[i]:split.offsets[i + 1]].astype(np.float64)
        hours.append(p2_alarm_hour(Fi, repair=repair))
        margins.append(margin(Fi, repair=repair))
    return {"p1": p1, "p2": alarms_from_hours(hours, split.lengths), "margin": np.concatenate(margins)}


def eval_run(cond: str, seed: int, bundle, cfg: dict) -> dict:
    run = RUNS / cond / f"seed{seed}"
    meta = json.loads((run / "meta.json").read_text())
    c = cfg["conditions"][cond]
    model = SepsisModel(c["head"], c["K"], hidden_size=c["hidden"], input_size=120 if c["mask_delta"] else 40, seed=seed)
    model.load_state_dict(torch.load(run / "model.pt"))
    assert weights_sha256(model) == meta["weights_sha256"], "checkpoint differs from the weights used in training"

    F_test_saved = np.load(run / "preds_test.npz")["F"]
    F_val_saved = np.load(run / "preds_val.npz")["F"]
    F_test = score_split(model, bundle.test, c["mask_delta"])
    assert np.allclose(F_test, F_test_saved, atol=1e-5), "saved predictions do not come from this checkpoint"

    repair = c["head"] == "independent"
    out = {}
    for name, F, split in (("val", F_val_saved, bundle.val), ("test", F_test_saved, bundle.test)):
        r = derive_rules(F, split, meta["tau_p1"], repair)
        out[f"p1_{name}"], out[f"p2_{name}"], out[f"margin_{name}"] = r["p1"], r["p2"], r["margin"]
    # shared-weights check, recorded: both rules read the same F array and the same hash
    out["weights_sha256_p1"] = out["weights_sha256_p2"] = weights_sha256(model)
    EVAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL / f"{cond}_seed{seed}.npz", tau=meta["tau_p1"], **out)
    return out


def eval_baselines(bundle) -> dict:
    test = bundle.test
    tab = UtilityTable.from_split(test)
    res = {}
    hours = [qsofa_alarm_hour(load_patient_tensor(PROCESSED, pid)) for pid in test.ids]
    res["B0"] = alarms_from_hours(hours, test.lengths)
    ckpt = Path("results/grid/B1/checkpoints")
    for seed in (0, 1, 2):
        meta = json.loads((ckpt / f"b1_seed{seed}_meta.json").read_text())
        model = B1Model(); model.load_state_dict(torch.load(ckpt / f"b1_seed{seed}.pt")); model.eval()
        scores = [None] * len(test)
        order = np.argsort(test.lengths)
        with torch.no_grad():
            for s in range(0, len(order), 256):
                idx = [int(i) for i in order[s:s + 256]]
                r = model(make_batch(test, idx, K=None)["Z"])
                for j, i in enumerate(idx):
                    scores[i] = r[j, :int(test.lengths[i])].numpy().astype(np.float64)
        res[f"B1_seed{seed}"] = tab.alarms_at_threshold(scores, meta["tau"])
    EVAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL / "baselines.npz", **res)
    return res


def main():
    cfg = load_cfg()
    bundle, _ = get_bundle(cfg)
    cmd = sys.argv[1]
    if cmd == "baselines":
        res = eval_baselines(bundle)
        tab = UtilityTable.from_split(bundle.test)
        ref = {}
        import pandas as pd
        t = pd.read_csv("results/grid/results_table.csv")
        for _, row in t.iterrows():
            key = "B0" if row["condition"] == "B0" else f"B1_seed{int(float(row['seed']))}"
            ref[key] = row["utility"]
        for k, a in res.items():
            u = tab.normalised_for(a)
            print(f"{k}: utility {u:.6f}  phase1 results_table {ref[k]:.6f}  diff {u - ref[k]:+.2e}")
    elif cmd == "runs":
        conds = sys.argv[2:] or list(cfg["conditions"])
        for cond in conds:
            for seed in cfg["seeds"]:
                if (RUNS / cond / f"seed{seed}" / "meta.json").exists():
                    eval_run(cond, seed, bundle, cfg)
                    print("evaluated", cond, seed)


if __name__ == "__main__":
    main()
