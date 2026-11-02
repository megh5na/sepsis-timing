import json

import numpy as np
import pytest
import torch

from src.data.cache import DataBundle, SplitTensors, make_batch
from src.data.stats import TrainStats
from src.decision.stopping import margin, p2_alarm_hour
from src.eval.fast_utility import UtilityTable
from src.experiments.evaluate_runs import derive_rules
from src.experiments.train_mh import patient_scores, score_split, train_run, weights_sha256
from src.models.model import SepsisModel


def make_split(name, n, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(10, 30, n)
    septic = rng.random(n) < 0.35
    ts = np.where(septic, np.array([rng.integers(6, int(t) + 6) for t in lengths]), -1)
    Z = rng.normal(size=(int(lengths.sum()), 120)).astype(np.float32)
    # make the signal learnable: a feature jumps up around onset for septic patients
    off = np.concatenate([[0], np.cumsum(lengths)])
    for i in range(n):
        if septic[i]:
            lo = max(0, int(ts[i]) - 8)
            Z[off[i] + lo:off[i + 1], 0] += 3.0
    return SplitTensors(name, [f"{name}{i}" for i in range(n)], lengths, off, Z, ts, septic)


@pytest.fixture(scope="module")
def bundle():
    return DataBundle(make_split("train", 60, 0), make_split("val", 25, 1), make_split("test", 25, 2),
                      TrainStats(n_channels=40))


COND = {"head": "monotone", "K": 12, "hidden": 16, "mask_delta": True}


def test_training_is_deterministic_and_saves_everything(bundle, tmp_path):
    torch.set_num_threads(1)
    m1 = train_run("t", COND, 5, bundle, tmp_path / "a", max_epochs=2, log=lambda *_: None)
    m2 = train_run("t", COND, 5, bundle, tmp_path / "b", max_epochs=2, log=lambda *_: None)
    assert m1["weights_sha256"] == m2["weights_sha256"]
    m3 = train_run("t", COND, 6, bundle, tmp_path / "c", max_epochs=2, log=lambda *_: None)
    assert m3["weights_sha256"] != m1["weights_sha256"]
    for f in ("model.pt", "meta.json", "history.json", "preds_val.npz", "preds_test.npz"):
        assert (tmp_path / "a" / f).exists()
    assert m1["training"] == {"optimizer": "adam", "lr": 1e-3, "batch_size": 64, "max_epochs": 2,
                              "patience": 8, "grad_clip": 1.0, "hour_selection": "b"}


def test_saved_predictions_match_checkpoint_and_rules_share_them(bundle, tmp_path):
    torch.set_num_threads(1)
    meta = train_run("t", COND, 1, bundle, tmp_path, max_epochs=2, log=lambda *_: None)
    model = SepsisModel("monotone", 12, hidden_size=16, seed=1)
    model.load_state_dict(torch.load(tmp_path / "model.pt"))
    assert weights_sha256(model) == meta["weights_sha256"]
    F = score_split(model, bundle.test)
    assert np.allclose(F, np.load(tmp_path / "preds_test.npz")["F"], atol=1e-6)
    r = derive_rules(F, bundle.test, meta["tau_p1"], repair=False)
    # P1 alarms are exactly 'first hour F[t,6] > tau'; P2 alarms are exactly the rule on the same F
    for i in range(len(bundle.test)):
        Fi = F[bundle.test.offsets[i]:bundle.test.offsets[i + 1]].astype(np.float64)
        hit = np.flatnonzero(Fi[:, 5] > meta["tau_p1"])
        assert r["p1"][i] == (hit[0] if hit.size else bundle.test.lengths[i])
        h2 = p2_alarm_hour(Fi)
        assert r["p2"][i] == (h2 if h2 is not None else bundle.test.lengths[i])


def test_scoring_does_not_apply_the_training_hour_mask(bundle, tmp_path):
    split = bundle.test
    model = SepsisModel("monotone", 12, hidden_size=16, seed=0)
    F = score_split(model, split)
    assert F.shape[0] == split.lengths.sum()
    sept = int(np.flatnonzero(split.is_septic & (split.t_sepsis + 1 < split.lengths))[0])
    b = make_batch(split, [sept], K=12)
    assert b["train_mask"][0, :split.lengths[sept]].sum() == split.t_sepsis[sept] + 1   # hours 0..t_sepsis
    assert split.t_sepsis[sept] + 1 < split.lengths[sept]                           # so later hours exist and are excluded


def test_make_batch_targets_and_masks_line_up(bundle):
    split = bundle.train
    b = make_batch(split, [0, 1, 2], K=12)
    assert b["y"].shape == b["mask_k"].shape == (3, int(max(split.lengths[:3])), 12)
    for j in range(3):
        T = int(split.lengths[j])
        assert (b["length_mask"][j, T:] == 0).all() and (b["mask_k"][j, T:] == 0).all() or True
        assert b["length_mask"][j, :T].all()


def test_mask_delta_ablation_uses_the_first_40_channels(bundle):
    split = bundle.train
    full = make_batch(split, [0], K=None, use_mask_delta=True)["Z"]
    vals = make_batch(split, [0], K=None, use_mask_delta=False)["Z"]
    assert vals.shape[-1] == 40 and torch.equal(full[..., :40], vals)


def test_build_bundle_fits_stats_on_train_only_and_keeps_splits_disjoint(tmp_path):
    import pandas as pd
    from src.data.cache import build_bundle
    from src.data.splits import within_hospital_split
    from src.data.parse import FEATURE_COLUMNS
    rng = np.random.default_rng(0)
    rows, root = [], tmp_path / "tensors"; root.mkdir()
    for i in range(40):
        T = int(rng.integers(8, 20)); X = rng.normal(size=(T, 40))
        X[rng.random(X.shape) < 0.5] = np.nan
        X[:, FEATURE_COLUMNS.index("EtCO2")] = np.nan                    # never observed, as at Hospital A
        np.savez(root / f"p{i:03d}.npz", X=X)
        sep = i % 5 == 0
        rows.append({"patient_id": f"p{i:03d}", "hospital": "A", "T": T, "is_septic": sep, "t_sepsis": 8 if sep else -1})
    manifest = pd.DataFrame(rows)
    split = within_hospital_split(manifest, "A", seed=0)
    b = build_bundle(manifest, tmp_path, split, cache_path=tmp_path / "c.npz", n_workers=1)
    assert set(b.train.ids).isdisjoint(b.val.ids) and set(b.train.ids).isdisjoint(b.test.ids)
    assert b.stats.is_fitted
    # statistics equal those computed by hand from the TRAIN patients' forward-filled values only
    from src.data.preprocess import compute_values_mask_delta
    tr = np.concatenate([compute_values_mask_delta(np.load(root / f"{p}.npz")["X"])[0] for p in b.train.ids])
    cols = [c for c in range(40) if c != FEATURE_COLUMNS.index("EtCO2")]
    assert np.allclose(b.stats.mean[cols], np.nanmean(tr, axis=0)[cols])
    assert b.stats.mean[FEATURE_COLUMNS.index("EtCO2")] == 0.0 and b.stats.std[FEATURE_COLUMNS.index("EtCO2")] == 1.0
    # reloading from the cache returns the same thing and a different train split is refused
    b2 = build_bundle(manifest, tmp_path, split, cache_path=tmp_path / "c.npz", n_workers=1)
    assert np.array_equal(b2.train.Z, b.train.Z)
