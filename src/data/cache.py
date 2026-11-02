# builds and caches the tensors used for training/validation/testing
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.parse import FEATURE_COLUMNS, load_patient_tensor
from src.data.preprocess import build_multi_horizon_targets, compute_values_mask_delta
from src.data.splits import SplitAssignment, assert_no_patient_in_multiple_splits
from src.data.stats import TrainStats


def _one_patient(args):
    processed_root, pid = args
    X = load_patient_tensor(Path(processed_root), pid)
    values, mask, delta = compute_values_mask_delta(X)
    return pid, values, mask.astype(np.float32), delta.astype(np.float32)


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


@dataclass
class SplitTensors:
    name: str
    ids: list[str]
    lengths: np.ndarray        # (N,)
    offsets: np.ndarray        # (N+1,)
    Z: np.ndarray              # (sum T, 120) float32, [VALUES_z | MASK | DELTA]
    t_sepsis: np.ndarray       # (N,) int, -1 if non-septic
    is_septic: np.ndarray      # (N,) bool

    def __len__(self):
        return len(self.ids)

    def patient_Z(self, i: int, use_mask_delta: bool = True) -> np.ndarray:
        z = self.Z[self.offsets[i]:self.offsets[i + 1]]
        return z if use_mask_delta else z[:, :40]


@dataclass
class DataBundle:
    train: SplitTensors
    val: SplitTensors
    test: SplitTensors
    stats: TrainStats


def build_bundle(manifest: pd.DataFrame, processed_root: Path, split: SplitAssignment,
                 cache_path: Path | None = None, n_workers: int = 8) -> DataBundle:
    assert_no_patient_in_multiple_splits(split)             # guard 1, enforced again at use
    train_ids, val_ids, test_ids = split.patients("train"), split.patients("val"), split.patients("test")
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids)) and not (set(val_ids) & set(test_ids)), \
        "a patient appears in more than one split"
    digest = _ids_digest(train_ids)

    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as d:
            if str(d["train_digest"]) == digest:
                return _bundle_from_npz(d)

    all_ids = train_ids + split.patients("val") + split.patients("test")
    with Pool(n_workers) as pool:
        res = pool.map(_one_patient, [(str(processed_root), p) for p in all_ids], chunksize=64)
    by_id = {r[0]: r for r in res}

    # guard #2: statistics from the train patients' forward-filled values only
    stats = TrainStats(n_channels=len(FEATURE_COLUMNS))
    stats.fit([by_id[p][1] for p in train_ids], channel_names=FEATURE_COLUMNS)

    mi = manifest.set_index("patient_id")
    splits = {}
    for name in ("train", "val", "test"):
        ids = split.patients(name)
        zs, lengths = [], []
        for p in ids:
            _, values, mask, delta = by_id[p]
            vz = stats.impute_and_zscore(values).astype(np.float32)
            zs.append(np.concatenate([vz, mask, delta], axis=-1))
            lengths.append(values.shape[0])
        lengths = np.array(lengths)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        rows = mi.loc[ids]
        splits[name] = SplitTensors(
            name=name, ids=ids, lengths=lengths, offsets=offsets, Z=np.concatenate(zs, axis=0),
            t_sepsis=np.where(rows["is_septic"].to_numpy(), rows["t_sepsis"].to_numpy(), -1).astype(int),
            is_septic=rows["is_septic"].to_numpy().astype(bool),
        )
        assert np.array_equal(lengths, rows["T"].to_numpy()), "manifest T disagrees with parsed lengths"

    bundle = DataBundle(train=splits["train"], val=splits["val"], test=splits["test"], stats=stats)
    if cache_path is not None:
        _save(bundle, cache_path, digest)
    return bundle


def _save(b: DataBundle, path: Path, digest: str) -> None:
    out = {"train_digest": np.array(digest),
           "stats_median": b.stats.median, "stats_mean": b.stats.mean, "stats_std": b.stats.std}
    for s in (b.train, b.val, b.test):
        out[f"{s.name}_ids"] = np.array(s.ids)
        out[f"{s.name}_lengths"] = s.lengths
        out[f"{s.name}_Z"] = s.Z
        out[f"{s.name}_t_sepsis"] = s.t_sepsis
        out[f"{s.name}_is_septic"] = s.is_septic
    np.savez(path, **out)


def _bundle_from_npz(d) -> DataBundle:
    stats = TrainStats(n_channels=len(FEATURE_COLUMNS))
    stats.median_, stats.mean_, stats.std_, stats._fitted = d["stats_median"], d["stats_mean"], d["stats_std"], True
    sp = {}
    for name in ("train", "val", "test"):
        lengths = d[f"{name}_lengths"]
        sp[name] = SplitTensors(
            name=name, ids=[str(x) for x in d[f"{name}_ids"]], lengths=lengths,
            offsets=np.concatenate([[0], np.cumsum(lengths)]), Z=d[f"{name}_Z"],
            t_sepsis=d[f"{name}_t_sepsis"], is_septic=d[f"{name}_is_septic"],
        )
    return DataBundle(train=sp["train"], val=sp["val"], test=sp["test"], stats=stats)


def make_batch(split: SplitTensors, idx: list[int], K: int | None = None,
               use_mask_delta: bool = True) -> dict:
    D = 120 if use_mask_delta else 40
    B = len(idx)
    T_max = int(max(split.lengths[i] for i in idx))
    Z = torch.zeros(B, T_max, D)
    length_mask = torch.zeros(B, T_max)
    train_mask = torch.zeros(B, T_max)
    y = torch.zeros(B, T_max, K) if K else None
    mask_k = torch.zeros(B, T_max, K) if K else None
    for b, i in enumerate(idx):
        T = int(split.lengths[i])
        Z[b, :T] = torch.from_numpy(split.patient_Z(i, use_mask_delta))
        length_mask[b, :T] = 1.0
        ts = int(split.t_sepsis[i]) if split.is_septic[i] else None
        train_mask[b, :T] = 1.0 if ts is None else torch.from_numpy((np.arange(T) <= ts).astype(np.float32))
        if K:
            yy, mm = build_multi_horizon_targets(T, ts, K)
            y[b, :T] = torch.from_numpy(yy.astype(np.float32))
            mask_k[b, :T] = torch.from_numpy(mm.astype(np.float32))
    return {"Z": Z, "length_mask": length_mask, "train_mask": train_mask, "y": y, "mask_k": mask_k,
            "lengths": [int(split.lengths[i]) for i in idx]}
