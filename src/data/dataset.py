"""PyTorch Dataset/collate glue for B0/B1 (Phase 1). Bridges the pure
functions in parse.py / preprocess.py / stats.py to torch tensors and
batches.

Training-hour selection (spec §3.3, option (b), adopted — see
configs/base.yaml): for septic patients, only hours t <= t_sepsis
contribute to the training loss (post-onset hours are a degenerate
prediction task — SepsisLabel is trivially 1 — and dilute the gradient);
all hours contribute for non-septic patients. This is a *training-time*
mask only. Evaluation always covers every hour of every patient's stay
(spec §3.3: "Evaluation must still cover all hours, because the official
scorer does"), and nothing in this module or in src/eval/harness.py
applies train_mask at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.parse import load_patient_tensor
from src.data.preprocess import build_Z
from src.data.stats import TrainStats


@dataclass
class PatientExample:
    patient_id: str
    Z: np.ndarray            # (T, 120)
    sepsis_label: np.ndarray  # (T,) reconstructed from t_sepsis
    train_mask: np.ndarray    # (T,) training-hour-selection mask, §3.3 option (b)
    T: int
    t_sepsis: int | None
    is_septic: bool


def build_patient_example(
    patient_id: str, manifest_row: pd.Series, processed_root: Path, stats: TrainStats
) -> PatientExample:
    X_raw = load_patient_tensor(processed_root, patient_id)
    Z = build_Z(X_raw, stats)

    T = int(manifest_row["T"])
    is_septic = bool(manifest_row["is_septic"])
    t_sepsis = int(manifest_row["t_sepsis"]) if is_septic else None

    sepsis_label = np.zeros(T, dtype=np.float64)
    if is_septic:
        sepsis_label[max(0, t_sepsis - 6):] = 1.0

    if is_septic:
        train_mask = (np.arange(T) <= t_sepsis).astype(np.float64)
    else:
        train_mask = np.ones(T, dtype=np.float64)

    return PatientExample(
        patient_id=patient_id, Z=Z, sepsis_label=sepsis_label, train_mask=train_mask,
        T=T, t_sepsis=t_sepsis, is_septic=is_septic,
    )


class SepsisDataset(Dataset):
    """Caches each patient's built example (disk read + VALUES/MASK/DELTA
    computation) after the first access. This is safe specifically because
    `stats` is fixed for the lifetime of one SepsisDataset instance (fit
    once before construction, spec §3.6 guard #2) — Z is a deterministic
    function of (raw X, stats), so nothing changes between epochs and
    recomputing it from disk on every epoch (the model trains for up to 50
    epochs, spec §8) would be pure waste. A fresh TrainStats/SepsisDataset
    per training run still fits and reads the training split fresh, so this
    does not weaken guard #2 — it only avoids repeating identical work
    within one already-guarded run."""

    def __init__(self, patient_ids: list[str], manifest: pd.DataFrame, processed_root: Path, stats: TrainStats):
        self.patient_ids = patient_ids
        self.manifest = manifest.set_index("patient_id")
        self.processed_root = processed_root
        self.stats = stats
        self._cache: dict[str, PatientExample] = {}

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> PatientExample:
        pid = self.patient_ids[idx]
        cached = self._cache.get(pid)
        if cached is not None:
            return cached
        row = self.manifest.loc[pid]
        example = build_patient_example(pid, row, self.processed_root, self.stats)
        self._cache[pid] = example
        return example


def collate_examples(examples: list[PatientExample]) -> dict:
    B = len(examples)
    T_max = max(e.T for e in examples)
    C = examples[0].Z.shape[1]

    Z = torch.zeros(B, T_max, C, dtype=torch.float32)
    sepsis_label = torch.zeros(B, T_max, dtype=torch.float32)
    train_mask = torch.zeros(B, T_max, dtype=torch.float32)
    length_mask = torch.zeros(B, T_max, dtype=torch.float32)

    for i, e in enumerate(examples):
        t = e.T
        Z[i, :t] = torch.from_numpy(e.Z).float()
        sepsis_label[i, :t] = torch.from_numpy(e.sepsis_label).float()
        train_mask[i, :t] = torch.from_numpy(e.train_mask).float()
        length_mask[i, :t] = 1.0

    return {
        "Z": Z,
        "sepsis_label": sepsis_label,
        "train_mask": train_mask,
        "length_mask": length_mask,
        "patient_ids": [e.patient_id for e in examples],
        "lengths": [e.T for e in examples],
    }
