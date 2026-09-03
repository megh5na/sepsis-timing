"""Train a single condition, single seed (spec repo layout: experiments/train.py).

Phase 1 conditions:
  B0 — rule-based qSOFA proxy, no training (src/decision/qsofa.py).
  B1 — GRU(64) encoder + sigmoid head, BCE against per-hour SepsisLabel,
       tuned threshold. This is the only condition that actually trains.

Training-hour selection, checkpoint-on-val-utility, early stopping,
gradient clipping, optimiser: all per spec §8 / configs/base.yaml.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SepsisDataset, collate_examples
from src.data.splits import within_hospital_split
from src.data.stats import TrainStats
from src.data.parse import load_patient_tensor, FEATURE_COLUMNS
from src.data.preprocess import compute_values_mask_delta
from src.decision.threshold import apply_threshold, first_alarm_hour, tune_threshold
from src.eval.harness import evaluate_fast
from src.models.encoder import GRUEncoder
from src.models.heads import SigmoidHead


def fit_train_stats(train_patient_ids: list[str], processed_root: Path) -> TrainStats:
    """Guard #2: fit strictly on the training split's raw VALUES block
    (forward-filled, pre-z-score), before any val/test data is read."""
    X_list = []
    for pid in train_patient_ids:
        X_raw = load_patient_tensor(processed_root, pid)
        values, _, _ = compute_values_mask_delta(X_raw)
        X_list.append(values)
    stats = TrainStats(n_channels=len(FEATURE_COLUMNS))
    stats.fit(X_list, channel_names=FEATURE_COLUMNS)
    return stats


class B1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GRUEncoder()
        self.head = SigmoidHead(hidden_size=self.encoder.hidden_size)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        h = self.encoder(Z)
        return self.head(h)  # (B, T) risk scores in [0, 1]


def masked_bce_loss(r: torch.Tensor, sepsis_label: torch.Tensor, train_mask: torch.Tensor,
                     length_mask: torch.Tensor) -> torch.Tensor:
    combined_mask = train_mask * length_mask
    bce = nn.functional.binary_cross_entropy(r, sepsis_label, reduction="none")
    denom = combined_mask.sum().clamp_min(1.0)
    return (bce * combined_mask).sum() / denom


@torch.no_grad()
def score_all_patients(model: B1Model, patient_ids: list[str], manifest: pd.DataFrame,
                        processed_root: Path, stats: TrainStats,
                        dataset: "SepsisDataset | None" = None) -> dict[str, np.ndarray]:
    """r[t] for every hour of every patient (no train_mask applied — this
    is inference, spec §3.3: evaluation covers all hours).

    Pass a pre-built `dataset` (e.g. the validation SepsisDataset built
    once outside the training loop) to reuse its cache across repeated
    calls — this is called once per epoch during training, and rebuilding
    a fresh SepsisDataset internally on every call would re-read every
    validation patient from disk and recompute VALUES/MASK/DELTA every
    single epoch for no reason (see SepsisDataset's docstring on why
    caching is safe here). If omitted, a fresh uncached dataset is built
    (fine for one-off scoring, e.g. in src/experiments/analyse.py)."""
    model.eval()
    ds = dataset if dataset is not None else SepsisDataset(patient_ids, manifest, processed_root, stats)
    scores = {}
    for i in range(len(ds)):
        ex = ds[i]
        Z = torch.from_numpy(ex.Z).float().unsqueeze(0)
        r = model(Z).squeeze(0).numpy()
        scores[ex.patient_id] = r
    return scores


def val_normalized_utility(scores: dict[str, np.ndarray], tau: float, manifest: pd.DataFrame) -> float:
    """Fast, in-memory path (src/eval/harness.py:evaluate_fast) — called up
    to n_threshold_candidates times per epoch, so the disk-based evaluate()
    (validated against the official scorer in tests/test_harness_validation.py,
    and agreeing with evaluate_fast exactly there) would dominate runtime."""
    patient_ids = list(scores.keys())

    def predict_fn(pid):
        return first_alarm_hour(scores[pid], tau)

    metrics = evaluate_fast(predict_fn, patient_ids, manifest, utility_only=True)
    return metrics["utility"]


def train_b1(
    manifest: pd.DataFrame,
    processed_root: Path,
    seed: int,
    max_epochs: int = 50,
    patience: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    grad_clip_norm: float = 1.0,
    n_threshold_candidates: int = 40,
    out_dir: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    split = within_hospital_split(manifest, hospital="A", seed=0)  # split assignment fixed across seeds
    train_ids = split.patients("train")
    val_ids = split.patients("val")

    stats = fit_train_stats(train_ids, processed_root)  # guard #2, enforced

    train_ds = SepsisDataset(train_ids, manifest, processed_root, stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_examples)
    # Built once and reused every epoch (see SepsisDataset / score_all_patients
    # docstrings) — rebuilding this from disk every epoch was the dominant
    # per-epoch cost before this fix.
    val_ds = SepsisDataset(val_ids, manifest, processed_root, stats)

    model = B1Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_utility = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            r = model(batch["Z"])
            loss = masked_bce_loss(r, batch["sepsis_label"], batch["train_mask"], batch["length_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        val_scores = score_all_patients(model, val_ids, manifest, processed_root, stats, dataset=val_ds)
        candidate_taus = np.linspace(0.01, 0.99, n_threshold_candidates)
        tune_result = tune_threshold(
            candidate_taus,
            lambda tau: val_normalized_utility(val_scores, tau, manifest),
            split_name="val",
        )
        val_utility = tune_result.objective_value

        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_utility": val_utility})
        print(f"[seed {seed}] epoch {epoch}: train_loss={np.mean(epoch_losses):.4f} val_utility={val_utility:.4f}")

        if val_utility > best_val_utility:
            best_val_utility = val_utility
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "tau": tune_result.threshold,
                "epoch": epoch,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[seed {seed}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state["model"])

    result = {
        "seed": seed,
        "best_epoch": best_state["epoch"],
        "best_val_utility": best_val_utility,
        "tau": best_state["tau"],
        "history": history,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"b1_seed{seed}.pt")
        with open(out_dir / f"b1_seed{seed}_meta.json", "w") as f:
            json.dump({"tau": best_state["tau"], "best_epoch": best_state["epoch"],
                       "best_val_utility": best_val_utility}, f, indent=2)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--condition", choices=["B0", "B1"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/checkpoints"))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.processed / "manifest.parquet")

    if args.condition == "B1":
        t0 = time.time()
        result = train_b1(manifest, args.processed, seed=args.seed, out_dir=args.out)
        print(f"done in {time.time() - t0:.1f}s: {result['best_val_utility']=:.4f} tau={result['tau']:.4f}")
    else:
        print("B0 is rule-based; nothing to train. See src/decision/qsofa.py.")


if __name__ == "__main__":
    main()
