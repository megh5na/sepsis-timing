from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.cache import DataBundle, SplitTensors, make_batch
from src.decision.stopping import P1_HORIZON
from src.decision.threshold import tune_threshold
from src.eval.fast_utility import UtilityTable
from src.models.losses import masked_multihorizon_bce
from src.models.model import SepsisModel

VAL_SELECTION_TAUS = np.linspace(0.01, 0.99, 99)
FINAL_TAUS = np.linspace(0.001, 0.999, 999)


def weights_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def score_split(model: SepsisModel, split: SplitTensors, use_mask_delta: bool = True,
                batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = np.zeros((int(split.offsets[-1]), model.K), dtype=np.float32)
    order = np.argsort(split.lengths)
    for s in range(0, len(order), batch_size):
        idx = [int(i) for i in order[s:s + batch_size]]
        b = make_batch(split, idx, K=None, use_mask_delta=use_mask_delta)
        F, _ = model(b["Z"])
        for j, i in enumerate(idx):
            T = int(split.lengths[i])
            out[split.offsets[i]:split.offsets[i] + T] = F[j, :T].numpy()
    return out


def patient_scores(F_all: np.ndarray, split: SplitTensors, col: int) -> list[np.ndarray]:
    return [F_all[split.offsets[i]:split.offsets[i + 1], col] for i in range(len(split))]


def p1_sweep(F_all: np.ndarray, split: SplitTensors, table: UtilityTable, taus: np.ndarray) -> np.ndarray:
    return table.sweep_thresholds(patient_scores(F_all, split, P1_HORIZON - 1), taus)


def train_run(name: str, cond: dict, seed: int, bundle: DataBundle, out_dir: Path,
              max_epochs: int = 50, patience: int = 8, lr: float = 1e-3, batch_size: int = 64,
              clip: float = 1.0, log=print) -> dict:
    K, hidden, use_md = cond["K"], cond["hidden"], cond["mask_delta"]
    D = 120 if use_md else 40
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SepsisModel(cond["head"], K, hidden_size=hidden, input_size=D, seed=seed)
    n_params = sum(p.numel() for p in model.parameters())
    torch.manual_seed(seed)                       # same dropout stream whatever the head
    rng = np.random.default_rng(seed)             # same batch order whatever the head
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    val_table = UtilityTable.from_split(bundle.val)
    train, val = bundle.train, bundle.val
    best = {"u": -np.inf, "epoch": -1, "state": None}
    history, bad, t0 = [], 0, time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(len(train))
        losses = []
        for s in range(0, len(perm), batch_size):
            b = make_batch(train, [int(i) for i in perm[s:s + batch_size]], K=K, use_mask_delta=use_md)
            _, logits = model(b["Z"])
            loss = masked_multihorizon_bce(logits, b["y"], b["mask_k"], b["length_mask"], b["train_mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())

        F_val = score_split(model, val, use_md)
        u = float(p1_sweep(F_val, val, val_table, VAL_SELECTION_TAUS).max())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_p1_utility": u,
                        "elapsed_s": round(time.time() - t0, 1)})
        log(f"[{name} seed{seed}] epoch {epoch:2d} loss {np.mean(losses):.4f} val P1 utility {u:.4f} "
            f"({time.time() - t0:.0f}s)")
        if u > best["u"]:
            best = {"u": u, "epoch": epoch, "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"[{name} seed{seed}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best["state"])
    F_val = score_split(model, val, use_md)
    F_test = score_split(model, bundle.test, use_md)

    # tau for P1: tuned on validation only (guard 4), fine grid
    sweep = dict(zip(FINAL_TAUS.tolist(), p1_sweep(F_val, val, val_table, FINAL_TAUS).tolist()))
    tuned = tune_threshold(FINAL_TAUS, lambda t: sweep[float(t)], split_name="val")

    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez_compressed(out_dir / "preds_val.npz", F=F_val)
    np.savez_compressed(out_dir / "preds_test.npz", F=F_test)
    meta = {"name": name, "seed": seed, "cond": cond, "n_params": n_params,
            "best_epoch": best["epoch"], "best_val_p1_utility_selection_grid": best["u"],
            "tau_p1": tuned.threshold, "val_p1_utility_at_tau": tuned.objective_value,
            "weights_sha256": weights_sha256(model), "epochs_run": len(history),
            "torch_threads": torch.get_num_threads(), "train_seconds": round(time.time() - t0, 1),
            "training": {"optimizer": "adam", "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
                         "patience": patience, "grad_clip": clip, "hour_selection": "b"}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return meta
