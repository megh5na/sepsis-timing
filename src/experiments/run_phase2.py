from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.cache import build_bundle
from src.data.splits import within_hospital_split
from src.experiments.train_mh import train_run

ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
CACHE = PROCESSED / "bundle_A_seed0.npz"
RUNS = ROOT / "results" / "phase2" / "runs"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "phase2.yaml").read_text())


def get_bundle(cfg: dict | None = None):
    cfg = cfg or load_cfg()
    manifest = pd.read_parquet(PROCESSED / "manifest.parquet")
    split = within_hospital_split(manifest, hospital=cfg["split"]["hospital"], seed=cfg["split"]["seed"])
    return build_bundle(manifest, PROCESSED, split, cache_path=CACHE), manifest


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("cond"); t.add_argument("seed", type=int)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.cmd == "prepare":
        bundle, _ = get_bundle(cfg)
        print("train/val/test patients:", len(bundle.train), len(bundle.val), len(bundle.test))
        return

    torch.set_num_threads(args.threads)
    out = RUNS / args.cond / f"seed{args.seed}"
    if (out / "meta.json").exists() and not args.force:
        print(f"{out} already trained, skipping (use --force to retrain)")
        return
    bundle, _ = get_bundle(cfg)
    meta = train_run(args.cond, cfg["conditions"][args.cond], args.seed, bundle, out)
    print({k: meta[k] for k in ("name", "seed", "best_epoch", "tau_p1", "val_p1_utility_at_tau", "train_seconds")})


if __name__ == "__main__":
    main()
