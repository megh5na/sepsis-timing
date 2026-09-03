"""Fetch and verify the PhysioNet/CinC Challenge 2019 training archives.

Downloads every per-patient .psv file from the two official training sets
(Hospital A, Hospital B) directly from PhysioNet's open-access file server.
No credentialing or data-use agreement is required for this dataset (see
spec §2.1).

Usage:
    python -m src.data.download --out data/raw [--workers 32]

The PhysioNet server exposes a plain Apache-style directory listing for
    https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/
    https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/
each row giving a filename and a byte size. We parse that listing to get
the authoritative file list (rather than hardcoding an expected patient
count), download each file, and verify it against the size reported in the
listing. This also gives us an early, independent check on the published
patient counts (20,336 for Hospital A, 20,000 for Hospital B) before any
parsing happens.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training"
HOSPITALS = {"A": "training_setA", "B": "training_setB"}


def list_remote_files(hospital: str, session: requests.Session) -> dict[str, int]:
    """Return {filename: expected_size_bytes} parsed from the directory index."""
    url = f"{BASE_URL}/{HOSPITALS[hospital]}/"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    files = {}
    for line in resp.text.splitlines():
        m = re.search(r'href="(p\d{6}\.psv)"', line)
        if not m:
            continue
        fname = m.group(1)
        size_m = re.search(r"(\d+)\s*$", line)
        size = int(size_m.group(1)) if size_m else -1
        files[fname] = size
    if not files:
        raise RuntimeError(f"Parsed zero files from {url} — listing format may have changed.")
    return files


def download_one(hospital: str, fname: str, expected_size: int, out_dir: Path,
                  session: requests.Session, retries: int = 3) -> tuple[str, bool, str]:
    dest = out_dir / fname
    if dest.exists() and dest.stat().st_size == expected_size:
        return fname, True, "cached"
    url = f"{BASE_URL}/{HOSPITALS[hospital]}/{fname}"
    last_err = ""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            if expected_size >= 0 and dest.stat().st_size != expected_size:
                last_err = (
                    f"size mismatch: got {dest.stat().st_size}, expected {expected_size}"
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            return fname, True, "downloaded"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    return fname, False, last_err


def download_hospital(hospital: str, out_root: Path, workers: int) -> dict:
    out_dir = out_root / HOSPITALS[hospital]
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    remote_files = list_remote_files(hospital, session)
    print(f"[hospital {hospital}] listing parsed: {len(remote_files)} files")

    ok, failed = 0, []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, hospital, fname, size, out_dir, session): fname
            for fname, size in remote_files.items()
        }
        for i, fut in enumerate(cf.as_completed(futures), 1):
            fname, success, msg = fut.result()
            if success:
                ok += 1
            else:
                failed.append((fname, msg))
            if i % 2000 == 0 or i == len(futures):
                print(f"[hospital {hospital}] {i}/{len(futures)} processed, {len(failed)} failed")

    return {
        "hospital": hospital,
        "n_listed": len(remote_files),
        "n_ok": ok,
        "n_failed": len(failed),
        "failed": failed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--hospitals", nargs="+", default=["A", "B"], choices=["A", "B"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for h in args.hospitals:
        summaries.append(download_hospital(h, args.out, args.workers))

    print("\n=== download summary ===")
    all_ok = True
    for s in summaries:
        print(f"Hospital {s['hospital']}: listed={s['n_listed']} ok={s['n_ok']} failed={s['n_failed']}")
        if s["n_failed"]:
            all_ok = False
            for fname, msg in s["failed"][:10]:
                print(f"    FAILED {fname}: {msg}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
