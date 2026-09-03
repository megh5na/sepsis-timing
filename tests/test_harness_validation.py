"""Stage 4 gate: harness output must match the official scorer exactly
(spec §7.1, §10.1 stage 4).

No bundled "example data" ships with the official evaluation repos for this
purpose (checked directly — see src/decision/utility.py docstring and
third_party/physionet/PROVENANCE.md). Validation here instead runs the
harness's full write-files-then-call-vendored-scorer path against:

  1. Many synthetic patients under several prediction strategies (never
     alarm / always alarm from hour 0 / alarm exactly at t_sepsis / random
     alarm hour), checked two ways:
       (a) our harness's call to the vendored evaluate_sepsis_score()
           matches an INDEPENDENT subprocess invocation of the same
           vendored script's documented CLI, on the exact files our
           harness wrote — this validates the file-writing/matching logic,
           not just "we imported the right function".
       (b) our own utility.py-based cross-check (utility_cross_check)
           agrees with the vendored scorer's utility number — this
           validates utility.py independently of the vendored scorer.
  2. A real slice of downloaded PhysioNet patient files (skipped if the
     data hasn't been downloaded yet), under the same strategies, as a
     smoke test against real missingness/length patterns.

If ANY of these disagree, the spec's instruction is to stop and report
before writing model code — these tests are exactly that gate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.harness import evaluate, evaluate_fast

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_SCRIPT = REPO_ROOT / "third_party" / "physionet" / "evaluate_sepsis_score.py"


def _synthetic_manifest(n=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        T = int(rng.integers(10, 120))
        is_septic = bool(rng.random() < 0.4)
        if is_septic:
            first_positive = int(rng.integers(0, T))
            t_sepsis = first_positive + 6
        else:
            t_sepsis = -1
        rows.append(
            {
                "patient_id": f"syn_{i:04d}",
                "hospital": "A",
                "T": T,
                "t_sepsis": t_sepsis,
                "is_septic": is_septic,
            }
        )
    return pd.DataFrame(rows)


def _strategy_never_alarm(manifest):
    return lambda pid: None


def _strategy_always_alarm_hour_zero(manifest):
    return lambda pid: 0


def _strategy_alarm_at_true_t_sepsis(manifest):
    idx = manifest.set_index("patient_id")

    def fn(pid):
        row = idx.loc[pid]
        return int(row["t_sepsis"]) if row["is_septic"] else None

    return fn


def _strategy_random_alarm(manifest, seed=1):
    """Deterministic *per patient_id*, not a shared stateful RNG walked
    across calls — predict_fn must be idempotent (safe to call once per
    patient across multiple full passes, e.g. evaluate() then
    evaluate_fast() in the same test), not merely reproducible given a
    fixed call order."""
    idx = manifest.set_index("patient_id")

    def fn(pid):
        T = int(idx.loc[pid, "T"])
        rng = np.random.default_rng(abs(hash((pid, seed))) % (2**32))
        if rng.random() < 0.5:
            return None
        return int(rng.integers(0, T))

    return fn


STRATEGIES = {
    "never_alarm": _strategy_never_alarm,
    "always_alarm_hour_zero": _strategy_always_alarm_hour_zero,
    "alarm_at_true_t_sepsis": _strategy_alarm_at_true_t_sepsis,
    "random_alarm": _strategy_random_alarm,
}


def _run_official_cli(label_dir: Path, pred_dir: Path) -> tuple[float, float, float, float, float]:
    result = subprocess.run(
        [sys.executable, str(VENDORED_SCRIPT), str(label_dir), str(pred_dir)],
        capture_output=True, text=True, check=True,
    )
    lines = result.stdout.strip().splitlines()
    values = lines[-1].split("|")
    return tuple(float(v) for v in values)


class TestHarnessMatchesOfficialScorer:
    @pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
    def test_synthetic_patients_all_strategies(self, tmp_path, strategy_name):
        manifest = _synthetic_manifest(n=40, seed=hash(strategy_name) % 1000)
        predict_fn = STRATEGIES[strategy_name](manifest)
        patient_ids = manifest["patient_id"].tolist()

        workdir = tmp_path / strategy_name
        metrics = evaluate(predict_fn, patient_ids, manifest, workdir=workdir)

        official_cli = _run_official_cli(workdir / "labels", workdir / "predictions")
        auroc, auprc, accuracy, f_measure, utility = official_cli

        assert metrics["auroc"] == pytest.approx(auroc, abs=1e-9)
        assert metrics["auprc"] == pytest.approx(auprc, abs=1e-9)
        assert metrics["accuracy"] == pytest.approx(accuracy, abs=1e-9)
        assert metrics["f_measure"] == pytest.approx(f_measure, abs=1e-9)
        assert metrics["utility"] == pytest.approx(utility, abs=1e-9)

        # Independent utility.py cross-check must also agree.
        assert metrics["utility_agrees"], (
            f"our utility.py cross-check ({metrics['utility_cross_check']}) "
            f"disagrees with the vendored scorer ({metrics['utility']}) "
            f"for strategy={strategy_name}"
        )

    @pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
    def test_evaluate_fast_agrees_with_disk_based_evaluate(self, tmp_path, strategy_name):
        """The in-memory path used for per-epoch threshold tuning
        (src/experiments/train.py) must not be a second, divergent
        implementation — it must match the disk-based, officially-validated
        evaluate() exactly."""
        manifest = _synthetic_manifest(n=40, seed=(hash(strategy_name) + 1) % 1000)
        predict_fn = STRATEGIES[strategy_name](manifest)
        patient_ids = manifest["patient_id"].tolist()

        slow = evaluate(predict_fn, patient_ids, manifest, workdir=tmp_path / "slow")
        fast = evaluate_fast(predict_fn, patient_ids, manifest)

        for key in ("auroc", "auprc", "accuracy", "f_measure", "utility"):
            assert fast[key] == pytest.approx(slow[key], abs=1e-9), key

    @pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
    def test_evaluate_fast_utility_only_agrees_too(self, tmp_path, strategy_name):
        """The utility_only=True shortcut (used by per-epoch threshold
        tuning) must report the same utility as the full path."""
        manifest = _synthetic_manifest(n=40, seed=(hash(strategy_name) + 2) % 1000)
        predict_fn = STRATEGIES[strategy_name](manifest)
        patient_ids = manifest["patient_id"].tolist()

        full = evaluate_fast(predict_fn, patient_ids, manifest, utility_only=False)
        fast_only = evaluate_fast(predict_fn, patient_ids, manifest, utility_only=True)
        assert fast_only["utility"] == pytest.approx(full["utility"], abs=1e-9)
        assert set(fast_only.keys()) == {"utility", "n_patients"}

    def test_never_alarm_utility_is_exactly_zero(self, tmp_path):
        """Property check independent of the vendored script: inaction's
        normalised utility is 0 by construction of the normalisation
        formula (observed == inaction in this case)."""
        manifest = _synthetic_manifest(n=25, seed=7)
        metrics = evaluate(lambda pid: None, manifest["patient_id"].tolist(), manifest,
                            workdir=tmp_path / "never")
        assert metrics["utility"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["utility_cross_check"] == pytest.approx(0.0, abs=1e-9)


class TestHarnessOnRealDownloadedData:
    """Smoke test against real PhysioNet files, if they've been downloaded
    yet (this file may run before the full download finishes)."""

    def _real_slice_manifest(self, n=25):
        from src.data.parse import parse_patient_file

        raw_dir = REPO_ROOT / "data" / "raw" / "training_setA"
        if not raw_dir.exists():
            pytest.skip("raw data not downloaded yet")
        files = sorted(raw_dir.glob("p*.psv"))[:n]
        if len(files) < n:
            pytest.skip("not enough raw data downloaded yet")

        rows = []
        for f in files:
            p = parse_patient_file(f, hospital="A")
            rows.append(
                {
                    "patient_id": p.patient_id,
                    "hospital": p.hospital,
                    "T": p.T,
                    "t_sepsis": p.t_sepsis if p.t_sepsis is not None else -1,
                    "is_septic": p.is_septic,
                }
            )
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
    def test_real_patient_slice_all_strategies(self, tmp_path, strategy_name):
        manifest = self._real_slice_manifest(n=25)
        predict_fn = STRATEGIES[strategy_name](manifest)
        patient_ids = manifest["patient_id"].tolist()

        workdir = tmp_path / strategy_name
        metrics = evaluate(predict_fn, patient_ids, manifest, workdir=workdir)
        official_cli = _run_official_cli(workdir / "labels", workdir / "predictions")
        auroc, auprc, accuracy, f_measure, utility = official_cli

        assert metrics["auroc"] == pytest.approx(auroc, abs=1e-9)
        assert metrics["auprc"] == pytest.approx(auprc, abs=1e-9)
        assert metrics["accuracy"] == pytest.approx(accuracy, abs=1e-9)
        assert metrics["f_measure"] == pytest.approx(f_measure, abs=1e-9)
        assert metrics["utility"] == pytest.approx(utility, abs=1e-9)
        assert metrics["utility_agrees"]
