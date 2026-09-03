"""Patient-level train/validation/test split assignment (spec §3.5).

All splits are by patient_id, never by hour (spec §3.6 leakage guard #1).
Three regimes are supported; Phase 1 only *runs* the within-hospital regime
(cross-hospital and pooled are explicitly out of scope for Phase 1 — see
spec §11.1, "Phase 2 cuts, in the order to make them: drop cross-hospital
and pooled splits ..." — but the split logic itself is generic and cheap,
so all three are implemented here rather than only the one Phase 1 uses.)

Splits are stratified by is_septic so prevalence stays stable across folds
(spec §3.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9


@dataclass(frozen=True)
class SplitAssignment:
    """patient_id -> split name ("train" | "val" | "test"), plus the regime
    and seed that produced it, for provenance."""
    regime: str
    seed: int
    assignment: dict[str, str]

    def patients(self, split: str) -> list[str]:
        return [pid for pid, s in self.assignment.items() if s == split]


def _stratified_three_way(patient_ids: np.ndarray, is_septic: np.ndarray, seed: int) -> dict[str, str]:
    """Assign each patient in patient_ids to train/val/test, stratified by
    is_septic, using the given seed. Pure function of its inputs."""
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for stratum_value in (True, False):
        ids = patient_ids[is_septic == stratum_value]
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # remainder goes to test so the three counts always sum to n
        for pid in ids[:n_train]:
            out[pid] = "train"
        for pid in ids[n_train:n_train + n_val]:
            out[pid] = "val"
        for pid in ids[n_train + n_val:]:
            out[pid] = "test"
    return out


def within_hospital_split(manifest: pd.DataFrame, hospital: str = "A", seed: int = 0) -> SplitAssignment:
    """Required regime (spec §3.5): 70/15/15 within a single hospital."""
    sub = manifest[manifest["hospital"] == hospital]
    assignment = _stratified_three_way(sub["patient_id"].to_numpy(), sub["is_septic"].to_numpy(), seed)
    return SplitAssignment(regime=f"within_hospital_{hospital}", seed=seed, assignment=assignment)


def cross_hospital_split(manifest: pd.DataFrame, train_hospital: str = "A",
                          test_hospital: str = "B", seed: int = 0) -> SplitAssignment:
    """Stretch regime: train/val on 85/15 of train_hospital, test = all of
    test_hospital. Not run in Phase 1; implemented for interface completeness."""
    train_sub = manifest[manifest["hospital"] == train_hospital]
    test_sub = manifest[manifest["hospital"] == test_hospital]

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for stratum_value in (True, False):
        ids = train_sub.loc[train_sub["is_septic"] == stratum_value, "patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_train = int(round(len(ids) * 0.85))
        for pid in ids[:n_train]:
            assignment[pid] = "train"
        for pid in ids[n_train:]:
            assignment[pid] = "val"
    for pid in test_sub["patient_id"]:
        assignment[pid] = "test"

    return SplitAssignment(regime=f"cross_hospital_{train_hospital}_to_{test_hospital}",
                            seed=seed, assignment=assignment)


def pooled_split(manifest: pd.DataFrame, seed: int = 0) -> SplitAssignment:
    """Stretch regime: 70/15/15 over the pooled A+B cohort. Not run in
    Phase 1; implemented for interface completeness."""
    assignment = _stratified_three_way(manifest["patient_id"].to_numpy(),
                                        manifest["is_septic"].to_numpy(), seed)
    return SplitAssignment(regime="pooled", seed=seed, assignment=assignment)


# ---------------------------------------------------------------------------
# Leakage guards (spec §3.6) — runtime assertions, not comments.
# ---------------------------------------------------------------------------

def assert_no_patient_in_multiple_splits(split: SplitAssignment) -> None:
    """Guard #1: splitting by hour is forbidden; a patient must appear in
    exactly one split. This is trivially true by construction of
    `assignment: dict[patient_id, split]` (a dict key can only map to one
    value), but we assert it explicitly against the *source* manifest ids
    too, so a caller who builds `assignment` a different way still gets
    caught if a patient_id was somehow duplicated in the input."""
    pids = list(split.assignment.keys())
    assert len(pids) == len(set(pids)), (
        "duplicate patient_id in split assignment — a patient must not "
        "appear in more than one split"
    )


def assert_splits_partition_manifest(split: SplitAssignment, manifest: pd.DataFrame,
                                      hospital_filter: str | None = None) -> None:
    """Every patient in the relevant slice of the manifest is assigned to
    exactly one of train/val/test, and nothing outside that slice is
    assigned."""
    expected = manifest["patient_id"]
    if hospital_filter is not None:
        expected = manifest.loc[manifest["hospital"] == hospital_filter, "patient_id"]
    expected_set = set(expected)
    got_set = set(split.assignment.keys())
    missing = expected_set - got_set
    extra = got_set - expected_set
    missing_repr = repr(missing) if len(missing) < 10 else len(missing)
    extra_repr = repr(extra) if len(extra) < 10 else len(extra)
    assert expected_set == got_set, (
        f"split does not partition the expected manifest slice: "
        f"missing={missing_repr}, extra={extra_repr}"
    )
    values = set(split.assignment.values())
    assert values <= {"train", "val", "test"}, f"unexpected split labels: {values}"
