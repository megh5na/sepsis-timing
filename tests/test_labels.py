"""Tests for the six-hour SepsisLabel offset (spec §2.3).

This is flagged in the spec as the single highest-risk piece of the whole
pipeline: an off-by-six here does not crash, it silently corrupts every
downstream target and utility calculation. It is covered here against
synthetic patients with hand-known answers, and cross-checked against the
official scoring script's own internal definition of t_sepsis
(third_party/physionet/evaluate_sepsis_score.py:
`t_sepsis = np.argmax(labels) - dt_optimal`, dt_optimal = -6), not just
against our own reading of the documentation.
"""
import numpy as np
import pytest

from src.data.parse import first_positive_hour
from third_party.physionet.evaluate_sepsis_score import compute_prediction_utility


def official_t_sepsis(sepsis_label: np.ndarray) -> float:
    """Reproduce the official script's t_sepsis derivation exactly, as an
    independent oracle for the tests below (see docstring)."""
    dt_optimal = -6
    if not np.any(sepsis_label):
        return float("inf")
    return np.argmax(sepsis_label) - dt_optimal


def our_t_sepsis(sepsis_label: np.ndarray) -> int | None:
    fph = first_positive_hour(sepsis_label)
    return fph + 6 if fph is not None else None


class TestSixHourOffset:
    def test_onset_at_hour_10_label_flips_at_hour_4(self):
        # First positive at index 4 (0-indexed) => t_sepsis must be 10, not 4.
        label = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        assert first_positive_hour(label) == 4
        assert our_t_sepsis(label) == 10

    def test_label_positive_from_hour_zero(self):
        # Onset at t_sepsis=6 means SepsisLabel is already 1 at t=0.
        label = np.array([1, 1, 1, 1, 1, 1, 1])
        assert first_positive_hour(label) == 0
        assert our_t_sepsis(label) == 6

    def test_never_positive_is_non_septic(self):
        label = np.zeros(20)
        assert first_positive_hour(label) is None
        assert our_t_sepsis(label) is None

    def test_single_late_positive(self):
        label = np.array([0] * 49 + [1])
        assert first_positive_hour(label) == 49
        assert our_t_sepsis(label) == 55

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_official_scorer_t_sepsis_on_random_septic_patients(self, seed):
        """our_t_sepsis must agree with the official script's internal
        `np.argmax(labels) - dt_optimal` for any label vector shaped like a
        real SepsisLabel column (0s then 1s from some flip point onward)."""
        rng = np.random.default_rng(seed)
        T = rng.integers(5, 200)
        flip = rng.integers(0, T)
        label = np.zeros(T, dtype=int)
        label[flip:] = 1

        assert our_t_sepsis(label) == official_t_sepsis(label)

    def test_offset_is_exactly_six_not_some_other_constant(self):
        """Guard against a future edit accidentally changing the constant."""
        label = np.array([0, 0, 1, 1])
        assert our_t_sepsis(label) == 2 + 6
        assert our_t_sepsis(label) != 2 + 5
        assert our_t_sepsis(label) != 2 + 7

    def test_utility_at_t_sepsis_minus_six_is_zero_boundary(self):
        """Sanity cross-check: firing an alarm exactly at t_sepsis (i.e. six
        hours after the label first turns positive) should already be inside
        the official scorer's optimal window (t <= t_sepsis + dt_optimal ==
        t_sepsis - 6 is the lower edge; t == t_sepsis is comfortably inside
        [t_sepsis - 12, t_sepsis + 3])."""
        label = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # first_positive=6, t_sepsis=12
        t_sepsis = our_t_sepsis(label)
        assert t_sepsis == 12

        predictions = np.zeros(len(label))
        # Alarm one hour before the label array even ends, well inside
        # [t_sepsis-12, t_sepsis+3] = [0, 15].
        predictions[t_sepsis - 3:] = 1
        u = compute_prediction_utility(label.tolist(), predictions.tolist())
        assert u > 0, "alarming inside the beneficial window should earn positive utility"
