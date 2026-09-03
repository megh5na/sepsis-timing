"""U() must reproduce the official scorer's output (spec §5.1, stage 5 gate).

No bundled "example data" ships with the PhysioNet evaluation repos (see
src/decision/utility.py docstring for what was actually checked). The
validation here is instead: for many random synthetic patients, summing
U() over every hour equals the vendored, unmodified
compute_prediction_utility's per-patient total, exactly — which is a
stronger check than agreement on one fixed example would be, since it
exercises every branch of the piecewise utility curve (early TP, optimal
TP, late TP, early FN, late FN, FP, TN) across many onset times and
prediction patterns.
"""
import numpy as np
import pytest

from src.data.parse import first_positive_hour
from src.decision.utility import U, reference_total_utility, total_utility_for_patient


def _random_septic_patient(rng, T):
    flip = int(rng.integers(0, T))
    label = np.zeros(T, dtype=int)
    label[flip:] = 1
    fph = first_positive_hour(label)
    t_sepsis = fph + 6 if fph is not None else None
    return label, t_sepsis


class TestUAgreesWithOfficialScorer:
    @pytest.mark.parametrize("seed", range(50))
    def test_random_septic_patient_all_ones_prediction(self, seed):
        rng = np.random.default_rng(seed)
        T = int(rng.integers(10, 150))
        label, t_sepsis = _random_septic_patient(rng, T)
        predictions = np.ones(T, dtype=int)

        ours = total_utility_for_patient(predictions, t_sepsis, T)
        official = reference_total_utility(label, predictions)
        assert ours == pytest.approx(official, abs=1e-9)

    @pytest.mark.parametrize("seed", range(50))
    def test_random_septic_patient_random_prediction(self, seed):
        rng = np.random.default_rng(seed)
        T = int(rng.integers(10, 150))
        label, t_sepsis = _random_septic_patient(rng, T)
        predictions = (rng.random(T) < 0.3).astype(int)

        ours = total_utility_for_patient(predictions, t_sepsis, T)
        official = reference_total_utility(label, predictions)
        assert ours == pytest.approx(official, abs=1e-9)

    @pytest.mark.parametrize("seed", range(50))
    def test_random_septic_patient_alarm_once_persists(self, seed):
        """The realistic case: an alarm fires once and stays on (spec §5.3
        semantics), not i.i.d. random per-hour predictions."""
        rng = np.random.default_rng(seed)
        T = int(rng.integers(10, 150))
        label, t_sepsis = _random_septic_patient(rng, T)
        alarm_hour = int(rng.integers(0, T))
        predictions = np.zeros(T, dtype=int)
        predictions[alarm_hour:] = 1

        ours = total_utility_for_patient(predictions, t_sepsis, T)
        official = reference_total_utility(label, predictions)
        assert ours == pytest.approx(official, abs=1e-9)

    @pytest.mark.parametrize("seed", range(30))
    def test_random_non_septic_patient(self, seed):
        rng = np.random.default_rng(seed)
        T = int(rng.integers(10, 150))
        label = np.zeros(T, dtype=int)
        t_sepsis = None
        predictions = (rng.random(T) < 0.2).astype(int)

        ours = total_utility_for_patient(predictions, t_sepsis, T)
        official = reference_total_utility(label, predictions)
        assert ours == pytest.approx(official, abs=1e-9)

    def test_non_septic_never_alarm_is_zero(self):
        T = 20
        predictions = np.zeros(T, dtype=int)
        assert total_utility_for_patient(predictions, None, T) == 0.0

    def test_non_septic_false_alarm_penalised_every_hour_it_persists(self):
        """Documents a real, easy-to-miss property of the official scorer:
        for non-septic patients, u_fp applies at EVERY hour predictions==1,
        not once per alarm. If a decision rule leaves the alarm on, it pays
        u_fp repeatedly."""
        T = 10
        predictions = np.ones(T, dtype=int)
        got = total_utility_for_patient(predictions, None, T)
        assert got == pytest.approx(10 * (-0.05), abs=1e-9)

    def test_alarm_in_optimal_window_earns_max_utility(self):
        # t_sepsis - 6 == t_sepsis + dt_optimal is inside the optimal region
        t_sepsis = 20
        predictions = np.zeros(30, dtype=int)
        predictions[t_sepsis - 6] = 1
        got = U(1, t_sepsis - 6, t_sepsis)
        assert got == pytest.approx(1.0, abs=1e-9)  # max_u_tp

    def test_alarm_too_early_earns_less_than_max(self):
        t_sepsis = 20
        # dt_early is the earliest hour that earns any positive utility at
        # all: the linear ramp evaluates to exactly 0 there (rising to
        # max_u_tp=1 at dt_optimal), so U at the boundary is 0 — strictly
        # less than the max_u_tp=1 earned inside the optimal window.
        got_at_dt_early = U(1, t_sepsis - 12, t_sepsis)
        got_at_dt_optimal = U(1, t_sepsis - 6, t_sepsis)
        assert got_at_dt_early == pytest.approx(0.0, abs=1e-9)
        assert got_at_dt_optimal == pytest.approx(1.0, abs=1e-9)
        assert got_at_dt_early < got_at_dt_optimal

    def test_alarm_before_dt_early_is_clipped_to_u_fp(self):
        # One hour earlier than dt_early: the raw linear ramp would go
        # negative, but the official formula clips it at u_fp (-0.05) via
        # max(..., u_fp) — this is what "predicting too early earns a small
        # penalty, not an unbounded one" means in practice.
        t_sepsis = 20
        got = U(1, t_sepsis - 13, t_sepsis)
        assert got == pytest.approx(-0.05, abs=1e-9)

    def test_missed_window_entirely_is_worst_case(self):
        t_sepsis = 20
        # far past dt_late -> contributes 0, but the FN inside the window
        # is the actually-penalised case:
        got_far_fn = U(0, t_sepsis + 3, t_sepsis)  # at the late boundary, FN
        assert got_far_fn == pytest.approx(-2.0, abs=1e-9)  # min_u_fn
