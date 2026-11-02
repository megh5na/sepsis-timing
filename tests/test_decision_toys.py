import numpy as np
import pytest

from src.decision.stopping import (alarm_now_from_V, expected_utility_V, margin, onset_distribution,
                                   p2_alarm_hour)
from third_party.physionet.evaluate_sepsis_score import compute_prediction_utility

K = 12


def official_V(p, p_none, K=K, t0=20):
    L = t0 + K + 3 + 10
    V = np.zeros(K + 1)
    for s in range(K + 1):
        pred = np.zeros(L); pred[t0 + s:] = 1
        for k in range(1, K + 1):
            lab = np.zeros(L); lab[t0 + k - 6:] = 1
            V[s] += p[k - 1] * compute_prediction_utility(list(lab), list(pred))
        V[s] += p_none * compute_prediction_utility(list(np.zeros(L)), list(pred))
    return V


def peaked(k):
    p = np.zeros(K); p[k - 1] = 1.0
    return p, 0.0


def check_against_official(p, p_none):
    V = expected_utility_V(p, p_none)
    Vo = official_V(p, p_none)
    assert np.allclose(V[0] - V, Vo[0] - Vo, atol=1e-9), "our V disagrees with the official scorer"
    return V


def row_from(p, p_none):
    assert p.sum() + p_none == pytest.approx(1.0)
    return np.cumsum(p)


# --------------------------------------------------------------------------
# Case 1: onset sharply peaked 6 hours ahead
# --------------------------------------------------------------------------
class TestCase1PeakedSixAhead:
    def test_alarm_now_with_hand_computed_margin(self):
        p, pn = peaked(6)
        V = check_against_official(p, pn)
        # hour t gain with onset at t+6 is d = -6: (-6+12)/6 = 1, and every later hour is positive,
        # so waiting only loses utility: V falls for s = 1..10 and is flat after (gain is 0 past onset+3).
        assert V[0] - V[1] == pytest.approx(1.0)
        assert np.all(np.diff(V)[:10] < 0) and np.all(np.diff(V) <= 1e-12)
        assert alarm_now_from_V(V)
        assert margin(row_from(p, pn)[None])[0] == pytest.approx(1.0)

    def test_trajectory_alarm_hour_known_analytically(self):
        tau, T = 20, 30
        F = np.zeros((T, K))
        for t in range(T):
            k = tau - t
            if 1 <= k <= K:
                F[t, k - 1:] = 1.0            # all mass at k, cumulative from k onward
        assert p2_alarm_hour(F) == 8


# --------------------------------------------------------------------------
# Case 2: onset distribution flat across all K
# --------------------------------------------------------------------------
class TestCase2Flat:
    def test_alarm_now_with_hand_computed_margin(self):
        p = np.full(K, 1 / K); pn = 0.0
        V = check_against_official(p, pn)
        # gain at hour t = mean over onsets k=1..12 of gain(d=-k):
        # d=-12..-6 -> 0,1,2,3,4,5,6 over 6 = 3.5 ; d=-5..-1 -> (10+11+12+13+14)/9 = 60/9
        expected_G0 = (3.5 + 60 / 9) / 12
        assert V[0] - V[1] == pytest.approx(expected_G0)
        assert expected_G0 == pytest.approx(0.8472222, abs=1e-6)
        # later hours (d up to +10) have gain >= 0 for every k, so V never rises: alarm now
        assert alarm_now_from_V(V)
        assert margin(row_from(p, pn)[None])[0] == pytest.approx(expected_G0)


# --------------------------------------------------------------------------
# Case 3: p_none near 1 -> never alarm
# --------------------------------------------------------------------------
class TestCase3NoOnset:
    def test_p_none_one_costs_exactly_u_fp_per_hour_of_alarm(self):
        p = np.zeros(K); pn = 1.0
        V = check_against_official(p, pn)
        assert np.allclose(V[0] - V, -0.05 * np.arange(K + 1))      # each earlier hour costs 0.05
        assert not alarm_now_from_V(V)
        assert margin(np.zeros((1, K)))[0] == pytest.approx(-0.05 * K)

    def test_never_alarms_on_a_whole_stay(self):
        assert p2_alarm_hour(np.zeros((48, K))) is None

    def test_p_none_099_still_never_alarms(self):
        p = np.full(K, 0.01 / K); pn = 0.99
        V = check_against_official(p, pn)
        assert V[0] - V[1] == pytest.approx(0.01 * 0.8472222 - 0.0495, abs=1e-6)
        assert not alarm_now_from_V(V)
        assert p2_alarm_hour(np.tile(np.cumsum(p), (40, 1))) is None


# --------------------------------------------------------------------------
# Case 4: onset peaked at k = 1 -> alarm immediately
# --------------------------------------------------------------------------
class TestCase4PeakedAtOne:
    def test_alarm_immediately(self):
        p, pn = peaked(1)
        V = check_against_official(p, pn)
        # d = -1: gain (-1 + 15) / 9 = 14/9, later hours d=0..: 15/9, 16/9, ... all positive
        assert V[0] - V[1] == pytest.approx(14 / 9)
        assert alarm_now_from_V(V)
        assert margin(row_from(p, pn)[None])[0] == pytest.approx(14 / 9)

    def test_alarm_hour_is_the_first_hour(self):
        F = np.ones((25, K))                      # belief at every hour: onset next hour
        assert p2_alarm_hour(F) == 0


# --------------------------------------------------------------------------
# Extra structure the spec describes in words
# --------------------------------------------------------------------------
class TestDeferralAndInvariants:
    def test_defers_while_expected_utility_rises_then_fires_at_the_peak(self):
        p = np.zeros(K); p[K - 1] = 0.08; pn = 0.92
        V = check_against_official(p, pn)
        assert int(np.argmax(V)) == 4
        assert V[0] - V[1] == pytest.approx(-0.046)
        assert V[1] - V[2] == pytest.approx(-0.0326667, abs=1e-6)
        assert V[3] - V[4] == pytest.approx(-0.0060, abs=1e-6)
        assert V[4] - V[5] == pytest.approx(+0.0073333, abs=1e-6)
        assert not alarm_now_from_V(V)            # still rising: defer

    def test_closed_form_equals_literal_definition(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            Kk = int(rng.choice([6, 12, 18]))
            w = rng.dirichlet(np.ones(Kk + 1) * rng.choice([0.1, 1.0, 5.0]))
            p, pn = w[:Kk], w[Kk]
            V = expected_utility_V(p, pn)
            m = margin(np.cumsum(p)[None])[0]
            assert (V[0] - V[1:]).min() == pytest.approx(m, abs=1e-9)
            assert alarm_now_from_V(V) == (m >= -1e-12)

    def test_at_most_one_alarm_per_stay(self):
        F = np.ones((30, K))
        assert isinstance(p2_alarm_hour(F), int)   # a single hour, never a list

    def test_decision_at_t_uses_only_row_t(self):
        rng = np.random.default_rng(2)
        F = np.sort(rng.random((20, K)), axis=1)
        F2 = F.copy(); F2[10:] = np.sort(rng.random((10, K)), axis=1)
        assert np.allclose(margin(F)[:10], margin(F2)[:10])

    def test_end_to_end_causality_through_the_encoder(self):
        import torch
        from src.data.leakage import assert_causal_in_time
        from src.models.model import SepsisModel
        m = SepsisModel("monotone", K, seed=0).eval()

        def pipeline(x):
            with torch.no_grad():
                F, _ = m(torch.tensor(x, dtype=torch.float32)[None])
            return margin(F[0].numpy().astype(np.float64))[:, None]

        assert_causal_in_time(pipeline, np.random.default_rng(0).normal(size=(40, 120)), n_trials=6, atol=1e-5)

    def test_onset_distribution_repair_makes_independent_head_usable(self):
        F = np.array([[0.2, 0.1, 0.5, 0.4]])           # non-monotone, as A1 can produce
        p_raw, _ = onset_distribution(F)
        assert (p_raw < 0).any()
        p, pn = onset_distribution(F, repair=True)
        assert (p >= 0).all() and pn[0] == pytest.approx(0.5)
