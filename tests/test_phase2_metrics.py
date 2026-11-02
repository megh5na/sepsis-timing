import numpy as np
import pytest

from src.eval.fast_utility import UtilityTable
from src.eval import phase2 as m


def test_operating_point_definitions():
    lengths = np.array([40, 40, 40, 40, 40])
    ts = np.array([20, 20, 20, -1, -1])
    #            in window  too early  never   alarmed(FP)  never
    alarm = np.array([12, 5, 40, 10, 40])
    op = m.operating_point(alarm, lengths, ts)
    assert op["sensitivity"] == pytest.approx(1 / 3)           # only the alarm at 12 is in [8, 23]
    assert op["n_alarmed"] == 3
    assert op["alarms_per_patient_day"] == pytest.approx(3 / (200 / 24))
    assert op["median_lead_time_h"] == pytest.approx(np.median([20 - 12, 20 - 5]))
    assert op["frac_alarm_before_window"] == pytest.approx(1 / 3)
    assert op["false_alarm_frac_nonseptic"] == pytest.approx(1 / 2)


def test_p1_curve_matches_pointwise_operating_points():
    rng = np.random.default_rng(0)
    lengths = rng.integers(10, 60, 40)
    ts = np.where(rng.random(40) < 0.4, np.array([rng.integers(6, t + 6) for t in lengths]), -1)
    scores = [rng.random(int(t)) for t in lengths]
    taus = np.linspace(0.3, 0.99, 12)
    curve = m.p1_curve(scores, lengths, ts, taus)
    tab = UtilityTable(lengths, ts)
    for j, tau in enumerate(taus):
        op = m.operating_point(tab.alarms_at_threshold(scores, tau), lengths, ts)
        assert curve["sensitivity"][j] == pytest.approx(op["sensitivity"])
        assert curve["alarms_per_patient_day"][j] == pytest.approx(op["alarms_per_patient_day"])
        a, b = curve["median_lead_time_h"][j], op["median_lead_time_h"]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)


def test_interp_reads_a_curve_and_refuses_to_extrapolate():
    x = np.array([0.1, 0.5, 0.9]); y = np.array([1.0, 2.0, 4.0])
    assert m.interp_on(x, y, 0.3) == pytest.approx(1.5)
    assert np.isnan(m.interp_on(x, y, 0.95))


def test_reliability_perfect_calibration_has_zero_ece_and_overconfidence_does_not():
    rng = np.random.default_rng(0)
    f = rng.random((20000, 3)); y = (rng.random((20000, 3)) < f).astype(float)
    w = np.ones_like(f)
    assert m.reliability(f, y, w)["ece"].max() < 0.02
    assert m.reliability(np.full((1000, 1), 0.9), np.zeros((1000, 1)), np.ones((1000, 1)))["ece"][0] == pytest.approx(0.9)
    # masked terms are ignored
    w0 = np.zeros_like(f)
    assert m.reliability(f, y, w0)["n"].sum() == 0


def test_coherence_violation_rate():
    F = np.array([[0.1, 0.2, 0.3], [0.1, 0.05, 0.3]])          # 4 adjacent pairs, 1 violation
    assert m.coherence_violation_rate(F) == pytest.approx(0.25)
    assert m.coherence_violation_rate(np.sort(np.random.rand(50, 12), axis=1)) == 0.0


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(0)
    N = 400
    best = rng.uniform(1, 2, N); inaction = np.zeros(N)
    a = np.stack([0.3 * best + rng.normal(0, 0.1, N), inaction, best], 1)
    b_better = a.copy(); b_better[:, 0] += 0.2 * best
    res = m.paired_bootstrap(a[None], b_better[None], n_boot=300)
    assert res["ci_excludes_zero"] and res["diff"] > 0.1
    b_same = a.copy()
    null = m.paired_bootstrap(a[None], b_same[None], n_boot=300)
    assert null["diff"] == 0 and not null["ci_excludes_zero"]
    # a noise-only difference should usually straddle zero
    b_noise = a.copy(); b_noise[:, 0] += rng.normal(0, 0.05, N)
    assert not m.paired_bootstrap(a[None], b_noise[None], n_boot=300)["ci_excludes_zero"]


def test_tail_ece_sees_overconfidence_that_plain_ece_hides():
    rng = np.random.default_rng(0)
    n = 20000
    f = np.where(rng.random(n) < 0.97, 0.01, 0.8)                 # mostly tiny predictions, a few confident ones
    y = np.where(f < 0.1, (rng.random(n) < 0.01), (rng.random(n) < 0.2)).astype(float)   # confident ones are right only 20%
    rel = m.reliability(f[:, None], y[:, None], np.ones((n, 1)))
    assert rel["ece"][0] < 0.03 < rel["ece_tail"][0]
    assert rel["ece_tail"][0] == pytest.approx(0.6, abs=0.05)
