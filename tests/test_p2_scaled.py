import numpy as np

from src.decision.stopping import margin, p2_alarm_hour

K = 12


def random_F(rng, T=60):
    z = np.sort(rng.normal(-3.0, 1.5, (T, K)), axis=1)
    return 1 / (1 + np.exp(-z))


def test_scale_one_is_the_primary_rule():
    rng = np.random.default_rng(0)
    for _ in range(20):
        F = random_F(rng)
        assert np.array_equal(margin(F), margin(F, fa_scale=1.0))
        assert p2_alarm_hour(F) == p2_alarm_hour(F, fa_scale=1.0)


def test_margin_never_increases_with_the_scale():
    rng = np.random.default_rng(1)
    F = random_F(rng, 200)
    m = [margin(F, fa_scale=s) for s in (0.25, 0.5, 1.0, 2.0, 8.0)]
    for lo, hi in zip(m[1:], m[:-1]):
        assert np.all(lo <= hi + 1e-12)


def test_larger_scale_never_alarms_earlier():
    rng = np.random.default_rng(2)
    for _ in range(30):
        F = random_F(rng)
        h1, h2 = p2_alarm_hour(F, fa_scale=1.0), p2_alarm_hour(F, fa_scale=4.0)
        assert h2 is None or (h1 is not None and h2 >= h1)


def test_hand_computed_no_onset_margin_scales_linearly():
    F = np.zeros((5, K))                     # p_none = 1: every hour of alarm costs 0.05 * scale
    for s in (0.5, 1.0, 3.0):
        assert np.allclose(margin(F, fa_scale=s), -0.05 * s * K)   # cumulative sum over the K-hour window
        assert p2_alarm_hour(F, fa_scale=s) is None
