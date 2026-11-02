import numpy as np
import pytest

from src.decision.stopping import event_alarm_flags, event_alarm_hour, event_values

K = 12


def tp(d):
    if d > 3:
        return 0.0
    if d <= -6:
        return max((d + 12) / 6, -0.05)
    return (3 - d) / 9


def hand_V(p, p_none):
    return np.array([sum(p[k - 1] * tp(j - k) for k in range(1, K + 1)) + p_none * -0.05 for j in range(K + 1)])


def row(p):
    return np.cumsum(p)[None]


def test_values_match_the_hand_written_tp_curve():
    rng = np.random.default_rng(0)
    for _ in range(30):
        w = rng.dirichlet(np.ones(K + 1) * 0.5)
        assert np.allclose(event_values(row(w[:K]))[0], hand_V(w[:K], w[K]), atol=1e-12)


def test_peaked_six_ahead_alarms_now():
    p = np.zeros(K); p[5] = 1
    V = hand_V(p, 0)
    assert V[0] == pytest.approx(1.0) and V.argmax() == 0
    assert event_alarm_flags(row(p))[0]


def test_peaked_twelve_ahead_waits_until_six_before_onset():
    p = np.zeros(K); p[11] = 1
    V = hand_V(p, 0)
    assert V.argmax() == 6 and V[0] == pytest.approx(0.0)      # best hour is tau - 6, not now
    assert not event_alarm_flags(row(p))[0]


def test_trajectory_alarm_is_six_hours_before_onset():
    tau, T = 20, 30
    F = np.zeros((T, K))
    for t in range(T):
        k = tau - t
        if 1 <= k <= K:
            F[t, k - 1:] = 1.0
    assert event_alarm_hour(F) == 14


def test_no_onset_never_alarms():
    assert event_alarm_hour(np.zeros((40, K))) is None
    p = np.full(K, 0.01 / K)
    assert event_alarm_hour(np.tile(np.cumsum(p), (30, 1))) is None   # V0 = 0.01*mean - 0.0495 < 0


def test_peaked_at_one_alarms_immediately_and_flat_waits_for_the_middle():
    p = np.zeros(K); p[0] = 1
    assert event_alarm_hour(np.tile(np.cumsum(p), (5, 1))) == 0
    flat = np.full(K, 1 / K)
    V = hand_V(flat, 0)
    assert V.argmax() > 0 and not event_alarm_flags(row(flat))[0]       # flat belief: still rising, defer
