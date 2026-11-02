from __future__ import annotations

import numpy as np

from src.decision.utility import U

TIE_TOL = 1e-12
P1_HORIZON = 6   # spec 5.2: r[t] = F[t, 6]


def onset_distribution(F: np.ndarray, repair: bool = False) -> tuple[np.ndarray, np.ndarray]:
    F = np.asarray(F, dtype=np.float64)
    if repair:
        F = np.maximum.accumulate(F, axis=-1)
    p = np.diff(F, axis=-1, prepend=0.0)
    return p, 1.0 - F[..., -1]


def _hourly_gain_table(K: int) -> tuple[np.ndarray, float]:
    base = 1000   # arbitrary origin; U only depends on the offset h - tau
    gain = np.zeros((K, K))
    for j in range(K):
        for k in range(1, K + 1):
            tau = base + k
            gain[j, k - 1] = U(1, base + j, tau) - U(0, base + j, tau)
    none_gain = U(1, base, None) - U(0, base, None)
    return gain, none_gain


_TABLE_CACHE: dict[int, tuple[np.ndarray, float]] = {}


def hourly_gain_table(K: int) -> tuple[np.ndarray, float]:
    if K not in _TABLE_CACHE:
        _TABLE_CACHE[K] = _hourly_gain_table(K)
    return _TABLE_CACHE[K]


def expected_utility_V(p: np.ndarray, p_none: float) -> np.ndarray:
    K = len(p)
    horizon = K + 3
    V = np.zeros(K + 1)
    for s in range(K + 1):
        for h in range(horizon + 1):
            pred = 1 if h >= s else 0
            for k in range(1, K + 1):
                V[s] += p[k - 1] * U(pred, h, k)
            V[s] += p_none * U(pred, h, None)
    return V


def alarm_now_from_V(V: np.ndarray) -> bool:
    return bool(V[0] >= V.max() - TIE_TOL)


def margin(F: np.ndarray, repair: bool = False, fa_scale: float = 1.0) -> np.ndarray:
    p, p_none = onset_distribution(F, repair=repair)
    K = p.shape[-1]
    gain, none_gain = hourly_gain_table(K)
    G = p @ gain.T + fa_scale * p_none[..., None] * none_gain   # (T, K): expected gain of alarming at hour t+j
    C = np.cumsum(G, axis=-1)                              # C[:, s-1] = V[t,t] - V[t,t+s]
    return C.min(axis=-1)


def p2_alarm_hour(F: np.ndarray, repair: bool = False, fa_scale: float = 1.0) -> int | None:
    hits = np.flatnonzero(margin(F, repair=repair, fa_scale=fa_scale) >= -TIE_TOL)
    return int(hits[0]) if hits.size else None


def p1_scores(F: np.ndarray) -> np.ndarray:
    return np.asarray(F)[:, P1_HORIZON - 1]


# ---------------------------------------------------------------------------
# Exploratory alternative reading (NOT the primary P2; added after the primary
# result was locked, see docs/REPORT.md section 8).
#
# "Event-level" reading of U(alarm at s, onset tau): the hourly utility of a
# POSITIVE PREDICTION at hour s, U(1, s, tau), i.e. the scorer's true-positive
# curve, which peaks at s = tau - 6. No persistence, no false-negative term.
#   V1[t, s] = sum_k p[t,k] U(1, s, t+k) + p_none U(1, s, no onset)      s = t..t+K
# Alarm at t iff now is the best hour (t = argmax_s V1[t, s], ties to the earlier
# hour) AND an alarm now has positive expected utility, V1[t, t] > 0. The second
# condition is needed because with p_none near 1 every V1 equals u_fp and the
# argmax alone would alarm on a tie.
# ---------------------------------------------------------------------------
def _event_table(K: int) -> tuple[np.ndarray, float]:
    base = 1000
    tab = np.zeros((K + 1, K))
    for j in range(K + 1):
        for k in range(1, K + 1):
            tab[j, k - 1] = U(1, base + j, base + k)
    return tab, U(1, base, None)


_EVENT_CACHE: dict[int, tuple[np.ndarray, float]] = {}


def event_values(F: np.ndarray, repair: bool = False) -> np.ndarray:
    p, p_none = onset_distribution(F, repair=repair)
    K = p.shape[-1]
    if K not in _EVENT_CACHE:
        _EVENT_CACHE[K] = _event_table(K)
    tab, none_u = _EVENT_CACHE[K]
    return p @ tab.T + p_none[..., None] * none_u


def event_alarm_flags(F: np.ndarray, repair: bool = False) -> np.ndarray:
    V = event_values(F, repair=repair)
    return (V[:, 0] > TIE_TOL) & (V[:, 0] >= V.max(axis=1) - TIE_TOL)


def event_alarm_hour(F: np.ndarray, repair: bool = False) -> int | None:
    hits = np.flatnonzero(event_alarm_flags(F, repair=repair))
    return int(hits[0]) if hits.size else None
