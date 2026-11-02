import numpy as np
import pandas as pd
import pytest

from src.decision.threshold import first_alarm_hour
from src.decision.utility import total_utility_for_patient
from src.eval.fast_utility import UtilityTable, alarms_from_hours, hours_from_alarms
from src.eval.harness import evaluate


def _cohort(n=60, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.integers(8, 120, n)
    septic = rng.random(n) < 0.4
    # label starts inside the record (t_sepsis - 6 < T), as in real data
    ts = np.where(septic, np.array([rng.integers(6, t + 6) for t in T]), -1)
    return T, ts


def test_observed_matches_hourly_utility_for_every_alarm_hour():
    T, ts = _cohort()
    tab = UtilityTable(T, ts)
    for i in range(len(T)):
        t_sep = int(ts[i]) if ts[i] >= 0 else None
        for a in range(0, int(T[i]) + 1):
            pred = np.zeros(T[i], dtype=int); pred[a:] = 1
            exp = total_utility_for_patient(pred, t_sep, int(T[i]))
            got = tab.observed(np.array([a if j == i else T[j] for j in range(len(T))]))[i]
            assert got == pytest.approx(exp, abs=1e-9)


def test_normalised_utility_matches_official_scorer(tmp_path):
    T, ts = _cohort(80, seed=3)
    ids = [f"p{i:04d}" for i in range(len(T))]
    manifest = pd.DataFrame({"patient_id": ids, "hospital": "A", "T": T, "t_sepsis": ts, "is_septic": ts >= 0})
    rng = np.random.default_rng(5)
    scores = [rng.random(int(t)) for t in T]
    tab = UtilityTable(T, ts)
    for tau in (0.5, 0.8, 0.95, 0.999):
        alarm = tab.alarms_at_threshold(scores, tau)
        hours = {ids[i]: first_alarm_hour(scores[i], tau) for i in range(len(T))}
        assert hours_from_alarms(alarm, T) == [hours[p] for p in ids]
        official = evaluate(lambda pid: hours[pid], ids, manifest, workdir=tmp_path / f"t{tau}")["utility"]
        assert tab.normalised_for(alarm) == pytest.approx(official, abs=1e-9)


def test_sweep_equals_individual_evaluations():
    T, ts = _cohort(50, seed=9)
    rng = np.random.default_rng(1)
    scores = [rng.random(int(t)) for t in T]
    tab = UtilityTable(T, ts)
    taus = np.linspace(0.05, 0.99, 25)
    sweep = tab.sweep_thresholds(scores, taus)
    for tau, v in zip(taus, sweep):
        assert tab.normalised_for(tab.alarms_at_threshold(scores, tau)) == pytest.approx(v, abs=1e-12)


def test_alarm_hour_roundtrip():
    T = np.array([5, 7])
    assert alarms_from_hours([None, 3], T).tolist() == [5, 3]
    assert hours_from_alarms(np.array([5, 3]), T) == [None, 3]
