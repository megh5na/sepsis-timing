from __future__ import annotations

import numpy as np

from src.data.preprocess import build_multi_horizon_targets
from src.decision.utility import DT_EARLY, DT_LATE
from src.eval.fast_utility import UtilityTable
from src.eval.harness import reconstruct_sepsis_label
from third_party.physionet.evaluate_sepsis_score import compute_auc


# ---------------------------------------------------------------- operating points
def operating_point(alarm: np.ndarray, lengths: np.ndarray, t_sepsis: np.ndarray) -> dict:
    septic = t_sepsis >= 0
    alarmed = alarm < lengths
    lo, hi = t_sepsis + DT_EARLY, t_sepsis + DT_LATE
    detected = septic & alarmed & (alarm >= lo) & (alarm <= hi)
    lead_mask = septic & alarmed & (alarm <= t_sepsis)
    return {
        "sensitivity": float(detected.sum() / max(1, septic.sum())),
        "alarms_per_patient_day": float(alarmed.sum() / (lengths.sum() / 24.0)),
        "median_lead_time_h": float(np.median((t_sepsis - alarm)[lead_mask])) if lead_mask.any() else float("nan"),
        "n_alarmed": int(alarmed.sum()),
        "frac_alarmed": float(alarmed.mean()),
        "frac_alarm_before_window": float((septic & alarmed & (alarm < lo)).sum() / max(1, septic.sum())),
        "frac_alarm_after_window": float((septic & alarmed & (alarm > hi)).sum() / max(1, septic.sum())),
        "false_alarm_frac_nonseptic": float((~septic & alarmed).sum() / max(1, (~septic).sum())),
    }


def p1_curve(scores: list[np.ndarray], lengths: np.ndarray, t_sepsis: np.ndarray, taus: np.ndarray) -> dict:
    N = len(lengths)
    A = np.zeros((N, len(taus)), dtype=int)
    for i, r in enumerate(scores):
        A[i] = np.searchsorted(np.maximum.accumulate(np.asarray(r, float)), taus, side="right")
    septic = t_sepsis >= 0
    alarmed = A < lengths[:, None]
    lo, hi = (t_sepsis + DT_EARLY)[:, None], (t_sepsis + DT_LATE)[:, None]
    detected = septic[:, None] & alarmed & (A >= lo) & (A <= hi)
    lead_mask = septic[:, None] & alarmed & (A <= t_sepsis[:, None])
    lead = np.full(len(taus), np.nan)
    for j in range(len(taus)):
        m = lead_mask[:, j]
        if m.any():
            lead[j] = np.median((t_sepsis - A[:, j])[m])
    return {"taus": taus, "sensitivity": detected.sum(0) / max(1, septic.sum()),
            "alarms_per_patient_day": alarmed.sum(0) / (lengths.sum() / 24.0), "median_lead_time_h": lead,
            "alarms": A}


def interp_on(curve_x: np.ndarray, curve_y: np.ndarray, x: float) -> float:
    ok = ~(np.isnan(curve_x) | np.isnan(curve_y))
    xs, ys = curve_x[ok], curve_y[ok]
    if len(xs) == 0 or x < xs.min() or x > xs.max():
        return float("nan")
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    # collapse duplicate x with the mean y so interpolation is well defined
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.bincount(inv, weights=ys) / np.bincount(inv)
    return float(np.interp(x, ux, uy))


def matched_comparison(p2_point: dict, curve: dict) -> dict:
    return {
        "p2_sensitivity": p2_point["sensitivity"],
        "p1_alarms_per_day_at_p2_sensitivity": interp_on(curve["sensitivity"], curve["alarms_per_patient_day"], p2_point["sensitivity"]),
        "p2_alarms_per_day": p2_point["alarms_per_patient_day"],
        "p1_lead_time_at_p2_alarm_rate": interp_on(curve["alarms_per_patient_day"], curve["median_lead_time_h"], p2_point["alarms_per_patient_day"]),
        "p2_lead_time": p2_point["median_lead_time_h"],
        "p1_sensitivity_at_p2_alarm_rate": interp_on(curve["alarms_per_patient_day"], curve["sensitivity"], p2_point["alarms_per_patient_day"]),
    }


# ---------------------------------------------------------------- discrimination
def sepsis_labels(lengths: np.ndarray, t_sepsis: np.ndarray) -> np.ndarray:
    return np.concatenate([reconstruct_sepsis_label(int(T), int(t) if t >= 0 else None)
                           for T, t in zip(lengths, t_sepsis)])


def auc_pair(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    s = scores if (scores.min() >= 0 and scores.max() <= 1) else _unit(scores)
    a, p = compute_auc(labels, s)
    return float(a), float(p)


def _unit(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# ---------------------------------------------------------------- calibration / coherence
def selected_hours_mask(lengths: np.ndarray, t_sepsis: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones(int(T), bool) if ts < 0 else (np.arange(int(T)) <= ts)
                           for T, ts in zip(lengths, t_sepsis)])


def targets_flat(lengths: np.ndarray, t_sepsis: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    ys, ms = [], []
    for T, ts in zip(lengths, t_sepsis):
        y, m = build_multi_horizon_targets(int(T), int(ts) if ts >= 0 else None, K)
        ys.append(y); ms.append(m)
    return np.concatenate(ys), np.concatenate(ms)


TAIL_MIN_PRED = 0.1


def reliability(F: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int = 10) -> dict:
    K = F.shape[1]
    edges = np.linspace(0, 1, n_bins + 1)
    out = {"edges": edges, "mean_pred": np.full((K, n_bins), np.nan), "frac_pos": np.full((K, n_bins), np.nan),
           "count": np.zeros((K, n_bins), int), "ece": np.zeros(K), "n": np.zeros(K, int),
           "ece_tail": np.full(K, np.nan), "n_tail": np.zeros(K, int)}
    for k in range(K):
        sel = w[:, k] > 0
        f, t = F[sel, k], y[sel, k]
        b = np.clip(np.digitize(f, edges[1:-1]), 0, n_bins - 1)
        n = len(f); out["n"][k] = n
        for j in range(n_bins):
            m = b == j
            c = int(m.sum()); out["count"][k, j] = c
            if c:
                out["mean_pred"][k, j] = f[m].mean(); out["frac_pos"][k, j] = t[m].mean()
        out["ece"][k] = float(np.nansum(out["count"][k] / max(1, n) * np.abs(out["frac_pos"][k] - out["mean_pred"][k])))
        tail_counts = out["count"][k, 1:]                      # bins [0.1, 1.0]
        out["n_tail"][k] = int(tail_counts.sum())
        if tail_counts.sum():
            gap = np.abs(out["frac_pos"][k, 1:] - out["mean_pred"][k, 1:])
            out["ece_tail"][k] = float(np.nansum(tail_counts * gap) / tail_counts.sum())
    return out


def coherence_violation_rate(F: np.ndarray) -> float:
    return float((F[:, 1:] < F[:, :-1]).mean())


# ---------------------------------------------------------------- paired bootstrap
def paired_bootstrap(comp_a: np.ndarray, comp_b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    comp_a, comp_b = np.asarray(comp_a, float), np.asarray(comp_b, float)
    assert comp_a.shape == comp_b.shape
    S, N, _ = comp_a.shape
    rng = np.random.default_rng(seed)

    def stat(idx):
        out = 0.0
        for s in range(S):
            out += UtilityTable.normalised(comp_b[s][idx]) - UtilityTable.normalised(comp_a[s][idx])
        return out / S

    full = np.arange(N)
    point = stat(full)
    boots = np.array([stat(rng.integers(0, N, N)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    per_patient = (comp_b[:, :, 0] - comp_a[:, :, 0]).mean(0)          # seed-averaged utility gain per patient
    sd = per_patient.std(ddof=1)
    per_seed = [UtilityTable.normalised(comp_b[s]) - UtilityTable.normalised(comp_a[s]) for s in range(S)]
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "boot_sd": float(boots.std(ddof=1)),
            "standardised_diff": float(point / boots.std(ddof=1)) if boots.std(ddof=1) > 0 else float("nan"),
            "cohens_dz": float(per_patient.mean() / sd) if sd > 0 else float("nan"),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "per_seed_diff": [float(x) for x in per_seed],
            "n_patients": int(N), "n_seeds": int(S), "n_boot": int(n_boot)}


def bootstrap_mean_utility(comp: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    comp = np.asarray(comp, float)
    S, N, _ = comp.shape
    rng = np.random.default_rng(seed)
    point = float(np.mean([UtilityTable.normalised(comp[s]) for s in range(S)]))
    boots = np.array([np.mean([UtilityTable.normalised(comp[s][idx]) for s in range(S)])
                      for idx in (rng.integers(0, N, N) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": point, "ci_low": float(lo), "ci_high": float(hi)}
