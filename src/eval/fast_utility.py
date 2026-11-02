# pre computes per patient utility tables for fast scoring of alarm hours. The official scorer is slow because it
# has to read the entire CSV and compute the utility for each patient. This module precomputes the utility tables and then can score any alarm hours in a single pass

from __future__ import annotations

import numpy as np

from src.decision.utility import DT_EARLY, DT_LATE, U


class UtilityTable:
    def __init__(self, lengths: np.ndarray, t_sepsis: np.ndarray):
        self.lengths = np.asarray(lengths, dtype=int)
        self.t_sepsis = np.asarray(t_sepsis, dtype=int)
        self.N = len(self.lengths)
        self.off = np.concatenate([[0], np.cumsum(self.lengths + 1)])
        total = int(self.off[-1])
        self.pre0 = np.zeros(total)     # pre0[off+a] = sum_{h<a} U(0,h)
        self.pre1 = np.zeros(total)     # pre1[off+a] = sum_{h<a} U(1,h)
        self.best = np.zeros(self.N)
        for i in range(self.N):
            T, ts = int(self.lengths[i]), (int(self.t_sepsis[i]) if self.t_sepsis[i] >= 0 else None)
            u0 = np.array([U(0, h, ts) for h in range(T)])
            u1 = np.array([U(1, h, ts) for h in range(T)])
            o = self.off[i]
            self.pre0[o + 1:o + T + 1] = np.cumsum(u0)
            self.pre1[o + 1:o + T + 1] = np.cumsum(u1)
            if ts is not None:
                lo, hi = max(0, ts + DT_EARLY), min(ts + DT_LATE + 1, T)
                self.best[i] = (self.pre0[o + lo] + (self.pre1[o + hi] - self.pre1[o + lo])
                                + (self.pre0[o + T] - self.pre0[o + hi]))
            else:
                self.best[i] = self.pre0[o + T]
        self.inaction = np.array([self.pre0[self.off[i] + self.lengths[i]] for i in range(self.N)])

    @classmethod
    def from_split(cls, split) -> "UtilityTable":
        return cls(split.lengths, split.t_sepsis)

    def observed(self, alarm: np.ndarray) -> np.ndarray:
        alarm = np.asarray(alarm, dtype=int)
        o = self.off[:-1]
        T = self.lengths
        return self.pre0[o + alarm] + (self.pre1[o + T] - self.pre1[o + alarm])

    def components(self, alarm: np.ndarray) -> np.ndarray:
        return np.stack([self.observed(alarm), self.inaction, self.best], axis=1)

    @staticmethod
    def normalised(components: np.ndarray) -> float:
        den = components[:, 2].sum() - components[:, 1].sum()
        return float((components[:, 0].sum() - components[:, 1].sum()) / den) if den else 0.0

    def normalised_for(self, alarm: np.ndarray) -> float:
        return self.normalised(self.components(alarm))

    def sweep_thresholds(self, scores: list[np.ndarray], taus: np.ndarray) -> np.ndarray:
        taus = np.asarray(taus, dtype=float)
        obs = np.zeros(len(taus))
        o = self.off[:-1]
        for i, r in enumerate(scores):
            T = int(self.lengths[i])
            cm = np.maximum.accumulate(np.asarray(r, dtype=float))
            a = np.searchsorted(cm, taus, side="right")        # first index with cummax > tau, T if none
            obs += self.pre0[o[i] + a] + (self.pre1[o[i] + T] - self.pre1[o[i] + a])
        den = self.best.sum() - self.inaction.sum()
        return (obs - self.inaction.sum()) / den

    def alarms_at_threshold(self, scores: list[np.ndarray], tau: float) -> np.ndarray:
        out = np.zeros(self.N, dtype=int)
        for i, r in enumerate(scores):
            cm = np.maximum.accumulate(np.asarray(r, dtype=float))
            out[i] = int(np.searchsorted(cm, tau, side="right"))
        return out


def alarms_from_hours(hours: list, lengths: np.ndarray) -> np.ndarray:
    return np.array([int(lengths[i]) if h is None else int(h) for i, h in enumerate(hours)], dtype=int)


def hours_from_alarms(alarm: np.ndarray, lengths: np.ndarray) -> list:
    return [None if int(a) >= int(T) else int(a) for a, T in zip(alarm, lengths)]
