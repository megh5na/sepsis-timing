# applies platt probability calibration to multi horizon predictions.

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6


def to_logit(F: np.ndarray) -> np.ndarray:
    F = np.clip(np.asarray(F, dtype=np.float64), EPS, 1 - EPS)
    return np.log(F / (1 - F))


def fit_platt(F: np.ndarray, y: np.ndarray, w: np.ndarray, per_horizon: bool) -> dict:
    z = torch.tensor(to_logit(F))
    yt, wt = torch.tensor(y, dtype=torch.float64), torch.tensor(w, dtype=torch.float64)
    K = z.shape[1]
    shape = (K,) if per_horizon else (1,)
    log_a = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_a, b], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = torch.exp(log_a) * z + b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="none") * wt).sum() / wt.sum()
        loss.backward()
        return loss

    opt.step(closure)
    a_, b_ = torch.exp(log_a).detach().numpy(), b.detach().numpy()
    return {"a": np.broadcast_to(a_, (K,)).copy(), "b": np.broadcast_to(b_, (K,)).copy(), "per_horizon": per_horizon}


def apply_platt(F: np.ndarray, params: dict) -> np.ndarray:
    out = 1.0 / (1.0 + np.exp(-(params["a"] * to_logit(F) + params["b"])))
    return np.maximum.accumulate(out, axis=-1) if params["per_horizon"] else out
