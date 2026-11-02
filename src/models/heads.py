from __future__ import annotations

import torch
import torch.nn as nn


class SigmoidHead(nn.Module): # produces a single risk score per hour, used by B1

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.linear(h).squeeze(-1)  # (B, T)
        return torch.sigmoid(logits)


class MonotoneHead(nn.Module): # produces K cumulative onset probabilities F[t, k], non-decreasing in k by construction. Used by P1 and P2 (same trained weights).

    def __init__(self, hidden_size: int = 64, K: int = 12):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(hidden_size, K)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.linear(h)                              # (B, T, K)
        base = raw[..., 0:1]                              # unconstrained
        incr = nn.functional.softplus(raw[..., 1:])       # (B, T, K-1), >= 0
        return torch.cat([base, base + torch.cumsum(incr, dim=-1)], dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(h))


class IndependentHead(nn.Module): # also produces K outputs, but unconstrained. Used only by A1, to test whether the monotone constraint matters (H3).

    def __init__(self, hidden_size: int = 64, K: int = 12):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(hidden_size, K)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(h))
