"""Prediction heads that consume the shared encoder's output h (B, T, 64).

Phase 1 implements only Head A (spec §4.2), the baseline sigmoid head used
by condition B1. Head B (monotone multi-horizon, §4.3) and Head C
(independent multi-horizon, §4.4) are Phase 2 and are deliberately absent
from this file — not stubbed, not raising NotImplementedError, simply not
here (see SCOPE CONTROL in the spec). When Phase 2 begins, both heads take
the same encoder output `h` this head takes, which is what makes the
encoder head-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SigmoidHead(nn.Module):
    """Head A (spec §4.2): Linear(64 -> 1) -> sigmoid -> r of shape (B, T).

    Trained with binary cross-entropy against per-hour SepsisLabel
    (condition B1). Alarm rule: r[t] > tau, tau tuned on validation to
    maximise normalised utility (src/decision/threshold.py).
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, hidden_size) -> r: (B, T)."""
        logits = self.linear(h).squeeze(-1)  # (B, T)
        return torch.sigmoid(logits)
