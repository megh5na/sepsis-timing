"""Shared GRU encoder (spec §4.1) — a control variable.

Must be byte-identical (same architecture, same init seed, same training
data) across every condition it's used in. In Phase 1 that means B0 (which
doesn't use it at all — it's a rule-based baseline) and B1. Phase 2 reuses
this exact class for P1/P2/A1; nothing about it may change between
conditions being compared, which is why it takes no condition-specific
arguments at all — it only knows its own architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn

INPUT_SIZE = 120   # spec §4.1: 3 * 40 channels (VALUES | MASK | DELTA)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class GRUEncoder(nn.Module):
    """Unidirectional, 2-layer GRU. Input (B, T, 120) -> output h (B, T, 64).

    Unidirectionality is exactly leakage guard #3 (spec §3.6): a
    bidirectional or otherwise backward-looking encoder would see future
    timesteps at every hour. `self.gru.bidirectional` is asserted False at
    construction, not just left as a default.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        assert self.gru.bidirectional is False, "encoder must be unidirectional (leakage guard #3)"
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: (B, T, input_size) -> h: (B, T, hidden_size)."""
        h, _ = self.gru(Z)
        return h
