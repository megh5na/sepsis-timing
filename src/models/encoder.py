# 2 layers, 64 hidden units, 0.2 dropout, unidirectional (no leakage)

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_SIZE = 120   # spec §4.1: 3 * 40 channels (VALUES | MASK | DELTA)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class GRUEncoder(nn.Module):
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
        h, _ = self.gru(Z)
        return h
