from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import GRUEncoder
from src.models.heads import IndependentHead, MonotoneHead


class SepsisModel(nn.Module):
    def __init__(self, head: str, K: int, hidden_size: int = 64, input_size: int = 120, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = GRUEncoder(input_size=input_size, hidden_size=hidden_size)
        if head == "monotone":
            self.head = MonotoneHead(hidden_size, K)
        elif head == "independent":
            self.head = IndependentHead(hidden_size, K)
        else:
            raise ValueError(f"unknown head {head!r}")
        self.K = K

    def forward(self, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(Z)
        logits = self.head.logits(h)
        return torch.sigmoid(logits), logits
