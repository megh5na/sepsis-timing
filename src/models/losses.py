from __future__ import annotations

import torch
import torch.nn.functional as Fn


def masked_multihorizon_bce(logits: torch.Tensor, y: torch.Tensor, mask_k: torch.Tensor,
                            length_mask: torch.Tensor,
                            train_mask: torch.Tensor | None = None) -> torch.Tensor:
    w = mask_k * length_mask.unsqueeze(-1)
    if train_mask is not None:
        w = w * train_mask.unsqueeze(-1)
    # BCE on the logit is the same quantity as BCE(sigmoid(logit), y), but stable
    bce = Fn.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (bce * w).sum() / w.sum().clamp_min(1.0)
