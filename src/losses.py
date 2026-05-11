from __future__ import annotations

import torch
import torch.nn.functional as F


def multitask_sequence_loss(
    mlm_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
    next_logits: torch.Tensor,
    next_labels: torch.Tensor,
    mlm_weight: float = 1.0,
    next_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    if mlm_logits.numel() == 0 or mlm_labels.numel() == 0:
        mlm_loss = next_logits.new_zeros(())
    else:
        mlm_loss = F.cross_entropy(mlm_logits, mlm_labels)
    next_loss = F.cross_entropy(next_logits, next_labels, ignore_index=-100)
    total_loss = mlm_weight * mlm_loss + next_weight * next_loss
    return {
        "loss_total": total_loss,
        "loss_mlm": mlm_loss.detach(),
        "loss_next": next_loss.detach(),
    }
