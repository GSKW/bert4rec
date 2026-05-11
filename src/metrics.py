from __future__ import annotations

import torch


@torch.no_grad()
def next_event_topk_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    valid_mask = labels >= 0
    if not valid_mask.any():
        return {f"hit_at_{k}": 0.0, f"mrr_at_{k}": 0.0}

    logits = logits[valid_mask]
    labels = labels[valid_mask]
    topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices
    matches = topk.eq(labels.unsqueeze(-1))
    hits = matches.any(dim=-1).float()

    ranks = torch.arange(1, topk.size(1) + 1, device=topk.device, dtype=torch.float32)
    reciprocal_ranks = torch.where(matches, 1.0 / ranks, torch.zeros_like(matches, dtype=torch.float32))
    mrr = reciprocal_ranks.max(dim=-1).values

    return {
        f"hit_at_{k}": float(hits.mean().detach().cpu()),
        f"mrr_at_{k}": float(mrr.mean().detach().cpu()),
    }


class MeanMetric:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.count += weight

    def compute(self) -> float:
        return self.total / self.count if self.count else 0.0
