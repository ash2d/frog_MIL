"""Lightweight probes over frozen Perch v2 embeddings.

The encoder never moves, so everything here is small by design: the point of
the experiment is the pooling function, and a heavy head would mask its effect.
Keep the probe identical across pooling variants or the comparison means
nothing.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import build_pooler


class InstanceProbe(nn.Module):
    """Per-instance scorer: linear, or one hidden layer.

    ``hidden=0`` gives the linear probe -- the honest baseline, and the one
    whose weights you can actually inspect against the Perch label space.
    """

    def __init__(self, dim: int = 1536, n_classes: int = 2, hidden: int = 0,
                 dropout: float = 0.2):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def forward(self, x):            # [B, N, D] -> [B, N, C]
        return self.net(x)


class OrdinalHead(nn.Module):
    """Cumulative-link head for the 0-3 calling index: P(y >= k) = sigmoid(s - b_k).

    ``s`` is the pooled bag logit -- the same single score per species that the
    presence task uses -- and b_1 = 0 <= b_2 <= b_3 are learned thresholds,
    ordered by construction (non-negative increments via softplus).

    Because b_1 is fixed at 0, P(y >= 1) = sigmoid(s) *is* the presence
    prediction. The ordinal task is therefore a strict extension of the binary
    one rather than a separate head bolted on: it adds 2 parameters per species,
    leaves the pooling comparison untouched, and asks one extra thing of the
    score -- that a chorus hour should score higher than a one-call hour.

    A 4-way softmax would treat "0 vs 3" as no worse than "0 vs 1" and would
    need its own representation; this does neither.
    """

    def __init__(self, n_classes: int = 2, n_levels: int = 4):
        super().__init__()
        # increments for b_2..b_K; init so thresholds start ~1 logit apart
        self.deltas = nn.Parameter(torch.full((n_classes, n_levels - 2), 0.55))

    def thresholds(self) -> torch.Tensor:              # [C, K-1], b_1 = 0
        inc = torch.nn.functional.softplus(self.deltas)
        zero = torch.zeros(inc.shape[0], 1, device=inc.device, dtype=inc.dtype)
        return torch.cat([zero, inc.cumsum(-1)], dim=-1)

    def forward(self, bag_logits: torch.Tensor) -> torch.Tensor:
        """[B, C] -> cumulative logits [B, C, K-1] for P(y>=1), P(y>=2), P(y>=3)."""
        return bag_logits.unsqueeze(-1) - self.thresholds().unsqueeze(0)


class MILModel(nn.Module):
    """instance probe -> pooling -> bag logits (-> optional ordinal thresholds).

    ``ordinal=True`` adds an :class:`OrdinalHead` on the pooled score. Presence
    is unchanged either way: ``logits`` is always P(y >= 1).
    """

    def __init__(self, dim: int = 1536, n_classes: int = 2, hidden: int = 0,
                 pooling: str = "attention", dropout: float = 0.2,
                 ordinal: bool = False, **pool_kw):
        super().__init__()
        self.probe = InstanceProbe(dim, n_classes, hidden, dropout)
        self.pool = build_pooler(pooling, dim=dim, n_classes=n_classes, **pool_kw)
        self.n_classes, self.pooling = n_classes, pooling
        self.ordinal = OrdinalHead(n_classes) if ordinal else None

    def forward(self, x, mask):
        inst = self.probe(x)                              # [B, N, C]
        bag, w = self.pool(inst, mask, x if self.pool.needs_embeddings else None)
        out = {"logits": bag, "instance_logits": inst, "weights": w}
        if self.ordinal is not None:
            out["cum_logits"] = self.ordinal(bag)          # [B, C, 3]
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
