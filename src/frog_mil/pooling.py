"""MIL pooling functions, compared under one interface.

Every pooler maps per-instance logits ``[B, N, C]`` plus a validity ``mask``
``[B, N]`` to bag logits ``[B, C]``, and also returns the per-instance weights
``[B, N, C]`` it used. The weights are what make the comparison interesting:
they are the model's guess at *which* 5 s window held the call, so the same
downstream localisation check works for all five methods.

References
----------
Wang, Li & Metze (2019), "A comparison of five multiple instance learning
pooling functions for sound event detection with weak labeling" -- source of
the linear-softmax pooler, which they found best for weakly-labelled SED.
Ilse, Tomczak & Welling (2018), "Attention-based deep MIL" -- gated attention.

Conventions
-----------
All poolers return *logits*, so a single ``BCEWithLogitsLoss(pos_weight=...)``
covers every variant and the comparison is not confounded by the loss. Linear
softmax is defined on probabilities, so it pools sigmoids internally and
converts back with a clamped logit; that clamp is the only place a pooler can
saturate, hence ``EPS``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6
NEG_INF = -1e9


def _masked_fill(x: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    """Apply a [B, N] validity mask to a [B, N, C] tensor."""
    return x.masked_fill(~mask.unsqueeze(-1), value)


def _safe_logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(EPS, 1.0 - EPS)
    return torch.log(p) - torch.log1p(-p)


class Pooler(nn.Module):
    """Base class. ``needs_embeddings`` says whether ``forward`` uses ``emb``."""

    needs_embeddings = False

    def forward(self, logits: torch.Tensor, mask: torch.Tensor,
                emb: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MeanPool(Pooler):
    """Uniform average over valid instances.

    The most conservative pooler: it assumes calls fill the bag. Under the
    actual label process (one call anywhere in the hour makes the hour
    positive) it systematically under-responds to sparse calls, so it is the
    floor the others should beat.
    """

    def forward(self, logits, mask, emb=None):
        n = mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        w = mask.unsqueeze(-1).float() / n
        return (logits * w).sum(1), w.expand_as(logits)


class MaxPool(Pooler):
    """Hard max -- the pooler that matches how the labels were actually made.

    The hourly 0-3 index is the max over the clips in that hour, so this is the
    assumption-free choice. Its weakness is gradient sparsity: one instance per
    bag per class gets a gradient, which with ~124 positive Gastrotheca bags is
    very little signal.
    """

    def forward(self, logits, mask, emb=None):
        masked = _masked_fill(logits, mask, NEG_INF)
        bag, idx = masked.max(dim=1)
        w = torch.zeros_like(logits).scatter_(1, idx.unsqueeze(1), 1.0)
        return bag, w


class LogMeanExpPool(Pooler):
    """(1/r) * log(mean_i exp(r * logit_i)).

    Smooth interpolation between mean (r -> 0) and max (r -> inf), so it keeps
    max's sensitivity to a single call while giving every instance a gradient.
    ``r`` is parameterised in log-space when learnable so it stays positive;
    a learned ``r`` is worth reporting, since it says how peaked the evidence
    actually is.
    """

    def __init__(self, r: float = 1.0, learnable: bool = False):
        super().__init__()
        log_r = torch.tensor(float(r)).log()
        self.log_r = nn.Parameter(log_r) if learnable else nn.Buffer(log_r)

    @property
    def r(self) -> torch.Tensor:
        return self.log_r.exp()

    def forward(self, logits, mask, emb=None):
        r = self.r.clamp(1e-3, 1e3)
        masked = _masked_fill(logits * r, mask, NEG_INF)
        n = mask.sum(1).clamp(min=1).log().unsqueeze(-1)
        bag = (torch.logsumexp(masked, dim=1) - n) / r
        w = torch.softmax(masked, dim=1)
        return bag, w


class LinearSoftmaxPool(Pooler):
    """sum_i p_i^2 / sum_i p_i, on probabilities -- Wang et al. (2019).

    Instances weight themselves by their own confidence, so a bag with one
    confident call scores near that call's probability while a bag of uniform
    low scores stays low. No temperature to tune, which is its practical appeal
    over LME.
    """

    def forward(self, logits, mask, emb=None):
        p = torch.sigmoid(logits) * mask.unsqueeze(-1)
        denom = p.sum(1).clamp(min=EPS)
        w = p / denom.unsqueeze(1)
        return _safe_logit((p * w).sum(1)), w


class AttentionPool(Pooler):
    """Gated attention over instance embeddings (Ilse et al., 2018).

    Weights come from the *embedding*, not the logit, so attention can learn
    "this window is worth listening to" separately from "this window is a
    frog". ``per_class=True`` gives each species its own attention: the two
    frogs call at different times of night and 100 hours are positive for both,
    so a shared weight would force one species' call to explain the other's
    label.
    """

    needs_embeddings = True

    def __init__(self, dim: int, hidden: int = 128, n_classes: int = 2,
                 per_class: bool = True, dropout: float = 0.0):
        super().__init__()
        heads = n_classes if per_class else 1
        self.per_class, self.heads = per_class, heads
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, logits, mask, emb=None):
        if emb is None:
            raise ValueError("AttentionPool needs instance embeddings")
        h = self.drop(torch.tanh(self.V(emb)) * torch.sigmoid(self.U(emb)))
        a = self.w(h)                                        # [B, N, heads]
        if not self.per_class:
            a = a.expand(-1, -1, logits.shape[-1])
        a = _masked_fill(a, mask, NEG_INF)
        w = torch.softmax(a, dim=1)
        return (logits * w).sum(1), w


POOLERS = {
    "mean": MeanPool,
    "max": MaxPool,
    "lme": LogMeanExpPool,
    "linear_softmax": LinearSoftmaxPool,
    "attention": AttentionPool,
}


def build_pooler(name: str, dim: int, n_classes: int = 2, **kw) -> Pooler:
    """``kw`` is filtered per pooler so one config block can drive all five."""
    if name not in POOLERS:
        raise KeyError(f"unknown pooling '{name}'; choose from {sorted(POOLERS)}")
    cls = POOLERS[name]
    if cls is AttentionPool:
        return cls(dim=dim, n_classes=n_classes,
                   hidden=kw.get("attn_hidden", 128),
                   per_class=kw.get("attn_per_class", True),
                   dropout=kw.get("attn_dropout", 0.0))
    if cls is LogMeanExpPool:
        return cls(r=kw.get("lme_r", 1.0), learnable=kw.get("lme_learnable", False))
    return cls()
