"""Metrics and training-free baselines.

Average precision is the headline: with a few % (Gastrotheca) to ~12%
(Oreobates) positive hours, ROC-AUC flatters everything and accuracy is
meaningless. Confidence intervals live in ``stats``.
"""
from __future__ import annotations

import numpy as np

from .config import MAX_INDEX, SPECIES


def ap_score(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Average precision, tie-aware; matches sklearn.average_precision_score.

    ``w`` are non-negative integer bag weights (bootstrap counts): a bag with
    weight 2 counts as two copies. NaN when there are no positives.
    """
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    y = np.asarray(y, float)
    if (w * y).sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    y, s, w = y[o], s[o], w[o]
    tp, fp = np.cumsum(w * y), np.cumsum(w * (1 - y))
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]   # end of each tied group
    tp, fp = tp[last], fp[last]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def species_ap(y: np.ndarray, s: np.ndarray) -> dict:
    """{species_ap, macro_ap} for [n, C] labels and scores."""
    out = {f"{sp}_ap": ap_score(y[:, c], s[:, c]) for c, sp in enumerate(SPECIES)}
    aps = [v for v in out.values() if not np.isnan(v)]
    out["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return out


def by_intensity(idx: np.ndarray, s: np.ndarray) -> dict:
    """AP for each calling level against silent hours only.

    For level L, positives are hours with index == L and negatives are hours
    with index == 0; hours at other positive levels are dropped rather than
    counted as negatives, which would punish a model for detecting them.
    Annotators scored only the recorded clips, so index-1 bags are clean
    positives: weak index-1 AP means isolated calls are being missed.
    """
    out = {}
    for c, name in enumerate(SPECIES):
        neg = idx[:, c] == 0
        for level in range(1, MAX_INDEX + 1):
            pos = idx[:, c] == level
            keep = neg | pos
            out[f"{name}_idx{level}_n"] = int(pos.sum())
            if pos.sum() and neg.sum():
                out[f"{name}_idx{level}_ap"] = ap_score(pos[keep], s[keep, c])
    return out


def zero_shot_baseline(rows: np.ndarray, emb_meta: dict, logits_path) -> dict[str, np.ndarray]:
    """Training-free scores from Perch's own logits, max-pooled over the bag.

    ``rows`` is ``BagData.rows`` ([n_bags, max_n] cache rows, -1 = pad). Two
    variants, [n_bags, C] each:
      ``congeneric``: max over the species' genus columns (8 Gastrotheca spp.;
                      Oreobates quixensis)
      ``frog``:       the generic Frog class, same score for both species

    If ``congeneric`` is close to the trained probes, the probe is mostly
    re-deriving what Perch already knows about the genus.
    """
    names = emb_meta.get("logit_names") or []
    if not names or not logits_path.exists():
        return {}
    logits = np.asarray(np.load(logits_path, mmap_mode="r"), np.float32)
    genus = {"gastrotheca": "Gastrotheca ", "oreobates": "Oreobates "}
    cols = {s: [i for i, n in enumerate(names) if n.startswith(genus[s])] for s in SPECIES}
    frog = [i for i, n in enumerate(names) if n == "Frog"]
    mask = rows >= 0
    lg = np.where(mask[..., None], logits[np.where(mask, rows, 0)], -np.inf)  # [n, N, cols]
    out = {}
    if all(cols.values()):
        out["congeneric"] = np.stack([lg[..., cols[s]].max(axis=(1, 2)) for s in SPECIES], -1)
    if frog:
        f = lg[..., frog].max(axis=(1, 2))
        out["frog"] = np.stack([f] * len(SPECIES), -1)
    return out


def clock_baseline(bags: list[dict], train: np.ndarray) -> np.ndarray:
    """Hour-of-day x month presence rate, fit on the ``train`` bags (a mask).

    The trap this guards against: both species are strongly nocturnal and
    seasonal, so a model that knows nothing but the clock scores well. Any
    audio model that does not clearly beat this has learned nothing about frogs.
    """
    scores = np.zeros((len(bags), len(SPECIES)))
    tr = [b for b, t in zip(bags, train) if t]
    for c, name in enumerate(SPECIES):
        rate: dict[tuple, list] = {}
        for b in tr:
            rate.setdefault((int(b["hour"]), b["date"][5:7]), []).append(
                int(b[f"{name}_present"]))
        prior = np.mean([int(b[f"{name}_present"]) for b in tr])
        for i, b in enumerate(bags):
            v = rate.get((int(b["hour"]), b["date"][5:7]))
            scores[i, c] = np.mean(v) if v else prior
    return scores
