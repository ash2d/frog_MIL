"""Block bootstrap for AP, shared by ``frog-report`` and the figures.

Neighbouring hours share a night, weather and calling individuals, so they are
not independent. Resampling single hours would make every CI too narrow.
Resamples here draw whole 3-day blocks (the unit the folds are dealt in) with
replacement, separately within each recording regime, so every resample keeps
the regime mix of the data.

Every model is scored on the same resamples, so the difference between two
models gets its own (paired) CI. A resample is a vector of bag counts, and AP
is computed with those counts as weights, which is identical to AP on the
resampled bag list and much faster.

Cells: AP is computed for
    ("presence", c)          species c, all bags
    ("level", c, L)          index == L vs index == 0 (other levels dropped)
    ("regime", c, r)         presence within recording regime r
Results are cached in the run store, keyed on the prediction files.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MAX_INDEX, SPECIES

MIN_LEVEL_N = 5


def block_draws(blocks: np.ndarray, strata: np.ndarray, B: int, seed: int) -> np.ndarray:
    """[B, n_bags] bag counts: blocks resampled with replacement within each stratum."""
    rng = np.random.default_rng(seed)
    clusters = np.unique(np.stack([strata.astype(str), blocks.astype(str)], 1), axis=0)
    key = {tuple(c): i for i, c in enumerate(map(tuple, clusters))}
    bag_cluster = np.array([key[(s, b)] for s, b in zip(strata.astype(str),
                                                        blocks.astype(str))])
    counts = np.zeros((B, len(clusters)), np.int32)
    for s in np.unique(clusters[:, 0]):
        members = np.flatnonzero(clusters[:, 0] == s)
        picks = rng.integers(0, len(members), size=(B, len(members)))
        np.add.at(counts, (np.arange(B)[:, None], members[picks]), 1)
    return counts[:, bag_cluster]


def ap_weighted(y: np.ndarray, s: np.ndarray, W: np.ndarray, chunk: int = 500) -> np.ndarray:
    """AP of scores ``s`` against labels ``y`` under each row of count weights ``W``.

    Tie-aware like ``evaluate.ap_score``; NaN for a row with no weighted positives.
    """
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o].astype(np.float32), s[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    out = np.empty(len(W))
    for a in range(0, len(W), chunk):
        w = W[a:a + chunk, o].astype(np.float32)
        tp = np.cumsum(w * y, axis=1)[:, last]
        fp = np.cumsum(w * (1 - y), axis=1)[:, last]
        P = tp[:, -1]
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        drec = np.diff(tp, axis=1, prepend=0) / np.where(P > 0, P, 1)[:, None]
        r = (drec * prec).sum(1)
        out[a:a + chunk] = np.where(P > 0, r, np.nan)
    return out


def make_cells(idx: np.ndarray, regime: np.ndarray) -> list[tuple]:
    cells = [("presence", c) for c in range(len(SPECIES))]
    for c in range(len(SPECIES)):
        for lvl in range(1, MAX_INDEX + 1):
            if (idx[:, c] == lvl).sum() >= MIN_LEVEL_N:
                cells.append(("level", c, lvl))
    for c in range(len(SPECIES)):
        for r in sorted(set(regime)):
            if (idx[regime == r, c] > 0).any():
                cells.append(("regime", c, r))
    return cells


def cell_rows(cell: tuple, idx: np.ndarray, regime: np.ndarray):
    """(bags kept, positive label) of a cell."""
    c = cell[1]
    if cell[0] == "presence":
        return np.ones(len(idx), bool), idx[:, c] > 0
    if cell[0] == "level":
        return (idx[:, c] == 0) | (idx[:, c] == cell[2]), idx[:, c] == cell[2]
    return regime == cell[2], idx[:, c] > 0


def _model_job(args):
    scores, idx, regime, W, cells = args
    ones = np.ones((1, W.shape[1]), np.int32)
    point, boots = np.zeros(len(cells)), np.zeros((len(W), len(cells)))
    for k, cell in enumerate(cells):
        keep, pos = cell_rows(cell, idx, regime)
        c = cell[1]
        point[k] = np.nanmean([ap_weighted(pos[keep], sc[keep, c], ones[:, keep])[0]
                               for sc in scores])
        boots[:, k] = np.nanmean([ap_weighted(pos[keep], sc[keep, c], W[:, keep])
                                  for sc in scores], axis=0)
    return point, boots


@dataclass
class Stats:
    ids: list[str]
    cells: list[tuple]
    point: dict          # model_id -> [cells]
    boots: dict          # model_id -> [B, cells]
    W: np.ndarray        # [B, n_bags] the shared resamples

    def k(self, cell) -> int:
        return self.cells.index(tuple(cell))

    def ap(self, mid, cell) -> float:
        return float(self.point[mid][self.k(cell)])

    def ap_boot(self, mid, cell) -> np.ndarray:
        return self.boots[mid][:, self.k(cell)]

    def has(self, cell) -> bool:
        return tuple(cell) in self.cells

    def macro(self, mid) -> float:
        return float(np.mean([self.ap(mid, ("presence", c)) for c in range(len(SPECIES))]))

    def macro_boot(self, mid) -> np.ndarray:
        return np.mean([self.ap_boot(mid, ("presence", c)) for c in range(len(SPECIES))], 0)

    def get(self, mid, what):
        """what: 'macro' or a species index -> (point, boots)."""
        if what == "macro":
            return self.macro(mid), self.macro_boot(mid)
        return self.ap(mid, ("presence", what)), self.ap_boot(mid, ("presence", what))


def ci(b: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.nanpercentile(b, [2.5, 97.5]))


def compute(models, blocks: np.ndarray, regime: np.ndarray, B: int, seed: int,
            cache_dir: Path | None = None, workers: int = 16) -> Stats:
    """Point AP and block-bootstrap AP for every model x cell (cached)."""
    idx = models[0].idx
    cells = make_cells(idx, regime)
    W = block_draws(blocks, regime, B, seed)
    ids = [m.model_id for m in models]
    files = [str(p) for m in models for p in ([m.path / "predictions.npz"] if m.path else [])]
    key = json.dumps([B, seed, ids, [(f, Path(f).stat().st_mtime_ns) for f in files],
                      hashlib.sha1(np.ascontiguousarray(blocks).tobytes()).hexdigest(),
                      [list(map(str, c)) for c in cells],
                      [hashlib.sha1(m.scores.tobytes()).hexdigest() for m in models
                       if not m.path]])
    cf = (cache_dir / f"bootstrap_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
          if cache_dir else None)
    if cf is not None and cf.exists():
        d = np.load(cf)
        return Stats(ids, cells, dict(zip(ids, d["point"])), dict(zip(ids, d["boots"])), W)

    print(f"bootstrap: {len(ids)} models x B={B} x {len(cells)} cells ...", flush=True)
    jobs = [(m.scores, m.idx, regime, W, cells) for m in models]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_model_job, jobs))
    point, boots = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
    if cf is not None:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cf, point=point, boots=boots)
    return Stats(ids, cells, dict(zip(ids, point)), dict(zip(ids, boots)), W)
