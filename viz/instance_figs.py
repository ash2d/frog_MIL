"""Window-level figures: what each trained model thinks happens inside a bag.

Window logits and pooling weights of every bag come from ``windows.npz``, saved
at training time by the fold model that tested the bag. Only the slide variant
that keeps the first clip (``max_windows``) re-runs the checkpoints, because it
changes the bag; that needs the embedding cache of the same dataset.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    AXIS,
    BLUE_RAMP,
    INK,
    POOL_ORDER,
    SP_SHORT,
    SURFACE,
    Info,
    pcolor,
    plabel,
    scale_fonts,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from frog_mil.config import SPECIES
from frog_mil.data import FoldView, load_bags
from frog_mil.runs import load_model

BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd"] + BLUE_RAMP)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(np.float32)))


def infer_run(runs: Path, dataset_id: str, run_id: str, poolings: list[str], cache: Path,
              max_windows: int | None = None, device="cpu"):
    """{pooling: dict(inst_prob [seeds, bags, N, C], weights [...], bag_prob, bag_ids, mask)}.

    ``max_windows`` masks every window after the first ``max_windows`` (12 = the first
    1 min clip) and re-runs each bag's test-fold checkpoint, so pooling weights and
    bag scores are those of that shorter bag."""
    run_dir = runs / dataset_id / run_id
    out = {}
    if not max_windows:
        for p in poolings:
            w = np.load(run_dir / p / "windows.npz")
            pr = np.load(run_dir / p / "predictions.npz")
            out[p] = dict(inst_prob=_sigmoid(w["logits"]), weights=w["weights"].astype(np.float32),
                          bag_prob=pr["scores"], bag_ids=w["bag_ids"], mask=w["mask"])
        return out

    tag = f"_first{max_windows}"
    todo = []
    for p in poolings:
        f = cache / f"instances_{run_id}_{p}{tag}.npz"
        ckpts = sorted((run_dir / p).glob("seed*_fold*.pt"))
        if f.exists() and all(f.stat().st_mtime > c.stat().st_mtime for c in ckpts):
            out[p] = dict(np.load(f))
        else:
            todo.append(p)
    if not todo:
        return out
    print(f"  first-{max_windows}-window inference for {run_id}: {', '.join(todo)}")
    data = load_bags(device=device, dataset_id=dataset_id)
    views = [FoldView(data, f) for f in range(data.n_folds)]
    mask = data.mask.clone()
    mask[:, max_windows:] = False
    n, N, C = data.x.shape[0], data.x.shape[1], len(SPECIES)
    cache.mkdir(parents=True, exist_ok=True)
    for p in todo:
        seeds = sorted({int(c.stem.split("_")[0][4:]) for c in (run_dir / p).glob("seed*_fold*.pt")})
        ip = np.zeros((len(seeds), n, N, C), np.float32)
        ww, bp = np.zeros_like(ip), np.zeros((len(seeds), n, C), np.float32)
        for k in seeds:
            for v in views:
                model = load_model(run_dir, p, k, v.f, data.dim, device)
                t = v.idx["test"]
                b = v.batch(t)
                m = mask[t]
                with torch.no_grad():
                    o = model(b["x"] * m.unsqueeze(-1), m)
                ti = t.cpu().numpy()
                ip[k, ti] = torch.sigmoid(o["instance_logits"]).cpu().numpy()
                ww[k, ti] = o["weights"].cpu().numpy()
                bp[k, ti] = torch.sigmoid(o["logits"]).cpu().numpy()
        d = dict(inst_prob=ip, weights=ww, bag_prob=bp, bag_ids=data.bag_ids,
                 mask=mask.cpu().numpy())
        np.savez_compressed(cache / f"instances_{run_id}_{p}{tag}.npz", **d)
        out[p] = d
    return out


def _check(inst, info: Info):
    """Saved window outputs must belong to the same bags as the predictions."""
    assert list(inst["bag_ids"]) == list(info.m.bag_ids), "window bags differ from predictions"


# --------------------------------------------------------------------------- I2
def pooling_example(save, inst: dict, infos: list[Info], run_id: str, bag_id=None,
                    species=None, name="pooling_example", width=12.0, title=True,
                    max_windows=None, capitalise=False, font=1.0, short_titles=False,
                    seed_range=False, labels=None):
    """``max_windows`` expects ``inst`` from ``infer_run(..., max_windows=...)``: only
    those windows are drawn, and bag P comes from that re-run, not the saved hour."""
    pools = [p for p in POOL_ORDER if p in inst] + sorted(set(inst) - set(POOL_ORDER))
    by = {i.pooling: i for i in infos if i.run_id == run_id}
    if not max_windows:
        for p in pools:
            _check(inst[p], by[p])
    idx = infos[0].m.idx
    bag_ids = list(infos[0].m.bag_ids)
    c = SPECIES.index(species) if species else 0
    if bag_id is None:
        # an isolated-call hour where the poolers disagree most on the bag score
        cand = np.flatnonzero(idx[:, c] == 1) if (idx[:, c] == 1).any() else \
            np.flatnonzero(idx[:, c] > 0)
        spread = np.std([inst[p]["bag_prob"][:, cand, c].mean(0) for p in pools], axis=0)
        b = cand[np.argmax(spread)]
    else:
        b = bag_ids.index(bag_id)
    n_n = max_windows or int(inst[pools[0]]["mask"][b].sum())
    fig, axes = plt.subplots(len(pools), 2, figsize=(width, 1.25 * len(pools) + 1.2),
                             sharex=True)
    xs = np.arange(n_n)
    for r, p in enumerate(pools):
        d = inst[p]
        prob = d["inst_prob"][:, b, :n_n, c].mean(0)
        w = d["weights"][:, b, :n_n, c].mean(0)
        bag = d["bag_prob"][:, b, c].mean()
        for k, (a, v, lab) in enumerate([(axes[r, 0], prob, "window P(call)"),
                                         (axes[r, 1], w, "pooling weight")]):
            a.bar(xs, v, color=pcolor(p), width=0.8)
            for x in range(12, n_n, 12):        # clip boundaries (12 windows per clip)
                a.axvline(x - 0.5, color=AXIS, lw=0.8)
            a.grid(axis="x", visible=False)
            a.set_ylim(0, (1.55 if seed_range else 1.05) if k == 0 else max(0.3, v.max() * 1.15))
            if k == 0 and seed_range:
                a.set_yticks([0, 0.5, 1])
            a.set_xlim(-0.6, n_n - 0.4)
            if short_titles and not (k == 0 and seed_range):
                a.yaxis.set_major_locator(MaxNLocator(3))
            if r == 0:
                a.set_title((["Window P(call)", "Pooling weight"] if short_titles else
                             ["Window probability (mean over seeds)",
                              "Weight the pooler gives each window"])[k])
        lab = (labels or {}).get(p, plabel(p))
        axes[r, 0].set_ylabel(lab[:1].upper() + lab[1:] if capitalise else lab, rotation=0, ha="right", va="center", color=INK,
                              fontsize=9)
        axes[r, 0].text(n_n - 0.5, 1.53 if seed_range else 1.0, f"bag P = {bag:.2f}" + (
            f" ({d['bag_prob'][:, b, c].min():.2f}–{d['bag_prob'][:, b, c].max():.2f})"
            if seed_range else ""), ha="right", va="top",
                        fontsize=8, color=INK, weight="bold",
                        bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.5))
    for a in axes[-1]:
        t = [0] + list(range(11, n_n, 12))
        a.set_xticks(t, [str(v + 1) for v in t])
        a.set_xlabel("5 s window")
    if title:
        fig.suptitle(f"One held-out hour through every pooler: {bag_ids[b]}, {SP_SHORT[SPECIES[c]]} "
                     f"index {idx[b, c]} — {run_id}", x=0.01, ha="left", weight="bold")
    scale_fonts(fig, font)
    fig.tight_layout()
    first = (f" Only the first {n_n} windows (the first 1 min clip) are used: later windows "
             "are masked, so weights and bag P are for that clip alone."
             if max_windows else "")
    if seed_range:
        first += (" Bars and bag P are means over the training seeds; the bracket is the "
                  "range of bag P across seeds.")
    save(fig, name,
         f"A real hour (`{bag_ids[b]}`, *{SP_SHORT[SPECIES[c]]}* index {idx[b, c]}), scored "
         "by the cross-validation models that never saw it, "
         "chosen automatically as the hour where the poolers disagree most. Left: each "
         "model's per-window probability. Right: the weight its pooler put on each window "
         "(max is one-hot, mean is uniform, attention is learned from the embedding)." + first)
