"""Method diagrams: bag structure, pipeline, pooling behaviour, ordinal head.

These are drawn with matplotlib patches so they regenerate with everything
else. Numbers that depend on the data (window counts, thresholds, test-bag
logits) are read from the runs, not hard-coded.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    AXIS,
    GRID,
    INK,
    INK2,
    MUTED,
    POOL_ORDER,
    SPECIES_COLOR,
    SURFACE,
    pcolor,
    plabel,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from frog_mil.data import SPECIES
from frog_mil.pooling import POOLERS

FROZEN, TRAINED = "#f0efec", "#cde2fb"


def _box(ax, x, y, w, h, text, fc=SURFACE, ec=AXIS, fs=8.5, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, color=color, linespacing=1.35)


def _arrow(ax, p0, p1, color=INK2, ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------- M1
def bag_structure(save):
    """1 hour -> 2 clips -> 24 windows; labels exist per bag only."""
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.1], hspace=1.0, wspace=0.08)
    rng = np.random.default_rng(3)

    # Row 1: the hour
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 60); ax.set_ylim(0, 1); ax.grid(False)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 61, 10), [str(m) for m in range(0, 61, 10)])
    ax.set_xlabel("minutes")
    ax.add_patch(Rectangle((0, 0.15), 60, 0.7, fc=GRID, ec="none", alpha=0.5))
    clips = [(0, "clip A", "0–1 min"), (30, "clip B", "30–31 min")]
    for x0, name, span in clips:
        ax.add_patch(Rectangle((x0, 0.15), 1.0, 0.7, fc=INK2, ec="none"))
        ax.text(x0 + 1.6, 0.5, f"{name}  ({span})", va="center", fontsize=8.5, color=INK2)
        ax.text(x0 + 15.5, 0.5, "not recorded", ha="center", va="center", fontsize=8,
                color=MUTED)
    ax.set_title("1 bag = 1 clock hour.  Label: hourly calling index 0–3 per species "
                 "(annotators heard only the two clips)", pad=10)

    # Row 2: each clip tiled into 12 contiguous 5 s windows (the instances)
    call_windows = {0: [4], 1: [7, 8]}           # toy calls, for illustration
    for k, (x0, name, _) in enumerate(clips):
        a = fig.add_subplot(gs[1, k])
        a.set_xlim(0, 60); a.set_ylim(0.1, 1.3); a.grid(False)
        a.set_yticks([]); a.spines["left"].set_visible(False)
        a.set_xticks(range(0, 61, 10), [f"{s}s" for s in range(0, 61, 10)])
        for j in range(12):
            call = j in call_windows[k]
            a.add_patch(Rectangle((j * 5 + 0.15, 0.2), 4.7, 0.8,
                                  fc=SPECIES_COLOR["gastrotheca"] if call else TRAINED,
                                  ec="none"))
            a.text(j * 5 + 2.5, 0.6, str(k * 12 + j + 1), ha="center", va="center",
                   fontsize=7, color=SURFACE if call else INK2)
        for j in call_windows[k]:                  # calls: ~0.3 s ticks
            for t in rng.uniform(j * 5 + 0.7, j * 5 + 4.3, 2):
                a.plot([t, t], [1.05, 1.22], color=SPECIES_COLOR["gastrotheca"], lw=1.5)
        a.set_title(f"{name}: 12 contiguous 5 s windows (instances)", fontsize=9)

    fig.text(0.5, -0.04,
             "We only have bag labels (one per hour). The model scores every instance and "
             "predicts the bag label from those instance predictions.",
             ha="center", fontsize=10, color=INK, weight="bold")
    save(fig, "bag_structure",
         "How an hour becomes a MIL bag: two 1-min clips per hour, each tiled into 12 "
         "contiguous 5 s windows (instances). Labels exist only per bag. The model "
         "predicts the bag label from its instance predictions.")


# --------------------------------------------------------------------------- M2
def pipeline(save, n_windows=24, dim=1536):
    lin = dim * 2 + 2
    att = 2 * (dim * 128 + 128) + 128 * 2 + 2
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0.5, 4.6); ax.axis("off")

    # row 1: frozen feature extraction
    y1, h = 3.1, 1.1
    _box(ax, 0.1, y1, 2.0, h, "1 hour of audio\n2 × 1 min clips\n44.1 kHz")
    _box(ax, 2.7, y1, 2.2, h, f"{n_windows} windows × 5 s\nresampled to 32 kHz")
    _box(ax, 5.5, y1, 2.4, h, "Perch v2\n(frozen)", fc=FROZEN,
         weight="bold")
    _box(ax, 8.5, y1, 3.3, h, f"$X$  [{n_windows} × {dim}]\nz-scored with train mean/std",
         fc=SURFACE)
    for a, b in [(2.1, 2.7), (4.9, 5.5), (7.9, 8.5)]:
        _arrow(ax, (a, y1 + h / 2), (b, y1 + h / 2))

    # row 2: trained MIL head
    y2 = 0.9
    _box(ax, 0.1, y2, 2.5, 1.3, f"instance probe\nlinear: {lin:,} params", fc=TRAINED, weight="bold")
    # instance logits glyph
    ax.text(4.25, y2 + 1.3, f"window logits  [{n_windows} × 2]", ha="center", fontsize=8.5,
            color=INK)
    vals = np.r_[np.full(6, 0.05), 0.9, np.full(9, 0.1), 0.6, np.full(7, 0.05)]
    for j in range(n_windows):
        for c, sp in enumerate(SPECIES):
            ax.add_patch(Rectangle((3.05 + j * 0.1, y2 + 0.72 - c * 0.3), 0.085, 0.26,
                                   fc=SPECIES_COLOR[sp], alpha=0.12 + 0.85 * (
                                       vals[j] if c == 0 else vals[::-1][j] * 0.5),
                                   ec="none"))
    ax.text(3.0, y2 + 0.85, "G", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.text(3.0, y2 + 0.55, "O", ha="right", va="center", fontsize=7.5, color=INK2)
    _box(ax, 6.0, y2 - 0.25, 2.3, 1.8, "", fc=TRAINED)
    ax.text(7.15, y2 + 1.3, "MIL pooling", ha="center", fontsize=9, weight="bold")
    for k, p in enumerate(POOL_ORDER):
        ax.text(6.35, y2 + 0.98 - k * 0.24, "■", color=pcolor(p), fontsize=9, va="center")
        ax.text(6.6, y2 + 0.98 - k * 0.24, plabel(p) + ("  (+%s)" % f"{att:,}"
                if p == "attention" else ""), fontsize=7.8, va="center", color=INK)
    _box(ax, 8.9, y2 + 0.3, 1.2, 0.9, "bag logit\n$s$  [2]")
    _box(ax, 10.6, y2 + 0.85, 1.3, 0.75, "$P$(present)\n= $\\sigma(s)$", fc=SURFACE, weight="bold")
    _box(ax, 10.6, y2 - 0.25, 1.3, 0.85, "ordinal (opt.)\n$P(\\mathrm{idx} \\geq k)$\n= $\\sigma(s - b_k)$",
         fc=SURFACE, fs=7.8)
    _arrow(ax, (2.6, y2 + 0.65), (3.0, y2 + 0.65))
    _arrow(ax, (5.5, y2 + 0.65), (6.0, y2 + 0.65))
    _arrow(ax, (8.3, y2 + 0.75), (8.9, y2 + 0.75))
    _arrow(ax, (10.1, y2 + 0.85), (10.6, y2 + 1.2))
    _arrow(ax, (10.1, y2 + 0.65), (10.6, y2 + 0.2))
    # X feeds the probe (elbow above the head row)
    ax.plot([10.15, 10.15, 1.35], [y1, 2.85, 2.85], color=INK2, lw=1.4,
            solid_joinstyle="round")
    _arrow(ax, (1.35, 2.86), (1.35, y2 + 1.3))
    ax.text(0.1, 4.45, "Frozen feature extraction (run once)", fontsize=10, weight="bold")
    ax.text(2.0, 2.4, "Trained MIL head", fontsize=10,
            weight="bold")
    save(fig, "pipeline",
         "Model pipeline: frozen Perch v2 embeddings per 5 s window, a per-window probe, "
         "a pooling function that turns 24 window logits into one bag logit per species, "
         "and an optional cumulative-link ordinal head on that same logit.")


# --------------------------------------------------------------------------- M3
def _pool_numpy(name, logits):
    """Run the project's own pooler on a [N] (one species) logit vector."""
    x = torch.tensor(logits, dtype=torch.float32)[None, :, None]
    mask = torch.ones(1, x.shape[1], dtype=torch.bool)
    bag, w = POOLERS[name]()(x, mask)
    return float(torch.sigmoid(bag)), w[0, :, 0].numpy()


def pooling_toy(save, n=24, pos=2.0, neg=-3.0):
    fixed = [p for p in POOL_ORDER if p != "attention"]
    fig = plt.figure(figsize=(12, 4.4))
    gs = fig.add_gridspec(len(fixed), 3, width_ratios=[1.1, 1.1, 1.3], hspace=0.35,
                          wspace=0.3)
    a1, a2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])

    ks = np.arange(0, n + 1)
    for p in fixed:
        ys = [_pool_numpy(p, np.r_[np.full(k, pos), np.full(n - k, neg)])[0] for k in ks]
        a1.plot(ks, ys, color=pcolor(p), label=plabel(p))
    a1.set(xlabel=f"windows containing a call (of {n})", ylabel="bag P(present)",
           ylim=(0, 1.02), xlim=(0, n))
    a1.set_title("A. More calls in the hour")
    a1.text(n * 0.98, 0.05, f"call window logit {pos:+g}\nsilent window logit {neg:+g}",
            ha="right", fontsize=7.5, color=MUTED)

    xs = np.linspace(-4, 10, 120)
    for p in fixed:
        ys = [_pool_numpy(p, np.r_[x, np.full(n - 1, neg)])[0] for x in xs]
        a2.plot(xs, ys, color=pcolor(p), label=plabel(p))
    a2.set(xlabel="logit of the single call window", ylabel="bag P(present)",
           ylim=(0, 1.02))
    a2.set_title("B. One call, growing confidence")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.14), fontsize=8)
    bag = np.full(n, neg); bag[[5, 6, 17]] = [pos, pos - 1.5, pos + 1]
    for r, p in enumerate(fixed):
        a = fig.add_subplot(gs[r, 2])
        prob, w = _pool_numpy(p, bag)
        a.bar(np.arange(n), w, color=pcolor(p), width=0.8)
        a.set_xlim(-0.6, n - 0.4); a.set_ylim(0, 1.05); a.set_yticks([0, 1])
        a.set_xticks([] if r < len(fixed) - 1 else [0, 11, 23],
                     [] if r < len(fixed) - 1 else ["1", "12", "24"])
        a.axvline(11.5, color=AXIS, lw=0.8)
        a.grid(axis="x", visible=False)
        a.text(n - 0.5, 0.95, f"{plabel(p)}: P = {prob:.2f}", ha="right", va="top",
               fontsize=7.5, color=INK)
        if r == 0:
            a.set_title("C. Weight each window gets (3 call windows)")
    fig.axes[-1].set_xlabel("window")
    save(fig, "pooling_toy",
         "How the fixed poolers turn window logits into a bag score, using the project's "
         "own pooling code on toy inputs. Mean needs calls to fill the hour. Max reacts "
         "to one confident window but gives gradient to that window only. LME and "
         "linear-softmax sit in between. Attention is learned, so it is shown on real "
         "data in figure 10.")
