"""Result figures, built from out-of-fold predictions and the shared block bootstrap."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import (
    AXIS,
    BASE_COLOR,
    BLUE_RAMP,
    GRID,
    INK,
    INK2,
    MUTED,
    POOL_ORDER,
    REGIME_COLOR,
    SP_SHORT,
    SURFACE,
    Info,
    Stats,
    ci,
    pcolor,
    plabel,
    pmarker,
    pooler_legend_handles,
    scale_fonts,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from frog_mil.config import SPECIES

WHATS = [("macro", "macro AP"), (0, f"{SP_SHORT['gastrotheca']} AP"),
         (1, f"{SP_SHORT['oreobates']} AP")]
BLUES = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)


def _point(ax, x, y, i: Info, size=6.5, zorder=3, **kw):
    """Colour = pooler, marker = pooler, hollow = non-linear probe, grey = baseline."""
    if not i.is_model:
        ax.plot(x, y, marker="D", color=BASE_COLOR, ms=size * 0.85, ls="", zorder=zorder, **kw)
        return
    hollow = i.hidden > 0
    ax.plot(x, y, marker=pmarker(i.pooling), ms=size, ls="", zorder=zorder,
            mfc=SURFACE if hollow else pcolor(i.pooling), mec=pcolor(i.pooling),
            mew=1.6 if hollow else 0.8, **kw)


def _prev(infos):
    y = infos[0].m.y
    return [float((y[:, c] > 0).mean()) for c in range(len(SPECIES))]


def _chance(infos, what):
    p = _prev(infos)
    return float(np.mean(p)) if what == "macro" else p[what]


def _legend(fig_or_ax, infos, extra=(), **kw):
    pools = [p for p in POOL_ORDER if any(i.pooling == p for i in infos if i.is_model)]
    hid = sorted({i.hidden for i in infos if i.is_model and i.hidden})
    hs = pooler_legend_handles(pools, hollow_probe=f"MLP-{hid[0]}" if hid else None)
    if any(not i.is_model for i in infos):
        hs.append(Line2D([], [], ls="", marker="D", color=BASE_COLOR, ms=6,
                         label="baseline (no training)"))
    fig_or_ax.legend(handles=hs + list(extra), **kw)


# --------------------------------------------------------------------------- R1
def forest(save, infos: list[Info], st: Stats, ref: str):
    order = sorted(infos, key=lambda i: -st.macro(i.model_id))
    n = len(order)
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.24 * n + 1.6), sharey=True)
    ys = np.arange(n)[::-1]
    for a, (what, lab) in zip(axes, WHATS):
        for y, i in zip(ys, order):
            p, b = st.get(i.model_id, what)
            lo, hi = ci(b)
            a.plot([lo, hi], [y, y], color=pcolor(i.pooling) if i.is_model else BASE_COLOR,
                   lw=1.3, alpha=0.55, solid_capstyle="round")
            _point(a, p, y, i)
        a.axvline(_chance(infos, what), color=MUTED, lw=1, zorder=1)
        a.text(_chance(infos, what), n - 0.2, " chance", fontsize=7, color=MUTED, va="bottom")
        a.set_xlim(0, 1); a.set_xlabel(lab); a.grid(axis="y", visible=False)
        bp = st.get(ref, what)[0]
        a.axvline(bp, color=GRID, lw=0.8, zorder=0)
    axes[0].set_yticks(ys, [i.model_id for i in order], fontsize=7.5)
    for t, i in zip(axes[0].get_yticklabels(), order):
        if i.model_id == ref:
            t.set_fontweight("bold"); t.set_color(INK)
    axes[0].set_ylim(-0.8, n - 0.2)
    _legend(fig, infos, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0 + 0.5 / n))
    fig.suptitle("Out-of-fold AP with 95% block-bootstrap CI, sorted by macro AP",
                 x=0.01, ha="left", y=1.0 + 1.6 / n, weight="bold")
    fig.tight_layout()
    save(fig, "forest_ap",
         "Out-of-fold average precision for every model and baseline (every hour is "
         "scored by the cross-validation model that never saw it), seed-averaged, with a "
         "95% block-bootstrap CI (3-day blocks resampled within each recording regime). The "
         "bold label and faint vertical line mark the validation-selected reference model. "
         "Colour and marker show the pooler, and hollow markers are the MLP probe.")


def forest_simple(save, infos: list[Info], st: Stats, run_id: str, font=1.8):
    """Slide version of ``forest``: the five poolers of one run plus the baselines,
    per-species AP only, in a fixed order (poolers, then baselines)."""
    pools = [i for p in ["max", "lme", "mean", "linear_softmax", "attention"]
             for i in infos if i.is_model and i.run_id == run_id and i.pooling == p]
    order = pools + [i for i in infos if not i.is_model]
    n = len(order)
    fig, axes = plt.subplots(1, 2, figsize=(7, 0.62 * n + 1.6), sharey=True)
    ys = np.arange(n)[::-1]
    for a, (what, lab) in zip(axes, WHATS[1:]):
        for y, i in zip(ys, order):
            p, b = st.get(i.model_id, what)
            lo, hi = ci(b)
            a.plot([lo, hi], [y, y], color=pcolor(i.pooling) if i.is_model else BASE_COLOR,
                   lw=2, alpha=0.55, solid_capstyle="round")
            _point(a, p, y, i, size=8)
        a.axvline(_chance(infos, what), color=MUTED, lw=1, zorder=1)
        a.text(_chance(infos, what), n - 0.3, " chance", fontsize=8, color=MUTED, va="bottom")
        a.axhline(n - len(pools) - 0.5, color=GRID, lw=0.8)
        a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_xlabel(lab)
        a.grid(axis="y", visible=False)
    labels = [plabel(i.pooling) if i.is_model else i.label.replace(", ", ",\n") for i in order]
    axes[0].set_yticks(ys, [t[:1].upper() + t[1:] for t in labels])
    axes[0].set_ylim(-0.7, n - 0.3)
    scale_fonts(fig, font)
    fig.tight_layout()
    save(fig, "forest_simple",
         f"Simplified forest plot for slides: out-of-fold AP per species for each pooler "
         f"in `{run_id}` (colours as in figure 5) and the three baselines (grey), as "
         f"seed-averaged AP with a 95% block-bootstrap CI.")


# --------------------------------------------------------------------------- R4
def _pairs(models):
    """Controlled pairs: (factor, context label, ref Info, alt Info)."""
    key = {(i.hidden, i.w, i.pooling): i for i in models}
    out = []
    for (h, w, p), a in key.items():
        if h == 0:
            for (h2, w2, p2), b in key.items():
                if h2 and (w2, p2) == (w, p):
                    out.append(("probe", f"MLP-{h2} − linear", f"w={w:g}", a, b))
        if w == 0:
            for (h2, w2, p2), b in key.items():
                if w2 and (h2, p2) == (h, p):
                    out.append(("ordinal", f"ordinal w={w2:g} − binary", a.probe, a, b))
    return out


def effects(save, infos, st: Stats):
    models = [i for i in infos if i.is_model]
    pairs = _pairs(models)
    factors = [f for f in ("probe", "ordinal") if any(p[0] == f for p in pairs)]
    if not factors:
        return
    groups = {f: sorted({(p[1], p[2]) for p in pairs if p[0] == f},
                     key=lambda g: (g[1].startswith("MLP"), g[1], g[0])) for f in factors}
    heights = [len(g) * 1.0 + 0.6 for g in groups.values()]
    fig, axes = plt.subplots(len(factors), 1, figsize=(10, sum(heights) * 0.55 + 1.2),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    pools = [p for p in POOL_ORDER if any(i.pooling == p for i in models)]
    for a, f in zip(axes[:, 0], factors):
        labels = []
        for g, (comp, ctx) in enumerate(groups[f]):
            y0 = len(groups[f]) - 1 - g
            labels.append((y0, f"{comp}\n({ctx})"))
            sel = [p for p in pairs if p[0] == f and (p[1], p[2]) == (comp, ctx)]
            sel.sort(key=lambda p: pools.index(p[3].pooling) if p[3].pooling in pools else 99)
            for k, (_, _, _, ref, alt) in enumerate(sel):
                y = y0 + 0.3 - 0.6 * k / max(len(pools) - 1, 1)
                d = st.macro(alt.model_id) - st.macro(ref.model_id)
                lo, hi = ci(st.macro_boot(alt.model_id) - st.macro_boot(ref.model_id))
                sig = lo > 0 or hi < 0
                a.plot([lo, hi], [y, y], color=pcolor(ref.pooling), lw=1.4, alpha=0.6)
                a.plot(d, y, marker=pmarker(ref.pooling), ms=6.5, ls="",
                       mfc=pcolor(ref.pooling) if sig else SURFACE, mec=pcolor(ref.pooling),
                       mew=1.5)
        a.axvline(0, color=INK2, lw=1)
        a.set_yticks([l[0] for l in labels], [l[1] for l in labels], fontsize=8)
        a.set_ylim(-0.6, len(groups[f]) - 0.4)
        a.grid(axis="y", visible=False)
        for y in range(len(groups[f]) - 1):
            a.axhline(y + 0.5, color=GRID, lw=0.8)
        a.set_title({"probe": "Effect of the probe",
                     "ordinal": "Effect of the ordinal loss"}[f])
        a.set_xlabel("Δ macro AP (paired, 95% CI)")
    hs = pooler_legend_handles(pools) + [
        Line2D([], [], ls="", marker="o", color=INK2, label="CI excludes 0"),
        Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=INK2, mew=1.5, label="CI includes 0")]
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.5, 1.0))
    save(fig, "effects",
         "Controlled comparisons: each Δ compares two models that differ in one factor "
         "only (probe or ordinal loss weight), with the same pooler, "
         "splits and seeds. Filled markers are CIs that exclude 0.")


# --------------------------------------------------------------------------- R5
def ordinal_sweep(save, infos, st: Stats):
    models = [i for i in infos if i.is_model]
    groups = {}
    for i in models:
        groups.setdefault(i.hidden, set()).add(i.w)
    groups = {k: sorted(v) for k, v in groups.items() if len(v) >= 3}
    for h, ws in groups.items():
        sub = [i for i in models if i.hidden == h]
        pools = [p for p in POOL_ORDER if any(i.pooling == p for i in sub)]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        x = np.arange(len(ws))
        for a, (what, lab) in zip(axes, WHATS):
            for k, p in enumerate(pools):
                dx = (k - (len(pools) - 1) / 2) * 0.07
                pts, los, his = [], [], []
                for w in ws:
                    i = next((j for j in sub if j.w == w and j.pooling == p), None)
                    if i is None:
                        pts.append(np.nan); los.append(np.nan); his.append(np.nan)
                        continue
                    v, b = st.get(i.model_id, what)
                    lo, hi = ci(b)
                    pts.append(v); los.append(lo); his.append(hi)
                a.vlines(x + dx, los, his, color=pcolor(p), lw=1.2, alpha=0.45)
                a.plot(x + dx, pts, color=pcolor(p), marker=pmarker(p), ms=6, lw=1.8,
                       label=plabel(p))
            a.set_xticks(x, [f"{w:g}" + (" (binary)" if w == 0 else "") for w in ws])
            a.set_xlabel("ordinal loss weight w"); a.set_title(lab)
            a.axhline(_chance(infos, what), color=MUTED, lw=1)
        axes[0].set_ylabel("out-of-fold AP (95% CI)")
        axes[-1].legend(loc="lower right", fontsize=7.5)
        probe = f"MLP-{h}" if h else "linear"
        name = f"ordinal_sweep_{'mlp' + str(h) if h else 'linear'}"
        fig.suptitle(f"Ordinal-loss weight sweep, {probe} probe", x=0.01, ha="left",
                     weight="bold")
        fig.tight_layout()
        save(fig, name,
             f"Out-of-fold AP against the ordinal loss weight w for the {probe} probe. Points are "
             "dodged sideways so the CIs stay readable. The grey line is chance. Use "
             "`effects` for paired Δ vs w = 0.")


# --------------------------------------------------------------------------- R6
def per_index(save, infos, st: Stats, run_id: str):
    sub = sorted([i for i in infos if i.run_id == run_id],
                 key=lambda i: POOL_ORDER.index(i.pooling) if i.pooling in POOL_ORDER else 9)
    base = [i for i in infos if not i.is_model]
    fig, axes = plt.subplots(1, len(SPECIES), figsize=(12, 3.9), sharey=True)
    for c, (a, sp) in enumerate(zip(axes, SPECIES)):
        lvls = [cell[2] for cell in st.cells if cell[0] == "level" and cell[1] == c]
        n_pos = {l: int((infos[0].m.idx[:, c] == l).sum()) for l in (1, 2, 3)}
        x = np.arange(len(lvls))
        for k, i in enumerate(sub):
            dx = (k - (len(sub) - 1) / 2) * 0.06
            pts = [st.ap(i.model_id, ("level", c, l)) for l in lvls]
            cis = [ci(st.ap_boot(i.model_id, ("level", c, l))) for l in lvls]
            a.vlines(x + dx, [q[0] for q in cis], [q[1] for q in cis], color=pcolor(i.pooling),
                     lw=1.2, alpha=0.45)
            a.plot(x + dx, pts, color=pcolor(i.pooling), marker=pmarker(i.pooling), ms=6.5,
                   ls="", label=plabel(i.pooling))
        for k, i in enumerate(base):
            a.plot(x + 0.34, [st.ap(i.model_id, ("level", c, l)) for l in lvls], ls="", marker="D",
                   ms=4.5, color=BASE_COLOR, alpha=0.4 + 0.2 * k,
                   label=i.label if c == 0 else None)
        n0 = int((infos[0].m.idx[:, c] == 0).sum())
        for xx, l in zip(x, lvls):
            npos = n_pos[l]
            a.plot([xx - 0.4, xx + 0.4], [npos / (npos + n0)] * 2, color=INK2, lw=1.2)
        a.set_xticks(x, [f"index {l}\n(n = {n_pos[l]})" for l in lvls])
        a.set_title(SP_SHORT[sp], style="italic"); a.set_ylim(0, 1.02)
        skipped = [l for l in (1, 2, 3) if not st.has(("level", c, l))]
        if skipped:
            a.text(0.99, 0.98, "omitted (n < 5): " + ", ".join(f"index {l}" for l in skipped),
                   transform=a.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    axes[0].set_ylabel("AP vs silent hours (95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=1.2)); l.append("chance")
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.05), fontsize=7.5)
    fig.suptitle(f"AP by calling index — {run_id}", x=0.01, ha="left", y=1.1, weight="bold")
    fig.tight_layout()
    save(fig, "per_index",
         f"AP of each calling index against silent hours, for every pooler in `{run_id}` "
         "and the baselines (grey diamonds). Short dark bars are chance. Index 1 (isolated "
         "calls) is the hard, pooling-sensitive case.")


# --------------------------------------------------------------------------- R7
def val_vs_test(save, infos, st: Stats):
    from scipy.stats import spearmanr
    models = [i for i in infos if i.is_model]
    v = np.array([i.m.val_ap.mean() for i in models])
    t = np.array([st.macro(i.model_id) for i in models])
    fig, a = plt.subplots(figsize=(7.8, 5.4))
    lo, hi = min(v.min(), t.min()) - 0.02, max(v.max(), t.max()) + 0.02
    a.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    a.text(hi, hi, "val = test ", ha="right", va="bottom", fontsize=7, color=MUTED,
           rotation=45, rotation_mode="anchor", transform_rotates_text=True)
    for i, vv, tt in zip(models, v, t):
        _point(a, vv, tt, i, size=7)
    r = spearmanr(v, t).statistic
    a.text(0.02, 0.97, f"Spearman ρ = {r:.2f}  ({len(models)} models)", transform=a.transAxes,
           va="top", fontsize=8.5, color=INK)
    a.set(xlabel="validation macro AP (mean over seeds × folds)",
          ylabel="out-of-fold test macro AP",
          xlim=(lo, hi), ylim=(lo, hi))
    a.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    a.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    _legend(a, models, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5)
    a.set_title("Does validation rank models like test?")
    scale_fonts(fig, 1.65)
    save(fig, "val_vs_test",
         "Validation vs out-of-fold test macro AP per model. Validation AP is each fold "
         "model's early-stopping optimum on its validation fold, averaged over folds and "
         "seeds, so it is optimistic. The report selects its reference model on this axis; "
         "a weak rank correlation means that choice is close to arbitrary among the top "
         "models.")


# --------------------------------------------------------------------------- R8
def per_regime(save, infos, st: Stats, run_id: str, regime: np.ndarray):
    """Presence AP within each recording regime, per pooler of one run."""
    sub = sorted([i for i in infos if i.run_id == run_id],
                 key=lambda i: POOL_ORDER.index(i.pooling) if i.pooling in POOL_ORDER else 9)
    base = [i for i in infos if not i.is_model]
    regs = [r for r in REGIME_COLOR if r in set(regime)] + sorted(set(regime) - set(REGIME_COLOR))
    fig, axes = plt.subplots(1, len(SPECIES), figsize=(12, 3.9), sharey=True)
    y = infos[0].m.idx > 0
    for c, (a, sp) in enumerate(zip(axes, SPECIES)):
        rr = [r for r in regs if st.has(("regime", c, r))]
        x = np.arange(len(rr))
        for k, i in enumerate(sub):
            dx = (k - (len(sub) - 1) / 2) * 0.06
            pts = [st.ap(i.model_id, ("regime", c, r)) for r in rr]
            cis = [ci(st.ap_boot(i.model_id, ("regime", c, r))) for r in rr]
            a.vlines(x + dx, [q[0] for q in cis], [q[1] for q in cis], color=pcolor(i.pooling),
                     lw=1.2, alpha=0.45)
            a.plot(x + dx, pts, color=pcolor(i.pooling), marker=pmarker(i.pooling), ms=6.5,
                   ls="", label=plabel(i.pooling))
        for k, i in enumerate(base):
            a.plot(x + 0.34, [st.ap(i.model_id, ("regime", c, r)) for r in rr], ls="",
                   marker="D", ms=4.5, color=BASE_COLOR, alpha=0.4 + 0.2 * k,
                   label=i.label if c == 0 else None)
        for xx, r in zip(x, rr):
            a.plot([xx - 0.4, xx + 0.4], [y[regime == r, c].mean()] * 2, color=INK2, lw=1.2)
        a.set_xticks(x, [f"{r}\n({int(y[regime == r, c].sum())} / {int((regime == r).sum())})"
                         for r in rr])
        a.set_title(SP_SHORT[sp], style="italic"); a.set_ylim(0, 1.02)
    axes[0].set_ylabel("presence AP within regime (95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=1.2)); l.append("chance")
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.05), fontsize=7.5)
    fig.suptitle(f"AP by recording regime — {run_id}", x=0.01, ha="left", y=1.1,
                 weight="bold")
    fig.tight_layout()
    save(fig, "per_regime",
         f"Presence AP within each recording regime for every pooler in `{run_id}` and the "
         "baselines (grey diamonds); tick labels give positive / total hours. `8k-3clip` = "
         "8 kHz, three 1 min clips per hour (Sep–Oct 2018); `44k-1clip` = 44.1 kHz, one clip "
         "(Nov–Dec 2018); `44k-2clip` = 44.1 kHz, two clips (2019). Short dark bars are "
         "chance, which differs between regimes, so compare each group with its own bar.")
