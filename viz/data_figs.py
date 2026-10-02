"""Data figures: the labelled calendar with its split blocks, and split counts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (AXIS, GRID, INDEX_COLOR, INDEX_LABEL, INK2, MUTED, SP_SHORT,
                    SPLIT_COLOR, SURFACE)
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from frog_mil.data import SPECIES


def load_bags(path) -> pd.DataFrame:
    b = pd.read_csv(path, parse_dates=["datetime"])
    b["date"] = pd.to_datetime(b["date"])
    return b


GAP_DAYS = 7       # a run of nights longer than this without audio is drawn as a break
GAP_COLS = 3       # width of that break, in night columns


def night_positions(nights: pd.Series):
    """x position of every night, with long stretches without audio collapsed.

    Returns ({night: column}, n_columns, [(first_col, last_col) per segment],
    [(break_col, first_missing_night, last_missing_night) per break]).
    """
    have = sorted(nights.unique())
    segs = [[have[0], have[0]]]
    for d in have[1:]:
        if (d - segs[-1][1]).days > GAP_DAYS:
            segs.append([d, d])
        else:
            segs[-1][1] = d
    pos, spans, breaks, x = {}, [], [], 0
    for k, (a, b) in enumerate(segs):
        if k:
            breaks.append((x, segs[k - 1][1] + pd.Timedelta(days=1), a - pd.Timedelta(days=1)))
            x += GAP_COLS
        days = pd.date_range(a, b, freq="D")
        pos.update({d: x + i for i, d in enumerate(days)})
        spans.append((x, x + len(days) - 1))
        x += len(days)
    return pos, x, spans, breaks


def calendar(save, bags: pd.DataFrame):
    # Nights are the unit of calling, so each column runs noon -> noon: hours
    # before 12:00 belong to the previous evening's column.
    bags = bags.assign(night=bags["date"] - pd.to_timedelta((bags["hour"] < 12).astype(int),
                                                            unit="D"),
                       y=(bags["hour"] + 12) % 24)
    dpos, ncol, spans, breaks = night_positions(bags["night"])
    cmap = ListedColormap([INDEX_COLOR[k] for k in range(4)])
    cmap.set_bad(SURFACE)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

    # Tick the first of each month inside a segment, and the segment's first night
    # unless a month start is close enough that the labels would collide.
    ticks = []
    for a, b in spans:
        seg = sorted((x, d) for d, x in dpos.items() if a <= x <= b)
        months = [(x, d) for x, d in seg if d.day == 1]
        if not months or months[0][0] - a > 8:
            ticks.append(seg[0])
        ticks += months

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.18, 1, 1], hspace=0.12)
    a0 = fig.add_subplot(gs[0])
    split_day = bags.groupby("date")["split"].first()
    for d, s in split_day.items():
        if d not in dpos:
            continue
        a0.add_patch(plt.Rectangle((dpos[d], 0), 1, 1, fc=SPLIT_COLOR[s], ec="none"))
    a0.set_xlim(0, ncol); a0.set_ylim(0, 1); a0.axis("off")
    a0.set_title("Labelled hours with audio, one column per night (noon → noon).  "
                 "Top strip: split of each 3-day block", pad=8)

    for c, sp in enumerate(SPECIES):
        a = fig.add_subplot(gs[c + 1], sharex=None)
        grid = np.full((24, ncol), np.nan)
        for d, h, v in bags[["night", "y", f"{sp}_index"]].itertuples(index=False):
            grid[h, dpos[d]] = v
        a.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, norm=norm,
                 origin="lower", extent=(0, ncol, 0, 24), interpolation="nearest")
        a.grid(False)
        for x, first, last in breaks:
            a.axvspan(x, x + GAP_COLS, fc=SURFACE, ec=GRID, lw=0, hatch="//")
            a.text(x + GAP_COLS / 2, 12, f"no audio  {first:%d %b} – {last:%d %b}",
                   rotation=90, ha="center", va="center", fontsize=7.5, color=MUTED,
                   bbox=dict(fc=SURFACE, ec="none", pad=1.5))
        a.set_yticks([0, 6, 12, 18, 24], ["12:00", "18:00", "00:00", "06:00", "12:00"])
        a.set_ylabel("time (noon → noon)")
        a.set_xticks([x + 0.5 for x, _ in ticks],
                     [d.strftime("%d %b") for _, d in ticks] if c == len(SPECIES) - 1 else [])
        pos = int((bags[f"{sp}_index"] > 0).sum())
        a.text(0.005, 0.97, f"{SP_SHORT[sp]}   ({pos} positive hours of {len(bags)})",
               transform=a.transAxes, va="top", fontsize=9, style="italic", color=INK2,
               bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.5))
        for sp_ in ("left", "bottom"):
            a.spines[sp_].set_color(AXIS)
    # Two legends: the train colour is also the index-2 colour, so keep them apart.
    l1 = fig.legend(handles=[Patch(fc=INDEX_COLOR[k], label=INDEX_LABEL[k]) for k in range(4)],
                    title="calling index", ncol=4, loc="lower right",
                    bbox_to_anchor=(0.62, -0.06), fontsize=8, title_fontsize=8)
    fig.legend(handles=[Patch(fc=c, label=s) for s, c in SPLIT_COLOR.items()],
               title="split (top strip)", ncol=3, loc="lower left",
               bbox_to_anchor=(0.66, -0.06), fontsize=8, title_fontsize=8)
    fig.add_artist(l1)
    save(fig, "calendar",
         "Every labelled hour with audio, coloured by calling index, for each species. Each "
         "column is one night (noon to noon), so a night of calling is one contiguous "
         "band. Stretches of more than a week without audio are collapsed into a hatched "
         "break. The top strip shows which split each 3-day block belongs to. Both frogs "
         "call at night and neither calls in the Feb–Apr recordings; Oreobates calls only "
         "from October on, which is why the clock baseline does well for Oreobates.")
