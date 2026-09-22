"""Data figures: the labelled calendar with its split blocks, and split counts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import AXIS, INDEX_COLOR, INDEX_LABEL, INK2, SP_SHORT, SPLIT_COLOR, SURFACE
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from frog_mil.data import SPECIES


def load_bags(path) -> pd.DataFrame:
    b = pd.read_csv(path, parse_dates=["datetime"])
    b["date"] = pd.to_datetime(b["date"])
    return b


def calendar(save, bags: pd.DataFrame):
    # Nights are the unit of calling, so each column runs noon -> noon: hours
    # before 12:00 belong to the previous evening's column.
    bags = bags.assign(night=bags["date"] - pd.to_timedelta((bags["hour"] < 12).astype(int),
                                                            unit="D"),
                       y=(bags["hour"] + 12) % 24)
    days = pd.date_range(bags["night"].min(), bags["night"].max(), freq="D")
    dpos = {d: i for i, d in enumerate(days)}
    cmap = ListedColormap([INDEX_COLOR[k] for k in range(4)])
    cmap.set_bad(SURFACE)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.18, 1, 1], hspace=0.12)
    a0 = fig.add_subplot(gs[0])
    split_day = bags.groupby("date")["split"].first()
    for d, s in split_day.items():
        if d not in dpos:
            continue
        a0.add_patch(plt.Rectangle((dpos[d], 0), 1, 1, fc=SPLIT_COLOR[s], ec="none"))
    a0.set_xlim(0, len(days)); a0.set_ylim(0, 1); a0.axis("off")
    a0.set_title("Labelled hours with audio, one column per night (noon → noon).  "
                 "Top strip: split of each 3-day block", pad=8)

    for c, sp in enumerate(SPECIES):
        a = fig.add_subplot(gs[c + 1], sharex=None)
        grid = np.full((24, len(days)), np.nan)
        for d, h, v in bags[["night", "y", f"{sp}_index"]].itertuples(index=False):
            grid[h, dpos[d]] = v
        a.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, norm=norm,
                 origin="lower", extent=(0, len(days), 0, 24), interpolation="nearest")
        a.grid(False)
        a.set_yticks([0, 6, 12, 18, 24], ["12:00", "18:00", "00:00", "06:00", "12:00"])
        a.set_ylabel("time (noon → noon)")
        month_starts = [i for i, d in enumerate(days) if d.day == 1 or i == 0]
        a.set_xticks(month_starts, [days[i].strftime("%d %b") for i in month_starts]
                     if c == len(SPECIES) - 1 else [])
        pos = int((bags[f"{sp}_index"] > 0).sum())
        a.text(0.005, 0.97, f"{SP_SHORT[sp]}   ({pos} positive hours of {len(bags)})",
               transform=a.transAxes, va="top", fontsize=9, style="italic", color=INK2,
               bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.5))
        for sp_ in ("left", "bottom"):
            a.spines[sp_].set_color(AXIS)
    fig.legend(handles=[Patch(fc=INDEX_COLOR[k], label=INDEX_LABEL[k]) for k in range(4)]
               + [Patch(fc=c, label=f"{s} block") for s, c in SPLIT_COLOR.items()],
               ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
    save(fig, "calendar",
         "Every labelled hour with audio, coloured by calling index, for each species. Each "
         "column is one night (noon to noon), so a night of calling is one contiguous "
         "band. "
         "The top strip shows which split each 3-day block belongs to. Both frogs call at "
         "night, and Oreobates only from October on, which is why the clock baseline does "
         "well for Oreobates.")
