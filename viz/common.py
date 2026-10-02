"""Shared style, loading and statistics for the figure scripts.

Everything is derived from ``runs/`` (predictions + configs) and
``outputs/bags.csv``, so the figures always match whatever runs exist. The
bootstrap mirrors ``frog_mil.report`` (same resampling scheme, same seed), so
CIs agree with ``results/RESULTS.md`` when both are built from the same runs.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from frog_mil.data import SPECIES
from frog_mil.report import Model, ap_score, load_models

warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)

# --------------------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

POOL_ORDER = ["mean", "max", "lme", "linear_softmax", "attention"]
POOL_COLOR = {"mean": "#2a78d6", "max": "#eb6834", "lme": "#1baf7a",
              "linear_softmax": "#eda100", "attention": "#e87ba4"}
POOL_LABEL = {"mean": "mean", "max": "max", "lme": "LME",
              "linear_softmax": "linear-softmax", "attention": "attention"}
# Secondary encoding: five hues are past the all-pairs CVD-safe count, so every
# pooler also has its own marker shape.
POOL_MARKER = {"mean": "o", "max": "^", "lme": "D", "linear_softmax": "s", "attention": "P"}
BASE_COLOR = MUTED
SPECIES_COLOR = {"gastrotheca": "#008300", "oreobates": "#4a3aa7"}
SP_SHORT = {"gastrotheca": "G. chrysosticta", "oreobates": "O. berdemenos"}
SPLIT_COLOR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
INDEX_COLOR = {0: "#e1e0d9", 1: "#86b6ef", 2: "#2a78d6", 3: "#0d366b"}
INDEX_LABEL = {0: "0 silent", 1: "1 isolated calls", 2: "2 overlapping", 3: "3 chorus"}


def pcolor(pooling: str) -> str:
    return POOL_COLOR.get(pooling, BASE_COLOR)


def plabel(pooling: str) -> str:
    return POOL_LABEL.get(pooling, pooling)


def pmarker(pooling: str) -> str:
    return POOL_MARKER.get(pooling, "o")


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "lines.markersize": 6,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def pooler_legend_handles(poolings, probes=("linear",), hollow_probe=None):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker=pmarker(p), color=pcolor(p), ms=7, label=plabel(p))
          for p in poolings]
    if hollow_probe:
        hs += [Line2D([], [], ls="", marker="o", color=INK2, ms=7, label="linear probe"),
               Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=INK2, mew=1.5, ms=7,
                      label=f"{hollow_probe} probe")]
    return hs


# The ten figures, in reading order. Output files are numbered by this list.
FIGURES = ["bag_structure", "pipeline", "pooling_toy", "calendar", "forest_ap", "effects",
           "ordinal_sweep", "per_index", "val_vs_test", "pooling_example", "forest_simple"]


def scale_fonts(fig, k: float) -> None:
    """Multiply every text size in ``fig`` (ticks, labels, legends, annotations) by
    ``k``. Call before the figure's own layout step."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * k)


class Saver:
    """Writes ``NN_<name>.<fmt>`` for every format and keeps a caption index."""

    def __init__(self, out: Path, formats=("png", "pdf")):
        self.out, self.formats, self.items = out, formats, []

    def __call__(self, fig, name: str, caption: str) -> None:
        num = next(k for k, f in enumerate(FIGURES, 1) if name.startswith(f))
        stem = f"{num:02d}_{name}"
        self.out.mkdir(parents=True, exist_ok=True)
        for f in self.formats:
            fig.savefig(self.out / f"{stem}.{f}")
        plt.close(fig)
        self.items.append((stem, caption))
        print(f"  {stem}")

    def write_index(self, header: str) -> None:
        lines = ["# Figures\n", header, ""]
        for stem, cap in sorted(self.items):
            lines += [f"## {stem}\n", f"![{stem}]({stem}.png)\n", f"{cap}\n"]
        (self.out / "index.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- models
@dataclass
class Info:
    """A model or baseline plus the config fields the figures group by."""
    m: Model
    model_id: str
    run_id: str
    pooling: str
    kind: str          # model | baseline
    hidden: int = 0
    w: float = 0.0     # ordinal weight
    stride: int = 2
    label: str = ""

    @property
    def probe(self) -> str:
        return f"MLP-{self.hidden}" if self.hidden else "linear"

    @property
    def is_model(self) -> bool:
        return self.kind == "model"


def load_infos(runs: Path) -> list[Info]:
    out = []
    for m in load_models(runs):
        if m.kind == "baseline":
            out.append(Info(m, m.model_id, "baseline", m.model_id.split("/")[1], "baseline",
                            label=m.pooling))
            continue
        run_id, pooling = m.model_id.split("/")
        cfg = json.loads((runs / run_id / "config.json").read_text())
        out.append(Info(m, m.model_id, run_id, pooling, "model",
                        hidden=int(cfg["hidden"]), w=float(cfg["ordinal_weight"]),
                        stride=int(cfg.get("stride", 2)), label=m.model_id))
    return out


# --------------------------------------------------------------------------- stats
def _boot_worker(args):
    """Point AP and bootstrap AP for one model: presence + each index level.

    Returns point[k], boots[B, k] where k runs over ``cells``: (c, lvl) with
    lvl=0 meaning presence (idx > 0 vs rest) and lvl>0 meaning idx==lvl vs 0.
    """
    scores, idx, draws, cells = args
    n = idx.shape[0]
    rows_all = [np.arange(n)] + list(draws)
    res = np.full((len(rows_all), len(cells)), np.nan)
    for k, (c, lvl) in enumerate(cells):
        if lvl == 0:
            keep, pos = np.ones(n, bool), idx[:, c] > 0
        else:
            keep, pos = (idx[:, c] == 0) | (idx[:, c] == lvl), idx[:, c] == lvl
        for j, rows in enumerate(rows_all):
            r = rows[keep[rows]]
            yy = pos[r].astype(float)
            res[j, k] = np.nanmean([ap_score(yy, sc[r, c]) for sc in scores])
    return res[0], res[1:]


@dataclass
class Stats:
    ids: list[str]
    cells: list[tuple[int, int]]
    point: dict      # model_id -> [cells]
    boots: dict      # model_id -> [B, cells]
    n_pos: dict      # (c, lvl) -> n positives

    def k(self, c, lvl=0):
        return self.cells.index((c, lvl))

    def ap(self, mid, c, lvl=0):
        return self.point[mid][self.k(c, lvl)]

    def ap_boot(self, mid, c, lvl=0):
        return self.boots[mid][:, self.k(c, lvl)]

    def macro(self, mid):
        return float(np.mean([self.ap(mid, c) for c in range(len(SPECIES))]))

    def macro_boot(self, mid):
        return np.mean([self.ap_boot(mid, c) for c in range(len(SPECIES))], axis=0)

    def get(self, mid, what):
        """what: 'macro' or a species index -> (point, boots)."""
        if what == "macro":
            return self.macro(mid), self.macro_boot(mid)
        return self.ap(mid, what), self.ap_boot(mid, what)


def ci(b) -> tuple[float, float]:
    return tuple(np.nanpercentile(b, [2.5, 97.5]))


def compute_stats(infos: list[Info], runs: Path, B: int, seed: int, workers: int,
                  cache: Path, min_level_n: int = 5) -> Stats:
    idx = infos[0].m.idx
    n = idx.shape[0]
    cells = [(c, 0) for c in range(len(SPECIES))]
    n_pos = {(c, 0): int((idx[:, c] > 0).sum()) for c in range(len(SPECIES))}
    for c in range(len(SPECIES)):
        for lvl in (1, 2, 3):
            n_pos[(c, lvl)] = int((idx[:, c] == lvl).sum())
            if n_pos[(c, lvl)] >= min_level_n:
                cells.append((c, lvl))

    files = sorted(runs.glob("*/*/predictions.npz")) + sorted(runs.glob("baselines/*.npz"))
    h = hashlib.sha1(json.dumps([B, seed, cells, [(str(f), f.stat().st_mtime_ns)
                                                  for f in files]]).encode()).hexdigest()[:16]
    cf = cache / f"bootstrap_{h}.npz"
    ids = [i.model_id for i in infos]
    if cf.exists():
        d = np.load(cf, allow_pickle=True)
        if list(d["ids"]) == ids:
            print(f"bootstrap: cached ({cf.name})")
            return Stats(ids, cells, dict(zip(ids, d["point"])), dict(zip(ids, d["boots"])),
                         n_pos)

    print(f"bootstrap: {len(ids)} models × B={B} × {len(cells)} cells ...", flush=True)
    draws = np.random.default_rng(seed).integers(0, n, size=(B, n))
    jobs = [(i.m.scores, i.m.idx, draws, cells) for i in infos]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_boot_worker, jobs))
    point = np.stack([r[0] for r in res])
    boots = np.stack([r[1] for r in res])
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cf, ids=np.array(ids), point=point, boots=boots)
    return Stats(ids, cells, dict(zip(ids, point)), dict(zip(ids, boots)), n_pos)
