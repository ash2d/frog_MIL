"""Shared style, loading and statistics for the figure scripts.

Everything is derived from the run store (``runs/<dataset_id>/``: predictions,
window scores, configs) and ``outputs/bags.csv``, so the figures always match
whatever runs exist. Statistics come from ``frog_mil.stats`` with the report's
resamples and cache, so CIs agree with ``results/RESULTS.md``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from frog_mil.config import SPECIES  # noqa: F401  (re-exported for the figure modules)
from frog_mil.report import BASELINE_NAMES, load_bag_table
from frog_mil.runs import Model, load_models
from frog_mil.stats import Stats, ci, compute  # noqa: F401

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
FOLD_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#898781"]
REGIME_COLOR = {"8k-3clip": "#b04a2a", "44k-1clip": "#6da7ec", "44k-2clip": "#184f95"}
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


# The figures, in reading order. Output files are numbered by this list.
FIGURES = ["bag_structure", "pipeline", "pooling_toy", "calendar", "forest_ap", "effects",
           "ordinal_sweep", "per_index", "val_vs_test", "pooling_example", "forest_simple",
           "per_regime"]


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
    label: str = ""

    @property
    def probe(self) -> str:
        return f"MLP-{self.hidden}" if self.hidden else "linear"

    @property
    def is_model(self) -> bool:
        return self.kind == "model"


def load_infos(dataset_id: str, runs: Path) -> list[Info]:
    out = []
    for m in load_models(dataset_id, runs):
        if m.kind == "baseline":
            out.append(Info(m, m.model_id, "baseline", m.pooling, "baseline",
                            label=BASELINE_NAMES.get(m.pooling, m.pooling)))
            continue
        out.append(Info(m, m.model_id, m.run_id, m.pooling, "model",
                        hidden=int(m.cfg["hidden"]), w=float(m.cfg["ordinal_weight"]),
                        label=m.model_id))
    return out


# --------------------------------------------------------------------------- stats
def compute_stats(infos: list[Info], dataset_id: str, runs: Path, bags_csv: Path, B: int,
                  seed: int, workers: int) -> Stats:
    """The block bootstrap of ``frog-report`` (same resamples, same cache)."""
    models = [i.m for i in infos]
    blocks, regime = load_bag_table(models, bags_csv)
    return compute(models, blocks, regime, B, seed, cache_dir=runs / dataset_id / ".cache",
                   workers=workers)
