"""Build the results tables from saved test predictions.

    frog-report                      # runs/ -> results/

Reads every ``runs/<run_id>/<pooling>/predictions.npz`` and
``runs/baselines/*.npz``, and writes ``results/RESULTS.md`` plus one CSV per
table. Nothing is retrained, so the tables always match the saved models.

Statistics
----------
Point estimate: test AP on the full test set, averaged over seeds.
95% CI: percentile bootstrap over test *bags* (B resamples). Each resample
recomputes the seed-averaged AP, so the interval reflects test-set sampling,
which dominates here; seed spread is reported separately.
Paired: every model is scored on the *same* resamples, so differences between
two models get their own CI. A difference whose CI excludes 0 is marked ``*``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import warnings

import numpy as np

from .data import SPECIES

# Resamples with no positives give NaN AP by design; they are dropped via nan*.
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

SP_SHORT = {"gastrotheca": "G. chrysosticta", "oreobates": "O. berdemenos"}
BASELINE_NAMES = {"clock": "clock (hour × month)",
                  "zeroshot_congeneric": "Perch zero-shot, congeneric",
                  "zeroshot_frog": "Perch zero-shot, 'Frog'"}
MIN_LEVEL_N = 5


def ap_score(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, tie-aware; matches sklearn.average_precision_score."""
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]   # end of each tied group
    tp, fp = tp[last], fp[last]
    prec, rec = tp / (tp + fp), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


@dataclass
class Model:
    model_id: str
    kind: str              # "model" | "baseline"
    probe: str
    target: str
    pooling: str
    params: int
    scores: np.ndarray     # [seeds, bags, 2]
    y: np.ndarray
    idx: np.ndarray
    bag_ids: np.ndarray
    val_ap: np.ndarray = field(default_factory=lambda: np.array([]))
    epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    thresholds: np.ndarray | None = None


def load_models(runs: Path) -> list[Model]:
    models = []
    for f in sorted(runs.glob("*/*/predictions.npz")):
        run_dir, pooling = f.parent.parent, f.parent.name
        cfg = json.loads((run_dir / "config.json").read_text())
        hidden, ow = int(cfg["hidden"]), float(cfg["ordinal_weight"])
        d = np.load(f)
        models.append(Model(
            model_id=f"{run_dir.name}/{pooling}", kind="model",
            probe=f"MLP-{hidden}" if hidden else "linear",
            target=f"ordinal (w={ow:g})" if ow else "binary",
            pooling=pooling, params=int(d["params"]), scores=d["scores"],
            y=d["y"], idx=d["idx"], bag_ids=d["bag_ids"],
            val_ap=d["val_macro_ap"], epochs=d["epochs"],
            thresholds=d["thresholds"] if "thresholds" in d else None))
    for f in sorted((runs / "baselines").glob("*.npz")):
        d = np.load(f)
        models.append(Model(
            model_id=f"baseline/{f.stem}", kind="baseline", probe="—",
            target="—", pooling=BASELINE_NAMES.get(f.stem, f.stem), params=0,
            scores=d["scores"], y=d["y"], idx=d["idx"], bag_ids=d["bag_ids"]))
    ref = models[0].bag_ids
    for m in models:
        assert np.array_equal(m.bag_ids, ref), f"{m.model_id}: different test set"
    return models


def seed_mean_ap(m: Model, rows: np.ndarray, sel=None) -> np.ndarray:
    """[species] AP averaged over seeds, on bag rows ``rows`` (may repeat)."""
    out = np.full(len(SPECIES), np.nan)
    for c in range(len(SPECIES)):
        r = rows if sel is None else rows[sel[c][rows]]
        y = (m.y[r, c] > 0).astype(float) if sel is None else (m.idx[r, c] > 0).astype(float)
        out[c] = np.nanmean([ap_score(y, sc[r, c]) for sc in m.scores])
    return out


def bootstrap(models, n_bags, B, seed, sel=None):
    """{model_id: (point[species], boots[B, species])} on shared resamples."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_bags, size=(B, n_bags))
    full = np.arange(n_bags)
    out = {}
    for m in models:
        point = seed_mean_ap(m, full, sel)
        boots = np.stack([seed_mean_ap(m, d, sel) for d in draws])
        out[m.model_id] = (point, boots)
    return out


def ci(b: np.ndarray) -> tuple[float, float]:
    return tuple(np.nanpercentile(b, [2.5, 97.5]))


def fmt(v, lo=None, hi=None, star=False):
    if v is None or np.isnan(v):
        return "–"
    s = f"{v:.3f}" if lo is None else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    return s + ("*" if star else "")


def fmt_d(v, lo, hi):
    """Signed difference with CI; * when the CI excludes 0."""
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if lo > 0 or hi < 0 else "")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    models = load_models(a.runs)
    ms = [m for m in models if m.kind == "model"]
    by_id = {m.model_id: m for m in models}
    n_bags = len(models[0].bag_ids)
    y, idx = models[0].y, models[0].idx
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ms)} models + {len(models) - len(ms)} baselines, {n_bags} test bags, "
          f"B={a.n_boot}")

    bs = bootstrap(models, n_bags, a.n_boot, a.seed)
    macro = {k: (p.mean(), b.mean(1)) for k, (p, b) in bs.items()}
    order = sorted(models, key=lambda m: -macro[m.model_id][0])
    best = next(m for m in order if m.kind == "model")

    # --- Table 1: overview
    h1 = ["model_id", "probe", "target", "pooling", "params", "macro AP [95% CI]",
          f"{SP_SHORT['gastrotheca']} AP [95% CI]", f"{SP_SHORT['oreobates']} AP [95% CI]",
          "seed SD", "val macro AP", "epochs"]
    rows1, csv1 = [], []
    for m in order:
        p, b = bs[m.model_id]
        mp, mb = macro[m.model_id]
        seed_macro = [np.mean([ap_score(m.y[:, c], sc[:, c]) for c in range(2)])
                      for sc in m.scores]
        sd = np.std(seed_macro) if len(seed_macro) > 1 else np.nan
        rows1.append([f"`{m.model_id}`", m.probe, m.target, m.pooling, f"{m.params:,}",
                      fmt(mp, *ci(mb)), fmt(p[0], *ci(b[:, 0])), fmt(p[1], *ci(b[:, 1])),
                      fmt(sd), fmt(m.val_ap.mean() if len(m.val_ap) else np.nan),
                      f"{m.epochs.mean():.0f}" if len(m.epochs) else "–"])
        csv1.append([m.model_id, m.kind, m.probe, m.target, m.pooling, m.params,
                     mp, *ci(mb), p[0], *ci(b[:, 0]), p[1], *ci(b[:, 1]), sd,
                     m.val_ap.mean() if len(m.val_ap) else "",
                     m.epochs.mean() if len(m.epochs) else "", len(m.scores)])
    write_csv(a.out / "overview.csv",
              ["model_id", "kind", "probe", "target", "pooling", "params", "macro_ap",
               "macro_lo", "macro_hi", "gastrotheca_ap", "gastrotheca_lo",
               "gastrotheca_hi", "oreobates_ap", "oreobates_lo", "oreobates_hi",
               "seed_sd_macro", "val_macro_ap", "epochs_mean", "n_seeds"], csv1)

    # --- Table 2: paired difference to the best model
    bp, bb = macro[best.model_id]
    rows2, csv2 = [], []
    for m in order:
        if m is best:
            continue
        mp, mb = macro[m.model_id]
        d, db = mp - bp, mb - bb
        lo, hi = ci(db)
        pg, bg = bs[m.model_id], bs[best.model_id]
        dg = [pg[0][c] - bg[0][c] for c in range(2)]
        dgb = [pg[1][:, c] - bg[1][:, c] for c in range(2)]
        rows2.append([f"`{m.model_id}`", fmt_d(d, lo, hi),
                      fmt_d(dg[0], *ci(dgb[0])), fmt_d(dg[1], *ci(dgb[1])),
                      "no" if (lo > 0 or hi < 0) else "yes"])
        csv2.append([m.model_id, d, lo, hi, dg[0], *ci(dgb[0]), dg[1], *ci(dgb[1]),
                     not (lo > 0 or hi < 0)])
    write_csv(a.out / "vs_best.csv",
              ["model_id", "d_macro", "lo", "hi", "d_gastrotheca", "lo", "hi",
               "d_oreobates", "lo", "hi", "indistinguishable_from_best"], csv2)

    # --- Tables 3 & 4: controlled effects (same everything except one factor)
    def effect(pairs, label_a, label_b, name):
        rows, rows_csv = [], []
        for key, ida, idb in pairs:
            if ida not in by_id or idb not in by_id:
                continue
            (pa, ba), (pb_, bb_) = macro[ida], macro[idb]
            d, db = pb_ - pa, bb_ - ba
            rows.append([key, fmt(pa), fmt(pb_), fmt_d(d, *ci(db))])
            rows_csv.append([key, ida, idb, pa, pb_, d, *ci(db)])
        write_csv(a.out / f"{name}.csv",
                  ["comparison", label_a, label_b, "macro_a", "macro_b", "d", "lo", "hi"],
                  rows_csv)
        return rows

    runs_seen = sorted({m.model_id.split("/")[0] for m in ms})
    poolings = [p for p in ("mean", "max", "lme", "linear_softmax", "attention")
                if any(m.pooling == p for m in ms)]
    probe_pairs, target_pairs = [], []
    for r in runs_seen:
        probe, target, geom = r.split("-")
        if probe == "linear":
            for q in (x for x in runs_seen if x.split("-")[1:] == [target, geom]
                      and x.split("-")[0] != "linear"):
                probe_pairs += [(f"{target} / {p}", f"{r}/{p}", f"{q}/{p}") for p in poolings]
        if target == "bin":
            for q in (x for x in runs_seen if x.split("-")[0] == probe
                      and x.split("-")[2] == geom and x.split("-")[1] != "bin"):
                target_pairs += [(f"{probe} / {p}", f"{r}/{p}", f"{q}/{p}") for p in poolings]
    rows3 = effect(probe_pairs, "linear", "mlp", "effect_probe")
    rows4 = effect(target_pairs, "binary", "ordinal", "effect_ordinal")

    # --- Table 5: AP by calling level (vs silent hours), with n
    cells = []
    for c, sp in enumerate(SPECIES):
        for lvl in (1, 2, 3):
            n = int((idx[:, c] == lvl).sum())
            cells.append((c, sp, lvl, n))
    lvl_boot = {}
    for c, sp, lvl, n in cells:
        if n < MIN_LEVEL_N:
            continue
        keep = (idx[:, c] == 0) | (idx[:, c] == lvl)
        mask = [keep if cc == c else np.ones(n_bags, bool) for cc in range(2)]
        idx_lvl = np.where(idx == lvl, 1, np.where(idx == 0, 0, -1))
        sub_models = []
        for m in models:
            mm = Model(**{**m.__dict__, "idx": idx_lvl})
            sub_models.append(mm)
        lvl_boot[(c, lvl)] = bootstrap(sub_models, n_bags, a.n_boot, a.seed, sel=mask)
    small = [f"{SP_SHORT[sp].split()[0]} idx{lvl} (n={n})"
             for c, sp, lvl, n in cells if n < MIN_LEVEL_N]
    cells = [cl for cl in cells if cl[3] >= MIN_LEVEL_N]
    h5 = ["model_id"]
    for c, sp, lvl, n in cells:
        chance = n / ((idx[:, c] == 0).sum() + n)
        h5.append(f"{SP_SHORT[sp].split()[0]} idx{lvl} (n={n}, chance {chance:.2f})")
    rows5, csv5 = [], []
    for m in order:
        row = [f"`{m.model_id}`"]
        for c, sp, lvl, n in cells:
            p, b = lvl_boot[(c, lvl)][m.model_id]
            row.append(fmt(p[c], *ci(b[:, c])))
            csv5.append([m.model_id, sp, lvl, n, p[c], *ci(b[:, c])])
        rows5.append(row)
    write_csv(a.out / "per_level.csv",
              ["model_id", "species", "index", "n_pos", "ap", "lo", "hi"], csv5)

    # --- Table 6: learned ordinal thresholds
    rows6 = []
    for m in order:
        if m.thresholds is None:
            continue
        t = m.thresholds.mean(0)
        rows6.append([f"`{m.model_id}`"] + [f"{t[c, 0]:.2f} / {t[c, 1]:.2f}" for c in range(2)])

    # --- RESULTS.md
    prev = [f"{SP_SHORT[s]} {int(y[:, c].sum())}/{n_bags} ({y[:, c].mean():.0%})"
            for c, s in enumerate(SPECIES)]
    md = ["# Results\n",
          f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `frog-report` from `runs/`. "
          f"Do not edit by hand; rerun `frog-report`.\n",
          f"**Test set:** {n_bags} hour-bags from held-out day-blocks. Positives: "
          f"{'; '.join(prev)}. Chance AP = prevalence.  ",
          "**Metric:** average precision (AP), averaged over seeds; macro = mean of "
          "the two species.  ",
          f"**95% CI:** bootstrap over test bags, B={a.n_boot}, same resamples for "
          f"every model (paired). `*` = difference whose CI excludes 0.  ",
          "**Seed SD:** spread of macro AP across training seeds (5 per model).  ",
          "**Baselines** (no audio training): *clock* = training presence rate for "
          "the bag's (hour of day × month); *Perch zero-shot* = max over the bag of "
          "Perch v2 logits for congeneric species or its 'Frog' class.\n",
          "Model IDs are `{probe}-{target}-s{stride}/{pooling}`: "
          "`linear`/`mlp256` = probe, `bin`/`ord0.5` = binary or ordinal loss "
          "(weight 0.5), `s2` = contiguous 5 s windows. Weights and predictions "
          "are in `runs/<model_id>/`.\n",
          "## 1. All models\n", md_table(h1, rows1),
          f"\n## 2. Difference from the best model (`{best.model_id}`)\n",
          "Δ = model − best, paired over the same test resamples. \"Tied\" = CI of Δ "
          "macro AP includes 0. No correction for multiple comparisons, so treat "
          "borderline `*` as suggestive.\n",
          md_table(["model_id", "Δ macro AP", f"Δ {SP_SHORT['gastrotheca']}",
                    f"Δ {SP_SHORT['oreobates']}", "tied with best"], rows2)]
    if rows3:
        md += ["\n## 3. Effect of the probe: MLP-256 − linear\n",
               "Same target, pooling, splits and seeds; only the probe differs.\n",
               md_table(["target / pooling", "linear", "MLP-256", "Δ macro AP"], rows3)]
    if rows4:
        md += ["\n## 4. Effect of the ordinal loss: ordinal − binary\n",
               "Same probe, pooling, splits and seeds; only the loss differs.\n",
               md_table(["probe / pooling", "binary", "ordinal", "Δ macro AP"], rows4)]
    md += ["\n## 5. AP by calling index\n",
           "Each column: hours at that index vs silent hours (other positive levels "
           f"excluded). n = positive test hours; chance = AP of a random ranking. "
           f"Index 1 = isolated calls, 3 = chorus. Omitted (n < {MIN_LEVEL_N}): "
           f"{', '.join(small) or 'none'}.\n",
           md_table(h5, rows5)]
    if rows6:
        md += ["\n## 6. Learned ordinal thresholds b₂ / b₃ (b₁ = 0, init 1.0 / 2.0)\n",
               md_table(["model_id", SP_SHORT["gastrotheca"], SP_SHORT["oreobates"]], rows6)]
    (a.out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print(f"written {a.out}/RESULTS.md and CSVs; best = {best.model_id} "
          f"({macro[best.model_id][0]:.3f})")


if __name__ == "__main__":
    main()
