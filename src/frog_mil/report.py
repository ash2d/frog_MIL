"""Build the results tables from saved predictions.

    frog-report                      # runs of the current manifest -> results/

Reads every finished model of ``runs/<dataset_id>/`` (in the group workspace)
and writes ``results/RESULTS.md`` plus one CSV per table. Nothing is retrained,
so the tables always match the saved models; ``results/dataset.json`` records
which dataset they describe.

Statistics
----------
Point estimate: AP on the out-of-fold test scores of all bags (every bag is
tested exactly once per seed, by the fold model that never saw it), averaged
over seeds.
95% CI: block bootstrap (``stats``): whole 3-day blocks resampled within each
recording regime, so correlated hours stay together. Every model is scored on
the same resamples, so differences get their own paired CI; ``*`` marks a
difference whose CI excludes 0.
Reference model: the one with the best *validation* macro AP. Choosing the
reference on test would make every "worse than best" claim optimistic.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np

from . import stats as st_
from .config import RESULTS, RUNS, SP_LABEL, SPECIES, STORE, current_dataset_id
from .data import fold_roles
from .runs import FACTORS, SPEC, load_models
from .stats import ci

# Resamples with no positives give NaN AP by design; they are dropped via nan*.
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)

BASELINE_NAMES = {"clock": "clock (hour × month)",
                  "zeroshot_congeneric": "Perch zero-shot, congeneric",
                  "zeroshot_frog": "Perch zero-shot, 'Frog'"}
POOL_ORDER = ("mean", "max", "lme", "linear_softmax", "attention")


def fmt(v, lo=None, hi=None, star=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
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


def load_bag_table(models, bags_csv: Path | None = None):
    """(block, regime) per bag, in the models' bag order, from the manifest."""
    from .config import BAGS_CSV
    with (bags_csv or BAGS_CSV).open() as fh:
        b = {r["bag_id"]: r for r in csv.DictReader(fh)}
    ids = models[0].bag_ids
    return (np.array([int(b[i]["block"]) for i in ids]),
            np.array([b[i]["regime"] for i in ids]))


def val_selected(models):
    """The model with the best mean validation macro AP (over seeds and folds).

    Models trained on some regimes only are validated on those regimes, so their
    val AP isn't comparable; they are candidates only if no other model is.
    """
    ms = [m for m in models if m.is_model]
    full = [m for m in ms if not m.cfg.get("train_regimes")]
    return max(full or ms, key=lambda m: float(np.mean(m.val_ap)))


def controlled_pairs(models, factor: str):
    """(context, model a, model b): same everything except ``factor`` (a = baseline level)."""
    ms = [m for m in models if m.is_model]
    others = [k for k in SPEC if k not in ("run_id", *FACTORS)]

    def key(m, drop):
        return (m.pooling, tuple(m.cfg.get(k) for k in others),
                tuple(m.cfg.get(f) for f in FACTORS if f != drop))

    out = []
    for a in ms:
        if a.cfg.get(factor):
            continue
        for b in ms:
            if b.cfg.get(factor) and key(a, factor) == key(b, factor):
                ctx = (f"{a.target.split(' ')[0]}" if factor == "hidden" else a.probe)
                out.append((f"{ctx} / {a.pooling}", a, b))
    order = {p: i for i, p in enumerate(POOL_ORDER)}
    return sorted(out, key=lambda t: (t[0].split(" / ")[0], t[2].model_id,
                                      order.get(t[1].pooling, 9)))


# --------------------------------------------------------------------------- recall@P
def val_cutoff(y, s, p):
    """Lowest threshold whose precision on (y, s) is >= p; inf if none reaches p."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]          # only cut between distinct scores
    ok = np.flatnonzero(last & (prec >= p))
    return ss[ok[-1]] if len(ok) else np.inf


def recall_at_precision(m, p: float, W: np.ndarray):
    """Recall on test with cutoffs chosen on validation, per fold model and seed.

    Model f's cutoff comes from its validation bags (fold f+1) and is applied to
    its test bags (fold f); flags are pooled over folds. Returns
    (recall [C], test precision [C], bootstrap recall [B, C], seeds x folds with no cutoff).
    """
    K = int(m.fold.max()) + 1
    y = (m.idx > 0).astype(int)
    rec, prec, boot, none = [], [], np.zeros((len(W), len(SPECIES))), 0
    for c in range(len(SPECIES)):
        r_s, p_s, b_s = [], [], []
        for k in range(len(m.scores)):
            flag = np.zeros(len(y), bool)
            for f in range(K):
                roles = fold_roles(m.fold, f, K)
                t = val_cutoff(y[roles["val"], c], m.val_scores[k, roles["val"], c], p)
                none += np.isinf(t)
                flag[roles["test"]] = m.scores[k, roles["test"], c] >= t
            pos = y[:, c] == 1
            r_s.append(flag[pos].mean())
            p_s.append(flag[pos].sum() / flag.sum() if flag.any() else np.nan)
            b_s.append((W @ (flag & pos)) / (W @ pos))
        rec.append(np.mean(r_s))
        prec.append(np.nanmean(p_s))
        boot[:, c] = np.mean(b_s, axis=0)
    return np.array(rec), np.array(prec), boot, int(none)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="dataset_id (default: the manifest's)")
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--bags-csv", type=Path, default=None,
                    help="manifest bags.csv of --dataset (default: outputs/bags.csv)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", type=float, nargs="+", default=[0.8, 0.9])
    a = ap.parse_args()

    did = a.dataset or current_dataset_id()
    models = load_models(did, a.runs)
    ms = [m for m in models if m.is_model]
    if not ms:
        raise SystemExit(f"no finished models for dataset {did} under {a.runs}")
    bags_csv = a.bags_csv
    if bags_csv is None and did != current_dataset_id():
        bags_csv = STORE / "manifests" / did / "bags.csv"     # kept by `frog run manifest`
    blocks, regime = load_bag_table(models, bags_csv)
    n_bags = len(blocks)
    idx = models[0].idx
    y = (idx > 0).astype(int)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"dataset {did}: {len(ms)} models + {len(models) - len(ms)} baselines, "
          f"{n_bags} bags, B={a.n_boot}")

    st = st_.compute(models, blocks, regime, a.n_boot, a.seed,
                     cache_dir=a.runs / did / ".cache")
    order = sorted(models, key=lambda m: -st.macro(m.model_id))
    ref = val_selected(models)
    test_best = next(m for m in order if m.is_model)

    # --- Table 1: overview
    h1 = ["model_id", "probe", "target", "pooling", "params", "macro AP [95% CI]",
          f"{SP_LABEL['gastrotheca']} AP [95% CI]", f"{SP_LABEL['oreobates']} AP [95% CI]",
          "seed SD", "val macro AP", "epochs"]
    rows1, csv1 = [], []
    for m in order:
        mp, mb = st.get(m.model_id, "macro")
        g, gb = st.get(m.model_id, 0)
        o, ob = st.get(m.model_id, 1)
        seed_macro = [np.mean([st_.ap_weighted(y[:, c], sc[:, c], np.ones((1, n_bags)))[0]
                               for c in range(len(SPECIES))]) for sc in m.scores]
        sd = float(np.std(seed_macro)) if len(seed_macro) > 1 else np.nan
        va = float(np.mean(m.val_ap)) if m.val_ap.size else np.nan
        mark = " ◆" if m is ref else ""
        rows1.append([f"`{m.model_id}`{mark}", m.probe, m.target,
                      BASELINE_NAMES.get(m.pooling, m.pooling), f"{m.params:,}",
                      fmt(mp, *ci(mb)), fmt(g, *ci(gb)), fmt(o, *ci(ob)), fmt(sd), fmt(va),
                      f"{m.epochs.mean():.0f}" if m.epochs.size else "–"])
        csv1.append([m.model_id, m.kind, m.probe, m.target, m.pooling, m.params,
                     mp, *ci(mb), g, *ci(gb), o, *ci(ob), sd, va,
                     m.epochs.mean() if m.epochs.size else "", len(m.scores),
                     m is ref])
    write_csv(a.out / "overview.csv",
              ["model_id", "kind", "probe", "target", "pooling", "params", "macro_ap",
               "macro_lo", "macro_hi", "gastrotheca_ap", "gastrotheca_lo",
               "gastrotheca_hi", "oreobates_ap", "oreobates_lo", "oreobates_hi",
               "seed_sd_macro", "val_macro_ap", "epochs_mean", "n_seeds", "val_selected"],
              csv1)

    # --- Table 2: paired difference to the val-selected model
    rp, rb = st.get(ref.model_id, "macro")
    rows2, csv2 = [], []
    for m in order:
        if m is ref:
            continue
        mp, mb = st.get(m.model_id, "macro")
        lo, hi = ci(mb - rb)
        ds = [(st.get(m.model_id, c)[0] - st.get(ref.model_id, c)[0],
               *ci(st.get(m.model_id, c)[1] - st.get(ref.model_id, c)[1])) for c in range(2)]
        tied = not (lo > 0 or hi < 0)
        rows2.append([f"`{m.model_id}`", fmt_d(mp - rp, lo, hi), fmt_d(*ds[0]),
                      fmt_d(*ds[1]), "tied" if tied else ("better" if lo > 0 else "worse")])
        csv2.append([m.model_id, mp - rp, lo, hi, *ds[0], *ds[1], tied])
    write_csv(a.out / "vs_selected.csv",
              ["model_id", "d_macro", "lo", "hi", "d_gastrotheca", "lo", "hi",
               "d_oreobates", "lo", "hi", "indistinguishable_from_selected"], csv2)

    # --- Tables 3 & 4: controlled effects (same everything except one factor)
    def effect(factor, name):
        rows, rows_csv = [], []
        for ctx, ma, mb_ in controlled_pairs(models, factor):
            (pa, ba), (pb, bb) = st.get(ma.model_id, "macro"), st.get(mb_.model_id, "macro")
            label = ctx + (f" (w={mb_.cfg['ordinal_weight']:g})" if factor != "hidden" else "")
            rows.append([label, fmt(pa), fmt(pb), fmt_d(pb - pa, *ci(bb - ba))])
            rows_csv.append([label, ma.model_id, mb_.model_id, pa, pb, pb - pa,
                             *ci(bb - ba)])
        write_csv(a.out / f"{name}.csv",
                  ["comparison", "model_a", "model_b", "macro_a", "macro_b", "d", "lo", "hi"],
                  rows_csv)
        return rows
    rows3 = effect("hidden", "effect_probe")
    rows4 = effect("ordinal_weight", "effect_ordinal")

    # --- Table 5: AP by calling level (vs silent hours), with n
    lvl_cells = [c for c in st.cells if c[0] == "level"]
    h5 = ["model_id"]
    for _, c, lvl in lvl_cells:
        n = int((idx[:, c] == lvl).sum())
        chance = n / ((idx[:, c] == 0).sum() + n)
        h5.append(f"{SP_LABEL[SPECIES[c]].split()[0]} idx{lvl} (n={n}, chance {chance:.2f})")
    small = [f"{SP_LABEL[sp].split()[0]} idx{lvl} (n={int((idx[:, c] == lvl).sum())})"
             for c, sp in enumerate(SPECIES) for lvl in (1, 2, 3)
             if not st.has(("level", c, lvl))]
    rows5, csv5 = [], []
    for m in order:
        row = [f"`{m.model_id}`"]
        for cell in lvl_cells:
            p, b = st.ap(m.model_id, cell), st.ap_boot(m.model_id, cell)
            row.append(fmt(p, *ci(b)))
            csv5.append([m.model_id, SPECIES[cell[1]], cell[2],
                         int((idx[:, cell[1]] == cell[2]).sum()), p, *ci(b)])
        rows5.append(row)
    write_csv(a.out / "per_level.csv",
              ["model_id", "species", "index", "n_pos", "ap", "lo", "hi"], csv5)

    # --- Table 6: AP by recording regime
    reg_cells = [c for c in st.cells if c[0] == "regime"]
    regimes = sorted(set(regime))
    h6 = ["model_id"]
    for _, c, r in reg_cells:
        sel = regime == r
        h6.append(f"{SP_LABEL[SPECIES[c]].split()[0]} {r} (n={int(y[sel, c].sum())}/"
                  f"{int(sel.sum())}, chance {y[sel, c].mean():.2f})")
    rows6, csv6 = [], []
    for m in order:
        row = [f"`{m.model_id}`"]
        for cell in reg_cells:
            p, b = st.ap(m.model_id, cell), st.ap_boot(m.model_id, cell)
            row.append(fmt(p, *ci(b)))
            sel = regime == cell[2]
            csv6.append([m.model_id, SPECIES[cell[1]], cell[2], int(sel.sum()),
                         int(y[sel, cell[1]].sum()), float(y[sel, cell[1]].mean()), p, *ci(b)])
        rows6.append(row)
    write_csv(a.out / "per_regime.csv",
              ["model_id", "species", "regime", "n_bags", "n_pos", "chance_ap", "ap", "lo",
               "hi"], csv6)

    # --- Table 7: fit per split (train / val / test AP of each fold model)
    rows7, csv7 = [], []
    for m in order:
        if not m.is_model:
            continue
        met = json.loads((m.path / "metrics.json").read_text())
        cells = []
        for r in ("train", "val", "test"):
            v = np.array([x[f"{r}_macro_ap"] for x in met])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")
        gap = np.array([x["train_macro_ap"] - x["val_macro_ap"] for x in met])
        rows7.append([f"`{m.model_id}`", *cells, f"{gap.mean():+.3f}"])
        for x in met:
            for r in ("train", "val", "test"):
                csv7.append([m.model_id, x["seed"], x["fold"], r,
                             *[x[f"{r}_{sp}_ap"] for sp in SPECIES], x[f"{r}_macro_ap"]])
    write_csv(a.out / "split_ap.csv",
              ["model_id", "seed", "fold", "split", *[f"{sp}_ap" for sp in SPECIES],
               "macro_ap"], csv7)

    # --- Table 8: recall at fixed precision, cutoffs chosen on validation
    rows8, csv8 = [], []
    for p in a.precision:
        res = {m.model_id: recall_at_precision(m, p, st.W) for m in ms}
        rr, _, rbt, _ = res[ref.model_id]
        for m in sorted(ms, key=lambda m: -res[m.model_id][0].mean()):
            rec, prec, bt, none = res[m.model_id]
            lo, hi = ci(bt.mean(1))
            d = rec.mean() - rr.mean()
            dlo, dhi = ci(bt.mean(1) - rbt.mean(1))
            rows8.append([f"{p:g}", f"`{m.model_id}`" + (" ◆" if m is ref else ""),
                          fmt(rec.mean(), lo, hi), fmt(rec[0]), fmt(rec[1]),
                          f"{prec[0]:.2f} / {prec[1]:.2f}",
                          "–" if m is ref else fmt_d(d, dlo, dhi)])
            csv8.append([p, m.model_id, rec.mean(), lo, hi, *rec, *prec, d, dlo, dhi, none])
    write_csv(a.out / "recall_at_precision.csv",
              ["precision", "model_id", "macro_recall", "macro_lo", "macro_hi",
               *[f"{s}_recall" for s in SPECIES], *[f"{s}_test_precision" for s in SPECIES],
               "d_vs_selected", "d_lo", "d_hi", "no_cutoff"], csv8)

    # --- Table 9: learned ordinal thresholds
    rows9 = []
    for m in order:
        if m.thresholds is None:
            continue
        t = m.thresholds.mean((0, 1))
        rows9.append([f"`{m.model_id}`"]
                     + [f"{t[c, 0]:.2f} / {t[c, 1]:.2f}" for c in range(2)])

    # --- RESULTS.md
    K = int(models[0].fold.max()) + 1
    prev = [f"{SP_LABEL[s]} {int(y[:, c].sum())}/{n_bags} ({y[:, c].mean():.1%})"
            for c, s in enumerate(SPECIES)]
    reg_txt = ", ".join(f"{r} {int((regime == r).sum())}" for r in regimes)
    md = ["# Results\n",
          f"Generated by `frog-report` from the runs of dataset `{did}`. Do not edit by "
          f"hand; rerun `frog run report`.\n",
          f"**Data:** {n_bags} hour-bags ({reg_txt}), {K}-fold cross-validation over "
          f"3-day blocks. Every bag is scored once per seed by the fold model that never "
          f"saw it (out-of-fold). Positives: {'; '.join(prev)}. Chance AP = prevalence.  ",
          "**Metric:** average precision (AP) on the out-of-fold scores, averaged over "
          "seeds; macro = mean of the two species.  ",
          f"**95% CI:** block bootstrap (whole 3-day blocks, resampled within each "
          f"recording regime), B={a.n_boot}, same resamples for every model (paired). `*` = "
          f"difference whose CI excludes 0.  ",
          f"**Reference model (◆):** `{ref.model_id}`, the best mean *validation* macro AP. "
          f"The best on test is `{test_best.model_id}`; it is not used for selection.  ",
          "**Seed SD:** spread of macro AP across training seeds.  ",
          "**Baselines** (no audio training): *clock* = training-fold presence rate for "
          "the bag's (hour of day × month); *Perch zero-shot* = max over the bag of "
          "Perch v2 logits for congeneric species or its 'Frog' class.\n",
          "Model IDs are `{probe}-{target}-s2/{pooling}`: `linear`/`mlp256` = probe, "
          "`bin`/`ord0.5` = binary or ordinal loss (weight 0.5), `s2` = contiguous 5 s "
          f"windows. Weights and predictions are in the run store, "
          f"`runs/{did}/<model_id>/`.\n",
          "## 1. All models\n", md_table(h1, rows1),
          f"\n## 2. Difference from the validation-selected model (`{ref.model_id}`)\n",
          "Δ = model − reference, paired over the same resamples. \"tied\" = CI of Δ "
          "macro AP includes 0. No correction for multiple comparisons, so treat "
          "borderline `*` as suggestive.\n",
          md_table(["model_id", "Δ macro AP", f"Δ {SP_LABEL['gastrotheca']}",
                    f"Δ {SP_LABEL['oreobates']}", "vs reference"], rows2)]
    if rows3:
        md += ["\n## 3. Effect of the probe: MLP − linear\n",
               "Same target, pooling, folds and seeds; only the probe differs.\n",
               md_table(["target / pooling", "linear", "MLP", "Δ macro AP"], rows3)]
    if rows4:
        md += ["\n## 4. Effect of the ordinal loss: ordinal − binary\n",
               "Same probe, pooling, folds and seeds; only the loss differs.\n",
               md_table(["probe / pooling (w)", "binary", "ordinal", "Δ macro AP"], rows4)]
    md += ["\n## 5. AP by calling index\n",
           "Each column: hours at that index vs silent hours (other positive levels "
           "excluded). n = positive hours; chance = AP of a random ranking. Index 1 = "
           f"isolated calls, 3 = chorus. Omitted (n < {st_.MIN_LEVEL_N}): "
           f"{', '.join(small) or 'none'}.\n",
           md_table(h5, rows5),
           "\n## 6. AP by recording regime\n",
           "Presence AP within each recording regime: `8k-3clip` = 8 kHz, 3 × 1 min per "
           "hour (2018-09/10, 36 windows); `44k-1clip` = 44.1 kHz, 1 clip (2018-11/12, "
           "12 windows); `44k-2clip` = 44.1 kHz, 2 clips (2019, 24 windows). n = "
           "positives/bags. Prevalence differs between regimes, so compare each column "
           "with its own chance level, not across columns.\n",
           md_table(h6, rows6),
           "\n## 7. Fit per split\n",
           "Macro AP of each fold model on its own train, validation and test folds, "
           "mean ± SD over seeds × folds. Validation AP is the early-stopping optimum, so "
           "it is optimistic.\n",
           md_table(["model_id", "train", "val", "test", "train − val"], rows7),
           "\n## 8. Recall at fixed precision\n",
           "Each fold model's score cutoff is the lowest one that reaches the target "
           "precision on its validation fold, applied unchanged to its test fold; flags "
           "are pooled over folds. Test precision shows how well the cutoff transfers. "
           "Δ is paired against the reference model.\n",
           md_table(["target precision", "model_id", "macro recall [95% CI]",
                     f"{SP_LABEL['gastrotheca']} recall", f"{SP_LABEL['oreobates']} recall",
                     "test precision G / O", "Δ macro recall"], rows8)]
    if rows9:
        md += ["\n## 9. Learned ordinal thresholds b₂ / b₃ (b₁ = 0, init 1.0 / 2.0)\n",
               md_table(["model_id", SP_LABEL["gastrotheca"], SP_LABEL["oreobates"]], rows9)]
    (a.out / "RESULTS.md").write_text("\n".join(md) + "\n")
    (a.out / "dataset.json").write_text(json.dumps({
        "dataset_id": did, "n_bags": n_bags, "folds": K, "n_models": len(ms),
        "reference_model": ref.model_id, "test_best_model": test_best.model_id,
        "generated": dt.datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    print(f"written {a.out}/RESULTS.md and CSVs; reference (val) = {ref.model_id} "
          f"({st.macro(ref.model_id):.3f}); test-best = {test_best.model_id} "
          f"({st.macro(test_best.model_id):.3f})")


if __name__ == "__main__":
    main()
