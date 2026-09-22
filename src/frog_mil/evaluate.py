"""Metrics for the pooling comparison.

Average precision is the headline: with 7% (Gastrotheca) and 22% (Oreobates)
positive hours, ROC-AUC flatters everything and accuracy is meaningless.

The bootstrap matters more than usual here. The test split holds on the order
of 20 positive Gastrotheca bags, so the gap between two pooling functions is
easily smaller than the sampling noise. Report intervals, or the comparison is
a coin flip dressed up as a result.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import MAX_INDEX, SPECIES


def bag_metrics(y: np.ndarray, s: np.ndarray, n_boot: int = 1000,
                seed: int = 0) -> dict:
    """AP and AUC per species with bootstrap CIs over bags."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for c, name in enumerate(SPECIES):
        yc, sc = y[:, c], s[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            out[f"{name}_ap"] = float("nan")
            continue
        out[f"{name}_ap"] = float(average_precision_score(yc, sc))
        out[f"{name}_auc"] = float(roc_auc_score(yc, sc))
        out[f"{name}_prevalence"] = float(yc.mean())
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yc), len(yc))
            if 0 < yc[i].sum() < len(i):
                boots.append(average_precision_score(yc[i], sc[i]))
        if boots:
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[f"{name}_ap_lo"], out[f"{name}_ap_hi"] = float(lo), float(hi)
    aps = [out[f"{s}_ap"] for s in SPECIES if not np.isnan(out.get(f"{s}_ap", np.nan))]
    out["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return out


def by_intensity(idx: np.ndarray, s: np.ndarray) -> dict:
    """AP for each calling level against silent hours only.

    For level L, positives are hours with index == L and negatives are hours
    with index == 0; hours at other positive levels are dropped rather than
    counted as negatives, which would punish a model for detecting them.

    This is where the pooling comparison earns its keep. On chorus hours
    (index 3) calls fill the bag and every pooler should do well; on index-1
    hours a single call sits in ~1 of 24 windows and the poolers should
    separate. Annotators scored only the recorded clips, so index-1 bags are
    clean positives: weak index-1 AP means isolated calls are being missed,
    not that the labels are wrong.
    """
    from scipy.stats import spearmanr

    out = {}
    for c, name in enumerate(SPECIES):
        neg = idx[:, c] == 0
        for level in range(1, MAX_INDEX + 1):
            pos = idx[:, c] == level
            keep = neg | pos
            out[f"{name}_idx{level}_n"] = int(pos.sum())
            if pos.sum() and neg.sum():
                out[f"{name}_idx{level}_ap"] = float(
                    average_precision_score(pos[keep], s[keep, c]))
                out[f"{name}_idx{level}_prev"] = float(pos[keep].mean())
        # Does the score order hours by intensity, among positives?
        p = idx[:, c] > 0
        if p.sum() > 2 and len(set(idx[p, c])) > 1:
            out[f"{name}_spearman_pos"] = float(spearmanr(idx[p, c], s[p, c])[0])
    return out


def format_intensity(results: dict[str, dict]) -> str:
    """AP by calling level; ``prev`` is the chance-level AP for that level."""
    lines = []
    for s in SPECIES:
        hdr = f"{s:28} " + " ".join(f"{'idx' + str(k) + ' AP':>14}" for k in (1, 2, 3)) \
              + f" {'rho(pos)':>9}"
        lines += [hdr, "-" * len(hdr)]
        for name, m in results.items():
            cells = []
            for k in (1, 2, 3):
                ap, pv = m.get(f"{s}_idx{k}_ap"), m.get(f"{s}_idx{k}_prev")
                cells.append(f"{ap:6.3f} ({pv:.2f})" if ap is not None else f"{'-':>14}")
            rho = m.get(f"{s}_spearman_pos")
            rho_s = f"{rho:9.3f}" if rho is not None else f"{'-':>9}"
            lines.append(f"{name:28} " + " ".join(f"{c:>14}" for c in cells) + " " + rho_s)
        lines.append("")
    return "\n".join(lines)


def zero_shot_baseline(ds, emb_dir) -> dict[str, np.ndarray]:
    """Training-free scores from Perch's own logits, max-pooled over the bag.

    Two variants, [n_bags, 2] each:
      ``congeneric``: max over the species' genus columns (8 Gastrotheca spp.;
                      Oreobates quixensis)
      ``frog``:       the generic Frog class, same score for both species

    If ``congeneric`` is close to the trained probes, the probe is mostly
    re-deriving what Perch already knows about the genus.
    """
    import json

    meta = json.loads((emb_dir / "meta.json").read_text())
    names = meta.get("logit_names") or []
    lp = emb_dir / "logits.f16.npy"
    if not names or not lp.exists():
        return {}
    logits = np.load(lp, mmap_mode="r")
    genus = {"gastrotheca": "Gastrotheca ", "oreobates": "Oreobates "}
    cols = {s: [i for i, n in enumerate(names) if n.startswith(genus[s])] for s in SPECIES}
    frog = [i for i, n in enumerate(names) if n == "Frog"]

    out = {"congeneric": np.zeros((len(ds), len(SPECIES))),
           "frog": np.zeros((len(ds), len(SPECIES)))}
    for i, b in enumerate(ds.bag_ids):
        lg = np.asarray(logits[ds.rows[b]], np.float32)          # [n_inst, n_cols]
        for c, sp in enumerate(SPECIES):
            if cols[sp]:
                out["congeneric"][i, c] = lg[:, cols[sp]].max()
            if frog:
                out["frog"][i, c] = lg[:, frog].max()
    return {k: v for k, v in out.items() if v.any()}


def metadata_baseline(bags: list[dict], train_bags: list[dict]) -> np.ndarray:
    """Hour-of-day + month prior, fit on train. The trap this guards against:

    both species are strongly nocturnal and strongly seasonal, and the splits
    are day-blocks, so a model that knows nothing but the clock scores well.
    Any audio model that does not clearly beat this has learned nothing about
    frogs.
    """
    scores = np.zeros((len(bags), len(SPECIES)))
    for c, name in enumerate(SPECIES):
        rate: dict[tuple, list] = {}
        for b in train_bags:
            k = (int(b["hour"]), b["date"][5:7])
            rate.setdefault(k, []).append(int(b[f"{name}_present"]))
        prior = np.mean([int(b[f"{name}_present"]) for b in train_bags])
        for i, b in enumerate(bags):
            v = rate.get((int(b["hour"]), b["date"][5:7]))
            scores[i, c] = np.mean(v) if v else prior
    return scores


def format_table(results: dict[str, dict]) -> str:
    """results: {run_name: metrics}. Sorted by macro AP."""
    hdr = f"{'run':28} {'macro AP':>9}"
    for s in SPECIES:
        hdr += f" | {s[:11]:>11} AP (95% CI)      "
    lines = [hdr, "-" * len(hdr)]
    for name, m in sorted(results.items(), key=lambda kv: -kv[1].get("macro_ap", 0)):
        row = f"{name:28} {m.get('macro_ap', float('nan')):9.3f}"
        for s in SPECIES:
            ap = m.get(f"{s}_ap", float("nan"))
            lo, hi = m.get(f"{s}_ap_lo", float("nan")), m.get(f"{s}_ap_hi", float("nan"))
            row += f" | {ap:11.3f} [{lo:.3f}, {hi:.3f}]"
        lines.append(row)
    return "\n".join(lines)
