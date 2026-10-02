"""Window-level figures: what each trained model thinks happens inside a bag.

Re-runs the saved ``seed{k}.pt`` checkpoints on the cached test embeddings (CPU
is fine; the heads are tiny) to recover per-window probabilities and pooling
weights. Needs ``embeddings/`` and ``outputs/bags.csv`` matching the runs.
"""
from __future__ import annotations

import json
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
    Info,
    pcolor,
    plabel,
    scale_fonts,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from frog_mil.data import SPECIES, BagEmbeddingDataset, DataConfig, collate
from frog_mil.models import MILModel

BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd"] + BLUE_RAMP)


def infer_run(runs: Path, run_id: str, poolings: list[str], cache: Path, device="cpu",
              max_windows: int | None = None):
    """{pooling: dict(inst_prob [seeds, bags, N, C], weights [...], bag_prob, bag_ids, mask)}.

    ``max_windows`` masks every window after the first ``max_windows`` (12 = the first
    1 min clip), so pooling weights and bag scores are those of that shorter bag."""
    tag = f"_first{max_windows}" if max_windows else ""
    cfg = json.loads((runs / run_id / "config.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for p in poolings:
        f = cache / f"instances_{run_id}_{p}{tag}.npz"
        ckpts = sorted((runs / run_id / p).glob("seed*.pt"))
        if f.exists() and all(f.stat().st_mtime > c.stat().st_mtime for c in ckpts):
            out[p] = dict(np.load(f))
        else:
            todo.append(p)
    if not todo:
        return out

    dc = DataConfig(emb_dir=Path(cfg["emb_dir"]), bags_csv=Path(cfg["bags_csv"]))
    print(f"  instance inference for {run_id}: {', '.join(todo)} (loading embeddings)")
    tr = BagEmbeddingDataset(dc, "train")
    te = BagEmbeddingDataset(dc, "test", stats=tr.stats)
    batch = collate([te[k] for k in range(len(te))])
    x, mask = batch["x"].to(device), batch["mask"].to(device)
    if max_windows:
        mask[:, max_windows:] = False
    for p in todo:
        ip, ww, bp = [], [], []
        for ck in sorted((runs / run_id / p).glob("seed*.pt"),
                         key=lambda f: int(f.stem[4:])):
            model = MILModel(dim=te.dim, n_classes=len(SPECIES), hidden=int(cfg["hidden"]),
                             pooling=p, dropout=float(cfg["dropout"]),
                             ordinal=float(cfg["ordinal_weight"]) > 0,
                             lme_r=float(cfg["lme_r"]),
                             lme_learnable=cfg["lme_learnable"] == "True",
                             attn_hidden=int(cfg["attn_hidden"]))
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval().to(device)
            with torch.no_grad():
                o = model(x, mask)
            ip.append(torch.sigmoid(o["instance_logits"]).cpu().numpy())
            ww.append(o["weights"].cpu().numpy())
            bp.append(torch.sigmoid(o["logits"]).cpu().numpy())
        d = dict(inst_prob=np.stack(ip), weights=np.stack(ww), bag_prob=np.stack(bp),
                 bag_ids=np.array(te.bag_ids), mask=mask.cpu().numpy())
        np.savez_compressed(cache / f"instances_{run_id}_{p}{tag}.npz", **d)
        out[p] = d
    return out


def _check(inst, info: Info):
    """The re-run must reproduce the saved test predictions."""
    assert list(inst["bag_ids"]) == list(info.m.bag_ids), "test bags differ from predictions"
    err = np.abs(inst["bag_prob"] - info.m.scores).max()
    if err > 1e-3:
        print(f"  WARNING {info.model_id}: re-inferred scores differ from saved by {err:.2g}")


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
    n_n = max_windows or inst[pools[0]]["inst_prob"].shape[2]
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
            if not max_windows:
                a.axvline(n_n / 2 - 0.5, color=AXIS, lw=0.8)
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
                        fontsize=8, color=INK, weight="bold")
    for a in axes[-1]:
        a.set_xticks([0, n_n // 2 - 1, n_n - 1], ["1", f"{n_n // 2}", f"{n_n}"])
        a.set_xlabel("5 s window")
    if title:
        fig.suptitle(f"One test hour through every pooler: {bag_ids[b]}, {SP_SHORT[SPECIES[c]]} "
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
         f"A real test hour (`{bag_ids[b]}`, *{SP_SHORT[SPECIES[c]]}* index {idx[b, c]}), "
         "chosen automatically as the hour where the poolers disagree most. Left: each "
         "model's per-window probability. Right: the weight its pooler put on each window "
         "(max is one-hot, mean is uniform, attention is learned from the embedding)." + first)
