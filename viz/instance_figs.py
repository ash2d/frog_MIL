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
)
from matplotlib.colors import LinearSegmentedColormap

from frog_mil.data import SPECIES, BagEmbeddingDataset, DataConfig, collate
from frog_mil.models import MILModel

BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd"] + BLUE_RAMP)


def infer_run(runs: Path, run_id: str, poolings: list[str], cache: Path, device="cpu"):
    """{pooling: dict(inst_prob [seeds, bags, N, C], weights [...], bag_prob, bag_ids, mask)}."""
    cfg = json.loads((runs / run_id / "config.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for p in poolings:
        f = cache / f"instances_{run_id}_{p}.npz"
        ckpts = sorted((runs / run_id / p).glob("seed*.pt"))
        if f.exists() and all(f.stat().st_mtime > c.stat().st_mtime for c in ckpts):
            out[p] = dict(np.load(f))
        else:
            todo.append(p)
    if not todo:
        return out

    dc = DataConfig(emb_dir=Path(cfg["emb_dir"]), bags_csv=Path(cfg["bags_csv"]),
                    stride=int(cfg["stride"]))
    print(f"  instance inference for {run_id}: {', '.join(todo)} (loading embeddings)")
    tr = BagEmbeddingDataset(dc, "train")
    te = BagEmbeddingDataset(dc, "test", stats=tr.stats)
    batch = collate([te[k] for k in range(len(te))])
    x, mask = batch["x"].to(device), batch["mask"].to(device)
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
        np.savez_compressed(cache / f"instances_{run_id}_{p}.npz", **d)
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
                    species=None):
    pools = [p for p in POOL_ORDER if p in inst] + sorted(set(inst) - set(POOL_ORDER))
    by = {i.pooling: i for i in infos if i.run_id == run_id}
    for p in pools:
        _check(inst[p], by[p])
    idx = infos[0].m.idx
    bag_ids = list(infos[0].m.bag_ids)
    c = SPECIES.index(species) if species else 0
    if bag_id is None:
        # an isolated-call hour where the poolers disagree most on the bag score
        cand = np.flatnonzero(idx[:, c] == 1) if (idx[:, c] == 1).any() else \
            np.flatnonzero(idx[:, c] > 0)
        spread = np.std([by[p].m.scores[:, cand, c].mean(0) for p in pools], axis=0)
        b = cand[np.argmax(spread)]
    else:
        b = bag_ids.index(bag_id)
    n_n = inst[pools[0]]["inst_prob"].shape[2]
    fig, axes = plt.subplots(len(pools), 2, figsize=(12, 1.25 * len(pools) + 1.2),
                             sharex=True)
    xs = np.arange(n_n)
    for r, p in enumerate(pools):
        d = inst[p]
        prob = d["inst_prob"][:, b, :, c].mean(0)
        w = d["weights"][:, b, :, c].mean(0)
        bag = by[p].m.scores[:, b, c].mean()
        for k, (a, v, lab) in enumerate([(axes[r, 0], prob, "window P(call)"),
                                         (axes[r, 1], w, "pooling weight")]):
            a.bar(xs, v, color=pcolor(p), width=0.8)
            a.axvline(n_n / 2 - 0.5, color=AXIS, lw=0.8)
            a.grid(axis="x", visible=False)
            a.set_ylim(0, 1.05 if k == 0 else max(0.3, v.max() * 1.15))
            a.set_xlim(-0.6, n_n - 0.4)
            if r == 0:
                a.set_title(["Window probability (mean over seeds)",
                             "Weight the pooler gives each window"][k])
        axes[r, 0].set_ylabel(plabel(p), rotation=0, ha="right", va="center", color=INK,
                              fontsize=9)
        axes[r, 0].text(n_n - 0.5, 1.0, f"bag P = {bag:.2f}", ha="right", va="top",
                        fontsize=8, color=INK, weight="bold")
    for a in axes[-1]:
        a.set_xticks([0, n_n // 2 - 1, n_n - 1], ["1", f"{n_n // 2}", f"{n_n}"])
        a.set_xlabel("5 s window")
    fig.suptitle(f"One test hour through every pooler: {bag_ids[b]}, {SP_SHORT[SPECIES[c]]} "
                 f"index {idx[b, c]} — {run_id}", x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "pooling_example",
         f"A real test hour (`{bag_ids[b]}`, *{SP_SHORT[SPECIES[c]]}* index {idx[b, c]}), "
         "chosen automatically as the hour where the poolers disagree most. Left: each "
         "model's per-window probability. Right: the weight its pooler put on each window "
         "(max is one-hot, mean is uniform, attention is learned from the embedding).")
