"""Bag-level AP on train, val and test for every trained model, without retraining.

Each runs/<run_id>/<pooling>/seed{k}.pt is rebuilt as in train_one and rescored on
all three splits. The train split uses an unshuffled loader so each bag is scored
once. Recomputed val and test macro AP must match metrics.json (val_macro_ap,
macro_ap), or the script stops.

    .venv/bin/python scripts/split_ap.py [--device cuda]

Writes outputs/split_ap.csv: one row per (run_id, pooling, seed, split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from frog_mil.data import SPECIES, DataConfig, collate, make_loaders
from frog_mil.evaluate import bag_metrics
from frog_mil.models import MILModel
from frog_mil.train import run_epoch

N_SEEDS = 5
SPLITS = ("train", "val", "test")
TOL = 1e-6


def load_cfg(run_dir: Path, num_workers: int) -> dict:
    """config.json stores every value as a string; restore what train_one reads."""
    raw = json.loads((run_dir / "config.json").read_text())
    return {
        "data": DataConfig(emb_dir=Path(raw["emb_dir"]), bags_csv=Path(raw["bags_csv"])),
        "batch_size": int(raw["batch_size"]), "num_workers": num_workers,
        "hidden": int(raw["hidden"]), "dropout": float(raw["dropout"]),
        "ordinal_weight": float(raw["ordinal_weight"]), "lme_r": float(raw["lme_r"]),
        "lme_learnable": raw["lme_learnable"] == "True",
        "attn_hidden": int(raw["attn_hidden"]),
    }


_LOADERS: dict = {}


def get_loaders(cfg, seed: int):
    """make_loaders as in train_one, cached: every run shares the same data."""
    key = (cfg["data"].emb_dir, cfg["data"].bags_csv, cfg["batch_size"], seed)
    if key not in _LOADERS:
        loaders, sets = make_loaders(cfg["data"], cfg["batch_size"], seed,
                                     cfg["num_workers"])
        # make_loaders shuffles train; score every train bag once, in a fixed order
        loaders["train"] = DataLoader(sets["train"], batch_size=cfg["batch_size"],
                                      shuffle=False, drop_last=False, collate_fn=collate,
                                      num_workers=cfg["num_workers"])
        _LOADERS[key] = loaders, sets
    return _LOADERS[key]


def score_seed(cfg, pooling: str, seed: int, weights: Path, device: str) -> dict:
    loaders, sets = get_loaders(cfg, seed)
    model = MILModel(dim=sets["train"].dim, n_classes=len(SPECIES),
                     hidden=cfg["hidden"], pooling=pooling, dropout=cfg["dropout"],
                     ordinal=cfg["ordinal_weight"] > 0,
                     lme_r=cfg["lme_r"], lme_learnable=cfg["lme_learnable"],
                     attn_hidden=cfg["attn_hidden"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    pw = sets["train"].pos_weight()
    out = {}
    for split in SPLITS:
        _, y, s, _ = run_epoch(model, loaders[split], device, pos_weight=pw)
        assert len(y) == len(sets[split]), f"{split}: scored {len(y)} of {len(sets[split])} bags"
        out[split] = bag_metrics(y, s, n_boot=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/split_ap.csv"))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    models = sorted((r.name, p.name) for r in a.runs.iterdir()
                    if r.name != "baselines" and (r / "config.json").exists()
                    for p in r.iterdir()
                    if all((p / f"seed{k}.pt").exists() for k in range(N_SEEDS)))
    print(f"{len(models)} models", flush=True)

    rows = []
    for run_id, pooling in models:
        cfg = load_cfg(a.runs / run_id, a.num_workers)
        mdir = a.runs / run_id / pooling
        ref = {r["seed"]: r for r in json.loads((mdir / "metrics.json").read_text())}
        for k in range(N_SEEDS):
            m = score_seed(cfg, pooling, k, mdir / f"seed{k}.pt", a.device)
            for split, key in (("val", "val_macro_ap"), ("test", "macro_ap")):
                err = abs(m[split]["macro_ap"] - ref[k][key])
                if not err <= TOL:
                    raise SystemExit(
                        f"{run_id}/{pooling} seed{k}: {split} macro AP "
                        f"{m[split]['macro_ap']:.8f} != metrics.json {key} "
                        f"{ref[k][key]:.8f} (off by {err:.2e})")
            for split in SPLITS:
                rows.append({"run_id": run_id, "pooling": pooling, "seed": k,
                             "split": split,
                             **{f"{s}_ap": m[split][f"{s}_ap"] for s in SPECIES},
                             "macro_ap": m[split]["macro_ap"]})
        print(f"scored {run_id}/{pooling}", flush=True)

    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"written {a.out} ({len(df)} rows)\n")

    wide = df.pivot_table(index=["run_id", "pooling", "seed"], columns="split",
                          values="macro_ap")
    wide["gap"] = wide["train"] - wide["val"]
    g = wide.groupby(level=["run_id", "pooling"])
    mean, sd = g.mean(), g.std(ddof=1)
    print(f"{'model_id':34} " + " ".join(f"{c:>15}" for c in (*SPLITS, "train-val")))
    for idx in mean.index:
        cells = [f"{mean.at[idx, c]:.3f} ± {sd.at[idx, c]:.3f}"
                 for c in (*SPLITS, "gap")]
        print(f"{'/'.join(idx):34} " + " ".join(f"{c:>15}" for c in cells))


if __name__ == "__main__":
    main()
