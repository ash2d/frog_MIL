"""Train probes over frozen Perch v2 embeddings, one sweep of poolers per run.

    frog-train --pooling all --seeds 5                        # linear-bin-s2
    frog-train --pooling all --seeds 5 --hidden 256 --ordinal-weight 0.5

Naming: a *run* is one probe/target/geometry setting; a *model* is a run plus a
pooler. IDs are derived from the config, never typed by hand:

    run_id    {probe}-{target}-s2          e.g. linear-bin-s2, mlp256-ord0.5-s2
    model_id  {run_id}/{pooling}           e.g. mlp256-ord0.5-s2/max

Output, per model: ``runs/<run_id>/<pooling>/``
    predictions.npz   test scores for every seed + per-seed diagnostics
    seed{k}.pt        best-on-val weights for seed k
    metrics.json      quick per-seed metrics (the report recomputes its own)
plus ``runs/<run_id>/config.json`` and ``runs/baselines/*.npz``.

Tables with CIs come from ``frog-report``, not from here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SPECIES, BagEmbeddingDataset, DataConfig, make_loaders
from .evaluate import (bag_metrics, by_intensity, format_intensity, format_table,
                       metadata_baseline, zero_shot_baseline)
from .models import MILModel, count_params
from .pooling import POOLERS


def run_epoch(model, loader, device, opt=None, pos_weight=None,
              cum_pos_weight=None, ordinal_weight: float = 0.0):
    """One pass. Presence is always the primary loss; the ordinal thresholds
    y>=2 and y>=3 are added on top when the model has an ordinal head (y>=1 is
    the presence term itself, so it is not counted twice)."""
    train = opt is not None
    model.train(train)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    levels = torch.arange(1, 4, device=device)
    tot, n, ys, ss, ids = 0.0, 0, [], [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            x, mask = b["x"].to(device), b["mask"].to(device)
            y = b["presence"].to(device)
            out = model(x, mask)
            loss = bce(out["logits"], y)
            if ordinal_weight and "cum_logits" in out:
                cum_t = (b["intensity"].to(device).unsqueeze(-1) >= levels).float()
                loss = loss + ordinal_weight * F.binary_cross_entropy_with_logits(
                    out["cum_logits"][..., 1:], cum_t[..., 1:],
                    pos_weight=cum_pos_weight[:, 1:].to(device))
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tot += loss.item() * len(y); n += len(y)
            ys.append(b["presence"].numpy())
            ids.append(b["intensity"].numpy())
            ss.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
    return tot / max(n, 1), np.concatenate(ys), np.concatenate(ss), np.concatenate(ids)


def train_one(cfg, pooling: str, seed: int, device: str) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    loaders, sets = make_loaders(cfg["data"], cfg["batch_size"], seed,
                                 cfg["num_workers"])
    model = MILModel(dim=sets["train"].dim, n_classes=len(SPECIES),
                     hidden=cfg["hidden"], pooling=pooling, dropout=cfg["dropout"],
                     ordinal=cfg["ordinal_weight"] > 0,
                     lme_r=cfg["lme_r"], lme_learnable=cfg["lme_learnable"],
                     attn_hidden=cfg["attn_hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    pw, cpw = sets["train"].pos_weight(), sets["train"].cum_pos_weight()

    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        run_epoch(model, loaders["train"], device, opt, pw, cpw, cfg["ordinal_weight"])
        _, yv, sv, _ = run_epoch(model, loaders["val"], device, pos_weight=pw)
        val_ap = bag_metrics(yv, sv, n_boot=0)["macro_ap"]
        if val_ap > best:
            best, bad = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)

    _, yt, st, it = run_epoch(model, loaders["test"], device, pos_weight=pw)
    test_ids = list(sets["test"].bag_ids)
    m = bag_metrics(yt, st, n_boot=cfg["n_boot"], seed=seed)
    m.update(by_intensity(it, st))
    m.update({"pooling": pooling, "seed": seed, "val_macro_ap": best,
              "params": count_params(model), "epochs_run": ep + 1})
    if pooling == "lme":
        m["lme_r"] = float(model.pool.r.detach())
    if model.ordinal is not None:
        for c, sp in enumerate(SPECIES):
            b = model.ordinal.thresholds()[c].detach().cpu().tolist()
            m[f"{sp}_thr2"], m[f"{sp}_thr3"] = b[1], b[2]
    return m, {"scores": st, "y": yt, "idx": it, "bag_ids": test_ids,
               "state": {k: v.cpu() for k, v in best_state.items()}}


def aggregate(runs: list[dict]) -> dict:
    """Mean over seeds, with the spread -- a single seed cannot separate these."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    for s in SPECIES:
        out[f"{s}_ap_seedstd"] = float(np.std([r[f"{s}_ap"] for r in runs]))
    out["n_seeds"] = len(runs)
    return out


def run_id(a) -> str:
    probe = f"mlp{a.hidden}" if a.hidden else "linear"
    target = f"ord{a.ordinal_weight:g}" if a.ordinal_weight else "bin"
    # "s2" names the only window geometry, contiguous 5 s windows (stride 2 of the
    # original 2.5 s grid). It stays in the ID so runs keep their earlier names.
    return f"{probe}-{target}-s2"


def save_model(out: Path, runs: list[dict], preds: list[dict]) -> None:
    """Everything the report needs, and nothing it has to re-derive by training."""
    out.mkdir(parents=True, exist_ok=True)
    extra = {}
    if "gastrotheca_thr2" in runs[0]:
        extra["thresholds"] = np.array([[[r[f"{s}_thr2"], r[f"{s}_thr3"]] for s in SPECIES]
                                        for r in runs])
    np.savez(out / "predictions.npz",
             scores=np.stack([p["scores"] for p in preds]),        # [seeds, bags, 2]
             y=preds[0]["y"], idx=preds[0]["idx"],
             bag_ids=np.array(preds[0]["bag_ids"]),
             val_macro_ap=np.array([r["val_macro_ap"] for r in runs]),
             epochs=np.array([r["epochs_run"] for r in runs]),
             params=np.array(runs[0]["params"]), **extra)
    for k, p in enumerate(preds):
        torch.save(p["state"], out / f"seed{k}.pt")
    (out / "metrics.json").write_text(json.dumps(runs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, default=Path("embeddings"))
    ap.add_argument("--bags-csv", type=Path, default=Path("outputs/bags.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--tag", default=None,
                    help="run_id override; required when changing settings not in "
                         "the run_id (lr, epochs, dropout, ...)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing run whose config differs")
    ap.add_argument("--pooling", default="all",
                    help="'all' or one of " + ", ".join(POOLERS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--attn-hidden", type=int, default=128)
    ap.add_argument("--lme-r", type=float, default=1.0)
    ap.add_argument("--lme-learnable", action="store_true")
    ap.add_argument("--ordinal-weight", type=float, default=0.0,
                    help="weight of the y>=2, y>=3 threshold losses; 0 = binary only")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = vars(a).copy()
    cfg["data"] = DataConfig(emb_dir=a.emb_dir, bags_csv=a.bags_csv)
    poolings = list(POOLERS) if a.pooling == "all" else [a.pooling]
    rid = a.tag or run_id(a)
    out = a.out_dir / rid
    # run_id encodes probe/target only. Any other change (lr, epochs, ...)
    # must get its own --tag, or it would silently overwrite a different run.
    cfg_now = {"run_id": rid, **{k: str(v) for k, v in vars(a).items()
                                 if k not in ("device", "num_workers", "overwrite")}}
    cfg_path = out / "config.json"
    if cfg_path.exists() and not a.overwrite:
        old = json.loads(cfg_path.read_text())
        old = {k: v for k, v in old.items() if k not in ("device", "num_workers", "overwrite")}
        diff = {k: (old.get(k), v) for k, v in cfg_now.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{out} exists with a different config {diff}.\n"
                             f"Use --tag <new_run_id> or --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_now, indent=2))

    results = {}
    for p in poolings:
        t0 = time.time()
        both = [train_one(cfg, p, s, a.device) for s in range(a.seeds)]
        runs, preds = [b[0] for b in both], [b[1] for b in both]
        results[p] = aggregate(runs)
        save_model(out / p, runs, preds)
        print(f"{p:16} macro AP {results[p]['macro_ap']:.3f}  "
              f"({time.time() - t0:.0f}s, {results[p]['params']:,.0f} params)",
              flush=True)

    # Baselines. If a pooled audio model cannot beat the clock, stop and think.
    tr = BagEmbeddingDataset(cfg["data"], "train")
    te = BagEmbeddingDataset(cfg["data"], "test")
    y = np.array([[int(b[f"{s}_present"]) for s in SPECIES] for b in te.bags])
    idx = np.array([[int(b[f"{s}_index"]) for s in SPECIES] for b in te.bags])
    clock = metadata_baseline(te.bags, tr.bags)
    results["baseline_hour_month"] = {**bag_metrics(y, clock, n_boot=a.n_boot),
                                      **by_intensity(idx, clock)}
    base = {"clock": clock, **{f"zeroshot_{k}": v
                               for k, v in zero_shot_baseline(te, a.emb_dir).items()}}
    for name, sc in base.items():
        if name != "clock":
            results[f"baseline_{name}"] = {**bag_metrics(y, sc, n_boot=a.n_boot),
                                           **by_intensity(idx, sc)}
        bdir = a.out_dir / "baselines"
        bdir.mkdir(parents=True, exist_ok=True)
        np.savez(bdir / f"{name}.npz", scores=sc[None], y=y, idx=idx,
                 bag_ids=np.array(te.bag_ids), params=np.array(0))

    table = format_table(results) + "\n\n" + format_intensity(results)
    print("\n" + table)
    (out / "table.txt").write_text(table + "\n")
    print(f"\nwritten to {out}  (run `frog-report` for the CI tables)")


if __name__ == "__main__":
    main()
