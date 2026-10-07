"""Train probes over frozen Perch v2 embeddings: every pooler x seed x fold of one run.

    frog-train --seeds 5                                     # linear-bin-s2
    frog-train --seeds 5 --hidden 256 --ordinal-weight 0.5   # mlp256-ord0.5-s2

Cross-validation: the manifest deals 3-day blocks into K folds. Model f is
tested on fold f, early-stopped on fold (f+1) mod K and trained on the rest,
so every bag gets exactly one out-of-fold test score and one validation score
per seed.

Output, per model (``runs.py`` has the layout), in the group workspace:
    predictions.npz   OOF test + val bag scores for every seed, labels, folds
    windows.npz       window logits and pooling weights of the test-fold model
    metrics.json      per seed x fold: epochs, best val AP, train/val/test AP
    seed{k}_fold{f}.pt
plus ``config.json`` per run and ``baselines/*.npz`` per dataset.

A finished pooler is skipped when rerun with the same config, so an
interrupted run resumes. Tables with CIs come from ``frog-report``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import runs as rs
from .config import EMB_DIR, OUTPUTS, RUNS, SPECIES
from .data import BagData, FoldView, load_bags
from .evaluate import clock_baseline, species_ap, zero_shot_baseline
from .models import count_params
from .pooling import POOLERS

EVAL_BATCH = 512


def train_epoch(model, view: FoldView, opt, gen, batch_size, pw, cpw, ordinal_weight):
    """One pass over the train bags. Presence is always the primary loss; the
    ordinal thresholds y>=2 and y>=3 are added on top when the model has an
    ordinal head (y>=1 is the presence term itself, so it is not counted twice)."""
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    levels = torch.arange(1, 4, device=pw.device)
    for b in view.batches("train", batch_size, gen):
        out = model(b["x"], b["mask"])
        loss = bce(out["logits"], b["presence"])
        if ordinal_weight and "cum_logits" in out:
            cum_t = (b["intensity"].unsqueeze(-1) >= levels).float()
            loss = loss + ordinal_weight * F.binary_cross_entropy_with_logits(
                out["cum_logits"][..., 1:], cum_t[..., 1:], pos_weight=cpw[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


@torch.no_grad()
def score(model, view: FoldView, role: str, windows: bool = False) -> dict:
    """Bag probabilities (and optionally window logits + pooling weights) for a role."""
    model.eval()
    out = {"prob": [], "logit_w": [], "weight_w": []}
    for b in view.batches(role, EVAL_BATCH):
        o = model(b["x"], b["mask"])
        out["prob"].append(torch.sigmoid(o["logits"]))
        if windows:
            out["logit_w"].append(o["instance_logits"])
            out["weight_w"].append(o["weights"])
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items() if v}


def train_fold(data: BagData, view: FoldView, cfg: dict, pooling: str, seed: int):
    """Train one seed x fold; restore the epoch with the best val macro AP."""
    f = view.f
    # Same init and batch order for every pooler at a given (seed, fold).
    torch.manual_seed(seed * 1000 + f)
    gen = torch.Generator().manual_seed(seed * 1000 + f)
    model = rs.build_model(cfg, pooling, data.dim).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pw, cpw = view.pos_weight(), view.cum_pos_weight()
    y = data.y()
    yv = y[view.idx["val"].cpu().numpy()]

    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        train_epoch(model, view, opt, gen, cfg["batch_size"], pw, cpw, cfg["ordinal_weight"])
        val_ap = species_ap(yv, score(model, view, "val")["prob"])["macro_ap"]
        if val_ap > best:
            best, bad = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)

    res = {r: score(model, view, r, windows=(r == "test")) for r in ("train", "val", "test")}
    m = {"seed": seed, "fold": f, "epochs": ep + 1, "val_macro_ap": best}
    for r in ("train", "val", "test"):
        ap = species_ap(y[view.idx[r].cpu().numpy()], res[r]["prob"])
        m.update({f"{r}_{k}": v for k, v in ap.items()})
    if pooling == "lme":
        m["lme_r"] = float(model.pool.r.detach())
    if model.ordinal is not None:
        thr = model.ordinal.thresholds().detach().cpu().numpy()
        for c, sp in enumerate(SPECIES):
            m[f"{sp}_thr2"], m[f"{sp}_thr3"] = float(thr[c, 1]), float(thr[c, 2])
    state = {k: v.cpu() for k, v in best_state.items()}
    return m, res, state, count_params(model)


def train_model(data: BagData, views: list[FoldView], cfg: dict, pooling: str, out) -> dict:
    """All seeds x folds of one pooler; writes the model directory."""
    S, K = cfg["seeds"], data.n_folds
    n, N, C = len(data.bag_ids), data.x.shape[1], len(SPECIES)
    test, val = np.full((S, n, C), np.nan), np.full((S, n, C), np.nan)
    wl = np.zeros((S, n, N, C), np.float16)
    ww = np.zeros((S, n, N, C), np.float16)
    val_ap, epochs, thr, metrics = np.zeros((S, K)), np.zeros((S, K), int), [], []
    out.mkdir(parents=True, exist_ok=True)
    for s in range(S):
        for view in views:
            m, res, state, params = train_fold(data, view, cfg, pooling, s)
            ti, vi = view.idx["test"].cpu().numpy(), view.idx["val"].cpu().numpy()
            test[s, ti], val[s, vi] = res["test"]["prob"], res["val"]["prob"]
            wl[s, ti], ww[s, ti] = res["test"]["logit_w"], res["test"]["weight_w"]
            val_ap[s, view.f], epochs[s, view.f] = m["val_macro_ap"], m["epochs"]
            if f"{SPECIES[0]}_thr2" in m:
                thr.append((s, view.f, [[m[f"{sp}_thr2"], m[f"{sp}_thr3"]] for sp in SPECIES]))
            metrics.append(m)
            torch.save(state, out / f"seed{s}_fold{view.f}.pt")
    assert not np.isnan(test).any() and not np.isnan(val).any(), "a bag was never scored"
    extra = {}
    if thr:
        t = np.zeros((S, K, C, 2))
        for s, f, v in thr:
            t[s, f] = v
        extra["thresholds"] = t
    rs.savez_atomic(out / "windows.npz", logits=wl, weights=ww, mask=data.mask.cpu().numpy(),
                    rows=data.rows, bag_ids=data.bag_ids)
    rs.write_json_atomic(out / "metrics.json", metrics)
    # Written last: its presence marks the model as finished.
    rs.savez_atomic(out / "predictions.npz", scores=test, val_scores=val, y=data.y(),
                    idx=data.idx(), bag_ids=data.bag_ids, fold=data.fold,
                    val_macro_ap=val_ap, epochs=epochs, params=np.array(params),
                    dataset_id=np.array(data.dataset_id), **extra)
    return {"macro_ap": float(np.mean([species_ap(data.y(), test[s])["macro_ap"]
                                       for s in range(S)])), "params": params}


def ensure_baselines(data: BagData, root, overwrite: bool = False) -> None:
    """Clock and Perch zero-shot scores for every bag (out-of-fold for the clock).

    Zero-shot scores depend on the embedding cache, so those from a band-limited
    cache are named ``zeroshot_<variant>_bl<Hz>`` and sit beside the full-band ones.
    """
    bdir = rs.dataset_dir(data.dataset_id, root) / "baselines"
    meta = json.loads((data.emb_dir / "meta.json").read_text())
    bl = meta.get("band_limit_hz")
    suffix = f"_bl{bl}" if bl else ""
    names = ["clock", f"zeroshot_congeneric{suffix}", f"zeroshot_frog{suffix}"]
    if not overwrite and all((bdir / f"{n}.npz").exists() for n in names):
        return
    bdir.mkdir(parents=True, exist_ok=True)
    clock = np.zeros((len(data.bag_ids), len(SPECIES)))
    for f in range(data.n_folds):
        r = data.roles(f)
        clock[r["test"]] = clock_baseline(data.bags, r["train"])[r["test"]]
    base = {"clock": clock, **{f"zeroshot_{k}{suffix}": v for k, v in zero_shot_baseline(
        data.rows, meta, data.emb_dir / "logits.f16.npy").items()}}
    for name, sc in base.items():
        rs.savez_atomic(bdir / f"{name}.npz", scores=sc[None], y=data.y(), idx=data.idx(),
                        bag_ids=data.bag_ids, fold=data.fold, params=np.array(0),
                        dataset_id=np.array(data.dataset_id))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--manifest", type=Path, default=OUTPUTS,
                    help="directory with the bags.csv and summary.json to train on "
                         "(default: the current manifest; a subset for experiments)")
    ap.add_argument("--out-dir", type=Path, default=RUNS)
    ap.add_argument("--tag", default=None,
                    help="run_id override; required when changing settings not in "
                         "the run_id (lr, epochs, dropout, ...)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing run whose config differs")
    ap.add_argument("--force", action="store_true", help="retrain finished poolers too")
    ap.add_argument("--pooling", default="all",
                    help="'all' or a comma list of " + ", ".join(POOLERS))
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def config_from_args(a, dataset_id: str, folds: int,
                     band_limit_hz: int | None = None) -> dict:
    """The run config (``runs.SPEC`` fields) that these arguments train."""
    return {"run_id": a.tag or rs.run_id_for(a.hidden, a.ordinal_weight),
            "dataset_id": dataset_id, "band_limit_hz": band_limit_hz,
            "hidden": a.hidden, "dropout": a.dropout,
            "attn_hidden": a.attn_hidden, "lme_r": a.lme_r, "lme_learnable": a.lme_learnable,
            "ordinal_weight": a.ordinal_weight, "lr": a.lr, "weight_decay": a.weight_decay,
            "epochs": a.epochs, "patience": a.patience, "batch_size": a.batch_size,
            "seeds": a.seeds, "folds": folds}


def main() -> None:
    a = build_parser().parse_args()

    t0 = time.time()
    data = load_bags(a.emb_dir, a.manifest / "bags.csv", a.manifest / "summary.json",
                     device=a.device)
    print(f"dataset {data.dataset_id}: {len(data.bag_ids)} bags, {data.n_folds} folds, "
          f"loaded to {a.device} in {time.time() - t0:.0f}s", flush=True)
    poolings = list(POOLERS) if a.pooling == "all" else a.pooling.split(",")
    emb_meta = json.loads((data.emb_dir / "meta.json").read_text())
    cfg = config_from_args(a, data.dataset_id, data.n_folds, emb_meta.get("band_limit_hz"))
    rid = cfg["run_id"]
    out = rs.dataset_dir(data.dataset_id, a.out_dir) / rid
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = rs.read_config(out)
        diff = {k: (old.get(k), v) for k, v in rs.spec(cfg).items() if old.get(k) != v}
        if diff and not a.overwrite:
            raise SystemExit(f"{out} exists with a different config {diff}.\n"
                             f"Use --tag <new_run_id> or --overwrite.")
        if not diff:
            poolings = list(dict.fromkeys(old.get("poolings", []) + poolings))
        elif a.overwrite:
            a.force = True
    todo = [p for p in poolings if a.force or not (out / p / "predictions.npz").exists()]
    out.mkdir(parents=True, exist_ok=True)
    rs.write_json_atomic(cfg_path, {**cfg, "poolings": poolings, "status": "running",
                                    "started": dt.datetime.now().isoformat(timespec="seconds"),
                                    "code": rs.code_version()})
    ensure_baselines(data, a.out_dir)
    if len(todo) < len(poolings):
        print(f"already finished: {', '.join(p for p in poolings if p not in todo)}")

    views = [FoldView(data, f) for f in range(data.n_folds)]
    for p in todo:
        t0 = time.time()
        r = train_model(data, views, cfg, p, out / p)
        print(f"{p:16} OOF macro AP {r['macro_ap']:.3f}  "
              f"({time.time() - t0:.0f}s, {r['params']:,} params)", flush=True)
    finished = dt.datetime.now().isoformat(timespec="seconds")
    rs.write_json_atomic(cfg_path, {**rs.read_config(out), "status": "complete",
                                    "finished": finished})
    print(f"written to {out}  (run `frog-report` for the tables)")


if __name__ == "__main__":
    main()
