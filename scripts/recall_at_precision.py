"""Recall at a fixed precision, with the score cutoff chosen on validation.

For every model and seed, the saved checkpoint rescores val and test (predictions.npz
holds test scores only). Per species, the cutoff is the lowest val score at which
val precision >= P, i.e. the highest val recall that still meets the target. That
cutoff is applied unchanged to test, giving the recall and precision you'd actually
get in use. "oracle" = best test recall at test precision >= P (threshold chosen on
test): a ranking-quality ceiling, not an achievable number.

Seeds are averaged as in frog-report. CIs come from a paired bootstrap over test bags
(B = 2000, the same resamples for every model), with cutoffs held fixed.

    .venv/bin/python scripts/recall_at_precision.py [--precision 0.8 0.9]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frog_mil.data import SPECIES, DataConfig, make_loaders
from frog_mil.models import MILModel
from frog_mil.train import run_epoch


def val_cutoff(y, s, p):
    """Lowest threshold whose precision on (y, s) is >= p; inf if none reaches p."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]          # only cut between distinct scores
    ok = np.where(last & (prec >= p))[0]
    return ss[ok[-1]] if len(ok) else np.inf


def oracle_recall(y, s, p):
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]
    ok = last & (prec >= p)
    return (tp[ok].max() / ys.sum()) if ok.any() else 0.0


def score_model(run_dir: Path, pooling: str, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    dc = DataConfig(emb_dir=Path(cfg["emb_dir"]), bags_csv=Path(cfg["bags_csv"]))
    loaders, sets = make_loaders(dc, int(cfg["batch_size"]), 0, 0)
    pw = sets["train"].pos_weight()
    pred = np.load(run_dir / pooling / "predictions.npz")
    out = {"val": [], "test": []}
    for k in range(pred["scores"].shape[0]):
        m = MILModel(dim=sets["train"].dim, n_classes=len(SPECIES), hidden=int(cfg["hidden"]),
                     pooling=pooling, dropout=float(cfg["dropout"]),
                     ordinal=float(cfg["ordinal_weight"]) > 0, lme_r=float(cfg["lme_r"]),
                     lme_learnable=cfg["lme_learnable"] == "True",
                     attn_hidden=int(cfg["attn_hidden"])).to(device)
        m.load_state_dict(torch.load(run_dir / pooling / f"seed{k}.pt", map_location=device))
        for split in ("val", "test"):
            _, y, s, _ = run_epoch(m, loaders[split], device, pos_weight=pw)
            out[split].append(s)
        err = np.abs(out["test"][-1] - pred["scores"][k]).max()
        assert err < 1e-4, f"{run_dir.name}/{pooling} seed{k}: test rescoring off by {err}"
    y_val = np.concatenate([b["presence"].numpy() for b in loaders["val"]])
    return np.stack(out["val"]), np.stack(out["test"]), y_val, pred["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--precision", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    models = sorted(f"{r.name}/{p.name}" for r in a.runs.iterdir()
                    if (r / "config.json").exists()
                    for p in r.iterdir() if (p / "predictions.npz").exists())
    data = {}
    for mid in models:
        run, pool = mid.split("/")
        data[mid] = score_model(a.runs / run, pool, a.device)
        print(f"scored {mid}", flush=True)
    y_test = data[models[0]][3]
    n = len(y_test)
    idx = np.random.default_rng(0).integers(0, n, (a.n_boot, n))
    C = np.stack([np.bincount(r, minlength=n) for r in idx])     # [B, n] resample counts

    rows = []
    boot = {}                                  # (P, mid) -> [B, species] recall
    for p in a.precision:
        for mid in models:
            sv, st, yv, yt = data[mid]
            rec, prec, orc, none = [], [], [], 0
            rb = np.zeros((a.n_boot, len(SPECIES)))
            for c in range(len(SPECIES)):
                r_s, p_s, o_s, rb_s = [], [], [], []
                for k in range(st.shape[0]):
                    t = val_cutoff(yv[:, c], sv[k, :, c], p)
                    none += np.isinf(t)
                    flag = st[k, :, c] >= t
                    pos = yt[:, c] == 1
                    r_s.append(flag[pos].mean())
                    p_s.append(flag[pos].sum() / flag.sum() if flag.any() else np.nan)
                    o_s.append(oracle_recall(yt[:, c], st[k, :, c], p))
                    rb_s.append((C @ (flag & pos)) / (C @ pos))
                rec.append(np.mean(r_s)); prec.append(np.nanmean(p_s)); orc.append(np.mean(o_s))
                rb[:, c] = np.mean(rb_s, axis=0)
            boot[p, mid] = rb
            lo, hi = np.percentile(rb.mean(1), [2.5, 97.5])
            rows.append(dict(precision=p, model_id=mid, macro_recall=np.mean(rec),
                             macro_lo=lo, macro_hi=hi,
                             **{f"{s}_recall": rec[c] for c, s in enumerate(SPECIES)},
                             **{f"{s}_test_precision": prec[c] for c, s in enumerate(SPECIES)},
                             **{f"{s}_oracle_recall": orc[c] for c, s in enumerate(SPECIES)},
                             no_cutoff=int(none)))

    import pandas as pd
    df = pd.DataFrame(rows)
    for p in a.precision:
        sub = df[df.precision == p]
        # paired delta vs the best model, for the macro and for each species
        for c, col in [(None, "macro")] + list(enumerate(SPECIES)):
            key = f"{col}_recall"
            best = sub.loc[sub[key].idxmax(), "model_id"]
            bb = boot[p, best].mean(1) if c is None else boot[p, best][:, c]
            for i in sub.index:
                b = boot[p, df.at[i, "model_id"]]
                d = (b.mean(1) if c is None else b[:, c]) - bb
                df.loc[i, [f"{col}_best", f"{col}_d", f"{col}_d_lo", f"{col}_d_hi"]] = [
                    best, df.at[i, key] - sub[key].max(), *np.percentile(d, [2.5, 97.5])]
    df.sort_values(["precision", "macro_recall"], ascending=[True, False]).to_csv(
        a.out / "recall_at_precision.csv", index=False)
    print(f"written {a.out / 'recall_at_precision.csv'}")


if __name__ == "__main__":
    main()
