"""Cross-regime transfer: does a model that never heard the 8 kHz format still
rank the 8 kHz hours, and the silent ones, the same way?

The band-limit control (``scripts/bandlimit_experiment.py``) showed that the
format stays recognisable from the embeddings after band-limiting, so it can't
rule out a format shortcut. Here the reference model (``mlp256-ord0.5-s2/lme``)
is trained and early-stopped on one set of regimes only (``frog-train
--train-regimes``) and still scores every hour out of fold:

    all     every regime (the reference)
    tr44k   44.1 kHz hours only  -> its scores on 8 kHz hours are transfer
    tr8k    8 kHz hours only     -> its scores on 44.1 kHz hours are transfer

each on the embeddings as recorded (``full``) and on band-limited ones (``bl8k``,
from the band-limit experiment), where transfer is at matched bandwidth. A
transfer model can't use the 8 kHz format as a shortcut. Two questions:

1. AP within each regime, against the reference with the same embeddings.
2. On silent 8 kHz hours: do the transfer models pick out the same high-scoring
   hours as the reference, and do those hours sit next to labelled calls?
   If so, they more likely hold missed calls than a format artefact.

    .venv/bin/python scripts/regime_transfer.py [--gpu 3]    # idempotent

Runs: ``STORE/experiments/regime_transfer/runs/<dataset_id>/``; tables:
``results/experiments/regime_transfer/``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frog_mil import stats as st_  # noqa: E402
from frog_mil.config import (  # noqa: E402
    BAGS_CSV,
    EMB_DIR,
    RESULTS,
    ROOT,
    SP_LABEL,
    SPECIES,
    STORE,
    current_dataset_id,
)
from frog_mil.report import load_bag_table  # noqa: E402
from frog_mil.runs import load_models  # noqa: E402

EXP = STORE / "experiments" / "regime_transfer"
RUNS = EXP / "runs"
OUT = RESULTS / "experiments" / "regime_transfer"
EMB = {"full": EMB_DIR, "bl8k": STORE / "experiments" / "bandlimit_8k" / "emb_all_bl8000"}
TRAIN_ON = {"all": None, "tr44k": "44k-1clip,44k-2clip", "tr8k": "8k-3clip"}
LABEL = {"all": "all regimes", "tr44k": "44.1 kHz only", "tr8k": "8 kHz only"}
BASE = "mlp256-ord0.5-s2"
MODEL = ["--hidden", "256", "--ordinal-weight", "0.5", "--pooling", "lme"]
PY = ROOT / ".venv" / "bin" / "python"
B, SEED, TOP = 2000, 0, 0.05
REGIMES = ["8k-3clip", "44k-1clip", "44k-2clip"]


def run_id(train_on: str, emb: str) -> str:
    return "-".join([BASE] + ([train_on] if train_on != "all" else [])
                    + ([emb] if emb != "full" else []))


ARMS = [(t, e) for t in TRAIN_ON for e in EMB]


def train(gpu: str, did: str) -> None:
    if not (EMB["bl8k"] / "meta.json").exists():
        raise SystemExit(f"{EMB['bl8k']} missing; run scripts/bandlimit_experiment.py first")
    EXP.mkdir(parents=True, exist_ok=True)
    # One after the other: every arm writes the dataset's shared baselines.
    for t, e in ARMS:
        rid = run_id(t, e)
        if (RUNS / did / rid / "lme" / "predictions.npz").exists():
            continue
        log = EXP / f"train_{rid}.log"
        print(f"training {rid} on GPU {gpu} (log {log})", flush=True)
        cmd = [str(PY), "-m", "frog_mil.train", "--emb-dir", str(EMB[e]),
               "--out-dir", str(RUNS), "--tag", rid, *MODEL]
        if TRAIN_ON[t]:
            cmd += ["--train-regimes", TRAIN_ON[t]]
        with log.open("a") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        if r.returncode:
            raise SystemExit(f"training {rid} failed; see {log}")


def report(did: str) -> None:
    subprocess.run([str(PY), "-m", "frog_mil.report", "--dataset", did, "--runs", str(RUNS),
                    "--out", str(OUT)], check=True)


# --------------------------------------------------------------------------- summary
def fmt_d(d, lo, hi) -> str:
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if lo > 0 or hi < 0 else "")


def ap_table(st, models, regime) -> list[str]:
    """Per species: AP within each regime for every arm, Δ vs the same-embedding reference."""
    idx = next(iter(models.values())).idx
    out = []
    for c, sp in enumerate(SPECIES):
        heads = []
        for r in REGIMES:
            keep = regime == r
            npos = int((idx[keep, c] > 0).sum())
            heads.append(f"{r} ({npos} / {int(keep.sum())}, chance {npos / keep.sum():.2f})")
        out += [f"### {SP_LABEL[sp]}", "",
                "| trained on | embeddings | " + " | ".join(heads) + " |",
                "|---|---|" + "---|" * len(REGIMES)]
        for t, e in ARMS:
            mid, ref = f"{run_id(t, e)}/lme", f"{run_id('all', e)}/lme"
            cells = []
            for r in REGIMES:
                cell = ("regime", c, r)
                a = st.ap(mid, cell)
                unseen = TRAIN_ON[t] is not None and r not in TRAIN_ON[t].split(",")
                txt = f"**{a:.3f}**" if unseen else f"{a:.3f}"
                if t != "all":
                    txt += " " + fmt_d(a - st.ap(ref, cell),
                                       *st_.ci(st.ap_boot(mid, cell) - st.ap_boot(ref, cell)))
                cells.append(txt)
            out.append(f"| {LABEL[t]} | {e} | " + " | ".join(cells) + " |")
        for name in ("clock", "zeroshot_congeneric"):
            mid = f"baseline/{name}"
            out.append(f"| baseline: {name} | full | "
                       + " | ".join(f"{st.ap(mid, ('regime', c, r)):.3f}" for r in REGIMES)
                       + " |")
        out.append("")
    return out


def neighbours(bags: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per bag and species: a labelled positive on the same night (noon to noon), and
    within ±1 h; plus bag_id -> row."""
    when = [dt.datetime.fromisoformat(b["datetime"]) for b in bags]
    night = [(w - dt.timedelta(hours=12)).date() for w in when]
    pos = np.array([[int(b[f"{sp}_present"]) for sp in SPECIES] for b in bags], bool)
    at = {w: i for i, w in enumerate(when)}
    same_night, near = np.zeros_like(pos), np.zeros_like(pos)
    by_night: dict = {}
    for i, n in enumerate(night):
        by_night.setdefault(n, []).append(i)
    for rows in by_night.values():
        same_night[rows] = pos[rows].any(0)
    for i, w in enumerate(when):
        for d in (-1, 1):
            j = at.get(w + dt.timedelta(hours=d))
            if j is not None:
                near[i] |= pos[j]
    return same_night, near, {b["bag_id"]: i for i, b in enumerate(bags)}


def silent_table(models, regime) -> list[str]:
    with BAGS_CSV.open() as fh:
        bags = list(csv.DictReader(fh))
    same_night, near, row = neighbours(bags)
    m0 = next(iter(models.values()))
    order = np.array([row[b] for b in m0.bag_ids])
    same_night, near = same_night[order], near[order]
    out = [f"| species | trained on | embeddings | ρ with reference | top {TOP:.0%} overlap "
           f"with reference | top {TOP:.0%}: positive same night | top {TOP:.0%}: "
           "positive within ±1 h |", "|---|---|---|---|---|---|---|"]
    for c, sp in enumerate(SPECIES):
        silent = (regime == "8k-3clip") & (m0.idx[:, c] == 0)
        k = int(np.ceil(TOP * silent.sum()))
        rows = np.flatnonzero(silent)
        out.append(f"| {SP_LABEL[sp]} | *all {silent.sum()} silent 8 kHz hours* | | | | "
                   f"{same_night[rows, c].mean():.2f} | {near[rows, c].mean():.2f} |")

        def top(mid):
            s = models[mid].scores.mean(0)[rows, c]
            return s, rows[np.argsort(-s)[:k]]
        for e in EMB:
            ref_s, ref_top = top(f"{run_id('all', e)}/lme")
            for t in TRAIN_ON:
                s, tp = top(f"{run_id(t, e)}/lme")
                rho = spearmanr(ref_s, s).statistic
                ov = len(set(tp) & set(ref_top)) / k
                out.append(f"| {SP_LABEL[sp]} | {LABEL[t]} | {e} | "
                           + (f"{rho:.2f} | {ov:.2f}" if t != "all" else "– | –")
                           + f" | {same_night[tp, c].mean():.2f} | {near[tp, c].mean():.2f} |")
    return out


def summarize(did: str) -> None:
    models = {m.model_id: m for m in load_models(did, RUNS, verbose=False)}
    blocks, regime = load_bag_table(list(models.values()))
    st = st_.compute(list(models.values()), blocks, regime, B, SEED,
                     cache_dir=RUNS / did / ".cache")
    out = ["# Cross-regime transfer", "",
           "Generated by `scripts/regime_transfer.py`; do not edit by hand. Model: "
           "`mlp256-ord0.5-s2/lme`, trained and early-stopped on all regimes, on the 44.1 kHz "
           "hours only, or on the 8 kHz hours only (`frog-train --train-regimes`), each on "
           "embeddings as recorded (`full`) or with all audio band-limited to 8 kHz (`bl8k`). "
           "Every model scores every hour out of fold (same folds and seeds), so **bold** "
           "cells are transfer: a regime the model never trained on. Δ = model − the "
           "all-regimes model on the same embeddings, paired block bootstrap "
           f"(B={B}); `*` = CI excludes 0. Full tables: [RESULTS.md](RESULTS.md); "
           "ignore its validation selection, which can't compare these models.", "",
           "## AP within each regime", "", *ap_table(st, models, regime),
           "## Silent 8 kHz hours", "",
           f"Silent 8 kHz hours of each species, ranked by each model's seed-mean score. "
           f"ρ: Spearman correlation with the all-regimes model on the same embeddings. "
           f"Overlap: share of that model's top {TOP:.0%} also in the all-regimes model's "
           f"top {TOP:.0%} (chance {TOP:.2f}). The last two columns: share of the top "
           f"{TOP:.0%} with a labelled positive hour of the species on the same night "
           "(noon to noon) or within ±1 h, against all silent 8 kHz hours (first row).", "",
           *silent_table(models, regime), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text("\n".join(out))
    print(f"written {OUT / 'SUMMARY.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default="0", help="GPU for training")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    did = current_dataset_id()
    if not a.summary_only:
        train(a.gpu, did)
        report(did)
    summarize(did)


if __name__ == "__main__":
    main()
