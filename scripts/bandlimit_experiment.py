"""Band-limit control: is the model using the recording format as a shortcut?

The 8 kHz recordings (Sep–Oct 2018) hold nothing above 4 kHz, and *Gastrotheca*
is positive in 23% of their hours against ~4% of the 44.1 kHz hours. A model
that can tell the formats apart can score hours by format. Here the reference
model (``mlp256-ord0.5-s2/lme``) is trained twice per design, with the same
folds and seeds: once on the audio as recorded (``full``) and once with every
clip above 8 kHz resampled to 8 kHz first (``bl8k``, the same path the 8 kHz
recordings take, via ``embed_perch.py --band-limit 8000``).

Two designs:

``mixed``  all 4914 hours (8 kHz + 44.1 kHz). Band-limiting removes the
           bandwidth cue. If the full-band model used it, the 8 kHz silent
           hours stop scoring high and the 8 kHz share of the top scores drops.
``44k``    the 3647 44.1 kHz hours only, so no format cue exists. The paired
           difference is the cost of losing everything above 4 kHz. It tells
           the shortcut apart from lost call energy in the ``mixed`` result.

It also fits a linear classifier of regime (8 kHz vs 44.1 kHz) on single window
embeddings, before and after band-limiting: what format information is left for
a probe to find.

Steps are idempotent; rerun to resume:

    .venv/bin/python scripts/bandlimit_experiment.py [--gpu 3]

State (manifest subset, embedding caches, runs) lives in
``STORE/experiments/bandlimit_8k/``, so these models stay out of the main sweep
and its validation selection. Tables go to ``results/experiments/bandlimit_8k/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frog_mil import embcache  # noqa: E402
from frog_mil import stats as st_  # noqa: E402
from frog_mil.config import (  # noqa: E402
    BAGS_CSV,
    EMB_DIR,
    INSTANCES_CSV,
    OUTPUTS,
    RESULTS,
    ROOT,
    SP_LABEL,
    SPECIES,
    STORE,
    manifest_summary,
)
from frog_mil.manifest import hash_dataset, hash_instances  # noqa: E402
from frog_mil.report import load_bag_table  # noqa: E402
from frog_mil.runs import load_models  # noqa: E402

BAND_HZ = 8000
EXP = STORE / "experiments" / "bandlimit_8k"
RUNS = EXP / "runs"
OUT = RESULTS / "experiments" / "bandlimit_8k"
MODEL = ["--hidden", "256", "--ordinal-weight", "0.5", "--pooling", "lme"]
RID = {"full": "mlp256-ord0.5-s2", "bl8k": "mlp256-ord0.5-s2-bl8k"}
DESIGNS = {
    "mixed": {"manifest": OUTPUTS,
              "emb": {"full": EMB_DIR, "bl8k": EXP / f"emb_all_bl{BAND_HZ}"}},
    "44k": {"manifest": EXP / "manifest_44k",
            "emb": {"full": EXP / "emb_44k_full", "bl8k": EXP / f"emb_44k_bl{BAND_HZ}"}},
}
PY = ROOT / ".venv" / "bin" / "python"
PERCH = [str(ROOT / "scripts" / "perch_env.sh"), "python",
         str(ROOT / "scripts" / "embed_perch.py")]
B, SEED = 2000, 0


# --------------------------------------------------------------------------- manifest
def build_44k_manifest() -> None:
    """Every hour recorded above 8 kHz, with the main manifest's folds."""
    parent, out = manifest_summary(), DESIGNS["44k"]["manifest"]
    with BAGS_CSV.open() as fh:
        rd = csv.DictReader(fh)
        cols, bags = rd.fieldnames, [r for r in rd if int(r["sample_rate"]) > BAND_HZ]
    keep = {b["bag_id"] for b in bags}
    with INSTANCES_CSV.open() as fh:
        rd = csv.DictReader(fh)
        icols, inst = rd.fieldnames, [r for r in rd if r["bag_id"] in keep]
    out.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in ((out / "bags.csv", cols, bags),
                               (out / "instances.csv", icols, inst)):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    instances_id = hash_instances([r["instance_id"] for r in inst])
    regimes = sorted({b["regime"] for b in bags})
    summary = {"dataset_id": hash_dataset(out / "bags.csv", instances_id),
               "instances_id": instances_id, "folds": parent["folds"],
               "n_bags": len(bags), "n_instances": len(inst),
               "parent_dataset_id": parent["dataset_id"],
               "subset": f"hours recorded above {BAND_HZ} Hz: {', '.join(regimes)}",
               "regimes": {r: parent["regimes"][r] for r in regimes}}
    embcache.write_text_atomic(out / "summary.json", json.dumps(summary, indent=2))
    print(f"44k manifest: {len(bags)} bags, {len(inst)} windows, "
          f"dataset {summary['dataset_id']} (parent {parent['dataset_id']})")


def summary_of(design: str) -> dict:
    return json.loads((DESIGNS[design]["manifest"] / "summary.json").read_text())


# --------------------------------------------------------------------------- embed
def complete(emb_dir: Path, instances_id: str) -> bool:
    try:
        embcache.check_matches(emb_dir, instances_id)
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def embed_into(emb_dir: Path, design: str, gpu: str, band_limit: int = 0,
               seed_from: Path | None = None) -> None:
    """Bring ``emb_dir`` up to date for ``design``'s windows.

    ``seed_from``: a cache to start from; embed_perch re-keys it by instance_id,
    so only windows it lacks are embedded.
    """
    if complete(emb_dir, summary_of(design)["instances_id"]):
        return
    if seed_from is not None and not emb_dir.exists():
        print(f"copying {seed_from} -> {emb_dir}", flush=True)
        shutil.copytree(seed_from, emb_dir)
    cmd = [*PERCH, "--instances", str(DESIGNS[design]["manifest"] / "instances.csv"),
           "--out-dir", str(emb_dir), "--band-limit", str(band_limit)]
    subprocess.run(cmd, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
    if not complete(emb_dir, summary_of(design)["instances_id"]):
        raise SystemExit(f"{emb_dir} is still incomplete")


def embed(gpu: str) -> None:
    e44, emix = DESIGNS["44k"]["emb"], DESIGNS["mixed"]["emb"]
    embed_into(e44["full"], "44k", gpu, seed_from=EMB_DIR)
    embed_into(e44["bl8k"], "44k", gpu, band_limit=BAND_HZ)
    # All hours band-limited: the 44.1 kHz rows carry over from the 44k cache;
    # the 8 kHz clips are embedded again (band-limiting leaves them unchanged).
    embed_into(emix["bl8k"], "mixed", gpu, band_limit=BAND_HZ, seed_from=e44["bl8k"])


# --------------------------------------------------------------------------- train/report
def train(gpu: str) -> None:
    # One after the other: the two arms of a design write its shared baselines.
    for design, d in DESIGNS.items():
        did = summary_of(design)["dataset_id"]
        for arm, rid in RID.items():
            if (RUNS / did / rid / "lme" / "predictions.npz").exists():
                continue
            log = EXP / f"train_{design}_{rid}.log"
            print(f"training {design}/{rid} on GPU {gpu} (log {log})", flush=True)
            with log.open("a") as fh:
                r = subprocess.run(
                    [str(PY), "-m", "frog_mil.train", "--manifest", str(d["manifest"]),
                     "--emb-dir", str(d["emb"][arm]), "--out-dir", str(RUNS), "--tag", rid,
                     *MODEL], stdout=fh, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
            if r.returncode:
                raise SystemExit(f"training {design}/{rid} failed; see {log}")


def report() -> None:
    for design, d in DESIGNS.items():
        subprocess.run([str(PY), "-m", "frog_mil.report", "--dataset",
                        summary_of(design)["dataset_id"], "--runs", str(RUNS),
                        "--bags-csv", str(d["manifest"] / "bags.csv"),
                        "--out", str(OUT / design)], check=True)


# --------------------------------------------------------------------------- summary
def fmt_d(d, lo, hi) -> str:
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def compare(design: str) -> tuple[list[str], dict, np.ndarray]:
    """Markdown rows of bl8k vs full, overall and per regime; the models; their regimes."""
    d = DESIGNS[design]
    did = summary_of(design)["dataset_id"]
    models = load_models(did, RUNS, verbose=False)
    blocks, regime = load_bag_table(models, d["manifest"] / "bags.csv")
    st = st_.compute(models, blocks, regime, B, SEED, cache_dir=RUNS / did / ".cache")
    full, bl = (f"{RID[a]}/lme" for a in ("full", "bl8k"))
    idx = models[0].idx
    cells = [("macro", "all hours", "macro", None)]
    cells += [(SP_LABEL[s], "all hours", ("presence", c), None) for c, s in enumerate(SPECIES)]
    cells += [(SP_LABEL[SPECIES[c[1]]], c[2], c, c[2]) for c in st.cells if c[0] == "regime"]
    rows = []
    for name, where, cell, r in cells:
        if cell == "macro":
            (a, ab), (b, bb) = st.get(full, "macro"), st.get(bl, "macro")
            n = ""
        else:
            a, ab, b, bb = (st.ap(full, cell), st.ap_boot(full, cell), st.ap(bl, cell),
                            st.ap_boot(bl, cell))
            keep = np.ones(len(idx), bool) if r is None else regime == r
            n = f"{int((idx[keep, cell[1]] > 0).sum())} / {int(keep.sum())}"
        rows.append(f"| {name} | {where} | {n} | {a:.3f} | {b:.3f} | "
                    f"{fmt_d(b - a, *st_.ci(bb - ab))} |")
    return rows, {m.model_id: m for m in models}, regime


def shortcut_rows(models: dict, regime: np.ndarray) -> list[str]:
    """How the mixed-design models score silent hours, by regime."""
    rows = []
    for arm in ("full", "bl8k"):
        m = models[f"{RID[arm]}/lme"]
        s, y = m.scores.mean(0), m.y
        for c, sp in enumerate(SPECIES):
            tails = [f"{np.quantile(s[(regime == r) & (y[:, c] == 0), c], 0.95):.2f}"
                     for r in sorted(set(regime))]
            top = np.argsort(-s[:, c])[:int(y[:, c].sum())]
            rows.append(f"| {arm} | {SP_LABEL[sp]} | {' / '.join(tails)} | "
                        f"{np.mean(regime[top] == '8k-3clip'):.2f} | "
                        f"{np.mean(regime[y[:, c] == 1] == '8k-3clip'):.2f} |")
    return rows


def regime_decoding(emb_dir: Path, months: set[int] | None = None, n_train: int = 20000,
                    seed: int = 0) -> tuple[float, float]:
    """Fold-CV AUC and balanced accuracy of a linear 8 kHz-vs-44.1 kHz classifier on
    single window embeddings (``months``: only hours from those months)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    with BAGS_CSV.open() as fh:
        bags = {r["bag_id"]: r for r in csv.DictReader(fh)}
    with (emb_dir / "index.csv").open() as fh:
        idx = list(csv.DictReader(fh))
    rows = np.flatnonzero([months is None or int(bags[r["bag_id"]]["date"][5:7]) in months
                           for r in idx])
    lab = np.array([bags[idx[i]["bag_id"]]["regime"] == "8k-3clip" for i in rows])
    fold = np.array([int(bags[idx[i]["bag_id"]]["fold"]) for i in rows])
    x = np.asarray(np.load(emb_dir / "embeddings.f16.npy", mmap_mode="r")[rows], np.float32)
    rng = np.random.default_rng(seed)
    p = np.zeros(len(rows))
    for f in np.unique(fold):
        tr = np.flatnonzero(fold != f)
        tr = rng.choice(tr, min(n_train, len(tr)), replace=False)
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        clf.fit(sc.transform(x[tr]), lab[tr])
        te = fold == f
        p[te] = clf.predict_proba(sc.transform(x[te]))[:, 1]
    return roc_auc_score(lab, p), balanced_accuracy_score(lab, p > 0.5)


def summarize() -> None:
    out = ["# Band-limit control (8 kHz)", "",
           "Generated by `scripts/bandlimit_experiment.py`; do not edit by hand. Model: "
           "`mlp256-ord0.5-s2/lme` (the main reference), trained per design on the audio as "
           "recorded (`full`) and with every clip above 8 kHz resampled to 8 kHz first "
           "(`bl8k`), with the same folds and seeds. Out-of-fold AP, 5 seeds; "
           f"Δ = bl8k − full, paired block bootstrap (B={B}); `*` = CI excludes 0. "
           "Full tables per design: [`mixed/`](mixed/RESULTS.md), [`44k/`](44k/RESULTS.md).",
           ""]
    for design, title in (("mixed", "Mixed design: all 4914 hours, 44.1 kHz audio "
                                     "band-limited in `bl8k`"),
                          ("44k", "44.1 kHz hours only: the cost of losing > 4 kHz")):
        rows, models, regime = compare(design)
        out += [f"## {title}", "", "| AP | hours | positives / hours | full | bl8k | Δ |",
                "|---|---|---|---|---|---|", *rows, ""]
        if design == "mixed":
            mixed = (models, regime)
    regs = " / ".join(sorted(set(mixed[1])))
    out += ["## Do silent 8 kHz hours still score high? (mixed design)", "",
            "95th percentile of the seed-mean score on silent hours per regime, and the "
            "share of 8 kHz hours among the top-k scored hours (k = positives), against "
            "their share of the positives.", "",
            f"| model | species | silent-hour 95th pct ({regs}) | 8 kHz share of top k | "
            "8 kHz share of positives |", "|---|---|---|---|---|",
            *shortcut_rows(*mixed), ""]
    dec = []
    for name, emb in (("full", EMB_DIR), ("bl8k", DESIGNS["mixed"]["emb"]["bl8k"])):
        for label, months in (("all hours", None), ("Sep–Oct hours only", {9, 10})):
            auc, bacc = regime_decoding(emb, months)
            dec.append(f"| {name} | {label} | {auc:.3f} | {bacc:.3f} |")
    out += ["## Can a linear probe still tell the formats apart?", "",
            "Logistic regression of 8 kHz vs 44.1 kHz on single 5 s window embeddings, "
            "trained on 20k windows of the other folds and tested on each fold. "
            "*Sep–Oct only* compares Sep–Oct 2018 (8 kHz) with Sep–Oct 2019 (44.1 kHz), "
            "the same season a year apart.", "",
            "| embeddings | hours | AUC | balanced accuracy |", "|---|---|---|---|", *dec, ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text("\n".join(out))
    print(f"written {OUT / 'SUMMARY.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default="0", help="GPU for embedding and training")
    ap.add_argument("--summary-only", action="store_true",
                    help="only rewrite SUMMARY.md from finished runs")
    a = ap.parse_args()
    if not a.summary_only:
        build_44k_manifest()
        embed(a.gpu)
        train(a.gpu)
        report()
    summarize()


if __name__ == "__main__":
    main()
