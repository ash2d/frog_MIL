"""Band-limit control: does the probe need the audio above 4 kHz?

The 8 kHz recordings (Sep–Oct 2018) hold nothing above 4 kHz and are confounded
with the labels. This experiment takes every 44.1 kHz hour (the only hours that
band-limiting changes), embeds them a second time after resampling to 8 kHz
(``embed_perch.py --band-limit 8000``, the same path the 8 kHz recordings take),
and trains the reference model on each version with the same folds and seeds:

    full band    mlp256-ord0.5-s2/lme       embeddings copied from the main cache
    band-limited mlp256-ord0.5-s2-bl8k/lme  embeddings of the 8 kHz-resampled audio

The paired difference is the cost of losing everything above 4 kHz. Steps are
idempotent; rerun to resume:

    .venv/bin/python scripts/bandlimit_experiment.py [--gpu 0]

State lives in ``STORE/experiments/bandlimit_8k/`` (manifest, two embedding
caches), runs in ``runs/<subset dataset_id>/``, tables in
``results/experiments/bandlimit_8k/``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frog_mil import embcache  # noqa: E402
from frog_mil.config import (  # noqa: E402
    BAGS_CSV,
    EMB_DIR,
    INSTANCES_CSV,
    RESULTS,
    ROOT,
    RUNS,
    STORE,
    manifest_summary,
)
from frog_mil.manifest import hash_dataset, hash_instances  # noqa: E402

BAND_HZ = 8000
EXP = STORE / "experiments" / "bandlimit_8k"
MANIFEST = EXP / "manifest"
EMB_FULL, EMB_BL = EXP / "emb_full", EXP / f"emb_bl{BAND_HZ}"
OUT = RESULTS / "experiments" / "bandlimit_8k"
MODEL = ["--hidden", "256", "--ordinal-weight", "0.5", "--pooling", "lme"]
ARMS = {"full": (EMB_FULL, "mlp256-ord0.5-s2"), "bl": (EMB_BL, "mlp256-ord0.5-s2-bl8k")}
PY = ROOT / ".venv" / "bin" / "python"
PERCH = [str(ROOT / "scripts" / "perch_env.sh"), "python",
         str(ROOT / "scripts" / "embed_perch.py")]


def build_manifest() -> str:
    """Every hour recorded above 8 kHz, with the main manifest's folds; returns dataset_id."""
    parent = manifest_summary()
    with BAGS_CSV.open() as fh:
        rd = csv.DictReader(fh)
        cols, bags = rd.fieldnames, [r for r in rd if int(r["sample_rate"]) > BAND_HZ]
    keep = {b["bag_id"] for b in bags}
    with INSTANCES_CSV.open() as fh:
        rd = csv.DictReader(fh)
        icols, inst = rd.fieldnames, [r for r in rd if r["bag_id"] in keep]
    MANIFEST.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in ((MANIFEST / "bags.csv", cols, bags),
                               (MANIFEST / "instances.csv", icols, inst)):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    instances_id = hash_instances([r["instance_id"] for r in inst])
    dataset_id = hash_dataset(MANIFEST / "bags.csv", instances_id)
    regimes = sorted({b["regime"] for b in bags})
    summary = {"dataset_id": dataset_id, "instances_id": instances_id,
               "folds": parent["folds"], "n_bags": len(bags), "n_instances": len(inst),
               "parent_dataset_id": parent["dataset_id"],
               "subset": f"hours recorded above {BAND_HZ} Hz: {', '.join(regimes)}",
               "regimes": {r: parent["regimes"][r] for r in regimes}}
    embcache.write_text_atomic(MANIFEST / "summary.json", json.dumps(summary, indent=2))
    print(f"subset manifest: {len(bags)} bags, {len(inst)} windows ({', '.join(regimes)}), "
          f"dataset {dataset_id} (parent {parent['dataset_id']})")
    return dataset_id


def complete(emb_dir: Path, instances_id: str) -> bool:
    try:
        embcache.check_matches(emb_dir, instances_id)
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def embed(gpu: str, instances_id: str) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    inst = ["--instances", str(MANIFEST / "instances.csv")]
    if not complete(EMB_FULL, instances_id):
        # Full band: copy the main cache, then embed_perch re-keys it to the subset
        # by instance_id (and would embed any row it can't carry over).
        if not EMB_FULL.exists():
            print(f"copying {EMB_DIR} -> {EMB_FULL}", flush=True)
            shutil.copytree(EMB_DIR, EMB_FULL)
        subprocess.run([*PERCH, *inst, "--out-dir", str(EMB_FULL)], check=True, env=env)
    if not complete(EMB_BL, instances_id):
        subprocess.run([*PERCH, *inst, "--out-dir", str(EMB_BL), "--band-limit", str(BAND_HZ)],
                       check=True, env=env)
    for d in (EMB_FULL, EMB_BL):
        if not complete(d, instances_id):
            raise SystemExit(f"{d} is still incomplete")


def train(gpu: str, dataset_id: str) -> None:
    # One after the other: both arms write the dataset's shared baselines.
    for emb, rid in ARMS.values():
        if (RUNS / dataset_id / rid / "lme" / "predictions.npz").exists():
            print(f"{rid}: done")
            continue
        log = EXP / f"train_{rid}.log"
        print(f"training {rid} on GPU {gpu} (log {log})", flush=True)
        with log.open("a") as fh:
            r = subprocess.run(
                [str(PY), "-m", "frog_mil.train", "--manifest", str(MANIFEST),
                 "--emb-dir", str(emb), "--tag", rid, *MODEL], stdout=fh,
                stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        if r.returncode:
            raise SystemExit(f"training {rid} failed; see {log}")


def report(dataset_id: str) -> None:
    subprocess.run([str(PY), "-m", "frog_mil.report", "--dataset", dataset_id,
                    "--bags-csv", str(MANIFEST / "bags.csv"), "--out", str(OUT)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default="0", help="GPU for embedding and training")
    a = ap.parse_args()
    dataset_id = build_manifest()
    instances_id = json.loads((MANIFEST / "summary.json").read_text())["instances_id"]
    embed(a.gpu, instances_id)
    train(a.gpu, dataset_id)
    report(dataset_id)


if __name__ == "__main__":
    main()
