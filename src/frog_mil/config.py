"""Paths and constants shared by every stage, so they are defined once.

Large or machine-specific artefacts (embedding cache, trained runs, window-level
scores, archives of old data versions) live in the group workspace, ``STORE``;
the repo keeps the manifest (``outputs/``), the tracked tables (``results/``) and
figures (``viz/figures/``). Override the two roots with ``FROG_STORE`` and
``FROG_AUDIO_DIR``.

Paths are absolute (anchored on the repo root), so every command works from any
working directory.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GWS = Path("/gws/ssde/j25b/iecdt/dash/frogs")

AUDIO_DIR = Path(os.environ.get("FROG_AUDIO_DIR", GWS / "data" / "2019_Rsync"))
DROPBOX_DIR = GWS / "data" / "dropbox"           # staging for scripts/ingest.py
STORE = Path(os.environ.get("FROG_STORE", GWS / "frog_mil"))

ANNOTATIONS = ROOT / "data" / "df_helechos_with2020.csv"
OUTPUTS = ROOT / "outputs"                       # manifest: bags, instances, summary
BAGS_CSV = OUTPUTS / "bags.csv"
INSTANCES_CSV = OUTPUTS / "instances.csv"
SUMMARY_JSON = OUTPUTS / "summary.json"
LOGS = OUTPUTS / "logs"
STAMPS = OUTPUTS / ".stamps"                     # frog pipeline bookkeeping
RESULTS = ROOT / "results"
FIGURES = ROOT / "viz" / "figures"
ARCHIVE = ROOT / "archive"                       # tracked tables/figures of old data versions
SWEEP = ROOT / "sweep.toml"

EMB_DIR = STORE / "embeddings"
RUNS = STORE / "runs"                            # runs/<dataset_id>/<run_id>/<pooling>/
STORE_ARCHIVE = STORE / "archive"                # untracked state of old data versions

SPECIES_FULL = ["Gastrotheca chrysosticta", "Oreobates berdemenos"]
SPECIES = ["gastrotheca", "oreobates"]
SPECIES_SHORT = dict(zip(SPECIES_FULL, SPECIES))
SP_LABEL = {"gastrotheca": "G. chrysosticta", "oreobates": "O. berdemenos"}
MAX_INDEX = 3

# Recorder file names: HELECHOS_YYYYMMDD_HHMMSS.wav
FNAME_RE = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<date>\d{8})_(?P<time>\d{6})\.wav$")


def manifest_summary(path: Path = SUMMARY_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path.parent}; run frog-manifest first")
    return json.loads(path.read_text())


def current_dataset_id() -> str:
    return manifest_summary()["dataset_id"]
