"""The run store: where trained models live, their configs, and loading them back.

    config.RUNS/<dataset_id>/<run_id>/config.json
    config.RUNS/<dataset_id>/<run_id>/<pooling>/predictions.npz   bag scores (OOF test + val)
                                               /windows.npz       window logits + weights
                                               /metrics.json      per seed x fold
                                               /seed{k}_fold{f}.pt
    config.RUNS/<dataset_id>/baselines/<name>.npz

Runs are keyed by the dataset they were trained on, so a new manifest starts an
empty directory and old runs stay where they are. ``predictions.npz`` is
written last and atomically, so its presence marks a finished model; a run's
``config.json`` says ``"status": "complete"`` once every pooler is done.

Naming: a *run* is one probe/target setting, ``{probe}-{target}-s2`` (``s2`` =
contiguous 5 s windows, kept so names match earlier runs); a *model* is a run
plus a pooler, ``{run_id}/{pooling}``. Settings outside the name (lr, epochs,
...) need ``--tag``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ROOT, RUNS, SPECIES
from .models import MILModel

# Config fields that change what is trained. Two configs with equal SPEC fields
# describe the same run; anything else (status, timestamps, code version) is
# bookkeeping.
# ``band_limit_hz`` is the embedding cache's (None = full band; absent in older configs).
SPEC = ["run_id", "dataset_id", "band_limit_hz", "hidden", "dropout", "attn_hidden", "lme_r",
        "lme_learnable", "ordinal_weight", "lr", "weight_decay", "epochs", "patience",
        "batch_size", "seeds", "folds"]
# Fields that may differ between two runs in a controlled comparison.
FACTORS = ("hidden", "ordinal_weight")


def run_id_for(hidden: int, ordinal_weight: float) -> str:
    probe = f"mlp{hidden}" if hidden else "linear"
    target = f"ord{ordinal_weight:g}" if ordinal_weight else "bin"
    return f"{probe}-{target}-s2"


def dataset_dir(dataset_id: str, root: Path = RUNS) -> Path:
    return root / dataset_id


def read_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def savez_atomic(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def spec(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in SPEC}


# The code that decides what a trained model is; its hash is recorded per run so
# `frog status` can say a run predates a change (without forcing a retrain).
TRAIN_CODE = ["data.py", "models.py", "pooling.py", "train.py"]


def train_code_hash() -> str:
    h = hashlib.sha256()
    for f in TRAIN_CODE:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def code_version() -> dict:
    """Git commit, uncommitted changes and the training-code hash, for provenance."""
    out = {"train_code": train_code_hash(), "commit": None, "dirty": None}
    try:
        out["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True,
                                       check=True).stdout.strip()
        out["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def build_model(cfg: dict, pooling: str, dim: int) -> MILModel:
    return MILModel(dim=dim, n_classes=len(SPECIES), hidden=cfg["hidden"], pooling=pooling,
                    dropout=cfg["dropout"], ordinal=cfg["ordinal_weight"] > 0,
                    lme_r=cfg["lme_r"], lme_learnable=cfg["lme_learnable"],
                    attn_hidden=cfg["attn_hidden"])


def load_model(run_dir: Path, pooling: str, seed: int, fold: int, dim: int,
               device: str | torch.device = "cpu") -> MILModel:
    """The checkpoint of one seed x fold, in eval mode."""
    m = build_model(read_config(run_dir), pooling, dim)
    m.load_state_dict(torch.load(run_dir / pooling / f"seed{seed}_fold{fold}.pt",
                                 map_location=device))
    return m.to(device).eval()


@dataclass
class Model:
    """Saved predictions of one model or baseline, as the report reads them."""
    model_id: str
    kind: str              # model | baseline
    run_id: str
    pooling: str
    cfg: dict
    params: int
    scores: np.ndarray     # [seeds, bags, C] out-of-fold test scores
    val_scores: np.ndarray | None
    y: np.ndarray
    idx: np.ndarray
    bag_ids: np.ndarray
    fold: np.ndarray
    val_ap: np.ndarray     # [seeds, folds] best val macro AP (early-stopping optimum)
    epochs: np.ndarray
    thresholds: np.ndarray | None = None
    path: Path | None = None

    @property
    def probe(self) -> str:
        h = self.cfg.get("hidden", 0)
        return f"MLP-{h}" if h else ("linear" if self.kind == "model" else "—")

    @property
    def target(self) -> str:
        if self.kind != "model":
            return "—"
        w = self.cfg.get("ordinal_weight", 0)
        return f"ordinal (w={w:g})" if w else "binary"

    @property
    def is_model(self) -> bool:
        return self.kind == "model"


def load_models(dataset_id: str, root: Path = RUNS, verbose: bool = True) -> list[Model]:
    """Every finished model and baseline trained on ``dataset_id``."""
    base = dataset_dir(dataset_id, root)
    models = []
    for cfg_path in sorted(base.glob("*/config.json")):
        run_dir = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("dataset_id") != dataset_id:
            raise RuntimeError(f"{run_dir} says dataset {cfg.get('dataset_id')}")
        if verbose and cfg.get("status") != "complete":
            print(f"note: {run_dir.name} is not complete; using its finished poolers")
        for pooling in cfg["poolings"]:
            f = run_dir / pooling / "predictions.npz"
            if not f.exists():
                continue
            d = np.load(f)
            assert str(d["dataset_id"]) == dataset_id, f"{f}: wrong dataset"
            models.append(Model(
                model_id=f"{run_dir.name}/{pooling}", kind="model", run_id=run_dir.name,
                pooling=pooling, cfg=cfg, params=int(d["params"]), scores=d["scores"],
                val_scores=d["val_scores"], y=d["y"], idx=d["idx"], bag_ids=d["bag_ids"],
                fold=d["fold"], val_ap=d["val_macro_ap"], epochs=d["epochs"],
                thresholds=d["thresholds"] if "thresholds" in d else None,
                path=run_dir / pooling))
    for f in sorted((base / "baselines").glob("*.npz")):
        d = np.load(f)
        assert str(d["dataset_id"]) == dataset_id, f"{f}: wrong dataset"
        models.append(Model(
            model_id=f"baseline/{f.stem}", kind="baseline", run_id="baseline",
            pooling=f.stem, cfg={}, params=0, scores=d["scores"], val_scores=None,
            y=d["y"], idx=d["idx"], bag_ids=d["bag_ids"], fold=d["fold"],
            val_ap=np.array([]), epochs=np.array([])))
    if models:
        ref = models[0].bag_ids
        for m in models:
            assert np.array_equal(m.bag_ids, ref), f"{m.model_id}: different bags"
    return models
