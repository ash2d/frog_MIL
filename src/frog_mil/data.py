"""Bags of frozen Perch v2 embeddings, assembled from the manifest.

The embedding cache is built at the finest hop (2.5 s). ``stride`` selects a
coarser geometry from it without re-embedding, because ``win_idx % stride == 0``
is an exact subset: ``stride=2`` is the contiguous 5 s tiling, ``stride=1``
keeps the 50%-overlap version. That is the whole reason the manifest uses a hop
that divides the window.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPECIES = ["gastrotheca", "oreobates"]
MAX_INDEX = 3


@dataclass
class DataConfig:
    emb_dir: Path = Path("embeddings")
    bags_csv: Path = Path("outputs/bags.csv")
    stride: int = 2          # 2 = contiguous 5 s tiling, 1 = 2.5 s overlap
    max_instances: int = 0   # 0 = keep the whole bag (24 instances fits easily)
    standardize: bool = True
    allow_incomplete: bool = False   # see the guard in BagEmbeddingDataset


class BagEmbeddingDataset(Dataset):
    """One item per annotated hour: [n_instances, 1536] plus both targets."""

    def __init__(self, cfg: DataConfig, split: str, stats: tuple | None = None,
                 train: bool = False, seed: int = 0):
        self.cfg, self.split, self.train = cfg, split, train
        self.rng = np.random.default_rng(seed)

        meta_path = cfg.emb_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no embedding cache at {cfg.emb_dir}. Build it with:\n"
                f"  scripts/perch_env.sh python scripts/embed_perch.py "
                f"--out-dir {cfg.emb_dir}")
        meta = json.loads(meta_path.read_text())
        # Unembedded rows are zeros, not missing, so a half-finished cache trains
        # silently and scores like noise. Fail loudly instead.
        done, total = meta.get("n_clips_done", 0), meta.get("n_clips_total", 0)
        if done < total and not cfg.allow_incomplete:
            raise RuntimeError(
                f"embedding cache is incomplete ({done}/{total} clips). Finish it "
                f"with scripts/embed_perch.py, or set allow_incomplete=True if you "
                f"really want zero vectors for the rest.")
        self.dim = meta["dim"]
        self.emb = np.load(cfg.emb_dir / "embeddings.f16.npy", mmap_mode="r")

        with (cfg.bags_csv).open() as fh:
            bags = {r["bag_id"]: r for r in csv.DictReader(fh) if r["split"] == split}
        rows_by_bag: dict[str, list[int]] = defaultdict(list)
        with (cfg.emb_dir / "index.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r["bag_id"] in bags and int(r["win_idx"]) % cfg.stride == 0:
                    rows_by_bag[r["bag_id"]].append(int(r["row"]))

        self.bag_ids = sorted(b for b in bags if rows_by_bag[b])
        self.rows = {b: np.array(sorted(rows_by_bag[b])) for b in self.bag_ids}
        self.bags = [bags[b] for b in self.bag_ids]
        self.mean, self.std = stats if stats is not None else self._fit_stats()

    def _fit_stats(self):
        if not self.cfg.standardize:
            return np.zeros(self.dim, np.float32), np.ones(self.dim, np.float32)
        idx = np.concatenate([self.rows[b] for b in self.bag_ids])
        x = np.asarray(self.emb[idx], np.float32)
        return x.mean(0), x.std(0) + 1e-6

    @property
    def stats(self):
        return self.mean, self.std

    def pos_weight(self) -> torch.Tensor:
        """neg/pos per species. Gastrotheca is ~7% of hours, so without this
        the probe learns the constant-negative solution and still scores well
        on accuracy."""
        out = []
        for s in SPECIES:
            pos = sum(int(b[f"{s}_present"]) for b in self.bags)
            out.append((len(self.bags) - pos) / max(pos, 1))
        return torch.tensor(out, dtype=torch.float32)

    def cum_pos_weight(self) -> torch.Tensor:
        """[C, 3] neg/pos for each threshold y>=1, y>=2, y>=3.

        Positives thin out fast up the scale (index-3 Gastrotheca is ~1% of
        hours), so each threshold gets its own weight rather than inheriting the
        presence one.
        """
        out = []
        for s in SPECIES:
            idx = np.array([int(b[f"{s}_index"]) for b in self.bags])
            out.append([(len(idx) - (idx >= k).sum()) / max((idx >= k).sum(), 1)
                        for k in range(1, MAX_INDEX + 1)])
        return torch.tensor(out, dtype=torch.float32)

    def __len__(self):
        return len(self.bag_ids)

    def __getitem__(self, i):
        bag_id = self.bag_ids[i]
        rows = self.rows[bag_id]
        k = self.cfg.max_instances
        if k and len(rows) > k:
            rows = (np.sort(self.rng.choice(rows, k, replace=False)) if self.train
                    else rows[np.linspace(0, len(rows) - 1, k).round().astype(int)])
        x = (np.asarray(self.emb[rows], np.float32) - self.mean) / self.std
        b = self.bags[i]
        return {
            "x": torch.from_numpy(x),
            "presence": torch.tensor([float(b[f"{s}_present"]) for s in SPECIES]),
            "intensity": torch.tensor([int(b[f"{s}_index"]) for s in SPECIES]),
            "bag_id": bag_id,
            "hour": int(b["hour"]),
            "date": b["date"],
        }


def collate(batch: list[dict]) -> dict:
    """Pad ragged bags and emit the validity mask every pooler needs."""
    n = max(b["x"].shape[0] for b in batch)
    x = torch.zeros(len(batch), n, batch[0]["x"].shape[1])
    mask = torch.zeros(len(batch), n, dtype=torch.bool)
    for i, b in enumerate(batch):
        x[i, :b["x"].shape[0]] = b["x"]
        mask[i, :b["x"].shape[0]] = True
    return {
        "x": x, "mask": mask,
        "presence": torch.stack([b["presence"] for b in batch]),
        "intensity": torch.stack([b["intensity"] for b in batch]),
        "bag_id": [b["bag_id"] for b in batch],
        "hour": torch.tensor([b["hour"] for b in batch]),
        "date": [b["date"] for b in batch],
    }


def make_loaders(cfg: DataConfig, batch_size: int = 32, seed: int = 0,
                 num_workers: int = 4):
    from torch.utils.data import DataLoader

    train = BagEmbeddingDataset(cfg, "train", train=True, seed=seed)
    sets = {"train": train}
    for split in ("val", "test"):
        sets[split] = BagEmbeddingDataset(cfg, split, stats=train.stats, seed=seed)
    return {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                         collate_fn=collate, num_workers=num_workers,
                         drop_last=False)
        for name, ds in sets.items()
    }, sets
