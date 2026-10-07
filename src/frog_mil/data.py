"""Bags of frozen Perch v2 embeddings, held as one padded tensor.

The whole dataset is small (~5k bags x <= 36 windows x 1536 dims, ~0.5 GB in
float16), so it is loaded once, onto the GPU if there is one, and every fold,
seed and pooler trains from it without a DataLoader.

A bag is every cached row of its hour, in manifest order: 12 windows per 1 min
clip, so 12, 24 or 36 per bag depending on the recording regime. Padded
windows are zero and masked; every pooler must ignore them.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import embcache
from .config import BAGS_CSV, EMB_DIR, MAX_INDEX, SPECIES, SUMMARY_JSON, manifest_summary

ROLES = ("train", "val", "test")


def fold_roles(fold: np.ndarray, f: int, k: int) -> dict[str, np.ndarray]:
    """Bag masks for model ``f``: test = fold f, val = fold (f+1) mod k, train = rest."""
    return {"test": fold == f, "val": fold == (f + 1) % k,
            "train": (fold != f) & (fold != (f + 1) % k)}


@dataclass
class BagData:
    x: torch.Tensor              # [n_bags, max_n, D] float16, zero-padded
    mask: torch.Tensor           # [n_bags, max_n] bool
    presence: torch.Tensor       # [n_bags, C] float
    intensity: torch.Tensor      # [n_bags, C] long, 0-3
    rows: np.ndarray             # [n_bags, max_n] cache row of each window, -1 = pad
    bags: list[dict]             # bags.csv rows, in bag_ids order
    bag_ids: np.ndarray
    fold: np.ndarray
    n_folds: int
    dataset_id: str
    emb_dir: Path

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.x.device

    def column(self, name: str) -> np.ndarray:
        return np.array([b[name] for b in self.bags])

    def roles(self, f: int) -> dict[str, np.ndarray]:
        return fold_roles(self.fold, f, self.n_folds)

    def y(self) -> np.ndarray:
        return self.presence.cpu().numpy().astype(int)

    def idx(self) -> np.ndarray:
        return self.intensity.cpu().numpy()


def load_bags(emb_dir: Path = EMB_DIR, bags_csv: Path = BAGS_CSV,
              summary_json: Path = SUMMARY_JSON, device: str | torch.device = "cpu",
              dataset_id: str | None = None) -> BagData:
    """Every bag of the manifest, checked against the embedding cache.

    ``dataset_id``, when given, must match the manifest's (a run must be scored
    on the data it was trained on).
    """
    summary = manifest_summary(summary_json)
    if dataset_id is not None and summary["dataset_id"] != dataset_id:
        raise RuntimeError(f"the manifest is dataset {summary['dataset_id']}, not "
                           f"{dataset_id}; rebuild it or use that dataset's archive")
    meta = embcache.check_matches(emb_dir, summary["instances_id"])

    with bags_csv.open() as fh:
        bags = sorted(csv.DictReader(fh), key=lambda r: r["bag_id"])
    rows_by_bag: dict[str, list[int]] = defaultdict(list)
    with (emb_dir / "index.csv").open() as fh:
        for r in csv.DictReader(fh):
            rows_by_bag[r["bag_id"]].append(int(r["row"]))
    missing = [b["bag_id"] for b in bags if not rows_by_bag[b["bag_id"]]]
    if missing:
        raise RuntimeError(f"{len(missing)} bags of {bags_csv} have no cached windows "
                           f"(e.g. {missing[0]}); run `frog run embed`")

    n = max(len(v) for v in rows_by_bag.values())
    rows = np.full((len(bags), n), -1, np.int64)
    for i, b in enumerate(bags):
        r = sorted(rows_by_bag[b["bag_id"]])
        rows[i, :len(r)] = r
    mask = rows >= 0
    emb = np.load(emb_dir / "embeddings.f16.npy", mmap_mode="r")
    x = np.zeros((*rows.shape, meta["dim"]), np.float16)
    x[mask] = np.asarray(emb)[rows[mask]]

    def cols(fmt, dtype):
        return torch.tensor([[int(b[fmt.format(s)]) for s in SPECIES] for b in bags],
                            dtype=dtype)

    return BagData(
        x=torch.from_numpy(x).to(device), mask=torch.from_numpy(mask).to(device),
        presence=cols("{}_present", torch.float32).to(device),
        intensity=cols("{}_index", torch.long).to(device),
        rows=rows, bags=bags, bag_ids=np.array([b["bag_id"] for b in bags]),
        fold=np.array([int(b["fold"]) for b in bags]), n_folds=int(summary["folds"]),
        dataset_id=summary["dataset_id"], emb_dir=emb_dir)


class FoldView:
    """One fold's train/val/test bags, z-scored with that fold's train windows."""

    def __init__(self, data: BagData, f: int, standardize: bool = True):
        self.data, self.f = data, f
        self.idx = {k: torch.from_numpy(np.flatnonzero(v)).to(data.device)
                    for k, v in data.roles(f).items()}
        if standardize:
            tr = self.idx["train"]
            w = data.x[tr][data.mask[tr]].float()          # [n_windows, D]
            self.mean, self.std = w.mean(0), w.std(0) + 1e-6
        else:
            self.mean = torch.zeros(data.dim, device=data.device)
            self.std = torch.ones(data.dim, device=data.device)

    def batch(self, idx: torch.Tensor) -> dict:
        d = self.data
        m = d.mask[idx]
        x = (d.x[idx].float() - self.mean) / self.std * m.unsqueeze(-1)
        return {"x": x, "mask": m, "presence": d.presence[idx],
                "intensity": d.intensity[idx], "idx": idx}

    def batches(self, role: str, batch_size: int, gen: torch.Generator | None = None):
        """Shuffled when ``gen`` is given, else in bag order."""
        idx = self.idx[role]
        if gen is not None:
            idx = idx[torch.randperm(len(idx), generator=gen).to(idx.device)]
        for s in range(0, len(idx), batch_size):
            yield self.batch(idx[s:s + batch_size])

    def pos_weight(self) -> torch.Tensor:
        """neg/pos per species on train bags. Gastrotheca is a few % of hours, so
        without this the probe learns the constant-negative solution."""
        y = self.data.presence[self.idx["train"]]
        pos = y.sum(0)
        return (len(y) - pos) / pos.clamp(min=1)

    def cum_pos_weight(self) -> torch.Tensor:
        """[C, 3] neg/pos for each threshold y>=1, y>=2, y>=3 on train bags.

        Positives thin out fast up the scale, so each threshold gets its own
        weight rather than inheriting the presence one.
        """
        it = self.data.intensity[self.idx["train"]]
        out = []
        for k in range(1, MAX_INDEX + 1):
            pos = (it >= k).sum(0).float()
            out.append((len(it) - pos) / pos.clamp(min=1))
        return torch.stack(out, -1)
