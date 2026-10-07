"""The on-disk embedding cache, keyed by ``instance_id`` (numpy only).

Layout of ``config.EMB_DIR``:
    embeddings.f16.npy   [n_instances, 1536] memmap, row i = row i of instances.csv
    logits.f16.npy       [n_instances, n_logit_columns] zero-shot logits, same rows
    done.npy             [n_instances] bool, rows already embedded
    index.csv            row, instance_id, bag_id, win_idx, file_sig  (row order)
    meta.json            model identity, instances_id, logit columns, progress

``file_sig`` is ``size:mtime_ns`` of the source WAV when the row was embedded. A
file that is replaced under the same name gets a new signature, so its rows are
re-embedded instead of silently reused.

``scripts/embed_perch.py`` (TF env) fills the cache; ``data.py`` (torch env)
reads it and checks ``meta["instances_id"]`` against the manifest.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

DIM = 1536
INDEX_COLS = ["row", "instance_id", "bag_id", "win_idx", "file_sig"]


def file_sig(path: str | Path) -> str:
    st = os.stat(path)
    return f"{st.st_size}:{st.st_mtime_ns}"


def save_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_index(path: Path, rows: list[dict], sigs: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, INDEX_COLS)
        w.writeheader()
        for i, (r, s) in enumerate(zip(rows, sigs)):
            w.writerow({"row": i, "instance_id": r["instance_id"], "bag_id": r["bag_id"],
                        "win_idx": r["win_idx"], "file_sig": s})
    os.replace(tmp, path)


def read_meta(emb_dir: Path) -> dict | None:
    p = emb_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_cache(emb_dir: Path):
    """(instance_id -> row, file_sig per row or None, done mask, meta), or None."""
    need = [emb_dir / f for f in ("embeddings.f16.npy", "index.csv", "meta.json")]
    if not all(f.exists() for f in need):
        return None
    meta = read_meta(emb_dir)
    with (emb_dir / "index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["instance_id"] for r in rows]
    sigs = [r["file_sig"] for r in rows] if rows and "file_sig" in rows[0] else None
    if (emb_dir / "done.npy").exists():
        done = np.load(emb_dir / "done.npy")
    elif meta.get("n_clips_done") == meta.get("n_clips_total"):
        done = np.ones(len(ids), bool)      # complete cache from before done.npy existed
    else:
        raise SystemExit(f"{emb_dir} is an incomplete cache without done.npy; "
                         f"embed into a fresh --out-dir")
    return {i: k for k, i in enumerate(ids)}, sigs, done, meta


def rekey(emb_dir: Path, rows: list[dict], cur_sigs: list[str], old, n_logits: int,
          meta_extra: dict) -> np.ndarray:
    """Rebuild the cache in manifest order, carrying over every still-valid row.

    A row is carried over if its instance_id is cached, embedded, and its source
    file signature is unchanged (caches from before signatures existed are
    trusted and stamped with the current signature). Built in a sibling
    directory and swapped in, so an interruption never leaves embeddings and
    index out of step. Returns the new ``done`` mask.
    """
    pos, old_sigs, old_done, meta = old
    src = np.array([pos.get(r["instance_id"], -1) for r in rows])
    keep = np.flatnonzero(src >= 0)
    keep = keep[old_done[src[keep]]]
    changed = 0
    if old_sigs is not None:
        same = np.array([old_sigs[src[i]] == cur_sigs[i] for i in keep], bool)
        changed = int((~same).sum())
        keep = keep[same]
    tmp = emb_dir.with_name(emb_dir.name + ".rekey")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    old_emb = np.load(emb_dir / "embeddings.f16.npy", mmap_mode="r")
    emb = np.lib.format.open_memmap(tmp / "embeddings.f16.npy", mode="w+", dtype=np.float16,
                                    shape=(len(rows), DIM))
    emb[keep] = old_emb[src[keep]]
    emb.flush()
    del emb, old_emb
    if n_logits:
        old_lg = np.load(emb_dir / "logits.f16.npy", mmap_mode="r")
        lg = np.lib.format.open_memmap(tmp / "logits.f16.npy", mode="w+", dtype=np.float16,
                                       shape=(len(rows), n_logits))
        lg[keep] = old_lg[src[keep]]
        lg.flush()
        del lg, old_lg
    done = np.zeros(len(rows), bool)
    done[keep] = True
    save_atomic(tmp / "done.npy", done)
    write_index(tmp / "index.csv", rows, cur_sigs)
    meta = {k: meta[k] for k in ("model", "model_identity", "logit_columns", "logit_names")
            if k in meta}
    meta.update(meta_extra, n_instances=len(rows), n_done=int(done.sum()))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    stale = len(pos) - len(keep) - changed
    print(f"re-keyed cache: kept {len(keep)} rows, {changed} rows from changed files "
          f"to re-embed, dropped {max(stale, 0)} rows no longer in the manifest, "
          f"{len(rows) - len(keep)} rows to embed")
    # Memmaps are closed above: on NFS an open file can't be removed, only renamed.
    old_dir = emb_dir.with_name(emb_dir.name + ".old")
    shutil.rmtree(old_dir, ignore_errors=True)
    emb_dir.rename(old_dir)
    tmp.rename(emb_dir)
    shutil.rmtree(old_dir)
    return done


def check_matches(emb_dir: Path, instances_id: str) -> dict:
    """Meta of a complete cache that matches the manifest, else a clear error."""
    meta = read_meta(emb_dir)
    if meta is None:
        raise FileNotFoundError(
            f"no embedding cache at {emb_dir}. Build it with `frog run embed` or\n"
            f"  scripts/perch_env.sh python scripts/embed_perch.py")
    if meta.get("hop_s") != meta.get("window_s"):
        raise RuntimeError(f"{emb_dir} holds overlapping windows (an old 2.5 s-hop cache); "
                           f"rerun scripts/embed_perch.py")
    if meta.get("instances_id") != instances_id:
        raise RuntimeError(
            f"{emb_dir} was built for instances {meta.get('instances_id')}, the manifest "
            f"has {instances_id}. Run `frog run embed` (it re-keys and embeds only new rows).")
    # Unembedded rows are zeros, not missing, so a half-finished cache trains
    # silently and scores like noise. Fail loudly instead.
    done = np.load(emb_dir / "done.npy")
    if not done.all():
        raise RuntimeError(f"embedding cache is incomplete ({int(done.sum())}/{len(done)} "
                           f"windows); finish it with `frog run embed`")
    return meta
