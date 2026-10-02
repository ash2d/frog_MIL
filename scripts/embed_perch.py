"""Precompute frozen Perch v2 embeddings for every instance in the manifest.

Runs in the TensorFlow environment, not the training one:

    UV_PROJECT_ENVIRONMENT=.venv-perch uv sync --only-group perch
    .venv-perch/bin/python scripts/embed_perch.py --out-dir embeddings

Perch v2 wants 32 kHz mono in 5.0 s windows and returns 1536-d embeddings, so
``manifest.py`` must have been run with ``--window-s 5.0``. Audio here is
44.1 kHz stereo; 44100 -> 32000 is exactly 320/441, so ``resample_poly`` is
exact rather than an interpolation fudge.

Output (``--out-dir``):
    embeddings.f16.npy   [n_instances, 1536] memmap, row i = row i of instances.csv
    logits.f16.npy       [n_instances, n_logit_columns] zero-shot logits, same rows
    done.npy             [n_instances] bool, rows already embedded
    index.csv            instance_id, bag_id, split, win_idx  (row order)
    meta.json            model name, dim, dtype, window, source manifest, progress

Incremental: the cache is keyed by ``instance_id``, not by row position. After
new audio lands, rebuild the manifest and rerun this script: rows already in the
cache are carried over into the new manifest order, rows no longer in the
manifest are dropped, and only the missing rows are embedded. Nothing is
embedded twice, wherever the new clips sort. An interrupted run resumes the same
way, from ``done.npy``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

PERCH_SR = 32_000
PERCH_WINDOW_S = 5.0
PERCH_DIM = 1536


def load_perch(name: str = "perch_v2"):
    from perch_hoplite.zoo import model_configs

    print(f"loading {name} ...", flush=True)
    model = model_configs.load_model_by_name(name)
    print("loaded", flush=True)
    return model


def read_clip_32k_mono(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != PERCH_SR:
        from math import gcd
        g = gcd(sr, PERCH_SR)
        x = resample_poly(x, PERCH_SR // g, sr // g).astype(np.float32)
    return x


def cut_windows(audio: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Exact 5 s windows at the manifest's offsets, zero-padded at the tail."""
    n = int(round(PERCH_WINDOW_S * PERCH_SR))
    out = np.zeros((len(rows), n), dtype=np.float32)
    for i, r in enumerate(rows):
        a = int(round(float(r["clip_start_s"]) * PERCH_SR))
        seg = audio[a:a + n]
        out[i, :len(seg)] = seg
    return out


def _flatten(a: np.ndarray) -> np.ndarray:
    """[B, time, channels, D] -> [B, D] by averaging the spatial axes."""
    a = np.asarray(a, dtype=np.float32)
    return a.reshape(a.shape[0], -1, a.shape[-1]).mean(axis=1) if a.ndim > 2 else a


def embed_batch(model, batch: np.ndarray, want_logits: bool):
    """Return (embeddings [B, 1536], logits [B, n_classes] or None).

    Keep the batch shape constant across calls: TF retraces per input shape,
    and retracing drops throughput from ~630 win/s to ~2.
    """
    if hasattr(model, "batch_embed"):
        out = model.batch_embed(batch)
        emb, lg = out.embeddings, (out.logits or {}).get("label")
    else:
        outs = [model.embed(w) for w in batch]
        emb = np.stack([o.embeddings for o in outs])
        lg = (np.stack([o.logits["label"] for o in outs])
              if outs[0].logits and "label" in outs[0].logits else None)
    return _flatten(emb), (_flatten(lg) if want_logits and lg is not None else None)


def find_label_columns(patterns: list[str]) -> tuple[list[int], list[str]]:
    """Locate Perch classes whose names match any pattern, from the model assets.

    perch_hoplite refuses to build ``model.class_list`` for perch_v2 (the
    shipped class list has duplicate entries), so read the asset directly.
    Perch v2 knows 8 congeneric *Gastrotheca*, *Oreobates quixensis* and a
    generic *Frog* class -- enough for a training-free baseline. Patterns are
    case-sensitive and anchored: "oreobates" alone also matches the bee-eater
    *Merops oreobates*.
    """
    import re
    from pathlib import Path as P

    cands = sorted(P.home().glob(
        ".cache/kagglehub/models/google/*/tensorFlow2/perch_v2/*/assets/labels.csv"))
    if not cands:
        return [], []
    names = cands[-1].read_text().splitlines()[1:]   # line 1 is a header
    rx = re.compile("|".join(patterns))
    hits = [(i, n) for i, n in enumerate(names) if rx.search(n)]
    return [i for i, _ in hits], [n for _, n in hits]


def load_cache(out_dir: Path):
    """(instance_id -> row, done mask, meta) of an existing cache, or None."""
    need = [out_dir / f for f in ("embeddings.f16.npy", "index.csv", "meta.json")]
    if not all(f.exists() for f in need):
        return None
    meta = json.loads((out_dir / "meta.json").read_text())
    with (out_dir / "index.csv").open() as fh:
        ids = [r["instance_id"] for r in csv.DictReader(fh)]
    if (out_dir / "done.npy").exists():
        done = np.load(out_dir / "done.npy")
    elif meta.get("n_clips_done") == meta.get("n_clips_total"):
        done = np.ones(len(ids), bool)      # complete cache from before done.npy existed
    else:
        sys.exit(f"{out_dir} is an incomplete cache without done.npy; "
                 f"embed into a fresh --out-dir")
    return {i: k for k, i in enumerate(ids)}, done, meta


def write_index(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["row", "instance_id", "bag_id", "split", "win_idx"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"row": i, "instance_id": r["instance_id"], "bag_id": r["bag_id"],
                        "split": r["split"], "win_idx": r["win_idx"]})


def rekey_cache(out_dir: Path, rows: list[dict], old, n_logits: int) -> None:
    """Rebuild the cache in manifest order, carrying over every embedded row.

    Built in a sibling directory and swapped in, so an interruption never leaves
    embeddings and index out of step.
    """
    pos, old_done, meta = old
    src = np.array([pos.get(r["instance_id"], -1) for r in rows])
    keep = np.flatnonzero(src >= 0)
    keep = keep[old_done[src[keep]]]
    tmp = out_dir.with_name(out_dir.name + ".rekey")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    old_emb = np.load(out_dir / "embeddings.f16.npy", mmap_mode="r")
    emb = np.lib.format.open_memmap(tmp / "embeddings.f16.npy", mode="w+", dtype=np.float16,
                                    shape=(len(rows), PERCH_DIM))
    emb[keep] = old_emb[src[keep]]
    emb.flush()
    del emb, old_emb
    if n_logits:
        old_lg = np.load(out_dir / "logits.f16.npy", mmap_mode="r")
        lg = np.lib.format.open_memmap(tmp / "logits.f16.npy", mode="w+", dtype=np.float16,
                                       shape=(len(rows), n_logits))
        lg[keep] = old_lg[src[keep]]
        lg.flush()
        del lg, old_lg
    done = np.zeros(len(rows), bool)
    done[keep] = True
    np.save(tmp / "done.npy", done)
    write_index(tmp / "index.csv", rows)
    meta = {k: meta[k] for k in ("model", "logit_columns", "logit_names") if k in meta}
    meta.update(n_instances=len(rows), n_done=int(done.sum()))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    stale = len(pos) - len(keep)
    print(f"re-keyed cache: kept {len(keep)} embedded rows, dropped {stale} rows no longer "
          f"in the manifest, {len(rows) - len(keep)} rows to embed")
    # Memmaps are closed above: on NFS an open file can't be removed, only renamed.
    old_dir = out_dir.with_name(out_dir.name + ".old")
    out_dir.rename(old_dir)
    tmp.rename(out_dir)
    shutil.rmtree(old_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, default=Path("outputs/instances.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    ap.add_argument("--model", default="perch_v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-clips", type=int, default=0,
                    help="embed only the first N clips (smoke test)")
    ap.add_argument("--list-labels", action="store_true",
                    help="print Perch label-set matches for our species and exit")
    ap.add_argument("--save-logits", action="store_true", default=True,
                    help="also cache Perch logits for the classes matching "
                         "--logit-patterns, for the zero-shot baseline")
    ap.add_argument("--no-save-logits", dest="save_logits", action="store_false")
    ap.add_argument("--logit-patterns", nargs="+",
                    default=[r"^Gastrotheca ", r"^Oreobates ", r"^Frog$"],
                    help="regexes over Perch class names; anchor on the genus, "
                         "since epithets are reused across unrelated taxa")
    args = ap.parse_args()

    with args.instances.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"no instances in {args.instances}")
    ids = [r["instance_id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit(f"duplicate instance_id in {args.instances}")
    starts = sorted({float(r["clip_start_s"]) for r in rows if r["filepath"] == rows[0]["filepath"]})
    hop = starts[1] - starts[0] if len(starts) > 1 else PERCH_WINDOW_S
    if abs(hop - PERCH_WINDOW_S) > 1e-6:
        sys.exit(f"{args.instances} has a {hop:g} s hop; the cache holds contiguous "
                 f"{PERCH_WINDOW_S:g} s windows only. Rebuild it with frog-manifest.")

    if args.list_labels:
        load_perch(args.model)
        cols, names = find_label_columns(args.logit_patterns)
        print(f"{len(names)} matches for {args.logit_patterns}:")
        for c, n in zip(cols, names):
            print(f"  [{c}] {n}")
        return

    model = None
    logit_cols, logit_names = (find_label_columns(args.logit_patterns)
                               if args.save_logits else ([], []))
    if args.save_logits and not logit_cols:     # label asset arrives with the model download
        model = load_perch(args.model)
        logit_cols, logit_names = find_label_columns(args.logit_patterns)
    emb_path, meta_path = args.out_dir / "embeddings.f16.npy", args.out_dir / "meta.json"
    old = load_cache(args.out_dir)
    if old is not None:
        meta = old[2]
        if meta.get("model") != args.model:
            sys.exit(f"{args.out_dir} holds {meta.get('model')} embeddings, not {args.model}")
        if meta.get("logit_columns", []) != logit_cols:
            sys.exit(f"{args.out_dir} caches logit columns {meta.get('logit_columns')}, "
                     f"not {logit_cols}; embed into a fresh --out-dir")
        if list(old[0]) != ids:
            rekey_cache(args.out_dir, rows, old, len(logit_cols))
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                  shape=(len(rows), PERCH_DIM)).flush()
        if logit_cols:
            np.lib.format.open_memmap(args.out_dir / "logits.f16.npy", mode="w+",
                                      dtype=np.float16,
                                      shape=(len(rows), len(logit_cols))).flush()
        np.save(args.out_dir / "done.npy", np.zeros(len(rows), bool))
        write_index(args.out_dir / "index.csv", rows)

    store = np.load(emb_path, mmap_mode="r+")
    logit_store = (np.load(args.out_dir / "logits.f16.npy", mmap_mode="r+")
                   if logit_cols else None)
    done = (np.load(args.out_dir / "done.npy") if (args.out_dir / "done.npy").exists()
            else np.ones(len(rows), bool))
    if logit_cols:
        print(f"caching {len(logit_cols)} zero-shot logit columns: "
              f"{', '.join(logit_names[:4])}...")

    by_clip: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_clip[r["filepath"]].append(i)
    todo = [c for c, idx in by_clip.items() if not done[idx].all()]
    if args.limit_clips:
        todo = todo[:args.limit_clips]
    print(f"{len(rows)} instances over {len(by_clip)} clips; {int(done.sum())} rows cached, "
          f"{len(todo)} clips to embed")

    def checkpoint() -> None:
        store.flush()
        if logit_store is not None:
            logit_store.flush()
        np.save(args.out_dir / "done.npy", done)
        meta_path.write_text(json.dumps({
            "model": args.model, "dim": PERCH_DIM, "dtype": "float16",
            "sample_rate": PERCH_SR, "window_s": PERCH_WINDOW_S, "hop_s": PERCH_WINDOW_S,
            "instances_csv": str(args.instances), "n_instances": len(rows),
            "n_done": int(done.sum()), "n_clips_total": len(by_clip),
            "logit_columns": logit_cols, "logit_names": logit_names,
        }, indent=2))

    if not todo:
        checkpoint()
        print("cache is complete")
        return
    model = model or load_perch(args.model)
    try:
        for n_done, clip in enumerate(tqdm(todo, unit="clip"), 1):
            idx = sorted((i for i in by_clip[clip] if not done[i]),
                         key=lambda i: int(rows[i]["win_idx"]))
            wins = cut_windows(read_clip_32k_mono(clip), [rows[i] for i in idx])
            parts = [embed_batch(model, wins[s:s + args.batch_size], bool(logit_cols))
                     for s in range(0, len(wins), args.batch_size)]
            if logit_cols and parts[0][1] is None:
                sys.exit(f"{args.model} returned no logits; rerun with --no-save-logits")
            store[idx] = np.concatenate([e for e, _ in parts]).astype(np.float16)
            if logit_store is not None:
                lg = np.concatenate([lg_ for _, lg_ in parts])
                logit_store[idx] = lg[:, logit_cols].astype(np.float16)
            done[idx] = True
            # Checkpoint so a hard kill costs minutes, not the whole run.
            if n_done % 200 == 0:
                checkpoint()
    finally:
        checkpoint()
        print(f"\n{int(done.sum())}/{len(rows)} rows embedded -> {emb_path}")


if __name__ == "__main__":
    main()
