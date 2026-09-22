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
    index.csv            instance_id, bag_id, split, win_idx  (row order)
    meta.json            model name, dim, dtype, source manifest, progress

Resumable: completed clips are recorded in meta.json, so re-running after an
interruption -- or after the rsync delivers more audio -- only does what is
left. Rebuild the manifest first and the new rows are appended.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
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

    model = load_perch(args.model)

    if args.list_labels:
        cols, names = find_label_columns(args.logit_patterns)
        print(f"{len(names)} matches for {args.logit_patterns}:")
        for c, n in zip(cols, names):
            print(f"  [{c}] {n}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.out_dir / "embeddings.f16.npy"
    meta_path = args.out_dir / "meta.json"

    # Row i of the memmap is row i of instances.csv, so the manifest is the index.
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    done = set(meta.get("done_clips", []))
    done_logits = set(meta.get("done_logit_clips", []))
    if meta.get("n_instances") not in (None, len(rows)):
        print(f"manifest grew {meta['n_instances']} -> {len(rows)} rows; extending")
    mode = "r+" if emb_path.exists() and meta.get("n_instances") == len(rows) else "w+"
    if mode == "w+" and emb_path.exists():
        old = np.load(emb_path, mmap_mode="r")
        store = np.lib.format.open_memmap(emb_path.with_suffix(".tmp.npy"), mode="w+",
                                          dtype=np.float16, shape=(len(rows), PERCH_DIM))
        store[:len(old)] = old[:min(len(old), len(rows))]
        del old, store
        emb_path.with_suffix(".tmp.npy").replace(emb_path)
        mode = "r+"
    store = (np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                       shape=(len(rows), PERCH_DIM))
             if mode == "w+" else np.load(emb_path, mmap_mode="r+"))

    with (args.out_dir / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["row", "instance_id", "bag_id", "split", "win_idx"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"row": i, "instance_id": r["instance_id"], "bag_id": r["bag_id"],
                        "split": r["split"], "win_idx": r["win_idx"]})

    logit_cols, logit_names = (find_label_columns(args.logit_patterns)
                               if args.save_logits else ([], []))
    logit_store = None
    if logit_cols:
        print(f"caching {len(logit_cols)} zero-shot logit columns: "
              f"{', '.join(logit_names[:4])}...")
        lp = args.out_dir / "logits.f16.npy"
        logit_store = (np.load(lp, mmap_mode="r+")
                       if lp.exists() and np.load(lp, mmap_mode="r").shape[0] == len(rows)
                       else np.lib.format.open_memmap(
                           lp, mode="w+", dtype=np.float16,
                           shape=(len(rows), len(logit_cols))))

    by_clip: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_clip[r["filepath"]].append((i, r))
    # A clip embedded without logits is not done when logits are wanted --
    # otherwise the logit memmap silently keeps zeros for those rows.
    todo = [c for c in by_clip
            if c not in done or (logit_cols and c not in done_logits)]
    if logit_cols and len(done - done_logits):
        print(f"{len(done - done_logits)} clips have embeddings but no logits; redoing")
    if args.limit_clips:
        todo = todo[:args.limit_clips]
    print(f"{len(rows)} instances over {len(by_clip)} clips; {len(todo)} clips to do")

    def write_meta() -> None:
        meta_path.write_text(json.dumps({
            "model": args.model, "dim": PERCH_DIM, "dtype": "float16",
            "sample_rate": PERCH_SR, "window_s": PERCH_WINDOW_S,
            "instances_csv": str(args.instances), "n_instances": len(rows),
            "done_clips": sorted(done), "n_clips_done": len(done),
            "n_clips_total": len(by_clip),
            "logit_columns": logit_cols, "logit_names": logit_names,
            "done_logit_clips": sorted(done_logits),
        }, indent=2))

    try:
        for n_done, clip in enumerate(tqdm(todo, unit="clip"), 1):
            items = sorted(by_clip[clip], key=lambda t: int(t[1]["win_idx"]))
            idx = np.array([i for i, _ in items])
            wins = cut_windows(read_clip_32k_mono(clip), [r for _, r in items])
            parts = [embed_batch(model, wins[s:s + args.batch_size], bool(logit_cols))
                     for s in range(0, len(wins), args.batch_size)]
            store[idx] = np.concatenate([e for e, _ in parts]).astype(np.float16)
            if logit_store is not None and parts[0][1] is not None:
                lg = np.concatenate([lg_ for _, lg_ in parts])
                logit_store[idx] = lg[:, logit_cols].astype(np.float16)
                done_logits.add(clip)
            done.add(clip)
            # Checkpoint so a hard kill costs minutes, not the whole run.
            if n_done % 200 == 0:
                store.flush()
                write_meta()
    finally:
        store.flush()
        if logit_store is not None:
            logit_store.flush()
        write_meta()
        print(f"\n{len(done)}/{len(by_clip)} clips embedded -> {emb_path}")


if __name__ == "__main__":
    main()
