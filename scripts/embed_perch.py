"""Precompute frozen Perch v2 embeddings for every instance in the manifest.

Runs in the TensorFlow environment, through the CUDA wrapper:

    scripts/perch_env.sh python scripts/embed_perch.py      # or: frog run embed

Perch v2 wants 32 kHz mono in 5.0 s windows and returns 1536-d embeddings.
Audio is resampled with ``resample_poly`` (44.1 kHz -> 32 kHz is exactly
320/441; 8 kHz -> 32 kHz is x4, so those windows hold nothing above 4 kHz).

The cache lives in ``config.EMB_DIR`` (group workspace); its layout is described
in ``frog_mil.embcache``. It is keyed by ``instance_id`` plus the source file's
signature: after the manifest changes, rows already embedded are carried over
into the new manifest order, rows no longer in it are dropped, and only missing
rows (or rows of files replaced since) are embedded. An interrupted run resumes
from ``done.npy``.

``--band-limit HZ`` first resamples any audio recorded above HZ down to HZ, then
on to 32 kHz as usual, so 44.1 kHz audio is processed exactly like the 8 kHz
recordings (``--band-limit 8000``: nothing above 4 kHz). It is recorded in the
cache's meta, and a cache never mixes band limits.

The exact model is pinned and recorded. ``perch_v2`` resolves to a *different*
Kaggle model (``perch_v2_cpu``) when TF sees no GPU, so the script refuses to
run without one unless ``--allow-cpu``, and refuses to add rows from a model
other than the one the cache was built with.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frog_mil import embcache  # noqa: E402
from frog_mil.config import EMB_DIR, INSTANCES_CSV, SUMMARY_JSON  # noqa: E402
from frog_mil.manifest import hash_instances  # noqa: E402

PERCH_SR = 32_000
PERCH_WINDOW_S = 5.0
PERCH_DIM = embcache.DIM


def model_identity(name: str, allow_cpu: bool) -> tuple[dict, Path]:
    """Which Kaggle model ``name`` resolves to on this machine, without loading it."""
    from perch_hoplite.zoo import kaggle_hub, model_configs

    preset = model_configs.get_preset_model_config(name)
    cfg = preset.model_config
    if "cpu" in cfg.tfhub_path and name == "perch_v2" and not allow_cpu:
        sys.exit("TF sees no GPU, so perch_v2 would load the different perch_v2_cpu model. "
                 "Run through scripts/perch_env.sh on a GPU host (or pass --allow-cpu "
                 "into a fresh --out-dir).")
    path = Path(kaggle_hub.resolve(cfg.tfhub_path, cfg.tfhub_version))
    sha = hashlib.sha256((path / "saved_model.pb").read_bytes()).hexdigest()[:16]
    return {"name": name, "kaggle": f"{cfg.tfhub_path}/{cfg.tfhub_version}",
            "saved_model_sha256": sha}, path


def load_perch(name: str):
    from perch_hoplite.zoo import model_configs

    print(f"loading {name} ...", flush=True)
    model = model_configs.load_model_by_name(name)
    print("loaded", flush=True)
    return model


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def read_clip_32k_mono(path: str, band_limit: int = 0) -> np.ndarray:
    """Mono audio at 32 kHz; first resampled to ``band_limit`` Hz if recorded above it."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if band_limit and sr > band_limit:
        x, sr = _resample(x, sr, band_limit), band_limit
    if sr != PERCH_SR:
        x = _resample(x, sr, PERCH_SR)
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


def find_label_columns(model_path: Path, patterns: list[str]) -> tuple[list[int], list[str]]:
    """Perch classes whose names match any pattern, read from *this* model's assets.

    perch_hoplite refuses to build ``model.class_list`` for perch_v2 (the
    shipped class list has duplicate entries), so read the asset directly.
    Patterns are case-sensitive and anchored on the genus: "oreobates" alone
    also matches the bee-eater *Merops oreobates*.
    """
    import re

    names = (model_path / "assets" / "labels.csv").read_text().splitlines()[1:]  # header
    rx = re.compile("|".join(patterns))
    hits = [(i, n) for i, n in enumerate(names) if rx.search(n)]
    return [i for i, _ in hits], [n for _, n in hits]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, default=INSTANCES_CSV)
    ap.add_argument("--out-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--model", default="perch_v2")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--band-limit", type=int, default=0, metavar="HZ",
                    help="resample audio recorded above HZ down to HZ first "
                         "(8000 = like the 8 kHz recordings); 0 = off")
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
    starts = sorted({float(r["clip_start_s"]) for r in rows
                     if r["filepath"] == rows[0]["filepath"]})
    hop = starts[1] - starts[0] if len(starts) > 1 else PERCH_WINDOW_S
    if abs(hop - PERCH_WINDOW_S) > 1e-6:
        sys.exit(f"{args.instances} has a {hop:g} s hop; the cache holds contiguous "
                 f"{PERCH_WINDOW_S:g} s windows only. Rebuild it with frog-manifest.")
    instances_id = hash_instances(ids)
    if args.instances == INSTANCES_CSV and SUMMARY_JSON.exists():
        want = json.loads(SUMMARY_JSON.read_text()).get("instances_id")
        if want and want != instances_id:
            sys.exit(f"{args.instances} doesn't match {SUMMARY_JSON}; rerun frog-manifest")

    ident, model_path = model_identity(args.model, args.allow_cpu)
    logit_cols, logit_names = (find_label_columns(model_path, args.logit_patterns)
                               if args.save_logits else ([], []))
    if args.list_labels:
        print(f"{ident['kaggle']}: {len(logit_names)} matches for {args.logit_patterns}:")
        for c, n in zip(logit_cols, logit_names):
            print(f"  [{c}] {n}")
        return

    # Signatures are read once here; a file replaced mid-run is caught next run.
    by_file = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_file[r["filepath"]].append(i)
    sig_of = {f: embcache.file_sig(f) for f in by_file}
    sigs = [sig_of[r["filepath"]] for r in rows]

    out = args.out_dir
    emb_path, meta_path = out / "embeddings.f16.npy", out / "meta.json"
    meta_extra = {"instances_id": instances_id, "dim": PERCH_DIM, "dtype": "float16",
                  "sample_rate": PERCH_SR, "window_s": PERCH_WINDOW_S,
                  "hop_s": PERCH_WINDOW_S, "instances_csv": str(args.instances),
                  "band_limit_hz": args.band_limit or None}
    old = embcache.load_cache(out)
    if old is not None:
        meta = old[3]
        if meta.get("model") != args.model:
            sys.exit(f"{out} holds {meta.get('model')} embeddings, not {args.model}")
        if meta.get("band_limit_hz") != (args.band_limit or None):
            sys.exit(f"{out} holds embeddings band-limited to {meta.get('band_limit_hz')} Hz, "
                     f"not {args.band_limit or None}; embed into a fresh --out-dir")
        if "model_identity" not in meta:
            print(f"note: {out} predates model identities; recording {ident['kaggle']}")
            meta["model_identity"] = ident
        elif meta["model_identity"] != ident:
            sys.exit(f"{out} was embedded with {meta['model_identity']}, this machine "
                     f"resolves {args.model} to {ident}; embed into a fresh --out-dir")
        if meta.get("logit_columns", []) != logit_cols:
            sys.exit(f"{out} caches logit columns {meta.get('logit_columns')}, "
                     f"not {logit_cols}; embed into a fresh --out-dir")
        if list(old[0]) != ids or old[1] != sigs:
            done = embcache.rekey(out, rows, sigs, old, len(logit_cols), meta_extra)
        else:
            done = old[2]
    else:
        out.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                  shape=(len(rows), PERCH_DIM)).flush()
        if logit_cols:
            np.lib.format.open_memmap(out / "logits.f16.npy", mode="w+", dtype=np.float16,
                                      shape=(len(rows), len(logit_cols))).flush()
        done = np.zeros(len(rows), bool)
        embcache.save_atomic(out / "done.npy", done)
        embcache.write_index(out / "index.csv", rows, sigs)

    store = np.load(emb_path, mmap_mode="r+")
    logit_store = (np.load(out / "logits.f16.npy", mmap_mode="r+") if logit_cols else None)
    if logit_cols:
        print(f"caching {len(logit_cols)} zero-shot logit columns: "
              f"{', '.join(logit_names[:4])}...")

    todo = [c for c, idx in by_file.items() if not done[idx].all()]
    if args.limit_clips:
        todo = todo[:args.limit_clips]
    print(f"{len(rows)} instances over {len(by_file)} clips; {int(done.sum())} rows cached, "
          f"{len(todo)} clips to embed")

    def checkpoint() -> None:
        store.flush()
        if logit_store is not None:
            logit_store.flush()
        embcache.save_atomic(out / "done.npy", done)
        embcache.write_text_atomic(meta_path, json.dumps({
            "model": args.model, "model_identity": ident, **meta_extra,
            "n_instances": len(rows), "n_done": int(done.sum()),
            "n_clips_total": len(by_file),
            "logit_columns": logit_cols, "logit_names": logit_names,
        }, indent=2))

    if not todo:
        checkpoint()
        print("cache is complete")
        return
    model = load_perch(args.model)
    # One TF input shape for every call: a clip's windows, padded to the longest clip.
    bs = min(args.batch_size, max(len(v) for v in by_file.values()))
    try:
        for n_done, clip in enumerate(tqdm(todo, unit="clip", mininterval=30), 1):
            idx = sorted((i for i in by_file[clip] if not done[i]),
                         key=lambda i: int(rows[i]["win_idx"]))
            wins = cut_windows(read_clip_32k_mono(clip, args.band_limit),
                               [rows[i] for i in idx])
            parts = []
            for s in range(0, len(wins), bs):
                chunk = wins[s:s + bs]
                n = len(chunk)
                if n < bs:
                    chunk = np.concatenate([chunk, np.zeros((bs - n, chunk.shape[1]),
                                                            np.float32)])
                e, lg = embed_batch(model, chunk, bool(logit_cols))
                parts.append((e[:n], lg[:n] if lg is not None else None))
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
