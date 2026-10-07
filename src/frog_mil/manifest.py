"""Join the hourly frog annotations to the recordings and emit an ML-ready manifest.

Data model
----------
One annotation row covers one clock hour. The recorder writes 1-3 one-minute
clips inside that hour, so the hour is a *bag* and each contiguous 5 s window
inside a clip (0-5 s, 5-10 s, ...) is an *instance*. The hourly label is the
max of the per-clip calling indices: bag_label = max(instance_labels).

Every clip passes ``audio.qc_wav`` first. Results are cached by (name, size,
mtime) in ``outputs/qc_cache.csv``, so only new or changed files are read. An
hour with any failed clip is dropped whole: annotators heard the intact
original, so a partial bag could hold a label without its call.

Each bag gets a recording **regime** (sample rate and nominal clips per hour,
e.g. ``44k-2clip``), and every 3-day block of days goes whole to one of K
cross-validation **folds**, dealt out per regime and balanced on positives.
Fold f is the test fold of model f, fold (f+1) mod K its validation fold, and
the rest train it (``data.fold_roles``).

The manifest content is hashed into a ``dataset_id``. Embeddings, runs, results
and figures all record the id they were built from, and refuse to mix.

Outputs (written to --out-dir):
    bags.csv       one row per annotated hour that has audio
    instances.csv  one row per contiguous window, keyed back to its bag
    summary.json   dataset_id, counts, folds, QC and the parameters used

Usage:
    frog-manifest [--audio-dir <wav dir>] [--folds 5]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

from .audio import QC_FIELDS, qc_wav
from .config import (
    ANNOTATIONS,
    AUDIO_DIR,
    FNAME_RE,
    OUTPUTS,
    SPECIES,
    SPECIES_FULL,
    SPECIES_SHORT,
)

QC_CACHE_COLS = ["filename", "size", "mtime_ns", *QC_FIELDS]


# --------------------------------------------------------------------------- QC + scan
def _qc_job(args):
    path, size, mtime = args
    return {"filename": Path(path).name, "size": size, "mtime_ns": mtime, **qc_wav(path)}


def load_qc_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return {r["filename"]: r for r in csv.DictReader(fh)}


def run_qc(files: list[Path], cache_path: Path, workers: int = 16) -> dict[str, dict]:
    """QC every file, reusing cached results whose (size, mtime) still match."""
    cache = load_qc_cache(cache_path)
    stat = {p.name: p.stat() for p in files}
    todo = [(str(p), stat[p.name].st_size, stat[p.name].st_mtime_ns) for p in files
            if p.name not in cache
            or (int(cache[p.name]["size"]), int(cache[p.name]["mtime_ns"]))
            != (stat[p.name].st_size, stat[p.name].st_mtime_ns)]
    if todo:
        print(f"QC: reading {len(todo)} new or changed files "
              f"({len(files) - len(todo)} cached)",
              flush=True)
        with Pool(workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_qc_job, todo, chunksize=4), 1):
                cache[r["filename"]] = {c: r[c] for c in QC_CACHE_COLS}
                if k % 1000 == 0:
                    print(f"  {k}/{len(todo)}", flush=True)
        names = {p.name for p in files}
        tmp = cache_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, QC_CACHE_COLS)
            w.writeheader()
            w.writerows(v for k, v in sorted(cache.items()) if k in names)
        os.replace(tmp, cache_path)
    return {p.name: cache[p.name] for p in files}


def scan_audio(audio_dir: Path, qc_cache: Path, workers: int = 16):
    """-> (clips by hour, skipped [(name, reason)], clip-level QC rows)."""
    by_hour: dict[dt.datetime, list[dict]] = collections.defaultdict(list)
    bad: list[tuple[str, str]] = []
    files = []
    for p in sorted(audio_dir.glob("*.wav")):
        if FNAME_RE.match(p.name):
            files.append(p)
        else:
            bad.append((p.name, "unparsed filename"))
    qc = run_qc(files, qc_cache, workers)
    for p in files:
        q, m = qc[p.name], FNAME_RE.match(p.name)
        stamp = dt.datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
        by_hour[stamp.replace(minute=0, second=0)].append(
            {"path": str(p), "filename": p.name, "site": m["site"], "start": stamp,
             "offset_in_hour_s": stamp.minute * 60 + stamp.second,
             "sample_rate": int(q["sample_rate"] or 0), "channels": int(q["channels"] or 0),
             "duration_s": float(q["duration_s"] or 0),
             "qc_ok": str(q["qc_ok"]) == "True", "qc_reason": q["qc_reason"],
             "clip_frac": float(q["clip_frac"] or 0)})
    return by_hour, bad


# --------------------------------------------------------------------------- regimes
def sr_label(sr: int) -> str:
    return f"{sr / 1000:g}k".replace("44.1k", "44k")


def assign_regimes(hours: dict[dt.datetime, list[dict]]) -> dict[dt.datetime, str]:
    """Regime = sample rate + the *nominal* clips per hour of that recording period.

    The nominal count is the most common clip count among hours with the same
    sample rate and year, so an hour that lost a clip keeps its period's regime.
    """
    groups = collections.defaultdict(list)
    for h, clips in hours.items():
        srs = {c["sample_rate"] for c in clips}
        sr = srs.pop() if len(srs) == 1 else 0
        groups[(sr, h.year)].append((h, len(clips)))
    out = {}
    for (sr, _), hs in groups.items():
        nominal = collections.Counter(n for _, n in hs).most_common(1)[0][0]
        for h, _ in hs:
            out[h] = f"{sr_label(sr)}-{nominal}clip" if sr else "mixed"
    return out


# --------------------------------------------------------------------------- folds
def make_blocks(dates, block_days: int) -> dict:
    """date -> block index; blocks are ``block_days`` calendar days from the first date."""
    origin = min(dates)
    return {d: (d - origin).days // block_days for d in dates}


def assign_folds(block_stats: dict[int, dict], block_regime: dict[int, str], k: int,
                 seed: int, n_tries: int = 2000) -> dict[int, int]:
    """Deal whole blocks to ``k`` folds, per regime, balanced on positives.

    Adjacent hours share weather, individuals and background, so blocks of days
    go whole to one fold. Within each regime, blocks are shuffled and dealt
    round-robin from a random starting fold, so every regime lands in every
    fold it has blocks for. Of ``n_tries`` deals, the one whose folds come
    closest to 1/k of the bags and of each species' positives (overall and per
    regime) is kept, then polished by swapping same-regime blocks.

    ``block_stats`` maps block -> {"bags": n, "<species>": n_positive, ...}.
    """
    rng = random.Random(seed)
    by_regime = collections.defaultdict(list)
    for b in sorted(block_stats):
        by_regime[block_regime[b]].append(b)
    metrics = sorted({m for v in block_stats.values() for m in v})
    keys = [(None, m) for m in metrics] + [(r, m) for r in by_regime for m in metrics]
    totals = {(r, m): sum(block_stats[b].get(m, 0) for b in block_stats
                          if r is None or block_regime[b] == r) for r, m in keys}

    def score(fold: dict[int, int]) -> float:
        got = collections.defaultdict(float)
        for b, f in fold.items():
            for m, v in block_stats[b].items():
                got[(None, m, f)] += v
                got[(block_regime[b], m, f)] += v
        out = 0.0
        for r, m in keys:
            if not totals[(r, m)]:
                continue
            # positives are the scarce resource, so weight them heavily
            w = (1.0 if m == "bags" else 5.0) * (1.0 if r is None else 0.5)
            out += w * sum(abs(got[(r, m, f)] / totals[(r, m)] - 1 / k) for f in range(k))
        return out

    best, best_score = None, float("inf")
    for _ in range(n_tries):
        fold = {}
        for r in sorted(by_regime):
            bl = by_regime[r][:]
            rng.shuffle(bl)
            start = rng.randrange(k)
            for i, b in enumerate(bl):
                fold[b] = (start + i) % k
        s = score(fold)
        if s < best_score:
            best, best_score = fold, s

    # Polish: swap two same-regime blocks between folds while that helps. Swaps
    # keep each fold's block count per regime, so the round-robin spread holds.
    improved = True
    while improved:
        improved = False
        for bl in by_regime.values():
            for i, a in enumerate(bl):
                for b in bl[i + 1:]:
                    if best[a] == best[b]:
                        continue
                    best[a], best[b] = best[b], best[a]
                    s = score(best)
                    if s < best_score - 1e-12:
                        best_score, improved = s, True
                    else:
                        best[a], best[b] = best[b], best[a]
    return best


# --------------------------------------------------------------------------- windows
def windows(duration_s: float, win: float, drop_partial: bool):
    """Contiguous, non-overlapping windows; the tail is kept if at least half full."""
    t, out, hop = 0.0, [], win
    while t + win <= duration_s + (0 if drop_partial else hop):
        end = min(t + win, duration_s)
        if end - t >= (win if drop_partial else win * 0.5):
            out.append((round(t, 3), round(end, 3)))
        t += hop
    return out


# --------------------------------------------------------------------------- ids
def hash_instances(instance_ids: list[str]) -> str:
    """Identity of the window set; the embedding cache must match it."""
    return hashlib.sha256("\n".join(instance_ids).encode()).hexdigest()[:12]


def hash_dataset(bags_csv: Path, instances_id: str) -> str:
    """Identity of everything a model sees: bags, labels, regimes, folds, windows."""
    h = hashlib.sha256(bags_csv.read_bytes())
    h.update(instances_id.encode())
    return h.hexdigest()[:10]


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ANNOTATIONS)
    ap.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUTPUTS)
    ap.add_argument("--window-s", type=float, default=5.0,
                    help="instance length; 5.0 is Perch v2's native window")
    ap.add_argument("--drop-partial", action="store_true",
                    help="drop the short trailing window instead of zero-padding it")
    ap.add_argument("--block-days", type=int, default=3,
                    help="day-block granularity for the cross-validation folds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16, help="parallel QC readers")
    args = ap.parse_args()

    with args.csv.open() as fh:
        ann = {dt.datetime.fromisoformat(r["DateTime"]): r for r in csv.DictReader(fh)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_hour, bad = scan_audio(args.audio_dir, args.out_dir / "qc_cache.csv", args.workers)
    if not by_hour:
        raise SystemExit(f"no usable audio under {args.audio_dir}")

    qc_dropped = {h: [f"{c['filename']}: {c['qc_reason']}" for c in clips if not c["qc_ok"]]
                  for h, clips in by_hour.items()}
    qc_dropped = {h: v for h, v in qc_dropped.items() if v}
    unlabelled = sorted(h for h in by_hour if h not in ann)
    hours = {h: sorted(by_hour[h], key=lambda c: c["start"]) for h in sorted(by_hour)
             if h in ann and h not in qc_dropped}
    regime = assign_regimes(hours)

    blocks = make_blocks({h.date() for h in hours}, args.block_days)
    block_stats = collections.defaultdict(lambda: collections.defaultdict(int))
    block_regimes = collections.defaultdict(collections.Counter)
    for h in hours:
        b = blocks[h.date()]
        block_stats[b]["bags"] += 1
        block_regimes[b][regime[h]] += 1
        for sp, short in SPECIES_SHORT.items():
            block_stats[b][short] += int(float(ann[h][sp]) > 0)
    block_regime = {b: c.most_common(1)[0][0] for b, c in block_regimes.items()}
    fold_of_block = assign_folds({b: dict(v) for b, v in block_stats.items()}, block_regime,
                                 args.folds, args.seed)

    bag_cols = (["bag_id", "datetime", "date", "hour", "year", "block", "fold", "regime",
                 "site", "sample_rate", "n_clips", "n_windows", "bag_duration_s"]
                + [f"{s}_index" for s in SPECIES] + [f"{s}_present" for s in SPECIES]
                + ["any_present", "clip_frac_max", "temp_c", "rh_pct"])
    inst_cols = ["instance_id", "bag_id", "fold", "filepath", "filename", "clip_start_s",
                 "clip_end_s", "win_idx", "sample_rate", "channels", "hour_offset_s"]
    bag_rows, inst_rows = [], []
    counts = collections.defaultdict(collections.Counter)
    for h, clips in hours.items():
        row = ann[h]
        bag_id = f"{clips[0]['site']}_{h:%Y%m%d_%H}"
        fold = fold_of_block[blocks[h.date()]]
        idx = {SPECIES_SHORT[s]: int(float(row[s])) for s in SPECIES_FULL}
        wins = [(c, s, e) for c in clips
                for s, e in windows(c["duration_s"], args.window_s, args.drop_partial)]
        rec = {"bag_id": bag_id, "datetime": h.isoformat(sep=" "),
               "date": h.date().isoformat(), "hour": h.hour, "year": h.year,
               "block": blocks[h.date()], "fold": fold, "regime": regime[h],
               "site": clips[0]["site"], "sample_rate": clips[0]["sample_rate"],
               "n_clips": len(clips), "n_windows": len(wins),
               "bag_duration_s": round(sum(c["duration_s"] for c in clips), 3),
               "any_present": int(any(v > 0 for v in idx.values())),
               "clip_frac_max": max(c["clip_frac"] for c in clips),
               "temp_c": row.get("Temp", ""), "rh_pct": row.get("RH%", "")}
        for k, v in idx.items():
            rec[f"{k}_index"] = v
            rec[f"{k}_present"] = int(v > 0)
        bag_rows.append(rec)
        key = f"fold{fold}"
        counts[key]["bags"] += 1
        counts[key]["instances"] += len(wins)
        counts[key][regime[h]] += 1
        for k, v in idx.items():
            counts[key][k] += int(v > 0)
        win_idx = collections.Counter()
        for c, s, e in wins:
            w = win_idx[c["filename"]]
            win_idx[c["filename"]] += 1
            inst_rows.append({
                "instance_id": f"{bag_id}_{c['start']:%H%M%S}_{s:07.2f}",
                "bag_id": bag_id, "fold": fold, "filepath": c["path"],
                "filename": c["filename"], "clip_start_s": s, "clip_end_s": e,
                "win_idx": w, "sample_rate": c["sample_rate"], "channels": c["channels"],
                "hour_offset_s": round(c["offset_in_hour_s"] + s, 2)})

    # Write next to the targets and rename, so a crash never leaves a half manifest.
    for name, cols, rows in (("bags.csv", bag_cols, bag_rows),
                             ("instances.csv", inst_cols, inst_rows)):
        tmp = args.out_dir / f".{name}.tmp"
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, args.out_dir / name)

    instances_id = hash_instances([r["instance_id"] for r in inst_rows])
    dataset_id = hash_dataset(args.out_dir / "bags.csv", instances_id)
    regimes = collections.Counter(r["regime"] for r in bag_rows)
    summary = {
        "dataset_id": dataset_id, "instances_id": instances_id,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "audio_dir": str(args.audio_dir), "annotation_csv": str(args.csv),
        "window_s": args.window_s, "hop_s": args.window_s, "drop_partial": args.drop_partial,
        "block_days": args.block_days, "folds": args.folds, "seed": args.seed,
        "species": SPECIES_FULL, "n_bags": len(bag_rows), "n_instances": len(inst_rows),
        "audio_window": [min(hours).isoformat(sep=" "), max(hours).isoformat(sep=" ")],
        "regimes": {r: {"bags": n, **{s: sum(b[f"{s}_present"] for b in bag_rows
                                             if b["regime"] == r) for s in SPECIES}}
                    for r, n in sorted(regimes.items())},
        "hours_with_audio_but_no_annotation": len(unlabelled),
        "skipped_files": [f"{n}: {r}" for n, r in bad],
        "qc_dropped_bags": {h.isoformat(sep=" "): v for h, v in sorted(qc_dropped.items())},
        "per_fold": {k: dict(v) for k, v in sorted(counts.items())},
    }
    tmp = args.out_dir / ".summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, args.out_dir / "summary.json")

    print(f"dataset_id {dataset_id}  (instances {instances_id})")
    print(f"bags      : {len(bag_rows)}  -> {args.out_dir / 'bags.csv'}")
    print(f"instances : {len(inst_rows)}  ({args.window_s}s contiguous windows)")
    for r, v in summary["regimes"].items():
        print(f"  regime {r:10} {v['bags']:>5} bags  "
              + "  ".join(f"{s}+={v[s]}" for s in SPECIES))
    for key, c in sorted(counts.items()):
        print(f"  {key} {c['bags']:>5} bags  {c['instances']:>6} instances  "
              + "  ".join(f"{s}+={c[s]}" for s in SPECIES) + "  "
              + " ".join(f"{r}={c[r]}" for r in sorted(regimes)))
    if bad:
        print(f"\nskipped {len(bad)} file(s) with unparsed names; see summary.json")
    if qc_dropped:
        print(f"dropped {len(qc_dropped)} hour(s) that failed audio QC; see summary.json")
    for s in SPECIES:
        if sum(c[s] for c in counts.values()) == 0:
            print(f"WARNING: zero positive bags for '{s}' in this audio window")


if __name__ == "__main__":
    main()
