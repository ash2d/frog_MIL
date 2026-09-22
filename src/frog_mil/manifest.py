"""Join the hourly frog annotations to the recordings and emit an ML-ready manifest.

Data model
----------
One annotation row covers one clock hour. The recorder writes 2-3 short clips
inside that hour, so the hour is a *bag* and each clip -- or, more usefully, each
fixed-length window inside a clip -- is an *instance*. The hourly label is the
merge (max) of the per-clip calling-intensity indices, which is exactly the
standard MIL assumption: bag_label = max(instance_labels).

Outputs (written to --out-dir):
    bags.csv       one row per annotated hour that has audio
    instances.csv  one row per fixed-length window, keyed back to its bag
    summary.json   counts, splits and the parameters used

Usage:
    frog-manifest --audio-dir <wav dir> --out-dir outputs
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import random
import re
import wave
from pathlib import Path

SPECIES = ["Gastrotheca chrysosticta", "Oreobates berdemenos"]
SPECIES_SHORT = {"Gastrotheca chrysosticta": "gastrotheca",
                 "Oreobates berdemenos": "oreobates"}
FNAME_RE = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<date>\d{8})_(?P<time>\d{6})\.wav$")


def probe(path: Path) -> dict | None:
    """Read WAV header + frame count. Returns None if the file is unreadable or
    truncated, which matters while the rsync is still in flight."""
    try:
        with wave.open(str(path)) as w:
            n, sr = w.getnframes(), w.getframerate()
            if n == 0 or sr == 0:
                return None
            expected = 44 + n * w.getnchannels() * w.getsampwidth()
            return {"sample_rate": sr, "channels": w.getnchannels(),
                    "duration_s": round(n / sr, 3),
                    "truncated": path.stat().st_size < expected}
    except Exception:
        return None


def scan_audio(audio_dir: Path) -> tuple[dict[dt.datetime, list[dict]], list[str]]:
    by_hour: dict[dt.datetime, list[dict]] = collections.defaultdict(list)
    bad: list[str] = []
    for p in sorted(audio_dir.glob("*.wav")):
        m = FNAME_RE.match(p.name)
        if not m:
            bad.append(f"{p.name}: unparsed filename")
            continue
        info = probe(p)
        if info is None or info["truncated"]:
            bad.append(f"{p.name}: unreadable or truncated (upload in flight?)")
            continue
        stamp = dt.datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
        by_hour[stamp.replace(minute=0, second=0)].append(
            {"path": str(p), "filename": p.name, "site": m["site"],
             "start": stamp, "offset_in_hour_s": stamp.minute * 60 + stamp.second,
             **info})
    return by_hour, bad


def assign_splits(day_pos: dict, block_days: int, ratios: tuple[float, float, float],
                  seed: int, n_tries: int = 500) -> dict:
    """Split by contiguous multi-day blocks, balanced on positives.

    Adjacent hours share weather, individuals and background, so an hour-level
    random split leaks badly. Whole blocks of days are dealt out instead. With
    only a few weeks of audio there are few blocks and a single shuffle easily
    strands every positive hour of a species in one split, so candidate
    shuffles are scored on how closely each split matches the target share of
    bags *and* of positive bags per species, and the best is kept.

    ``day_pos`` maps date -> {"bags": n, "<species>": n_positive, ...}.
    """
    if not day_pos:
        return {}
    origin = min(day_pos)
    blocks = collections.defaultdict(list)
    for d in day_pos:
        blocks[(d - origin).days // block_days].append(d)
    keys = sorted(blocks)
    names = ("train", "val", "test")
    metrics = sorted({k for v in day_pos.values() for k in v})
    totals = {m: sum(v.get(m, 0) for v in day_pos.values()) for m in metrics}

    # Largest-remainder allocation of blocks, so small block counts still give
    # every split at least one block where the ratios ask for one.
    raw = [len(keys) * r for r in ratios]
    alloc = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - alloc[i], reverse=True):
        if sum(alloc) >= len(keys):
            break
        alloc[i] += 1
    for i in (1, 2):
        if ratios[i] > 0 and alloc[i] == 0 and alloc[0] > 1:
            alloc[i], alloc[0] = 1, alloc[0] - 1

    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(n_tries):
        order = keys[:]
        rng.shuffle(order)
        cuts = (order[:alloc[0]], order[alloc[0]:alloc[0] + alloc[1]],
                order[alloc[0] + alloc[1]:])
        score = 0.0
        for name, target, ks in zip(names, ratios, cuts):
            for m in metrics:
                if totals[m] == 0:
                    continue
                got = sum(day_pos[d].get(m, 0) for k in ks for d in blocks[k])
                # positives are the scarce resource, so weight them heavily
                w = 1.0 if m == "bags" else 5.0
                score += w * abs(got / totals[m] - target)
        if score < best_score:
            best, best_score = cuts, score

    out = {}
    for name, ks in zip(names, best):
        for k in ks:
            for d in blocks[k]:
                out[d] = name
    return out


def windows(duration_s: float, win: float, hop: float, drop_partial: bool):
    t, out = 0.0, []
    while t + win <= duration_s + (0 if drop_partial else hop):
        end = min(t + win, duration_s)
        if end - t >= (win if drop_partial else win * 0.5):
            out.append((round(t, 3), round(end, 3)))
        t += hop
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/df_helechos_with2020.csv"))
    ap.add_argument("--audio-dir", type=Path,
                    default=Path("/gws/ssde/j25b/iecdt/dash/frogs/data/2019_Rsync"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--window-s", type=float, default=5.0,
                    help="instance length; 5.0 is Perch v2's native window")
    ap.add_argument("--hop-s", type=float, default=2.5,
                    help="instance hop. Keep window_s/hop_s an integer so coarser "
                         "hops stay exact subsets (win_idx %% k) of this cache")
    ap.add_argument("--drop-partial", action="store_true",
                    help="drop the short trailing window instead of zero-padding it")
    ap.add_argument("--block-days", type=int, default=3,
                    help="day-block granularity for the train/val/test split")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with args.csv.open() as fh:
        ann = {}
        for r in csv.DictReader(fh):
            ann[dt.datetime.fromisoformat(r["DateTime"])] = r

    by_hour, bad = scan_audio(args.audio_dir)
    if not by_hour:
        raise SystemExit(f"no usable audio under {args.audio_dir}")

    hours = sorted(h for h in by_hour if h in ann)
    unlabelled = sorted(h for h in by_hour if h not in ann)
    # Per-day tallies so the split can be balanced on positives, not just bags.
    day_pos: dict[dt.date, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    for h in hours:
        day_pos[h.date()]["bags"] += 1
        for sp, short in SPECIES_SHORT.items():
            day_pos[h.date()][short] += int(float(ann[h][sp]) > 0)
    splits = assign_splits(dict(day_pos), args.block_days,
                           tuple(args.ratios), args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bag_cols = (["bag_id", "datetime", "date", "hour", "split", "site",
                 "n_clips", "bag_duration_s"]
                + [f"{s}_index" for s in SPECIES_SHORT.values()]
                + [f"{s}_present" for s in SPECIES_SHORT.values()]
                + ["any_present", "temp_c", "rh_pct"])
    inst_cols = ["instance_id", "bag_id", "split", "filepath", "filename",
                 "clip_start_s", "clip_end_s", "win_idx", "sample_rate",
                 "channels", "hour_offset_s"]

    n_inst = 0
    counts = collections.defaultdict(collections.Counter)
    with (args.out_dir / "bags.csv").open("w", newline="") as bf, \
         (args.out_dir / "instances.csv").open("w", newline="") as inf:
        bw, iw = csv.DictWriter(bf, bag_cols), csv.DictWriter(inf, inst_cols)
        bw.writeheader()
        iw.writeheader()
        for h in hours:
            row, clips = ann[h], sorted(by_hour[h], key=lambda c: c["start"])
            bag_id = f"{clips[0]['site']}_{h:%Y%m%d_%H}"
            split = splits[h.date()]
            idx = {SPECIES_SHORT[s]: int(float(row[s])) for s in SPECIES}
            rec = {"bag_id": bag_id, "datetime": h.isoformat(sep=" "),
                   "date": h.date().isoformat(), "hour": h.hour, "split": split,
                   "site": clips[0]["site"], "n_clips": len(clips),
                   "bag_duration_s": round(sum(c["duration_s"] for c in clips), 3),
                   "any_present": int(any(v > 0 for v in idx.values())),
                   "temp_c": row.get("Temp", ""), "rh_pct": row.get("RH%", "")}
            for k, v in idx.items():
                rec[f"{k}_index"] = v
                rec[f"{k}_present"] = int(v > 0)
            bw.writerow(rec)
            for k, v in idx.items():
                counts[split][k] += int(v > 0)
            counts[split]["bags"] += 1

            for c in clips:
                for w, (s, e) in enumerate(windows(c["duration_s"], args.window_s,
                                                   args.hop_s, args.drop_partial)):
                    iw.writerow({
                        "instance_id": f"{bag_id}_{c['start']:%H%M%S}_{s:07.2f}",
                        "bag_id": bag_id, "split": split, "filepath": c["path"],
                        "filename": c["filename"], "clip_start_s": s,
                        "clip_end_s": e, "win_idx": w,
                        "sample_rate": c["sample_rate"], "channels": c["channels"],
                        "hour_offset_s": round(c["offset_in_hour_s"] + s, 2)})
                    n_inst += 1
                    counts[split]["instances"] += 1

    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "audio_dir": str(args.audio_dir), "annotation_csv": str(args.csv),
        "window_s": args.window_s, "hop_s": args.hop_s,
        "drop_partial": args.drop_partial,
        "block_days": args.block_days, "ratios": list(args.ratios),
        "seed": args.seed, "species": SPECIES,
        "n_bags": len(hours), "n_instances": n_inst,
        "audio_window": [hours[0].isoformat(sep=" "), hours[-1].isoformat(sep=" ")],
        "hours_with_audio_but_no_annotation": len(unlabelled),
        "skipped_files": bad,
        "per_split": {k: dict(v) for k, v in sorted(counts.items())},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"bags      : {len(hours)}  -> {args.out_dir/'bags.csv'}")
    print(f"instances : {n_inst}  ({args.window_s}s window, {args.hop_s}s hop)"
          f"  -> {args.out_dir/'instances.csv'}")
    for split, c in sorted(counts.items()):
        print(f"  {split:5} {c['bags']:>4} bags  {c['instances']:>6} instances  "
              + "  ".join(f"{s}+={c[s]}" for s in SPECIES_SHORT.values()))
    if bad:
        print(f"\nskipped {len(bad)} file(s); see summary.json")
    for s in SPECIES_SHORT.values():
        if sum(counts[k][s] for k in counts) == 0:
            print(f"WARNING: zero positive bags for '{s}' in this audio window")


if __name__ == "__main__":
    main()
