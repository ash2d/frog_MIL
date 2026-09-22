"""Summarise the Helechos hourly frog annotations.

Prints the facts that drive the MIL setup: label scale, seasonality, diel
pattern, co-occurrence, and how much of the annotation table is actually
covered by audio that has landed on disk.

Usage:
    python -m frog_mil.annotations [--csv data/df_helechos_with2020.csv]
                                   [--audio-dir <wav dir>]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import re
from pathlib import Path

SPECIES = ["Gastrotheca chrysosticta", "Oreobates berdemenos"]
FNAME_RE = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<date>\d{8})_(?P<time>\d{6})\.wav$")


def load_annotations(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_dt"] = dt.datetime.fromisoformat(r["DateTime"])
        for sp in SPECIES:
            r[sp] = int(float(r[sp]))
    return rows


def audio_hours(audio_dir: Path) -> dict[dt.datetime, list[str]]:
    """Map each hour to the recordings that fall inside it."""
    by_hour: dict[dt.datetime, list[str]] = collections.defaultdict(list)
    if not audio_dir.is_dir():
        return by_hour
    for p in sorted(audio_dir.glob("*.wav")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        stamp = dt.datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
        by_hour[stamp.replace(minute=0, second=0)].append(p.name)
    return by_hour


def _hist(rows, key, bucket, size):
    c = collections.Counter()
    for r in rows:
        if r[key] > 0:
            c[bucket(r)] += 1
    return [c.get(i, 0) for i in size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/df_helechos_with2020.csv", type=Path)
    ap.add_argument("--audio-dir", type=Path,
                    default=Path("/gws/ssde/j25b/iecdt/dash/frogs/data/2019_Rsync"))
    args = ap.parse_args()

    rows = load_annotations(args.csv)
    print(f"annotation rows : {len(rows)}")
    print(f"time range      : {rows[0]['_dt']}  ->  {rows[-1]['_dt']}")

    print("\n-- label scale (hourly calling-intensity index) --")
    for sp in SPECIES:
        by_year = collections.defaultdict(collections.Counter)
        for r in rows:
            by_year[r["_dt"].year][r[sp]] += 1
        print(f"  {sp}")
        for year, c in sorted(by_year.items()):
            tot = sum(c.values())
            pos = tot - c[0]
            print(f"    {year}: " + " ".join(f"{k}={c[k]:>5}" for k in range(4))
                  + f"   positive {pos}/{tot} ({pos / tot:.1%})")

    print("\n-- seasonality (positive hours per month, all years) --")
    for sp in SPECIES:
        h = _hist(rows, sp, lambda r: r["_dt"].month, range(1, 13))
        print(f"  {sp[:12]:12} " + " ".join(f"{m:>4}" for m in range(1, 13)))
        print(f"  {'':12} " + " ".join(f"{v:>4}" for v in h))

    print("\n-- diel pattern (positive hours per hour-of-day) --")
    for sp in SPECIES:
        h = _hist(rows, sp, lambda r: r["_dt"].hour, range(24))
        print(f"  {sp[:12]:12} " + " ".join(f"{v:>3}" for v in h))

    g, o = SPECIES
    both = sum(1 for r in rows if r[g] > 0 and r[o] > 0)
    print(f"\nco-occurring positive hours: {both} "
          f"(=> multi-label, not mutually exclusive)")

    by_hour = audio_hours(args.audio_dir)
    if not by_hour:
        print(f"\nno audio found under {args.audio_dir}")
        return

    print(f"\n-- audio on disk: {args.audio_dir} --")
    files = sum(len(v) for v in by_hour.values())
    hours = sorted(by_hour)
    print(f"  {files} files across {len(by_hour)} hours, "
          f"{hours[0]} -> {hours[-1]}")
    per_hour = collections.Counter(len(v) for v in by_hour.values())
    print(f"  files per hour: {dict(sorted(per_hour.items()))}")

    ann = {r["_dt"]: r for r in rows}
    matched = [ann[h] for h in hours if h in ann]
    print(f"  hours with a matching annotation row: {len(matched)}/{len(by_hour)}")
    print("  usable bag labels in this audio window:")
    for sp in SPECIES:
        c = collections.Counter(r[sp] for r in matched)
        pos = sum(v for k, v in c.items() if k > 0)
        flag = "   <-- NO POSITIVES: extend the upload" if pos == 0 else ""
        print(f"    {sp:26} positive {pos:>4}/{len(matched)}  "
              + " ".join(f"{k}={c.get(k, 0)}" for k in range(4)) + flag)

    print("\n  first positive hour per species in the full annotation table:")
    for sp in SPECIES:
        yr = [r for r in rows if r["_dt"].year == 2019 and r[sp] > 0]
        if yr:
            print(f"    {sp:26} 2019 season starts {yr[0]['_dt']}")


if __name__ == "__main__":
    main()
