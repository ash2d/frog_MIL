"""Pull new field recordings from Dropbox into the audio folder: status, pull, verify, merge.

Uploads land on the rclone remote under ``dropbox:File requests/Yungas/``, one folder
per upload, and the same files often appear twice (``58xxx-<date> - <name> Martin/``
and ``<name> Martin/``). Dropbox can't be reached with rsync, so everything goes
through ``rclone``. A remote wav is *new* when its Dropbox content hash matches no
file in the audio dir; one copy per hash is pulled, preferring the folder without the
``58xxx-... - `` prefix. Hash dedupe can't see a re-upload that keeps the name but
changes the content (HELECHOS_20190415_083000.wav, Oct 2026: the local copy was
zero-filled after 3 MiB), so a remote name that exists locally with another hash is
a *name clash*: reported, never pulled or merged silently.

    .venv/bin/python scripts/ingest.py status                # read-only diff
    .venv/bin/python scripts/ingest.py pull <batch> [--folder <substr>] [--include-clashes]
    .venv/bin/python scripts/ingest.py verify <batch>        # hashes + qc_wav; exit 1 if bad
    .venv/bin/python scripts/ingest.py merge <batch> [--replace]

``pull`` stages into ``DROPBOX_DIR/<batch>/``; ``merge`` moves the verified files into
the audio dir without overwriting (``mv -n``), and with ``--replace`` first moves a
clashing local file to ``DROPBOX_DIR/_replaced/``. Logs, file lists, the verify
report and the local hash cache (``local_hashes.tsv``: name, size, mtime_ns, hash;
only new or changed files are rehashed) live in ``DROPBOX_DIR/_logs/``, and every
merge appends a line to ``_logs/merge.log``. ``--audio-dir`` and ``--staging``
override the two roots (for tests). Never print ``~/.config/rclone/rclone.conf``:
it holds the OAuth token.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

try:
    from frog_mil.audio import qc_wav
    from frog_mil.config import AUDIO_DIR, DROPBOX_DIR, FNAME_RE
except ImportError:  # not installed: use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from frog_mil.audio import qc_wav
    from frog_mil.config import AUDIO_DIR, DROPBOX_DIR, FNAME_RE

RCLONE = os.environ.get("RCLONE", "/usr/bin/rclone")
REMOTE = "dropbox:File requests/Yungas"
HASH = "dropbox"                                  # Dropbox content hash, computed locally too
CACHE_NAME = "local_hashes.tsv"
CACHE_COLS = ["name", "size", "mtime_ns", "hash"]
COPY_FLAGS = ["--ignore-existing", "--transfers", "4", "--checkers", "8", "--retries", "5",
              "--low-level-retries", "20", "--stats", "1m", "--stats-one-line", "-v"]
PREFIXED = re.compile(r"^\d+-.* - ")              # "58702-04-06 18.33.09 - Helechos_..."
BATCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
QC_WORKERS = 16


# ---------------------------------------------------------------- pure logic

@dataclass(frozen=True)
class RemoteFile:
    hash: str
    size: int
    path: str                                     # relative to REMOTE

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def folder(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""


@dataclass
class Diff:
    """Remote wavs sorted into what to do with them. Dicts map hash -> chosen copy."""
    n_wav: int = 0
    local: list[RemoteFile] = field(default_factory=list)       # content already local
    replaced: list[RemoteFile] = field(default_factory=list)    # content in _replaced/
    nohash: list[RemoteFile] = field(default_factory=list)      # upload in flight?
    copies: dict[str, list[RemoteFile]] = field(default_factory=dict)
    new: dict[str, RemoteFile] = field(default_factory=dict)
    clash_local: dict[str, RemoteFile] = field(default_factory=dict)   # name local, new hash
    clash_remote: dict[str, RemoteFile] = field(default_factory=dict)  # name shared remotely
    bad_name: dict[str, RemoteFile] = field(default_factory=dict)      # not FNAME_RE


def parse_listing(text: str) -> list[RemoteFile]:
    """``rclone lsf --format hsp --separator '\\t'`` output -> RemoteFile list."""
    out = []
    for line in text.splitlines():
        if line.strip():
            h, size, path = line.split("\t", 2)
            out.append(RemoteFile(h.strip(), int(size), path))
    return out


def parse_hashsum(text: str) -> dict[str, str]:
    """``rclone hashsum`` output (``<hash>  <name>``) -> {name: hash}."""
    out = {}
    for line in text.splitlines():
        if line.strip():
            h, name = line.split("  ", 1)
            out[name] = h
    return out


def is_wav(name: str) -> bool:
    return name.lower().endswith(".wav")


def copy_rank(f: RemoteFile) -> tuple:
    """Sort key for the copy to pull: unprefixed upload folder, shallow, then by path."""
    return bool(PREFIXED.match(f.path.split("/", 1)[0])), f.path.count("/"), f.path


def choose_copy(copies: list[RemoteFile]) -> RemoteFile:
    return min(copies, key=copy_rank)


def diff_remote(remote: list[RemoteFile], local: dict[str, str],
                replaced_hashes: set[str] = frozenset()) -> Diff:
    """Compare a remote listing with the audio dir ({name: hash})."""
    d = Diff()
    local_hashes = set(local.values())
    for r in remote:
        if not is_wav(r.name):
            continue
        d.n_wav += 1
        if not r.hash:
            d.nohash.append(r)
        elif r.hash in local_hashes:
            d.local.append(r)
        elif r.hash in replaced_hashes:
            d.replaced.append(r)
        else:
            d.copies.setdefault(r.hash, []).append(r)
    cand = {}
    for h, copies in d.copies.items():
        c = choose_copy(copies)
        if not FNAME_RE.match(c.name):
            d.bad_name[h] = c
        elif c.name in local:
            d.clash_local[h] = c
        else:
            cand[h] = c
    n_name = Counter(c.name for c in cand.values())
    for h, c in cand.items():
        (d.clash_remote if n_name[c.name] > 1 else d.new)[h] = c
    return d


def select_for_pull(d: Diff, folder: str | None = None,
                    include_clashes: bool = False) -> list[RemoteFile]:
    """One copy per hash to pull, restricted to copies whose folder contains ``folder``."""
    hashes = list(d.new) + (list(d.clash_local) if include_clashes else [])
    out = []
    for h in hashes:
        copies = [c for c in d.copies[h] if folder is None or folder in c.folder]
        if copies:
            out.append(choose_copy(copies))
    return sorted(out, key=lambda f: f.path)


def name_time(name: str) -> datetime | None:
    m = FNAME_RE.match(name)
    if not m:
        return None
    # recorder clock time, deliberately naive
    return datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")  # noqa: DTZ007


def date_runs(names: list[str], gap_h: float = 48) -> list[tuple[datetime, datetime, int]]:
    """Split file times into runs separated by more than ``gap_h`` hours."""
    ts = sorted(t for t in map(name_time, names) if t)
    runs: list[list] = []
    for t in ts:
        if runs and t - runs[-1][1] <= timedelta(hours=gap_h):
            runs[-1][1] = t
            runs[-1][2] += 1
        else:
            runs.append([t, t, 1])
    return [tuple(r) for r in runs]


def fmt_range(a: datetime, b: datetime) -> str:
    return f"{a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M}"


def clips_per_hour(names: list[str]) -> str:
    """'3 clips/hour at :00/:20/:40' from file names."""
    ts = [t for t in map(name_time, names) if t]
    if not ts:
        return "no dated files"
    per_hour = Counter(t.replace(minute=0, second=0) for t in ts)
    lo, hi = min(per_hour.values()), max(per_hour.values())
    n = f"{lo}" if lo == hi else f"{lo}-{hi}"
    s = f"{n} clip{'s' if hi > 1 else ''}/hour"
    offs = sorted({(t.minute, t.second) for t in ts})
    if len(offs) <= 6:
        s += " at " + "/".join(f":{m:02d}" + (f":{x:02d}" if x else "") for m, x in offs)
    return s


def fmt_format(sr: int, ch: int, dur: float) -> str:
    khz = f"{sr / 1000:g} kHz"
    chs = {1: "mono", 2: "stereo"}.get(ch, f"{ch} ch")
    return f"{khz} {chs} {dur:.0f} s"


def describe_batch(files: list[dict]) -> str:
    """'2018-09-01 12:00 -> 2018-10-24 22:40, 3 clips/hour at :00/:20/:40, 8 kHz ...'."""
    names = [f["name"] for f in files]
    ts = [t for t in map(name_time, names) if t]
    fmts = Counter(fmt_format(f["sample_rate"], f["channels"], f["duration_s"])
                   for f in files if f.get("sample_rate"))
    fmt = (next(iter(fmts)) if len(fmts) == 1
           else " + ".join(f"{k} ({v})" for k, v in fmts.most_common()))
    parts = [fmt_range(min(ts), max(ts))] if ts else []
    return ", ".join(parts + [clips_per_hour(names), fmt])


def merge_log_line(ts: str, n: int, desc: str, src: Path, dst: Path, verify_name: str,
                   replaced: list[tuple[str, Path]], left_behind: int, n_after: int) -> str:
    """One merge.log line, in the style of the earlier manual merges."""
    clash = ("no name collisions" if not replaced
             else f"{len(replaced)} name collision{'s' * (len(replaced) > 1)} replaced")
    line = (f"{ts} moved {n} wavs ({desc}) from {src} to {dst} "
            f"(mv -n; {clash}; all verified by ingest.py verify, {verify_name})")
    for name, old in replaced:
        line += f"; {name} REPLACED: old copy moved to {old}"
    return line + f"; left behind {left_behind}; {dst.name} now {n_after} files"


def plan_merge(staged: list[str], local: set[str]) -> tuple[list[str], list[str]]:
    """(names to move, names that clash with the audio dir)."""
    return [n for n in staged if n not in local], [n for n in staged if n in local]


def stale(cache: dict[str, tuple[int, int, str]],
          stats: dict[str, tuple[int, int]]) -> list[str]:
    """Names whose (size, mtime_ns) is new or changed since the cache was written."""
    return sorted(n for n, st in stats.items() if n not in cache or cache[n][:2] != st)


# ---------------------------------------------------------------- rclone and files

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def rclone_hashsum(root: Path, names: list[str] | None = None) -> dict[str, str]:
    """Dropbox content hashes of files under ``root`` (all, or just ``names``)."""
    if names is not None and not names:
        return {}
    cmd = [RCLONE, "hashsum", HASH, "--checkers", "16", str(root)]
    if names is None:
        return parse_hashsum(run(cmd).stdout)
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write("".join(n + "\n" for n in names))
        f.flush()
        return parse_hashsum(run(cmd + ["--files-from-raw", f.name]).stdout)


def list_remote(remote: str, out: Path) -> list[RemoteFile]:
    cmd = [RCLONE, "lsf", "-R", "--hash", HASH, "--format", "hsp", "--separator", "\t",
           "--files-only", remote]
    print(f"listing {remote} ...", flush=True)
    text = run(cmd).stdout
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(out)
    print(f"  saved listing -> {out}")
    return parse_listing(text)


def wav_stats(root: Path) -> dict[str, tuple[int, int]]:
    """{name: (size, mtime_ns)} of the wavs directly in ``root``."""
    out = {}
    with os.scandir(root) as it:
        for e in it:
            if e.is_file() and is_wav(e.name):
                st = e.stat()
                out[e.name] = (st.st_size, st.st_mtime_ns)
    return out


def read_cache(path: Path, audio_dir: Path) -> dict[str, tuple[int, int, str]] | None:
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if lines[0] != f"# {audio_dir}":
        raise SystemExit(f"{path} is a cache for {lines[0][2:]}, not {audio_dir}")
    assert lines[1].split("\t") == CACHE_COLS, f"bad header in {path}"
    out = {}
    for line in lines[2:]:
        name, size, mtime, h = line.split("\t")
        out[name] = (int(size), int(mtime), h)
    return out


def write_cache(path: Path, audio_dir: Path, cache: dict[str, tuple[int, int, str]]) -> None:
    rows = [f"# {audio_dir}", "\t".join(CACHE_COLS)]
    rows += [f"{n}\t{s}\t{m}\t{h}" for n, (s, m, h) in sorted(cache.items())]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + "\n")
    tmp.replace(path)


def seed_cache(logs: Path, audio_dir: Path) -> dict[str, tuple[int, int, str]]:
    """Reuse the newest ``*_local_hashes.txt`` for files untouched since it was written.

    The old files carry no sizes, so an entry is kept only if the file's ctime predates
    the hash file (no write, rename or move since) and the size Dropbox reports for that
    hash in a saved listing equals the file's size.
    """
    seeds = sorted(logs.glob("*_local_hashes.txt"), key=lambda p: p.stat().st_mtime_ns)
    if not seeds:
        return {}
    seed, seed_t = seeds[-1], seeds[-1].stat().st_mtime_ns
    size_of: dict[str, int] = {}
    for tsv in logs.glob("*.tsv"):
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[1].isdigit():
                size_of[parts[0]] = int(parts[1])
    out = {}
    for name, h in parse_hashsum(seed.read_text()).items():
        try:
            st = (audio_dir / name).stat()
        except FileNotFoundError:
            continue
        if st.st_ctime_ns < seed_t and size_of.get(h) == st.st_size:
            out[name] = (st.st_size, st.st_mtime_ns, h)
    print(f"  seeded {len(out)} hashes from {seed.name}")
    return out


def local_hashes(audio_dir: Path, logs: Path) -> dict[str, str]:
    """{name: hash} for the audio dir, rehashing only new or changed files."""
    path = logs / CACHE_NAME
    cache = read_cache(path, audio_dir)
    if cache is None:
        cache = seed_cache(logs, audio_dir)
    stats = wav_stats(audio_dir)
    todo = stale(cache, stats)
    if todo:
        gb = sum(stats[n][0] for n in todo) / 1e9
        print(f"  hashing {len(todo)} new/changed local files ({gb:.1f} GB) ...", flush=True)
    hashed = rclone_hashsum(audio_dir, todo)
    missing = set(todo) - set(hashed)
    if missing:
        raise SystemExit(f"rclone hashsum skipped {len(missing)} files: {sorted(missing)[:5]}")
    cache = {n: cache[n] if n not in hashed else (*stats[n], hashed[n]) for n in stats}
    write_cache(path, audio_dir, cache)
    return {n: h for n, (_, _, h) in cache.items()}


def replaced_hashes(staging: Path) -> set[str]:
    d = staging / "_replaced"
    return set(rclone_hashsum(d).values()) if d.is_dir() and any(d.iterdir()) else set()


def read_files_tsv(path: Path) -> list[RemoteFile]:
    return parse_listing(path.read_text())


def write_files_tsv(path: Path, files: list[RemoteFile]) -> None:
    path.write_text("".join(f"{f.hash}\t{f.size}\t{f.path}\n" for f in files))


# ---------------------------------------------------------------- commands

def compute_diff(args) -> tuple[Diff, dict[str, str]]:
    logs = args.staging / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.listing:
        remote = parse_listing(Path(args.listing).read_text())
        print(f"using saved listing {args.listing}")
    else:
        out = logs / f"{today()}_remote_listing.tsv"
        remote = list_remote(args.remote, out)
    print(f"local hashes of {args.audio_dir} ...", flush=True)
    local = local_hashes(args.audio_dir, logs)
    return diff_remote(remote, local, replaced_hashes(args.staging)), local


def print_group(title: str, files: dict[str, RemoteFile]) -> None:
    if not files:
        return
    gb = sum(f.size for f in files.values()) / 1e9
    print(f"\n{title}: {len(files)} unique hashes, {gb:.2f} GB")
    by_folder: dict[str, list[RemoteFile]] = defaultdict(list)
    for f in files.values():
        by_folder[f.folder].append(f)
    for folder, fs in sorted(by_folder.items()):
        names = [f.name for f in fs]
        print(f"  {folder or '.'}: {len(fs)} files, {clips_per_hour(names)}")
        for a, b, n in date_runs(names):
            print(f"      {fmt_range(a, b)}  n={n}")
        undated = sorted(n for n in names if not name_time(n))
        if undated:
            print(f"      undated: {len(undated)}, e.g. {undated[:3]}")


def cmd_status(args) -> int:
    d, local = compute_diff(args)
    n_unique = len(set(d.copies) | {f.hash for f in d.local + d.replaced})
    print(f"\nremote wavs: {d.n_wav} ({n_unique} unique hashes); "
          f"audio dir: {len(local)} files")
    print(f"  already local (hash match): {len(d.local)} remote files")
    print(f"  previously replaced (hash in _replaced/): {len(d.replaced)}")
    if d.nohash:
        print(f"  no remote hash (still uploading?): {len(d.nohash)}, "
              f"e.g. {[f.path for f in d.nohash[:3]]}")
    print(f"  new unique hashes: {len(d.new)}")
    print_group("NEW", d.new)
    if d.clash_local:
        print(f"\nNAME CLASHES with the audio dir ({len(d.clash_local)}): same name, "
              "different content; pulled only with --include-clashes, merged only with "
              "--replace")
        for h, f in sorted(d.clash_local.items(), key=lambda x: x[1].name):
            print(f"  {f.name}  local {local[f.name][:12]}  remote {h[:12]}  {f.path}")
    if d.clash_remote:
        print(f"\nSAME NAME, DIFFERENT CONTENT within the remote ({len(d.clash_remote)}); "
              "never pulled, resolve by hand:")
        for h, f in sorted(d.clash_remote.items(), key=lambda x: x[1].name):
            print(f"  {f.name}  {h[:12]}  {f.path}")
    if d.bad_name:
        print(f"\nNON-STANDARD NAMES ({len(d.bad_name)}); never pulled:")
        for f in sorted(d.bad_name.values(), key=lambda f: f.path)[:20]:
            print(f"  {f.path}")
    return 0


def check_batch(batch: str) -> str:
    if not BATCH_RE.fullmatch(batch):
        raise SystemExit(f"bad batch name {batch!r}: letters, digits, _ . - only")
    return batch


def cmd_pull(args) -> int:
    batch = check_batch(args.batch)
    logs, dest = args.staging / "_logs", args.staging / batch
    d, _ = compute_diff(args)
    files = select_for_pull(d, args.folder, args.include_clashes)
    if not files:
        print("nothing to pull")
        return 0
    tsv = logs / f"{batch}_files.tsv"
    if tsv.exists() and read_files_tsv(tsv) != files:
        raise SystemExit(f"{tsv} exists with a different file list; use a new batch name")
    write_files_tsv(tsv, files)
    gb = sum(f.size for f in files) / 1e9
    print(f"\npulling {len(files)} files ({gb:.2f} GB) -> {dest}; list in {tsv}")
    dest.mkdir(parents=True, exist_ok=True)
    log = logs / f"{batch}_copy.log"
    by_folder: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_folder[f.folder].append(f.name)
    failed = 0
    for folder, names in sorted(by_folder.items()):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as lst:
            lst.write("".join(n + "\n" for n in names))
            lst.flush()
            src = f"{args.remote}/{folder}" if folder else args.remote
            print(f"  {len(names)} from {src} ...", flush=True)
            cmd = [RCLONE, "copy", "--files-from-raw", lst.name, src, str(dest), *COPY_FLAGS,
                   "--log-file", str(log)]
            rc = subprocess.run(cmd, check=False).returncode
        with log.open("a") as fh:
            fh.write(f"EXIT {rc} ({folder})\n")
        failed += rc != 0
    print(f"rclone log: {log}")
    if failed:
        print(f"{failed} rclone copies failed; rerun pull (it resumes) before verify")
        return 1
    print(f"next: .venv/bin/python scripts/ingest.py verify {batch}")
    return 0


def summarise(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("sample_rate")]
    names = [r["name"] for r in recs]
    ts = [t for t in map(name_time, names) if t]
    return {
        "n": len(recs),
        "formats": dict(Counter(fmt_format(r["sample_rate"], r["channels"], r["duration_s"])
                                + f" {8 * r['sampwidth']}-bit" for r in ok)),
        "clips_per_hour": clips_per_hour(names),
        "date_range": fmt_range(min(ts), max(ts)) if ts else "",
        "date_runs": [[fmt_range(a, b), n] for a, b, n in date_runs(names)],
        "gb": round(sum(r["size"] for r in recs) / 1e9, 3),
        "failures": dict(Counter(r["fail"] for r in recs if r["fail"])),
    }


def cmd_verify(args) -> int:
    batch = check_batch(args.batch)
    logs, dest = args.staging / "_logs", args.staging / batch
    expected = {f.name: f for f in read_files_tsv(logs / f"{batch}_files.tsv")}
    on_disk = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    print(f"hashing {len(on_disk)} staged files in {dest} ...", flush=True)
    got = rclone_hashsum(dest)
    on_disk_set = set(on_disk)
    present = [n for n in sorted(expected) if n in on_disk_set]
    with Pool(QC_WORKERS) as pool:
        qcs = pool.map(qc_wav, [dest / n for n in present], chunksize=4)
    qc_of = dict(zip(present, qcs))
    recs = []
    for name, f in sorted(expected.items()):
        qc = qc_of.get(name, {})
        size = (dest / name).stat().st_size if name in qc_of else None
        fails = [msg for bad, msg in [
            (name not in qc_of, "missing"),
            (name in qc_of and got.get(name) != f.hash, "hash mismatch"),
            (size is not None and size != f.size, "size mismatch"),
            (not FNAME_RE.match(name), "bad name"),
            (name in qc_of and not qc.get("qc_ok"), f"qc: {qc.get('qc_reason')}"),
        ] if bad]
        recs.append({"name": name, "remote_path": f.path, "expected_hash": f.hash,
                     "hash": got.get(name), "size": size or 0, **qc,
                     "ok": not fails, "fail": "; ".join(fails)})
    unexpected = [n for n in on_disk if n not in expected]
    bad_extra = [n for n in unexpected if is_wav(n)]
    summary = summarise(recs)
    ok = all(r["ok"] for r in recs) and not bad_extra and bool(recs)
    report = {"batch": batch, "staging": str(dest), "created": now(), "ok": ok,
              "summary": summary, "unexpected_files": unexpected, "files": recs}
    out = logs / f"{batch}_verify.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    print(f"expected {len(expected)}, on disk {len(on_disk)}, "
          f"failed {sum(not r['ok'] for r in recs)}")
    for k in ("formats", "clips_per_hour", "date_range", "gb", "failures"):
        print(f"  {k}: {summary[k]}")
    for r in [r for r in recs if not r["ok"]][:20]:
        print(f"  FAIL {r['name']}: {r['fail']}")
    if unexpected:
        print(f"  not in files.tsv: {len(unexpected)} ({len(bad_extra)} wavs), "
              f"e.g. {unexpected[:5]}")
    print(f"{'OK' if ok else 'FAILED'} -> {out}")
    if ok:
        print(f"next: .venv/bin/python scripts/ingest.py merge {batch}")
    return 0 if ok else 1


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def today() -> str:
    return now()[:10]


def cmd_merge(args) -> int:
    batch = check_batch(args.batch)
    logs, dest, audio = args.staging / "_logs", args.staging / batch, args.audio_dir
    vpath = logs / f"{batch}_verify.json"
    if not vpath.exists():
        raise SystemExit(f"no {vpath}; run verify first")
    report = json.loads(vpath.read_text())
    if not report["ok"] or not all(r["ok"] for r in report["files"]):
        raise SystemExit(f"{vpath} records failures; fix and rerun verify")
    vtime = vpath.stat().st_mtime_ns
    staged = [p for p in dest.rglob("*") if p.is_file()]
    newer = [p.name for p in staged
             if max(p.stat().st_mtime_ns, p.stat().st_ctime_ns) >= vtime]
    if newer:
        raise SystemExit(f"{len(newer)} staged files changed after verify, e.g. {newer[:5]}; "
                         "rerun verify")
    verified = {r["name"]: r for r in report["files"]}
    staged_wavs = {p.name for p in staged if is_wav(p.name) and p.parent == dest}
    if staged_wavs != set(verified):
        raise SystemExit(f"staged wavs differ from {vpath.name}: "
                         f"{sorted(staged_wavs ^ set(verified))[:10]}; rerun verify")

    moves, clashes = plan_merge(sorted(verified), set(wav_stats(audio)))
    if clashes and not args.replace:
        old = rclone_hashsum(audio, clashes)
        print(f"{len(clashes)} name clashes with {audio} (nothing moved):")
        for n in clashes:
            print(f"  {n}  local {old.get(n, '?')[:12]}  staged {verified[n]['hash'][:12]}")
        print("rerun with --replace to move the local copies to _replaced/ first")
        return 1
    replaced_dir = args.staging / "_replaced"
    stamp = today()
    targets = {n: replaced_dir / f"{n[:-4]}.replaced_{stamp}.wav" for n in clashes}
    taken = [str(t) for t in targets.values() if t.exists()]
    if taken:
        raise SystemExit(f"replacement targets already exist: {taken}")

    def mv(src: Path, dst: Path) -> None:
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        shutil.move(src, dst)

    replaced = []
    if clashes:
        replaced_dir.mkdir(exist_ok=True)
    for n in clashes:
        mv(audio / n, targets[n])
        replaced.append((n, targets[n]))
    for n in moves + clashes:
        mv(dest / n, audio / n)

    # the moved files keep size and mtime, so their verified hashes go straight into the cache
    cpath = logs / CACHE_NAME
    cache = read_cache(cpath, audio)
    if cache is not None:
        for n in verified:
            st = (audio / n).stat()
            cache[n] = (st.st_size, st.st_mtime_ns, verified[n]["hash"])
        write_cache(cpath, audio, cache)
    (logs / f"{batch}_merged.txt").write_text("".join(n + "\n" for n in sorted(verified)))
    left = sum(1 for p in dest.rglob("*") if p.is_file())
    n_after = sum(1 for p in audio.iterdir() if p.is_file())
    line = merge_log_line(now(), len(verified), describe_batch(list(verified.values())),
                          dest, audio, vpath.name, replaced, left, n_after)
    with (logs / "merge.log").open("a") as fh:
        fh.write(line + "\n")
    print(line)
    print("\nnext: .venv/bin/frog status, then .venv/bin/frog run "
          "(rebuilds the manifest and embeddings for the new audio)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    common.add_argument("--staging", type=Path, default=DROPBOX_DIR,
                        help="staging root; holds <batch>/, _logs/ and _replaced/")
    remote = argparse.ArgumentParser(add_help=False)
    remote.add_argument("--remote", default=REMOTE)
    remote.add_argument("--listing", help="reuse a saved remote listing TSV (no network)")

    sub.add_parser("status", parents=[common, remote], help="read-only diff, remote vs local")
    p = sub.add_parser("pull", parents=[common, remote], help="copy new files to staging")
    p.add_argument("batch")
    p.add_argument("--folder", help="only remote folders containing this substring")
    p.add_argument("--include-clashes", action="store_true",
                   help="also pull files whose name exists locally with other content")
    p = sub.add_parser("verify", parents=[common], help="hash + QC the staged batch")
    p.add_argument("batch")
    p = sub.add_parser("merge", parents=[common], help="move a verified batch into audio dir")
    p.add_argument("batch")
    p.add_argument("--replace", action="store_true",
                   help="move clashing local files to _replaced/ and take the staged copy")
    args = ap.parse_args(argv)
    cmds = {"status": cmd_status, "pull": cmd_pull, "verify": cmd_verify, "merge": cmd_merge}
    return cmds[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
