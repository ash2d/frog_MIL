"""Tests for scripts/ingest.py: diff/clash logic, the local hash cache, verify and merge.

No network: remote listings are faked, and local Dropbox hashes come from the
``rclone`` binary (skipped if it is missing) or the reference implementation below.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest", ROOT / "scripts" / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
sys.modules["ingest"] = ingest
spec.loader.exec_module(ingest)
RF = ingest.RemoteFile

needs_rclone = pytest.mark.skipif(not Path(ingest.RCLONE).exists(), reason="no rclone")


def dropbox_hash(path: Path) -> str:
    """Dropbox content hash: sha256 over the sha256 digests of 4 MiB blocks."""
    outer = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(4 << 20):
            outer.update(hashlib.sha256(block).digest())
    return outer.hexdigest()


def write_wav(path: Path, seed: int = 0, sr: int = 8000, secs: float = 2.0,
              zero_from: float | None = None) -> Path:
    x = np.random.default_rng(seed).integers(-3000, 3000, (int(sr * secs), 2), dtype=np.int16)
    if zero_from is not None:
        x[int(sr * zero_from):] = 0
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return path


# ---------------------------------------------------------------- pure logic

def test_parse_listing_and_hashsum():
    text = "aa\t10\tF Martin/HELECHOS_20190101_000000.wav\n\t5\tF Martin/x y.csv\n"
    assert ingest.parse_listing(text) == [
        RF("aa", 10, "F Martin/HELECHOS_20190101_000000.wav"), RF("", 5, "F Martin/x y.csv")]
    assert ingest.parse_hashsum("ab  a b.wav\ncd  c.wav\n") == {"a b.wav": "ab", "c.wav": "cd"}


def test_choose_copy_prefers_unprefixed_folder():
    a = RF("h", 1, "58702-04-06 18.33.09 - Up Martin/X.wav")
    b = RF("h", 1, "Up Martin/X.wav")
    c = RF("h", 1, "Up Martin/sub/X.wav")
    assert ingest.choose_copy([a, c, b]) == b
    assert ingest.choose_copy([a, c]) == c
    assert ingest.choose_copy([c, a]) == c


def test_diff_remote_categories():
    n = "HELECHOS_201909{:02d}_000000.wav".format
    remote = [
        RF("h1", 1, f"58700-x - Up Martin/{n(1)}"), RF("h1", 1, f"Up Martin/{n(1)}"),  # local
        RF("h2", 1, f"58700-x - Up Martin/{n(2)}"), RF("h2", 1, f"Up Martin/{n(2)}"),  # new
        RF("h3", 1, f"Up Martin/{n(3)}"),                     # name local, other content
        RF("h4", 1, f"A Martin/{n(4)}"), RF("h5", 1, f"B Martin/{n(4)}"),  # remote clash
        RF("h6", 1, "Up Martin/rec 1.wav"),                   # bad name
        RF("h7", 1, f"Up Martin/{n(7)}"),                     # known bad (in _replaced)
        RF("", 1, f"Up Martin/{n(8)}"),                       # no hash yet
        RF("h9", 1, "Up Martin/notes.csv"),                   # not a wav
    ]
    local = {n(1): "h1", n(3): "hX"}
    d = ingest.diff_remote(remote, local, {"h7"})
    assert d.n_wav == 10
    assert len(d.local) == 2 and {f.hash for f in d.local} == {"h1"}
    assert d.new == {"h2": RF("h2", 1, f"Up Martin/{n(2)}")}
    assert set(d.clash_local) == {"h3"}
    assert set(d.clash_remote) == {"h4", "h5"}
    assert set(d.bad_name) == {"h6"}
    assert [f.hash for f in d.replaced] == ["h7"] and len(d.nohash) == 1

    assert [f.hash for f in ingest.select_for_pull(d)] == ["h2"]
    assert {f.hash for f in ingest.select_for_pull(d, include_clashes=True)} == {"h2", "h3"}
    # --folder picks among the matching copies, even the prefixed one
    assert ingest.select_for_pull(d, folder="58700")[0].path.startswith("58700")
    assert ingest.select_for_pull(d, folder="Nope") == []


def test_date_runs_and_clips_per_hour():
    names = [f"HELECHOS_201809{d:02d}_{h:02d}{m:02d}00.wav"
             for d in (1, 2) for h in (10, 11) for m in (0, 20, 40)]
    names += ["HELECHOS_20181001_000000.wav", "junk.wav"]
    runs = ingest.date_runs(names)
    assert [(a.day, b.day, k) for a, b, k in runs] == [(1, 2, 12), (1, 1, 1)]
    assert ingest.clips_per_hour(names[:12]) == "3 clips/hour at :00/:20/:40"
    assert ingest.clips_per_hour(names[:1]) == "1 clip/hour at :00"
    assert ingest.clips_per_hour(names).startswith("1-3 clips/hour")


def test_describe_batch_and_merge_log_line():
    files = [{"name": f"HELECHOS_20180901_12{m}000.wav", "sample_rate": 8000, "channels": 2,
              "duration_s": 60.0} for m in (0, 2, 4)]
    desc = ingest.describe_batch(files)
    assert desc == ("2018-09-01 12:00 -> 2018-09-01 12:40, 3 clips/hour at :00/:20/:40, "
                    "8 kHz stereo 60 s")
    src, dst = Path("/d/dropbox/B"), Path("/d/2019_Rsync")
    line = ingest.merge_log_line("2026-10-07T10:25:27+01:00", 3, desc, src, dst,
                                 "B_verify.json", [], 0, 10)
    assert line.startswith(f"2026-10-07T10:25:27+01:00 moved 3 wavs ({desc}) from {src} "
                           f"to {dst} (mv -n; no name collisions;")
    assert line.endswith("; left behind 0; 2019_Rsync now 10 files")
    old = Path("/d/dropbox/_replaced/X.replaced_2026-10-07.wav")
    line = ingest.merge_log_line("t", 1, desc, src, dst, "v", [("X.wav", old)], 2, 11)
    assert "1 name collision replaced" in line
    assert f"X.wav REPLACED: old copy moved to {old}" in line


def test_plan_merge_and_stale():
    assert ingest.plan_merge(["a", "b", "c"], {"b", "z"}) == (["a", "c"], ["b"])
    cache = {"a": (1, 10, "ha"), "b": (2, 20, "hb")}
    assert ingest.stale(cache, {"a": (1, 10), "b": (2, 21), "c": (3, 30)}) == ["b", "c"]


# ---------------------------------------------------------------- with files

@pytest.fixture
def dirs(tmp_path):
    audio, staging = tmp_path / "audio", tmp_path / "staging"
    (staging / "_logs").mkdir(parents=True)
    audio.mkdir()
    return audio, staging


@needs_rclone
def test_rclone_hash_matches_reference(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(os.urandom((4 << 20) + 123))      # two blocks
    assert ingest.rclone_hashsum(tmp_path) == {"big.bin": dropbox_hash(p)}


@needs_rclone
def test_local_hash_cache_rehashes_only_changes(dirs, monkeypatch):
    audio, staging = dirs
    logs = staging / "_logs"
    for i in range(3):
        write_wav(audio / f"HELECHOS_20190901_0{i}0000.wav", seed=i)
    want = {p.name: dropbox_hash(p) for p in audio.iterdir()}
    assert ingest.local_hashes(audio, logs) == want

    calls = []
    real = ingest.rclone_hashsum
    monkeypatch.setattr(ingest, "rclone_hashsum", lambda d, names=None: (
        calls.append(names), real(d, names))[1])
    assert ingest.local_hashes(audio, logs) == want
    assert calls == [[]]                             # nothing rehashed
    p = write_wav(audio / "HELECHOS_20190901_000000.wav", seed=9, secs=3)
    want[p.name] = dropbox_hash(p)
    assert ingest.local_hashes(audio, logs) == want
    assert calls[-1] == [p.name]
    lines = (logs / "local_hashes.tsv").read_text().splitlines()
    assert lines[0] == f"# {audio}" and lines[1] == "name\tsize\tmtime_ns\thash"
    with pytest.raises(SystemExit):
        ingest.local_hashes(staging, logs)          # cache belongs to another dir


@needs_rclone
def test_seed_cache_checks_sizes(dirs):
    audio, staging = dirs
    logs = staging / "_logs"
    a = write_wav(audio / "HELECHOS_20190901_000000.wav", seed=1)
    b = write_wav(audio / "HELECHOS_20190901_010000.wav", seed=2)
    (logs / "old_local_hashes.txt").write_text(
        f"{dropbox_hash(a)}  {a.name}\nwronghash  {b.name}\n")
    # remote listing knows a's hash with the right size; nothing vouches for b
    (logs / "old_remote_listing.tsv").write_text(
        f"{dropbox_hash(a)}\t{a.stat().st_size}\tUp/{a.name}\nwronghash\t1\tUp/{b.name}\n")
    later = (logs / "old_local_hashes.txt").stat().st_mtime_ns + 10**9
    os.utime(logs / "old_local_hashes.txt", ns=(later, later))
    seeded = ingest.seed_cache(logs, audio)
    assert set(seeded) == {a.name}
    assert ingest.local_hashes(audio, logs)[b.name] == dropbox_hash(b)


def stage_batch(staging: Path, batch: str, files: dict[str, Path],
                remote_hash: dict[str, str] | None = None) -> Path:
    """Copy files into staging/<batch> and write a matching <batch>_files.tsv."""
    dest = staging / batch
    dest.mkdir()
    rows = []
    for name, src in files.items():
        shutil.copy(src, dest / name)
        h = (remote_hash or {}).get(name) or dropbox_hash(src)
        rows.append(RF(h, src.stat().st_size, f"Up Martin/{name}"))
    ingest.write_files_tsv(staging / "_logs" / f"{batch}_files.tsv", rows)
    return dest


def run_cli(*argv: str) -> int:
    return ingest.main(list(argv))


@needs_rclone
def test_verify_then_merge(dirs, tmp_path, capsys):
    audio, staging = dirs
    src = tmp_path / "src"
    src.mkdir()
    names = [f"HELECHOS_20190901_{h:02d}{m}000.wav" for h in (0, 1) for m in (0, 3)]
    files = {n: write_wav(src / n, seed=i) for i, n in enumerate(names)}
    dest = stage_batch(staging, "B1", files)
    (dest / "notes.csv").write_text("x")
    write_wav(audio / "HELECHOS_20190801_000000.wav", seed=99)
    common = ["--audio-dir", str(audio), "--staging", str(staging)]

    with pytest.raises(SystemExit, match="run verify first"):
        run_cli("merge", "B1", *common)
    assert run_cli("verify", "B1", *common) == 0
    rep = json.loads((staging / "_logs" / "B1_verify.json").read_text())
    assert rep["ok"] and rep["summary"]["clips_per_hour"] == "2 clips/hour at :00/:30"
    assert rep["summary"]["formats"] == {"8 kHz stereo 2 s 16-bit": 4}
    assert rep["unexpected_files"] == ["notes.csv"]

    assert run_cli("merge", "B1", *common) == 0
    assert sorted(p.name for p in dest.iterdir()) == ["notes.csv"]
    assert all((audio / n).exists() for n in names)
    log = (staging / "_logs" / "merge.log").read_text().splitlines()
    assert len(log) == 1 and "moved 4 wavs (2019-09-01 00:00 -> 2019-09-01 01:30" in log[0]
    assert log[0].endswith("left behind 1; audio now 5 files")
    assert "frog status" in capsys.readouterr().out


@needs_rclone
def test_verify_fails_on_hash_mismatch_qc_and_missing(dirs, tmp_path):
    audio, staging = dirs
    src = tmp_path / "src"
    src.mkdir()
    good = write_wav(src / "HELECHOS_20190901_000000.wav", seed=1)
    zero = write_wav(src / "HELECHOS_20190901_003000.wav", seed=2, zero_from=0.5)
    other = write_wav(src / "HELECHOS_20190901_010000.wav", seed=3)
    files = {good.name: good, zero.name: zero, other.name: other}
    dest = stage_batch(staging, "B2", files, {other.name: "not-the-remote-hash"})
    (dest / good.name).unlink()                      # missing
    write_wav(dest / "HELECHOS_20190901_020000.wav")  # stray wav
    common = ["--audio-dir", str(audio), "--staging", str(staging)]
    assert run_cli("verify", "B2", *common) == 1
    rep = json.loads((staging / "_logs" / "B2_verify.json").read_text())
    fail = {r["name"]: r["fail"] for r in rep["files"]}
    assert fail[good.name] == "missing"
    assert fail[zero.name].startswith("qc: digital zero")
    assert fail[other.name] == "hash mismatch"
    assert not rep["ok"]
    with pytest.raises(SystemExit, match="records failures"):
        run_cli("merge", "B2", *common)


@needs_rclone
def test_merge_refuses_stale_verify_and_handles_clashes(dirs, tmp_path):
    audio, staging = dirs
    src = tmp_path / "src"
    src.mkdir()
    new = write_wav(src / "HELECHOS_20190415_083000.wav", seed=1)
    old = write_wav(audio / new.name, seed=2, zero_from=0.2)   # same name, other content
    old_hash = dropbox_hash(old)
    dest = stage_batch(staging, "B3", {new.name: new})
    common = ["--audio-dir", str(audio), "--staging", str(staging)]
    assert run_cli("verify", "B3", *common) == 0

    # a staged file touched after verify -> refuse
    vjson = staging / "_logs" / "B3_verify.json"
    t = vjson.stat().st_mtime_ns - 10**9
    os.utime(vjson, ns=(t, t))
    with pytest.raises(SystemExit, match="changed after verify"):
        run_cli("merge", "B3", *common)
    assert run_cli("verify", "B3", *common) == 0

    assert run_cli("merge", "B3", *common) == 1      # clash, no --replace: nothing moved
    assert (dest / new.name).exists() and dropbox_hash(audio / new.name) == old_hash

    assert run_cli("merge", "B3", "--replace", *common) == 0
    stamp = ingest.today()
    moved_old = staging / "_replaced" / f"HELECHOS_20190415_083000.replaced_{stamp}.wav"
    assert dropbox_hash(moved_old) == old_hash
    assert dropbox_hash(audio / new.name) == dropbox_hash(new)
    assert not (dest / new.name).exists()
    assert "REPLACED: old copy moved to" in (staging / "_logs" / "merge.log").read_text()


@needs_rclone
def test_status_and_pull_planning_offline(dirs, tmp_path, capsys, monkeypatch):
    audio, staging = dirs
    a = write_wav(audio / "HELECHOS_20190901_000000.wav", seed=1)
    bad = write_wav(tmp_path / "bad.wav", seed=5)
    (staging / "_replaced").mkdir()
    shutil.copy(bad, staging / "_replaced" / "HELECHOS_20190415_083000.truncated.wav")
    listing = tmp_path / "listing.tsv"
    rows = [
        f"{dropbox_hash(a)}\t1\t58700-x - Up Martin/{a.name}",
        f"{dropbox_hash(a)}\t1\tUp Martin/{a.name}",
        "hnew\t2\t58700-x - Up Martin/HELECHOS_20190902_000000.wav",
        "hnew\t2\tUp Martin/HELECHOS_20190902_000000.wav",
        f"hclash\t3\tUp Martin/{a.name}",
        f"{dropbox_hash(bad)}\t4\tUp Martin/HELECHOS_20190415_083000.wav",
        "hcsv\t5\tUp Martin/notes.csv",
    ]
    listing.write_text("\n".join(rows) + "\n")
    common = ["--audio-dir", str(audio), "--staging", str(staging), "--listing", str(listing)]
    assert run_cli("status", *common) == 0
    out = capsys.readouterr().out
    assert "already local (hash match): 2" in out
    assert "previously replaced (hash in _replaced/): 1" in out
    assert "new unique hashes: 1" in out and "NAME CLASHES with the audio dir (1)" in out

    cmds, real_run = [], ingest.subprocess.run

    def fake_run(cmd, **kw):                          # record rclone copy, run everything else
        if cmd[1] != "copy":
            return real_run(cmd, **kw)
        cmds.append(cmd)
        return real_run(["true"], **kw)

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    assert run_cli("pull", "B4", *common) == 0
    tsv = (staging / "_logs" / "B4_files.tsv").read_text()
    assert tsv == "hnew\t2\tUp Martin/HELECHOS_20190902_000000.wav\n"
    (cmd,) = cmds
    assert cmd[:2] == [ingest.RCLONE, "copy"]
    assert cmd[4:6] == [f"{ingest.REMOTE}/Up Martin", str(staging / "B4")]
    assert "--ignore-existing" in cmd and cmd[-2] == "--log-file"
    with pytest.raises(SystemExit, match="different file list"):
        run_cli("pull", "B4", "--include-clashes", *common)
