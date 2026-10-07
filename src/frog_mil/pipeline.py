"""`frog`: the pipeline runner. Knows the stages, what each depends on, and reruns
only what is out of date.

    frog status                      # every stage: up to date / stale, and why
    frog run                         # run every stale stage, in order
    frog run report figures          # only these (and only if stale)
    frog run train --gpus 0,2        # stale sweep runs, in parallel, one per GPU
    frog run figures --force         # rerun even if up to date
    frog run --dry-run               # say what would run

Stages and what makes them stale
--------------------------------
manifest   the annotation CSV, the audio folder listing (name, size, mtime) or the
           manifest code changed since the last build.
embed      the embedding cache doesn't cover the manifest's windows, or the audio
           changed since it was last brought up to date (replaced files are
           re-embedded; see ``embcache``).
train:<id> one per entry of ``sweep.toml``: the run of the current dataset is
           missing, unfinished, or was trained with different settings. A run
           trained by older training code is reported, not retrained (``--force``).
report     the set of finished models, the manifest or the report code changed.
figures    the same, for the figure code.

A new manifest gets a new ``dataset_id``, so every train stage is stale (runs
live under ``runs/<dataset_id>/``; old ones stay where they are) and report and
figures follow. Before report or figures overwrite tables or figures of an
older dataset, those are copied to ``archive/<old dataset_id>/``.

Bookkeeping lives in ``outputs/.stamps/<stage>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import config as C

PY = sys.executable


# --------------------------------------------------------------------------- fingerprints
def sha_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        h.update(str(p.relative_to(C.ROOT) if p.is_relative_to(C.ROOT) else p).encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:16]


def audio_fingerprint(audio_dir: Path = C.AUDIO_DIR) -> str:
    h = hashlib.sha256()
    with os.scandir(audio_dir) as it:
        for e in sorted(it, key=lambda e: e.name):
            if e.name.endswith(".wav"):
                st = e.stat()
                h.update(f"{e.name}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def stamp_path(stage: str) -> Path:
    return C.STAMPS / f"{stage.replace(':', '_')}.json"


def read_stamp(stage: str) -> dict:
    p = stamp_path(stage)
    return json.loads(p.read_text()) if p.exists() else {}


def write_stamp(stage: str, **kw) -> None:
    C.STAMPS.mkdir(parents=True, exist_ok=True)
    stamp_path(stage).write_text(json.dumps(
        {**kw, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))


def dataset_id() -> str | None:
    try:
        return C.current_dataset_id()
    except (FileNotFoundError, KeyError):
        return None


def finished_models(did: str) -> list[Path]:
    return sorted((C.RUNS / did).glob("*/*/predictions.npz")) + \
        sorted((C.RUNS / did).glob("baselines/*.npz"))


def results_fingerprint(did: str, code: list[Path]) -> str:
    h = hashlib.sha256(did.encode())
    for p in finished_models(did):
        st = p.stat()
        h.update(f"{p.relative_to(C.RUNS)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    h.update(sha_files([C.BAGS_CSV, *code]).encode())
    return h.hexdigest()[:16]


SRC = Path(__file__).parent
MANIFEST_CODE = [SRC / "manifest.py", SRC / "audio.py", SRC / "config.py"]
REPORT_CODE = [SRC / f for f in ("report.py", "stats.py", "runs.py", "evaluate.py")]
FIG_CODE = REPORT_CODE + sorted((C.ROOT / "viz").glob("*.py"))


# --------------------------------------------------------------------------- stages
@dataclass
class Stage:
    name: str
    stale: bool
    reason: str
    cmd: list[str] = field(default_factory=list)
    gpu: bool = False
    note: str = ""
    stamp: dict = field(default_factory=dict)       # written after a successful run
    archive_from: str | None = None                 # dataset whose outputs get archived


def manifest_stage() -> Stage:
    fp = {"annotations": sha_files([C.ANNOTATIONS]), "audio": audio_fingerprint(),
          "code": sha_files(MANIFEST_CODE)}
    old = read_stamp("manifest")
    if not C.SUMMARY_JSON.exists():
        why = "no manifest"
    elif not old:
        why = "never built by frog"
    else:
        why = ", ".join(f"{k} changed" for k in fp if old.get(k) != fp[k])
    return Stage("manifest", bool(why), why or "up to date",
                 [PY, "-m", "frog_mil.manifest"], stamp=fp)


def embed_stage(audio: str) -> Stage:
    from .embcache import read_meta
    cmd = [str(C.ROOT / "scripts" / "perch_env.sh"), "python",
           str(C.ROOT / "scripts" / "embed_perch.py")]
    if not C.SUMMARY_JSON.exists():
        return Stage("embed", True, "waits for manifest", cmd, gpu=True)
    summ = C.manifest_summary()
    meta = read_meta(C.EMB_DIR)
    old = read_stamp("embed")
    if meta is None:
        why = "no embedding cache"
    elif meta.get("instances_id") != summ["instances_id"]:
        why = "cache doesn't match the manifest's windows"
    elif meta.get("n_done") != meta.get("n_instances"):
        why = f"cache incomplete ({meta.get('n_done')}/{meta.get('n_instances')})"
    elif old.get("audio") != audio:
        why = "audio changed since the cache was checked"
    else:
        why = ""
    return Stage("embed", bool(why), why or "up to date", cmd, gpu=True,
                 stamp={"audio": audio, "instances_id": summ["instances_id"]})


def sweep_entries() -> list[dict]:
    if not C.SWEEP.exists():
        return []
    return tomllib.loads(C.SWEEP.read_text()).get("run", [])


def entry_argv(e: dict) -> list[str]:
    argv = []
    for k, v in e.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            argv += [flag] if v else []
        else:
            argv += [flag, str(v)]
    return argv


def train_stages(did: str | None, n_folds: int | None) -> list[Stage]:
    from . import runs as rs
    from .train import build_parser, config_from_args
    out = []
    code = rs.train_code_hash()
    for e in sweep_entries():
        argv = entry_argv(e)
        a = build_parser().parse_args(argv)
        cmd = [PY, "-m", "frog_mil.train", *argv]
        rid = a.tag or rs.run_id_for(a.hidden, a.ordinal_weight)
        if did is None:
            out.append(Stage(f"train:{rid}", True, "waits for manifest", cmd, gpu=True))
            continue
        want = rs.spec(config_from_args(a, did, n_folds))
        d = rs.dataset_dir(did) / rid
        note = ""
        if not (d / "config.json").exists():
            why = "not trained on this dataset"
        else:
            cfg = rs.read_config(d)
            diff = [k for k in want if cfg.get(k) != want[k]]
            if diff:
                why = f"settings differ ({', '.join(diff)}); needs --overwrite or a --tag"
                cmd.append("--overwrite")
            elif cfg.get("status") != "complete":
                done = [p for p in cfg.get("poolings", [])
                        if (d / p / "predictions.npz").exists()]
                why = f"unfinished ({len(done)}/{len(cfg.get('poolings', []))} poolers)"
            else:
                why = ""
                if cfg.get("code", {}).get("train_code") != code:
                    note = "trained by older training code (--force to retrain)"
        out.append(Stage(f"train:{rid}", bool(why), why or "up to date", cmd, gpu=True,
                         note=note))
    return out


def results_stage(name: str, did: str | None, out_dir: Path, code: list[Path],
                  cmd: list[str], marker) -> Stage:
    if did is None:
        return Stage(name, True, "waits for manifest", cmd)
    if not any(p.name == "predictions.npz" for p in finished_models(did)):
        return Stage(name, True, "waits for trained models", cmd)
    fp = results_fingerprint(did, code)
    old = read_stamp(name)
    built_for = marker()
    if built_for != did:
        why = f"{out_dir.relative_to(C.ROOT)} describes dataset {built_for}"
    elif old.get("fingerprint") != fp:
        why = "models, manifest or code changed"
    else:
        why = ""
    archive = built_for if built_for and built_for != did else None
    return Stage(name, bool(why), why or "up to date", cmd,
                 stamp={"fingerprint": fp, "dataset_id": did}, archive_from=archive)


def results_dataset() -> str | None:
    p = C.RESULTS / "dataset.json"
    return json.loads(p.read_text()).get("dataset_id") if p.exists() else None


def figures_dataset() -> str | None:
    s = read_stamp("figures")
    if s:
        return s.get("dataset_id")
    return results_dataset() if (C.FIGURES / "index.md").exists() else None


def plan() -> list[Stage]:
    did = dataset_id()
    folds = C.manifest_summary().get("folds") if did else None
    man = manifest_stage()
    emb = embed_stage(man.stamp["audio"])
    if man.stale and not emb.stale:
        emb.stale, emb.reason = True, "after upstream stages"
    stages = [man, emb]
    upstream = any(s.stale for s in stages)
    trains = train_stages(did, folds)
    stages += trains
    stages.append(results_stage("report", did, C.RESULTS, REPORT_CODE,
                                [PY, "-m", "frog_mil.report"], results_dataset))
    stages.append(results_stage("figures", did, C.FIGURES, FIG_CODE,
                                [PY, str(C.ROOT / "viz" / "make_figures.py")],
                                figures_dataset))
    if upstream:   # anything downstream of a stale manifest/embed is stale too
        for s in stages[2:]:
            if not s.stale:
                s.stale, s.reason = True, "after upstream stages"
    return stages


def docs_note(did: str | None) -> str:
    """The prose isn't generated; flag docs that describe another dataset."""
    p = C.ROOT / "docs" / "findings.md"
    if not did or not p.exists():
        return ""
    import re
    m = re.search(r"<!--\s*dataset_id:\s*(\w+)\s*-->", p.read_text())
    if not m:
        return "docs/findings.md has no `<!-- dataset_id: ... -->` marker"
    if m.group(1) != did:
        return (f"docs/findings.md describes dataset {m.group(1)}, results are for {did}: "
                f"update the prose and its marker")
    return ""


# --------------------------------------------------------------------------- running
def free_gpus(max_used_mib: int = 2000) -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines()]
    return [i for i, mem, util in rows if int(mem) < max_used_mib and int(util) < 20]


def archive_outputs(old: str) -> None:
    """Copy tables/figures of dataset ``old`` to archive/<old>/ before they're replaced."""
    dest = C.ARCHIVE / old
    for src, name in ((C.RESULTS, "results"), (C.FIGURES, "figures")):
        if src.exists() and not (dest / name).exists():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns(".cache", "*.pdf"))
            print(f"archived {src.relative_to(C.ROOT)} -> {(dest / name).relative_to(C.ROOT)}")
    f = C.ROOT / "docs" / "findings.md"
    if f.exists() and not (dest / "findings.md").exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / "findings.md")


def log_path(stage: Stage) -> Path:
    did = dataset_id() or "no-dataset"
    p = C.LOGS / did / f"{stage.name.replace(':', '_')}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def launch(stage: Stage, gpu: str | None) -> subprocess.Popen:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log = log_path(stage)
    print(f"-> {stage.name}" + (f" on GPU {gpu}" if gpu is not None else "")
          + f"  (log: {log.relative_to(C.ROOT)})", flush=True)
    fh = log.open("a")
    fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(stage.cmd)}\n")
    fh.flush()
    return subprocess.Popen(stage.cmd, cwd=C.ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT)


def finish(stage: Stage, rc: int) -> None:
    if rc:
        tail = log_path(stage).read_text().splitlines()[-15:]
        raise SystemExit(f"{stage.name} failed (exit {rc}); last lines of its log:\n  "
                         + "\n  ".join(tail))
    if stage.stamp:
        if stage.name == "manifest":
            snapshot_manifest()
        write_stamp(stage.name, **stage.stamp)
    print(f"   {stage.name} done", flush=True)


def snapshot_manifest() -> None:
    """Keep every dataset's manifest by id, so old results can be regenerated."""
    did = dataset_id()
    dest = C.STORE / "manifests" / did
    dest.mkdir(parents=True, exist_ok=True)
    for f in (C.BAGS_CSV, C.INSTANCES_CSV, C.SUMMARY_JSON):
        shutil.copy2(f, dest / f.name)


def selected(stage: Stage, targets: list[str]) -> bool:
    if not targets:
        return True
    return any(stage.name == t or stage.name.split(":")[0] == t for t in targets)


def run(targets: list[str], force: bool, dry: bool, gpus: list[str]) -> None:
    order = ["manifest", "embed", "train", "report", "figures"]
    for group in order:
        stages = [s for s in plan() if s.name.split(":")[0] == group and selected(s, targets)]
        todo = [s for s in stages if s.stale or force]
        if not todo:
            continue
        blocked = [s for s in todo if s.reason.startswith("waits for")]
        if blocked and not dry:
            raise SystemExit(f"{blocked[0].name}: {blocked[0].reason}; run that stage first")
        for s in todo:
            print(f"{'would run' if dry else 'run'} {s.name}: "
                  f"{s.reason if s.stale else 'forced'}")
        if dry:
            continue
        for s in todo:
            if force and s.name.startswith("train:") and "--force" not in s.cmd:
                s.cmd.append("--force")
            if s.archive_from:
                archive_outputs(s.archive_from)
        if any(s.gpu for s in todo):
            pool = gpus or free_gpus()
            if not pool:
                raise SystemExit("no free GPU (nvidia-smi); pass --gpus")
            queue, running = list(todo), {}
            while queue or running:
                while queue and len(running) < len(pool):
                    busy = {g for _, g in running.values()}
                    g = next(g for g in pool if g not in busy)
                    s = queue.pop(0)
                    running[launch(s, g)] = (s, g)
                time.sleep(5)
                for p in [p for p in running if p.poll() is not None]:
                    s, _ = running.pop(p)
                    finish(s, p.returncode)
        else:
            for s in todo:
                finish(s, launch(s, None).wait())


def status() -> None:
    stages = plan()
    w = max(len(s.name) for s in stages)
    did = dataset_id()
    print(f"dataset {did or '—'}   store {C.STORE}")
    for s in stages:
        mark = "STALE" if s.stale else "ok"
        print(f"  {s.name:{w}}  {mark:5}  {s.reason}" + (f"  [{s.note}]" if s.note else ""))
    note = docs_note(did)
    if note:
        print(f"\nnote: {note}")
    if any(s.stale for s in stages):
        print("\n`frog run` brings everything up to date.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show which stages are out of date")
    r = sub.add_parser("run", help="run out-of-date stages")
    r.add_argument("targets", nargs="*",
                   help="manifest, embed, train, train:<run_id>, report, figures")
    r.add_argument("--force", action="store_true",
                   help="run selected stages even if up to date")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--gpus", default="", help="comma list; default: idle GPUs per nvidia-smi")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
    else:
        run(a.targets, a.force, a.dry_run, [g for g in a.gpus.split(",") if g])
        status()


if __name__ == "__main__":
    main()
