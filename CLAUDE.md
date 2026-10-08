# frog_MIL — working notes

MIL detection of two frogs from hourly labels: frozen Perch v2 embeddings, then
a per-window probe, then a pooling function. The overview is in `README.md`, the
design in `docs/methods.md`, results in `docs/findings.md` and `results/RESULTS.md`.

## Environments

Use two venvs, because TF and torch ship conflicting CUDA stacks. Embeddings
are cached on disk, so the two are never needed together.

```bash
uv sync --group dev                                             # .venv: torch, training, report, viz, tests
UV_PROJECT_ENVIRONMENT=.venv-perch uv sync --only-group perch   # .venv-perch: TF + perch_hoplite
```

**Always run Perch through `scripts/perch_env.sh`** (`frog run embed` does). Without
it TF can't see the GPU, and `perch_v2` then resolves to a *different* model
(`perch_v2_cpu`); `embed_perch.py` refuses that. The GPUs are shared, so check
`nvidia-smi`. Training and Perch run on gpuhost001 (hpxfer4 has no GPU).

## Pipeline: `frog`

```bash
.venv/bin/frog status                 # which stages are stale, and why
.venv/bin/frog run [--gpus 0,2]       # run every stale stage, in order
.venv/bin/frog run report figures     # only these, only if stale (--force to rerun)
.venv/bin/python -m pytest -q         # tests (tests/)
```

Stages: `manifest` → `embed` → `train:<run_id>` (one per `[[run]]` in
`sweep.toml`, parallel over GPUs) → `report` → `figures`. Each is fingerprinted
(inputs, code, dataset_id) in `outputs/.stamps/`, so `frog` reruns only what
changed and archives old tables/figures when the dataset changes. To add a run,
add it to `sweep.toml`; don't call `frog-train` by hand for sweep runs. Logs:
`outputs/logs/<dataset_id>/<stage>.log`.

New audio: `scripts/ingest.py status | pull <batch> | verify <batch> | merge <batch>`
(Dropbox via rclone; verify runs audio QC; merge never overwrites), then `frog run`.

The individual commands still exist: `frog-manifest`, `scripts/embed_perch.py`,
`frog-train`, `frog-report`, `viz/make_figures.py`, `python -m frog_mil.annotations`.

Experiments outside the sweep keep their own state and runs root
(`STORE/experiments/<name>/`, `--out-dir`), so they never enter the main tables
or the validation selection; tables go to `results/experiments/<name>/`. Both
are controls for the 8 kHz confound (method in `docs/methods.md`, results in
`docs/findings.md`):
- `scripts/bandlimit_experiment.py`: `embed_perch.py --band-limit 8000`
  (recorded in the cache meta and the run's `band_limit_hz`) and
  `frog-train --manifest <dir>` (a subset manifest).
- `scripts/regime_transfer.py`: `frog-train --train-regimes <list>` trains and
  early-stops on those regimes but scores every bag (recorded as `train_regimes`;
  needs `--tag`). Needs the band-limit experiment's `emb_all_bl8000` cache.

## Storage and IDs

- `src/frog_mil/config.py` holds every path. Large state lives in the group
  workspace, `STORE = /gws/ssde/j25b/iecdt/dash/frogs/frog_mil` (`FROG_STORE`):
  `embeddings/`, `runs/<dataset_id>/`, `manifests/<dataset_id>/`, `archive/`.
  Audio: `/gws/ssde/j25b/iecdt/dash/frogs/data/2019_Rsync` (`FROG_AUDIO_DIR`).
- The repo keeps the manifest (`outputs/`, untracked), `results/` and
  `viz/figures/*.png` (generated **and tracked**; never hand-edit) and
  `archive/<version>/` (tables/figures of old data versions, tracked).
- `dataset_id` = hash of `bags.csv` + the window list. Everything downstream records
  it and refuses to mix: `data.load_bags` checks the embedding cache's
  `instances_id`, runs live under `runs/<dataset_id>/`, `report` asserts it.
- `run_id = {probe}-{target}-s2` (`linear|mlp256`, `bin|ord<w>`; `s2` = contiguous
  5 s windows, kept for continuity). `model_id = {run_id}/{pooling}`. Settings not in
  the run_id (lr, epochs, ...) need `tag = "<run_id>"` in `sweep.toml`.
- Per model: `predictions.npz` (out-of-fold test + val bag scores, all seeds; written
  last, so it marks a finished model), `windows.npz` (window logits + pooling
  weights), `metrics.json` (per seed × fold), `seed{k}_fold{f}.pt`. Load a checkpoint
  with `runs.load_model`.

## Data facts

- Bag = 1 clock hour; instance = contiguous 5 s window, 12 per 1 min clip.
  Three recording **regimes** (`bags.csv: regime`):
  `8k-3clip` 2018-09-01 → 10-24, 8 kHz, clips at :00/:20/:40 (36 windows, 1267 h);
  `44k-1clip` 2018-11-14 → 12-16, one clip at :00 (12 windows, 762 h);
  `44k-2clip` 2019-02-27 → 04-17 (all silent) and 2019-09-23 → 12-03, :00 and :30
  (24 windows, 2885 h). 4914 hours, 123,876 windows.
- **Regime is confounded with the labels**: *Gastrotheca* is positive in 23% of
  8 kHz hours vs ~4% of 44.1 kHz hours (291 of its 429 positives). 8 kHz windows
  have nothing above 4 kHz and are ~10 dB louder. Always read `per_regime` results.
- Labels: hourly 0–3 calling index per species; target = presence (index > 0);
  the index enters through `OrdinalHead` (σ(s) = P(index ≥ 1)) and per-index
  evaluation. **Annotators heard only the recorded clips**, so positive bags
  contain a call. Species are multi-label.
- **Cross-validation**: 3-day blocks dealt to 5 folds, per regime, balanced on
  positives (`bags.csv: block, fold`). Model f: test fold f, val fold f+1, train
  the rest. Every bag has one out-of-fold test score per seed.
- Labels exist for 2019-01, 2019-04-17 → 09-22 and 2020, but that audio isn't here.

## Pitfalls

- **Contiguous windows only.** `embed_perch.py` and `embcache.check_matches` reject
  overlapping (2.5 s-hop) manifests and caches.
- **The embedding cache is keyed by `instance_id` + the WAV's size:mtime.** After a
  manifest change `embed` re-keys and embeds only missing rows or replaced files.
  Never copy rows by position. A partial cache is rejected (missing rows are zeros).
- **Audio QC**: `frog-manifest` QCs every WAV (`audio.qc_wav`, cached in
  `outputs/qc_cache.csv`) and drops hours with a truncated or zero-filled clip.
- **The audio is on NFS.** Move new audio into the audio dir (ingest `merge` does);
  a memmapped file can't be deleted while open.
- **Keep TF batch shapes constant** (one clip's windows per call). Retracing drops
  throughput from ~630 to ~2 windows/s.
- **Masks:** bags are padded to 36 windows; every pooler must ignore padding (tested).
- **Metrics:** AP, not accuracy. CIs are a block bootstrap (3-day blocks within
  regimes); compare models only through paired deltas. The reference model is
  chosen on **validation**; never select on test.
- **Genus-anchored logit patterns** (`^Oreobates `): the bare epithet also
  matches the bee-eater *Merops oreobates*.
- **Numpy indexing:** `a[:, rows, :, c]` (an index array and a scalar separated
  by a slice) moves the indexed axes to the front. Use `a[:, rows][..., c]`.
- `frog status` says "trained by older training code" whenever `data.py`,
  `models.py`, `pooling.py` or `train.py` changes, even if training is
  unaffected. The sweep runs predate the option-only changes of 2026-10-07 and
  were checked to reproduce bit for bit; retrain only if the training itself
  changed.
- `docs/findings.md` carries `<!-- dataset_id: ... -->`; `frog status` warns when
  it no longer matches the results. Update the prose, then the marker.
