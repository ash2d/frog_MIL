# frog_MIL — working notes

MIL detection of two frogs from hourly labels: frozen Perch v2 embeddings, then
a per-window probe, then a pooling function. The overview is in `README.md`, the
design in `docs/methods.md`, results in `docs/findings.md` and `results/RESULTS.md`.

## Environments

Use two venvs, because TF and torch ship conflicting CUDA stacks. Embeddings
are cached on disk, so the two are never needed together.

```bash
uv sync                                                         # .venv: torch, training, report, viz
UV_PROJECT_ENVIRONMENT=.venv-perch uv sync --only-group perch   # .venv-perch: TF + perch_hoplite
```

**Always run Perch through `scripts/perch_env.sh`.** Without it TF can't find
the pip-installed CUDA libraries and silently runs on CPU, about 50× slower. The
wrapper pins GPU 0 unless `CUDA_VISIBLE_DEVICES` is set. The GPUs are shared,
so check `nvidia-smi`.

On this machine, the audio is at `/gws/ssde/j25b/iecdt/dash/frogs/data/2019_Rsync`
(the `--audio-dir` default).

## Pipeline

```bash
.venv/bin/python -m frog_mil.annotations            # coverage report; rerun after new audio
.venv/bin/frog-manifest                             # outputs/bags.csv, instances.csv, summary.json
scripts/perch_env.sh python scripts/embed_perch.py  # embeddings/ (+ zero-shot logits), incremental
.venv/bin/frog-train --seeds 5 [--hidden 256] [--ordinal-weight 0.5]
.venv/bin/frog-report                               # results/ from runs/
.venv/bin/python viz/make_figures.py                # viz/figures/ from runs/
.venv/bin/python scripts/recall_at_precision.py     # results/recall_at_precision.csv
.venv/bin/python scripts/split_ap.py                # outputs/split_ap.csv (rescores checkpoints)
```

Run sweeps in parallel with `CUDA_VISIBLE_DEVICES=k`, logging to
`outputs/logs/<run_id>.log`. After every new run, rerun `frog-report` and
`viz/make_figures.py`.

## IDs and outputs

- `run_id = {probe}-{target}-s2` (`linear|mlp256`, `bin|ord<w>`). `s2` = contiguous
  5 s windows, the only geometry; it stays in the ID to match earlier runs.
  `model_id = {run_id}/{pooling}`. Both are derived from the config.
- `runs/<model_id>/` holds `predictions.npz` (test scores, all seeds),
  `seed{k}.pt` and `metrics.json`. `runs/<run_id>/config.json` holds the config.
- Changing a setting that isn't in the run_id (lr, epochs, dropout, …) needs
  `--tag <new_run_id>`. `train.py` refuses to overwrite a run whose config differs.
- `results/` and `viz/figures/*.png` are generated **and tracked**. Never
  hand-edit them. `runs/`, `embeddings/` and `outputs/` are not tracked.

## Data facts

- Bag = 1 hour = 2 × 1 min clips (44.1 kHz) = 24 contiguous 5 s windows.
- Audio covers 2019-02-27 → 04-15 (all silent, off-season) and 2019-09-23 → 12-03.
- Labels are an hourly 0–3 calling index per species. **Annotators heard only
  the recorded clips**, so positive bags always contain a call.
- The target is presence (index > 0). The index enters through `OrdinalHead`,
  where σ(s) = P(index ≥ 1), and through per-index evaluation.
- Species are multi-label. In the audio, *Gastrotheca* calls Sep–Nov and *Oreobates* Oct–Dec.
- Each pooler trains its own probe (same architecture and init per seed).

## Pitfalls

- **Contiguous windows only.** The manifest emits 0–5, 5–10, … s windows
  (12 per clip). `embed_perch.py` and `data.py` reject overlapping (2.5 s-hop)
  manifests and caches.
- **The embedding cache is keyed by `instance_id`, not row position.** After new
  audio, rebuild the manifest and rerun `embed_perch.py`. It re-keys the cache
  into manifest order and embeds only missing rows. Don't copy rows by position.
- **Rebuilding the manifest after new audio changes the splits.** Archive the
  old `runs/`, `results/` and `viz/figures/` (as in `*_previous_v1_sep-dec2019/`),
  rerun all runs, then `frog-report`. `report.py` asserts that every model shares
  one test set.
- **The audio is on NFS, at 94% full.** Move new audio into `--audio-dir` rather
  than copying it. On NFS a memmapped file can't be deleted while open.
- **A partial embedding cache is rejected** by `data.py`, because missing rows
  would be zeros.
- **Keep TF batch shapes constant** (12 windows per clip). Retracing drops
  throughput from ~630 to ~2 windows/s.
- **Masks:** bags are padded, and every pooler must ignore padded windows.
- **Metrics:** use AP, not accuracy (4% / 13% test prevalence). Compare models only
  through paired deltas (`results/vs_best.csv`, `effect_*.csv`). Test has 19
  *Gastrotheca* positives, and val ranks models differently from test, so
  don't select on test.
- **Genus-anchored logit patterns** (`^Oreobates `): the bare epithet also
  matches the bee-eater *Merops oreobates*.
- **Numpy indexing:** `a[:, rows, :, c]` (an index array and a scalar separated
  by a slice) moves the indexed axes to the front. Use `a[:, rows][..., c]`.
