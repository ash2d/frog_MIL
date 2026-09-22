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
scripts/perch_env.sh python scripts/embed_perch.py  # embeddings/ (+ zero-shot logits), resumable
.venv/bin/frog-train --seeds 5 [--hidden 256] [--ordinal-weight 0.5] [--stride 1]
.venv/bin/frog-report                               # results/ from runs/
.venv/bin/python viz/make_figures.py                # viz/figures/ from runs/
```

Run sweeps in parallel with `CUDA_VISIBLE_DEVICES=k`, logging to
`outputs/logs/<run_id>.log`. After every new run, rerun `frog-report` and
`viz/make_figures.py`.

## IDs and outputs

- `run_id = {probe}-{target}-s{stride}` (`linear|mlp256`, `bin|ord<w>`, `s2|s1`).
  `model_id = {run_id}/{pooling}`. Both are derived from the config.
- `runs/<model_id>/` holds `predictions.npz` (test scores, all seeds),
  `seed{k}.pt` and `metrics.json`. `runs/<run_id>/config.json` holds the config.
- Changing a setting that isn't in the run_id (lr, epochs, dropout, …) needs
  `--tag <new_run_id>`. `train.py` refuses to overwrite a run whose config differs.
- `results/` and `viz/figures/*.png` are generated **and tracked**. Never
  hand-edit them. `runs/`, `embeddings/` and `outputs/` are not tracked.

## Data facts

- Bag = 1 hour = 2 × 1 min clips (44.1 kHz) = 24 contiguous 5 s windows.
- Labels are an hourly 0–3 calling index per species. **Annotators heard only
  the recorded clips**, so positive bags always contain a call.
- The target is presence (index > 0). The index enters through `OrdinalHead`,
  where σ(s) = P(index ≥ 1), and through per-index evaluation.
- Species are multi-label. *Oreobates* calls only Oct–Dec.
- Each pooler trains its own probe (same architecture and init per seed).

## Pitfalls

- **The hop must divide the window.** The cache is at 2.5 s, and
  `win_idx % 2 == 0` gives the exact contiguous 5 s windows.
- **Rebuilding the manifest after new audio changes the splits.** Re-embed,
  rerun all runs, then `frog-report`. `report.py` asserts that every model shares
  one test set.
- **A partial embedding cache is rejected** by `data.py`, because missing rows
  would be zeros.
- **Keep TF batch shapes constant** (23 windows per clip). Retracing drops
  throughput from ~630 to ~2 windows/s.
- **Masks:** bags are padded, and every pooler must ignore padded windows.
- **Metrics:** use AP, not accuracy (7% / 29% prevalence). Compare models only
  through paired deltas (`results/vs_best.csv`, `effect_*.csv`). Test has 18
  *Gastrotheca* positives, and val ranks models differently from test, so
  don't select on test.
- **Genus-anchored logit patterns** (`^Oreobates `): the bare epithet also
  matches the bee-eater *Merops oreobates*.
- **Numpy indexing:** `a[:, rows, :, c]` (an index array and a scalar separated
  by a slice) moves the indexed axes to the front. Use `a[:, rows][..., c]`.
