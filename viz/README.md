# viz — figures for the MIL study

Twelve figures plus two slide variants, regenerated from the run store
(`runs/<dataset_id>/` in the group workspace) and `outputs/bags.csv`. Nothing is
retrained. Figures of earlier data versions are in `archive/<version>/figures/`.

```bash
.venv/bin/frog run figures                                    # when stale (or --force)
.venv/bin/python viz/make_figures.py --include 'linear-*' --out /tmp/figs_linear
.venv/bin/python viz/make_figures.py --focus-run linear-bin-s2 --example-bag HELECHOS_20191109_22
```

Output is `viz/figures/NN_<name>.{png,pdf}` plus `index.md`, a gallery with a
caption per figure (PNGs are tracked, PDFs are not).

| # | figure | shows |
|---|---|---|
| 01 | `bag_structure` | hour → clips → 5 s windows; bag label = max of unseen window labels |
| 02 | `pipeline` | frozen Perch v2 → probe → pooling → bag logit (→ ordinal head) |
| 02 | `pipeline_wytham` | slide version of the pipeline for one 1 min recording |
| 03 | `pooling_toy` | how each fixed pooler responds to sparse vs dense calls (runs `frog_mil.pooling`) |
| 04 | `calendar` | calling index per night × hour for both species, with each block's fold and the recording regime; gaps > 1 week collapsed |
| 05 | `forest_ap` | out-of-fold AP [95% CI] for every model and baseline |
| 06 | `effects` | paired Δ macro AP for one-factor changes (probe, ordinal weight) |
| 07 | `ordinal_sweep_*` | AP vs ordinal loss weight per pooler (probes with ≥ 3 weights) |
| 08 | `per_index` | AP by calling index per pooler, in the focus run |
| 09 | `val_vs_test` | validation vs out-of-fold test macro AP (model-selection check) |
| 10 | `pooling_example` | one real held-out hour: each pooler's window probabilities and weights |
| 10 | `pooling_example_wytham` | slide version: the first 1 min clip (12 windows) only |
| 11 | `forest_simple` | slide forest plot: per-species AP for each pooler of the focus run + baselines |
| 12 | `per_regime` | presence AP within each recording regime, per pooler of the focus run |

- **Which models:** every finished model of the current dataset plus the
  baselines. `--include`/`--exclude` filter by run_id glob; `--dataset` picks an
  older dataset in the store.
- **Reference / focus:** the reference model is the validation-selected one, as
  in `frog-report`. Figures 08, 10, 11 and 12 use its run unless `--focus-run` is set.
- **CIs:** `frog_mil.stats` block bootstrap with the report's resamples and cache
  (`runs/<dataset_id>/.cache/`), so the numbers match `results/RESULTS.md`.
- **Window data:** figure 10 reads `windows.npz` (saved at training). Only the
  first-clip variant re-runs the checkpoints; that result is cached too.

Encoding is the same in every figure. Colour and marker shape show the pooler
(mean blue ●, max orange ▲, LME green ◆, linear-softmax yellow ■, attention
pink ✚). Hollow markers are the MLP probe, and grey diamonds are baselines. The
list and order of figures live in `common.FIGURES`.
