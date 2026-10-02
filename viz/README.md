# viz — figures for the MIL study

Eleven figures plus two slide variants, regenerated from `runs/` (+ `outputs/bags.csv`,
and `embeddings/` for figure 10). Nothing is retrained. The v1 figures (Sep–Dec
audio only) are archived in `viz/figures_previous_v1_sep-dec2019/`.

```bash
.venv/bin/python viz/make_figures.py                          # all runs -> viz/figures/
.venv/bin/python viz/make_figures.py --include 'linear-*' --out viz/figures_linear
.venv/bin/python viz/make_figures.py --focus-run linear-bin-s2 --example-bag HELECHOS_20191104_05
```

Run from the project root. Output is `viz/figures/NN_<name>.{png,pdf}` plus
`index.md`, a gallery with a caption per figure.

| # | figure | shows |
|---|---|---|
| 01 | `bag_structure` | hour → 2 clips → 24 windows; bag label = max of unseen window labels |
| 02 | `pipeline` | frozen Perch v2 → probe → pooling → bag logit (→ ordinal head) |
| 02 | `pipeline_wytham` | slide version of the pipeline for one 1 min recording |
| 03 | `pooling_toy` | how each fixed pooler responds to sparse vs dense calls (runs `frog_mil.pooling`) |
| 04 | `calendar` | calling index per night × hour for both species, with split blocks; gaps > 1 week collapsed |
| 05 | `forest_ap` | test AP [95% CI] for every model and baseline |
| 06 | `effects` | paired Δ macro AP for one-factor changes (probe, ordinal weight) |
| 07 | `ordinal_sweep_*` | AP vs ordinal loss weight per pooler (probes with ≥ 3 weights) |
| 08 | `per_index` | AP by calling index per pooler, in the focus run |
| 09 | `val_vs_test` | validation vs test macro AP (model-selection caveat) |
| 10 | `pooling_example` | one real test hour: each pooler's window probabilities and weights |
| 10 | `pooling_example_wytham` | slide version: one 1 min clip (12 windows) |
| 11 | `forest_simple` | slide forest plot: per-species AP for each pooler of the focus run + baselines |

- **Which models:** by default, every `runs/<run_id>/<pooling>/predictions.npz`
  plus the baselines. `--include`/`--exclude` filter by run_id glob.
- **Best / focus:** "best" is the top test macro AP, as in `frog-report`.
  Figures 08, 10 and 11 use the best model's run unless `--focus-run` is set.
- **CIs:** percentile bootstrap over test bags, with the same scheme and seed as
  `frog-report` (B=2000, seed 0). The numbers match `results/RESULTS.md`.
- **Cache:** bootstrap and inference results go to `<out>/.cache/` (gitignored).
  They are keyed on file mtimes, so new or retrained runs invalidate the cache.

Encoding is the same in every figure. Colour and marker shape show the pooler
(mean blue ●, max orange ▲, LME green ◆, linear-softmax yellow ■, attention
pink ✚). Hollow markers are the MLP probe, and grey diamonds are baselines. The
list and order of figures live in `common.FIGURES`.
