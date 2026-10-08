<!-- dataset_id: ec7b516efa -->
# Findings

Dataset `ec7b516efa`: 4914 labelled hours from 2018-09-01 to 2019-12-03 in
three recording regimes (see [methods](methods.md#data)): `8k-3clip` (Sep–Oct
2018, 1267 h), `44k-1clip` (Nov–Dec 2018, 762 h) and `44k-2clip` (Feb–Apr and
Sep–Dec 2019, 2885 h). There are 429 *Gastrotheca* and 581 *Oreobates*
positive hours. Evaluation is 5-fold cross-validation over 3-day blocks, so
**every hour is scored once, out of fold**, by a model that never saw it. The
reference model (◆) is chosen on validation. Full tables are in
[`results/RESULTS.md`](../results/RESULTS.md) and figures in
[`viz/figures/`](../viz/figures/index.md). CIs are a block bootstrap (3-day
blocks, resampled within regimes). "Tied" means the paired 95% CI of the
difference includes 0. There is no multiple-comparison correction.

All 35 sweep models (7 runs × 5 poolers) come from the same training code; two
runs trained before a mid-session code change were retrained on 2026-10-07.
Later changes only added options (`--band-limit`, `--manifest`,
`--train-regimes`). A rerun with the new code reproduces the reference
model's predictions bit for bit, so the "older training code" notes in
`frog status` are harmless.

## Pooling comparison

- **Reference (val-selected): MLP-256 probe + LME pooling, ordinal loss
  w = 0.5**, macro AP 0.856 [0.813, 0.887] (`mlp256-ord0.5-s2/lme`;
  *G.* 0.837, *O.* 0.875). **The best on test** is `linear-bin-s2/mean`, 0.857
  [0.814, 0.890]. The two are tied: +0.001 [−0.013, +0.014].
- **Every trained model beats every baseline** by ≥ 0.34 macro AP. The clock
  (hour × month) scores 0.445, Perch zero-shot (congeneric) 0.397, and
  zero-shot 'Frog' 0.358. Zero-shot stays informative for *Gastrotheca* (0.692),
  but trails the reference there by −0.146 [−0.196, −0.102].
- **14 models are tied with the reference**: all 7 mean models, all 5
  linear-probe LME models, `mlp256-ord0.5-s2/attention` and
  `mlp256-ord0.5-s2/max`. The other 20 are significantly worse: every other max
  model, every linear-softmax model, every other attention model and
  `mlp256-bin-s2/lme`.
- **LME and mean lead, max and linear-softmax trail.** LME and mean are the top
  two by point estimate in all 5 linear-probe runs. In both MLP runs, attention
  is second. Ranges with the linear probe: mean 0.844–0.857, LME 0.846–0.855, max
  0.812–0.836, linear-softmax 0.811–0.830, attention 0.809–0.825.
- **The species pull in opposite directions.** Against the reference, every
  mean model is higher on *Gastrotheca* (+0.013 to +0.023, significant in 3) and
  significantly lower on *Oreobates* (−0.021 to −0.042, all 7). Linear-softmax
  is tied on *Gastrotheca* for the linear probe and loses −0.062 to −0.099 on
  *Oreobates*. So *Oreobates* still separates the poolers, and the CIs for
  *Gastrotheca* are wider.

## Probe and ordinal loss

- **The MLP-256 probe helps only attention**: +0.026 [+0.011, +0.039] (binary)
  and +0.028 (ordinal). It hurts mean (−0.012, −0.011), binary linear-softmax
  (−0.040) and binary LME (−0.017). The other 4 of 10 pairings are tied. The
  MLP fits train harder (train − val gap 0.061–0.108, against 0.024–0.054 for
  linear non-attention models) without testing better.
- **The ordinal loss helps peaked poolers and hurts flat ones.** 13 of 25
  ordinal − binary pairings are significantly positive. Max gains at every
  weight (+0.014 to +0.028), attention at w ≤ 1 (+0.009 to +0.016, MLP +0.017),
  LME at w ≤ 0.5 (+0.009, +0.010, MLP +0.027), and MLP linear-softmax +0.025.
  4 are significantly negative: mean (−0.006, −0.013) and linear-softmax
  (−0.014, −0.019), both at w = 1 and 2. Mean never gains.
- **The thresholds move most under linear max** (b₃ up to 3.25 from its
  initial 2.0), and barely under the MLP (b₃ ≤ 2.37) and attention (≤ 2.28).

## By calling index

- **Isolated calls (index 1) are the hard case** for both species: AP is
  0.37–0.50 for *Gastrotheca* and 0.33–0.56 for *Oreobates* (chance 0.03–0.04).
  Chorus hours (index 3) reach 0.98–0.99 for *Gastrotheca* with mean pooling.
- **Mean is best on *Gastrotheca* at every index** (index 1 up to 0.497,
  index 2 up to 0.826). **MLP probes are best on *Oreobates* index 1**: 0.556
  (`mlp256-bin-s2/lme`), 0.546 (reference and `mlp256-bin-s2/attention`).
  Linear linear-softmax is worst there (0.33–0.43).
- Max is weakest on *Gastrotheca* index 2–3 (0.65–0.75 and 0.74–0.90).

## Per regime: the 8 kHz confound

*Gastrotheca* is positive in 23% of the 8 kHz hours, against 2% (Nov–Dec 2018)
and 4% (2019) of the 44.1 kHz hours, and 291 of its 429 positives are 8 kHz.
8 kHz windows have nothing above 4 kHz and are ~10 dB louder. A score that knew
only the regime would get *Gastrotheca* AP 0.188 (chance 0.087) and
*Oreobates* 0.129 (chance 0.118).

| regime (G pos / hours) | *G.* chance | *G.* AP, all models | *O.* chance | *O.* AP, all models |
|---|---|---|---|---|
| `8k-3clip` (291 / 1267) | 0.23 | 0.83–0.91 | 0.08 | 0.72–0.87 |
| `44k-1clip` (14 / 762) | 0.02 | 0.65–0.92 | 0.13 | 0.77–0.90 |
| `44k-2clip` (124 / 2885) | 0.04 | 0.67–0.75 | 0.13 | 0.75–0.88 |

- **Within every regime, AP is far above that regime's chance**, so the models
  are not just detecting the recording format. The pooled AP (0.79–0.86 for
  *Gastrotheca*) is not mainly regime.
- **The 2019 hours are the hard *Gastrotheca* set**: 0.67–0.75 with 124
  positives. Zero-shot congeneric scores 0.514 there and 0.789 on 8 kHz hours,
  so the 8 kHz calls are easier for Perch too.
- **The 8 kHz negatives score high.** For the reference, the 95th
  percentile of *Gastrotheca* scores on silent hours is 0.86 at 8 kHz, against
  0.28 (2019) and 0.01 (Nov–Dec 2018). 8 kHz hours make up 72% of the top 429
  *Gastrotheca* scores and 68% of the positives. The controls below point to
  calls in those hours rather than a format shortcut.
- **Small cells have wide CIs.** *Gastrotheca* `44k-1clip` has 14 positives
  (CIs ≈ 0.5–1.0). *Oreobates* `8k-3clip` positives come mostly from a few
  late-October nights, so resamples that drop them give lower CI bounds as low as 0.09.
- **Pooler differences differ by regime.** On *Oreobates* `44k-1clip`
  (12-window bags), LME and max reach 0.89–0.90 and mean 0.77–0.84.

## Controls for the 8 kHz confound

Two experiments retrain the reference model (`mlp256-ord0.5-s2/lme`) with the
main folds and seeds and score every hour out of fold (method in
[methods](methods.md#controls-for-the-recording-regime)). Their models are kept
out of the main sweep. Their all-regimes, as-recorded model reproduces the
reference exactly (macro AP 0.856).

### Band-limit control

[`results/experiments/bandlimit_8k/`](../results/experiments/bandlimit_8k/SUMMARY.md):
every 44.1 kHz clip resampled to 8 kHz before embedding, as the 8 kHz
recordings are.

- **Band-limiting changes nothing measurable.** On all 4914 hours, macro AP is
  0.856 recorded and 0.853 band-limited: −0.003 [−0.016, +0.007]. Every
  per-species and per-regime difference is tied.
- **The audio above 4 kHz isn't needed.** On the 44.1 kHz hours alone (no
  8 kHz audio anywhere), band-limiting gives +0.004 [−0.012, +0.019].
- **Silent 8 kHz *Gastrotheca* hours still score high**: their 95th-percentile
  score is 0.86 both ways. 8 kHz hours are 69% of the top-ranked hours with
  band-limiting, against 72% without and 68% of the positives.
- **But band-limiting doesn't hide the format.** A linear classifier on single
  window embeddings tells 8 kHz from 44.1 kHz windows with AUC 1.000 before and
  after band-limiting. It stays at 1.000 comparing only Sep–Oct 2018 with
  Sep–Oct 2019, so the cue isn't the season. It is something else in the
  recording chain (recorder, gain, noise floor, filter) or the year. So this
  control alone can't rule out a format shortcut.

### Cross-regime transfer

[`results/experiments/regime_transfer/`](../results/experiments/regime_transfer/SUMMARY.md):
the model is trained and early-stopped on the 44.1 kHz hours only or on the
8 kHz hours only, and scores the other regime's hours without having heard that
format. A transfer model can't use the format as a shortcut. Δ is against the
all-regimes model on the same embeddings.

| *Gastrotheca* AP | 8 kHz (chance 0.23) | 44.1 kHz 2019 (chance 0.04) |
|---|---|---|
| trained on all regimes | 0.889 | 0.715 |
| trained on 44.1 kHz only | **0.843** (−0.046*) | 0.685 (−0.029) |
| trained on 8 kHz only | 0.861 (−0.027) | **0.541** (−0.173*) |
| zero-shot congeneric | 0.789 | 0.514 |

Bold = transfer; `*` = CI excludes 0; recorded embeddings.

- **A model that never heard 8 kHz audio still ranks the 8 kHz *Gastrotheca*
  hours well**: AP 0.843 (0.851 band-limited) against 0.889 for the
  all-regimes model, with chance at 0.23. *Oreobates* transfers too: 0.788
  (0.813 band-limited) against 0.845, with chance at 0.08. The 8 kHz result
  is not a format shortcut.
- **It flags the same silent 8 kHz hours.** On the 976 silent 8 kHz
  *Gastrotheca* hours, its scores correlate with the reference's at ρ = 0.85
  (0.87 band-limited). 73% (80%) of its top 5% are also in the reference's top
  5%, where chance is 5%.
- **Those hours sit next to labelled calls.** Of the top 5% silent 8 kHz
  *Gastrotheca* hours, 61–71% have a labelled positive hour within ±1 h, for
  every model including the transfer ones. Among all silent 8 kHz hours the
  rate is 14%. So the high-scoring silent hours are most likely hours with
  calls that the labels miss, or faint calls at the edge of a calling bout,
  not a format artefact. Listening to them would confirm it.
- **Training on the other regime helps.** Without the 8 kHz hours, *Oreobates*
  AP on the 44.1 kHz hours drops by −0.048* (Nov–Dec 2018) and −0.026* (2019).
  *Gastrotheca* 2019 drops by −0.029, not significant. The 8 kHz hours teach the
  model about calls, not only about the format.
- **Transfer from 8 kHz to 44.1 kHz is weaker.** The 8 kHz-only model reaches
  *Gastrotheca* 0.541 and *Oreobates* 0.643 on the 2019 hours (−0.173*,
  −0.236*), though still far above chance (0.04, 0.13). Band-limiting the
  44.1 kHz hours to match recovers part of that (0.597, 0.712). The 8 kHz data
  are one season with 101 *Oreobates* positives, and part of the gap is the
  missing high band.

## Validation vs test

- **Validation now ranks models like test**: Spearman ρ = 0.91 over 35 models
  (figure 09). The val-selected reference is tied with the test-best model.
- **Validation is optimistic by a small, uniform amount.** Fold-mean val AP
  exceeds fold-mean test AP by 0.005–0.018 for every model. The pooled
  out-of-fold AP is a little lower again, because each fold model's scores are
  on its own scale.

## Recall at fixed precision (cutoff chosen on val)

- **At P = 0.8 the reference recalls 0.785** [0.735, 0.824] of positive hours
  (*G.* 0.747, *O.* 0.823), the highest of all models. 6 are tied with it: the
  other MLP LME and attention models, and linear LME at w = 0, 0.25 and 0.5.
- **At P = 0.9, mean leads** (0.630–0.645, tied with the reference at 0.625).
  Max and linear-softmax drop to 0.38–0.58.
- **Val cutoffs transfer well for *Gastrotheca*, less so for *Oreobates*.**
  Test precision at P = 0.8 is 0.77–0.80 (*G.*) and 0.67–0.76 (*O.*); at
  P = 0.9 it is 0.85–0.88 and 0.78–0.88. In v3 the *Gastrotheca* range was
  0.55–0.88.

## Changes from v3 (one 576-hour test split)

v3 is archived in [`archive/v3_nov2018-dec2019/`](../archive/v3_nov2018-dec2019/findings.md).
This version adds 1267 hours of 8 kHz audio (Sep–Oct 2018, with 291
*Gastrotheca* positives) and replaces the single split with 5-fold
cross-validation. The test sets differ, so **v3 and this version can't be
compared as paired deltas**.

| | v3 | ec7b516efa |
|---|---|---|
| hours | 3594 | 4914 |
| evaluated hours (G / O positives) | 576 test (20 / 77) | 4914 out of fold (429 / 581) |
| model reported first | `linear-bin-s2/lme` (test-best) | `mlp256-ord0.5-s2/lme` (val-selected) |
| its macro AP | 0.829 [0.751, 0.893] | 0.856 [0.813, 0.887] |
| test-best macro AP | 0.829 | 0.857 (`linear-bin-s2/mean`) |
| models tied with it | 12 | 14 |
| clock baseline | 0.366 | 0.445 |
| val/test rank correlation ρ | 0.66 | 0.91 |

- **CIs are half as wide** (macro AP width 0.074 against 0.142), because every
  positive is now tested.
- **Conclusions that changed:** mean rose from third to joint first. Max is now
  significantly behind, where it was tied. The ordinal loss helps max, attention
  and LME at small w again, where v3 found no gain. The MLP probe has
  significant effects in both directions, where v3 found none. And validation
  now picks a model tied with the test-best, where v3's pick was significantly
  worse.
- **Conclusions that held:** every model beats the baselines by a wide margin,
  LME is in the top tier, linear-softmax and linear attention trail, and
  *Oreobates* is what separates the poolers.
- The training data, the evaluation and the regime mix all changed together,
  so none of these shifts can be attributed to one of them.

## Next steps

1. **Listen to the top-scored silent 8 kHz *Gastrotheca* hours** (the top 5%
   in `results/experiments/regime_transfer/`, flagged by every model and mostly
   within an hour of a labelled call). If they hold calls, relabel them; that
   also removes most of the apparent 8 kHz confound. Do the same for the
   top-weighted windows of isolated-call hours (`frog_mil.audio.read_segment`,
   weights in `windows.npz`), to separate missed labels from confusion.
2. **Pull the 693 pending Dropbox files** (8 kHz, 2018-10-24 23:00 → 11-03,
   `Helechos_09_11_2018 Martin`) with `scripts/ingest.py pull/verify/merge`,
   then `frog run`. They fill the gap between the two 2018 regimes.
3. **Find the audio for labelled periods without it**: 2018-12-16 → 2019-02-26,
   2019-04-17 → 09-22, and 2020. More 44.1 kHz *Gastrotheca* positives would
   help most: 2019 is its hardest regime (124 positives).
4. **1 s instances** from Perch's pre-pooling embeddings `(5, 3, 1536)`: a finer
   bag from the same forward pass. Isolated calls are where it could help.
