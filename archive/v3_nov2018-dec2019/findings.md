# Findings

Test set: 576 held-out hours (20 *Gastrotheca* and 77 *Oreobates* positives).
The audio runs from 2018-11-14 to 2018-12-16 (one clip per hour), 2019-02-27
to 2019-04-15 and 2019-09-23 to 2019-12-03 (two clips per hour). **Every test
hour is from 2019**: 216 Feb–Apr hours, when neither frog calls, and 360
Sep–Dec hours. The 2018 hours went to train and val only (see
[2018 vs 2019 hours](#2018-vs-2019-hours-val)). Full tables are in
[`results/RESULTS.md`](../results/RESULTS.md) and figures in
[`viz/figures/`](../viz/figures/index.md). "Tied" means the paired 95% CI of
the difference includes 0. There is no multiple-comparison correction.

## Pooling comparison

- **Every trained model beats every baseline** on macro AP. The best baseline
  is Perch zero-shot (congeneric) at 0.389. The clock (hour × month) scores 0.366.
- **Best: linear probe + LME pooling, binary loss**, 0.829 [0.751, 0.893]
  (`linear-bin-s2/lme`). 12 other models are tied with it, all of them LME,
  max or mean: 3 of the other 6 LME models, 5 of 7 max and 4 of 7 mean. Every
  attention model (−0.052 to −0.106) and every linear-softmax model (−0.078 to
  −0.169) is significantly worse.
- **The pooler order is nearly fixed.** By point estimate it is LME > max >
  mean > attention > linear-softmax in 6 of the 7 runs. In `mlp256-ord0.5-s2`,
  mean edges out max. Macro AP ranges with the linear probe: LME 0.79–0.83, max
  0.78–0.82, mean 0.77–0.80, attention 0.72–0.75, linear-softmax 0.66–0.69.
- ***Oreobates* separates the poolers.** *Gastrotheca* AP is 0.75–0.86 for every
  model, and no linear-softmax model differs significantly from the best on it.
  On *Oreobates*, linear-probe linear-softmax collapses to 0.54–0.55, against
  0.75–0.80 for linear LME.
- **The MLP-256 probe makes no significant difference** in any of the 10
  pairings (Δ −0.019 to +0.065). Its largest gains, for linear-softmax and
  attention, have CIs that include 0.
- **Perch zero-shot stays informative for *Gastrotheca*** (congeneric logits
  0.644), but it trails the best model by −0.219 [−0.398, −0.054] there.

## Ordinal loss (linear probe, w ∈ {0, 0.25, 0.5, 1, 2})

- **It no longer helps.** None of the 25 ordinal − binary pairings is
  significantly positive. Five are significantly negative, all at w ≥ 1 for the
  linear probe or at w = 0.5 for the MLP: linear LME at w = 1 (−0.030) and
  w = 2 (−0.041), linear mean (−0.035) and linear-softmax (−0.026) at w = 2,
  and MLP max at w = 0.5 (−0.025).
- Attention has the only positive point estimates (+0.020 at w = 0.25, +0.009
  at w = 0.5), and neither is significant.
- The thresholds still move under max and mean (b₃ up to 3.08 from its initial
  2.0), and barely under linear attention (b₃ ≤ 2.13).

## Model selection is still unreliable

- **Validation and test agree better than in v2, but not enough.** Across
  models, the rank correlation between val and test macro AP is ρ = 0.66
  (figure 09; it was 0.30 in v2). Validation prefers MLP models. The model it
  would pick, `mlp256-bin-s2/attention` (val 0.842), is significantly worse on
  test than the test-best: −0.052 [−0.090, −0.019].
- There are 23 *Gastrotheca* positives in val and 20 in test, all from a few
  nights.

## Train vs val vs test AP

Every checkpoint is rescored on all three splits (`scripts/split_ap.py`, which
writes `outputs/split_ap.csv`). The numbers are seed means without CIs, so they
describe the models; they don't test differences.

- ***Oreobates* val is no longer uniformly easier than test.** Val − test
  *Oreobates* AP ranges from −0.084 to +0.113, and val is lower for 16 of the 35
  models. *Gastrotheca* is 0.74–0.88 on val and 0.75–0.86 on test.
- **Linear max and LME have the smallest train − val gaps.** Their macro gaps are
  0.06–0.09 (max) and 0.08–0.11 (LME), against 0.12–0.14 for linear mean,
  0.13–0.17 for linear attention and 0.19–0.20 for linear-softmax. Linear max
  and linear-softmax have the lowest *Gastrotheca* train AP (0.76–0.81).
- **The MLP probe fits train harder without testing better.** MLP
  *Gastrotheca* train AP is 0.81–0.94, against 0.76–0.84 for the linear probe.
  MLP attention reaches 0.92–0.94 on train and 0.78 on test.

## 2018 vs 2019 hours (val)

A 2018 hour has one 1 min clip (12 windows), and its label covers only that
minute. A 2019 hour has two clips (24 windows). `scripts/year_ap.py` rescores
every checkpoint and computes AP on each year's hours separately, with a
bootstrap CI over that year's bags (`outputs/year_ap.csv`). **Test holds no
2018 hours, so this runs on val.** Val picks each seed's checkpoint, so these
APs are optimistic for both years.

| val hours | bags | *G.* positives (chance AP) | *O.* positives (chance AP) |
|---|---|---|---|
| 2018 (Nov 144, Dec 144) | 288 | 8 (0.028) | 42 (0.146) |
| 2019 (Apr 72, Oct 72, Nov 144) | 288 | 15 (0.052) | 36 (0.125) |

- ***Oreobates* scores higher on the 2018 hours for every model**: 2018 − 2019
  AP is +0.148 to +0.439, and the CI excludes 0 for all 35. 2018 AP is
  0.86–0.92 and 2019 AP is 0.45–0.77. The test-best `linear-bin-s2/lme` has the
  smallest gap, 0.918 vs 0.770. Linear-softmax has the largest gaps, because it
  fails on the 2019 hours (0.45–0.46 with the linear probe).
- ***Gastrotheca* shows no consistent difference**: −0.273 to +0.130, and no CI
  excludes 0. With 8 positive 2018 hours, the 2018 CIs are wide (median width
  0.33, up to 0.59).
- **Half the audio did not make 2018 hours harder to score.** For *Oreobates*
  they are easier, even though 28 of the 42 positive 2018 hours are index 1
  (isolated calls), against 14 of 36 in 2019. This analysis can't say why. The
  two years' val hours come from different nights. A 12-window silent hour
  gives max and LME half as many windows to fire on noise. And val also chose
  the checkpoints.

## Recall at fixed precision (cutoff chosen on val)

- ***Gastrotheca*: val cutoffs don't transfer.** At P = 0.8, test precision is
  0.55–0.88 and only 1 of 35 models reaches 0.8. At P = 0.9, 9 of 35 reach it
  (test precision 0.63–1.00). With the cutoff chosen on test, recall at
  precision ≥ 0.8 reaches 0.77 (`mlp256-bin-s2/mean`). It is 0.76 for the
  test-best `linear-bin-s2/lme`.
- ***Oreobates*: the ranking ceiling is low.** With the cutoff chosen on test,
  recall is at most 0.66 at P ≥ 0.8 and 0.45 at P ≥ 0.9. Val cutoffs give test
  precision 0.53–0.87 at P = 0.8 (6 of 35 models reach it) and 0.49–0.95 at
  P = 0.9 (8 of 35). Linear-probe linear-softmax never reaches either
  precision on *Oreobates* val, so it gets no cutoff and zero test recall.
- Table: [`results/recall_at_precision.csv`](../results/recall_at_precision.csv).

## Changes from v2 (Feb–Apr + Sep–Dec 2019)

v2 used 2832 hours from 2019 and a 432-hour test set. The archived tables and
figures are in
[`results_previous_v2_feb-dec2019/`](../results_previous_v2_feb-dec2019/RESULTS.md)
and [`viz/figures_previous_v2_feb-dec2019/`](../viz/figures_previous_v2_feb-dec2019/index.md).

v3 adds 762 hours from 14 Nov to 16 Dec 2018, recorded as one 1 min clip per
hour (12-window bags). They hold 14 *Gastrotheca* and 101 *Oreobates* positive
hours. Adding data recomputes the splits, so the v2 and v3 test sets differ,
and **v2 and v3 numbers can't be compared as paired deltas**.

| | v2 | v3 |
|---|---|---|
| hours (train / val / test) | 1968 / 432 / 432 | 2442 / 576 / 576 |
| test hours, Feb–Apr / Sep–Dec | 216 / 216 | 216 / 360 (no 2018) |
| test positives (G / O) | 19 / 58 | 20 / 77 |
| best model | `linear-ord2-s2/max` | `linear-bin-s2/lme` |
| best macro AP | 0.859 [0.784, 0.925] | 0.829 [0.751, 0.893] |
| best model's *G. chrysosticta* AP | 0.886 | 0.863 |
| best model's *O. berdemenos* AP | 0.833 | 0.796 |
| models tied with the best | 2 | 12 |
| clock baseline, macro AP | 0.482 | 0.366 |
| val/test rank correlation ρ | 0.30 | 0.66 |

- **The training data and the test nights both changed**, so the shifts below
  can't be attributed to the 2018 hours alone.
- **Conclusions that changed:** LME is now first, with max tied, where v2 had
  max clearly ahead and every LME model significantly behind. The ordinal loss
  no longer helps max or linear-softmax and hurts some poolers at w ≥ 1.
  Linear-softmax dropped to last, through its *Oreobates* AP. Attention no
  longer loses from the ordinal loss.
- **Conclusions that held:** a linear probe is enough, and the MLP-256 probe
  doesn't help. Every model beats the baselines, attention trails the best, and
  validation can't pick the pooler: it again picks a model that is
  significantly worse on test.

## Earlier: changes from v1 to v2 (adding Feb–Apr 2019)

v1 used 1707 hours from 2019-09-23 to 2019-12-03 and a 216-hour test set. The
archived v1 tables and figures are in
[`results_previous_v1_sep-dec2019/`](../results_previous_v1_sep-dec2019/RESULTS.md)
and [`viz/figures_previous_v1_sep-dec2019/`](../viz/figures_previous_v1_sep-dec2019/index.md).

v2 adds 1125 hours from 27 Feb to 15 Apr 2019. Every one of them is silent for
both species. Adding data recomputes the splits, so the v1 and v2 test sets
differ, and **v1 and v2 numbers can't be compared as paired deltas**. The v2
numbers below are in the v2 archive.

| | v1 | v2 |
|---|---|---|
| hours (train / val / test) | 1203 / 288 / 216 | 1968 / 432 / 432 |
| test positives (G / O) | 18 / 62 | 19 / 58 |
| best model | `linear-ord2-s2/max` | `linear-ord2-s2/max` |
| best macro AP | 0.809 [0.707, 0.894] | 0.859 [0.784, 0.925] |
| best *G. chrysosticta* AP | 0.662 | 0.886 |
| best *O. berdemenos* AP | 0.956 | 0.833 |

- **The silent Mar–Apr hours barely move test AP.** Scored on the 216 Sep–Dec
  test hours alone, every model's macro AP is 0.000–0.013 higher than on the
  full test set. The models rank those hours low easily.
- **The rest of the change comes from which Sep–Dec nights ended up in test.**
  *Oreobates* got harder for everything, including the clock baseline, which
  uses no audio. *Gastrotheca* rose for every model, but with 19 positives its
  CIs are wide.
- **Conclusions that changed:** LME is no longer tied with max. The ordinal loss
  now helps linear-softmax and max at w = 2 and hurts attention. Zero-shot Perch
  is no longer tied on *Gastrotheca*.
- **Conclusions that held:** max pooling with a linear probe is best, the
  MLP-256 probe doesn't help, every model beats the baselines, and validation
  can't pick the pooler.

## Next steps

1. **Cross-validation over day-blocks**, so every positive is tested once,
   including the 2018 hours. This addresses the val/test disagreement, the
   20-positive test set and the missing 2018 test hours.
2. **Separate bag size from year.** Score the 2019 hours from their first clip
   only (12 windows) and compare with the 2018 hours. Note that the 2019 labels
   still cover both clips.
3. **Listen to the top-weighted windows** in positive hours and in high-scoring
   silent hours (`frog_mil.audio.read_segment`), to separate missed labels
   from confusion.
4. **1 s instances** from Perch's pre-pooling embeddings `(5, 3, 1536)`: a finer
   bag from the same forward pass.
5. **More in-season audio.** Off-season hours add only easy negatives; the
   test set needs more calling nights.

## Adding 8 kHz recordings (1 min every 20 min)

More positives are what this study needs most, but mixing recording formats
without controls would confound the results:

1. **Bandwidth.** 8 kHz audio has nothing above 4 kHz, so its embeddings differ
   systematically. The probe could learn the recording format as a shortcut.
2. **Bag size.** 3 clips per hour gives 36 windows. Max pooling then gets more
   chances to fire on noise. The 2018 hours already mix 12-window bags with the
   24-window 2019 bags (see [2018 vs 2019 hours](#2018-vs-2019-hours-val)).
3. **Prevalence.** 3 sampled minutes catch more calls than 2, so the base rate
   per bag differs.

Before mixing them in, downsample the 2019 audio to 8 kHz and rerun the sweep
to measure the cost of the lower bandwidth. Then band-limit all audio the same
way, record the regime per bag, split so each regime appears in train and test,
and report AP per regime.
