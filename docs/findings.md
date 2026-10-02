# Findings

Test set: 432 held-out hours (19 *Gastrotheca* and 58 *Oreobates* positives),
using audio from 2019-02-27 to 2019-04-15 and 2019-09-23 to 2019-12-03. Half
the test hours (216) are Mar–Apr, when neither frog calls. Full tables are in
[`results/RESULTS.md`](../results/RESULTS.md) and figures in
[`viz/figures/`](../viz/figures/index.md). "Tied" means the paired 95% CI of
the difference includes 0. There is no multiple-comparison correction.

## Pooling comparison

- **Every trained model beats every baseline** on macro AP. The best baseline
  is the clock (hour × month) at 0.482.
- **Best: linear probe + max pooling**, 0.859 [0.784, 0.925] (`linear-ord2-s2/max`).
  Only max at w = 1 and w = 0.25 is tied with it. Max at w = 0.5 and binary max
  are just below (−0.022 and −0.029, CIs excluding 0 by < 0.003). Every LME
  model is significantly worse (−0.050 to −0.095).
- **Max is first at every ordinal weight** with the linear probe. The order of
  the rest depends on w: mean is second at w = 0 and 0.25, and LME or
  linear-softmax take over at w ≥ 0.5. Attention is last at every w.
  Macro AP ranges: max 0.75–0.86, LME 0.76–0.81, linear-softmax 0.70–0.81,
  mean 0.73–0.79, attention 0.69–0.75.
- ***Gastrotheca* drives the ranking.** Linear max models reach 0.83–0.89 on it. On
  *Oreobates*, every linear max/LME model scores 0.79–0.84 and most are tied.
- **The MLP-256 probe doesn't help.** It is significantly worse than linear in
  3 of 10 pairings (max: −0.072 and −0.087; mean at w = 0.5: −0.052) and never
  significantly better. With the MLP probe, LME is its best pooler (0.77).
- **Perch zero-shot is informative for *Gastrotheca* but no longer competitive:**
  congeneric logits reach 0.687, Δ −0.199 [−0.364, −0.073] against the best model.

## Ordinal loss (linear probe, w ∈ {0, 0.25, 0.5, 1, 2})

- **It helps linear-softmax at every weight:** +0.021 to +0.050, all four CIs
  excluding 0. With the MLP probe too (+0.048 at w = 0.5).
- **Max gains at w = 2:** +0.029 [+0.002, +0.059], which is how the best model
  gets to the top. At smaller w the max and LME gains (+0.004 to +0.045) are
  not significant. Mean stays flat (|Δ| ≤ 0.014).
- **It hurts attention at every weight:** −0.032 to −0.054, all significant.
- The thresholds move more than on the smaller v1 dataset (b₃ up to 2.87 from
  its initial 2.0), except under attention, where they barely move (≤ 0.16).

## Model selection is unreliable

- **Validation and test disagree.** Across models, the rank correlation between
  val and test macro AP is ρ = 0.30. Validation prefers LME (0.85–0.88) and
  mean. The model it would pick, `linear-bin-s2/lme`, is significantly worse on
  test than the test-best (−0.095 [−0.157, −0.027]).
- There are 18 *Gastrotheca* positives in val and 19 in test, all from a few
  nights, so a single held-out split can't reliably pick a pooler.

## Train vs val vs test AP

Every checkpoint is rescored on all three splits (`scripts/split_ap.py`, which
writes `outputs/split_ap.csv`). The numbers are seed means without CIs, so they
describe the models; they don't test differences.

- **The splits differ in difficulty, so train − val is not a clean overfitting
  measure.** *Oreobates* val is easy for every model: val AP is 0.85–0.96, and
  test AP is 0.10–0.24 lower for every model. For most models, *Gastrotheca* is
  hardest on val (0.64–0.80).
- **Linear max fits train least and tests best.** Its *Gastrotheca* train AP
  (0.77–0.79) is the lowest of all 35 models, and its test AP (0.83–0.89) is the
  highest. Its macro train − val gap is about 0 (−0.02 to 0.00), but only
  because the two species cancel: *Gastrotheca* is higher on train than val, and
  *Oreobates* is lower.
- **The MLP probe and attention overfit.** MLP models other than
  linear-softmax reach *Gastrotheca* train AP 0.90–0.96 but test AP 0.68–0.76.
  MLP macro train − val gaps are 0.06–0.15, against ≤ 0.05 for linear max, LME
  and mean. Linear attention has gaps of 0.08–0.11 and the largest seed SD on
  train (0.04–0.07).
- **Val favours whatever ranks *Oreobates* val well.** Linear LME and mean have
  the highest val macro AP (0.86–0.88), mostly from *Oreobates* val (≈ 0.96),
  and their *Oreobates* AP falls to 0.74–0.82 on test. This explains much of the
  val/test disagreement above.

## Recall at fixed precision (cutoff chosen on val)

- ***Gastrotheca*: the ranking is now good enough to be useful.** With the cutoff
  chosen on test, recall at precision ≥ 0.8 reaches 0.84 (`linear-ord1-s2/max`;
  it was 0.33 on v1). Cutoffs chosen on val give test precision 0.66–1.00, so
  some models hit the target and others miss it. Recall values of models that
  miss aren't comparable.
- ***Oreobates*: the val cutoffs no longer carry over.** Test precision is
  0.66–0.80 at P = 0.8 and 0.56–0.88 at P = 0.9, so most models miss the
  target. The ranking ceiling is low too: with the cutoff chosen on test, recall
  is at most 0.70 at P ≥ 0.8 and 0.46 at P ≥ 0.9. The clock baseline also falls
  for *Oreobates* (0.815 on v1, 0.543 now), which points to harder test nights
  rather than worse models.
- Table: [`results/recall_at_precision.csv`](../results/recall_at_precision.csv).

## Changes from v1 (Sep–Dec audio only)

v1 used 1707 hours from 2019-09-23 to 2019-12-03 and a 216-hour test set. The
archived tables and figures are in
[`results_previous_v1_sep-dec2019/`](../results_previous_v1_sep-dec2019/RESULTS.md)
and [`viz/figures_previous_v1_sep-dec2019/`](../viz/figures_previous_v1_sep-dec2019/index.md).

v2 adds 1125 hours from 27 Feb to 15 Apr 2019. Every one of them is silent for
both species. Adding data recomputes the splits, so the v1 and v2 test sets
differ, and **v1 and v2 numbers can't be compared as paired deltas**.

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

1. **Cross-validation over day-blocks**, so every positive is tested once. This
   addresses the val/test disagreement, the split difficulty differences and the
   19-positive test set.
2. **Listen to the top-weighted windows** in positive hours and in high-scoring
   silent hours (`frog_mil.audio.read_segment`), to separate missed labels
   from confusion.
3. **1 s instances** from Perch's pre-pooling embeddings `(5, 3, 1536)`: a finer
   bag from the same forward pass.
4. **Ordinal thresholds:** exclude them from weight decay or give them a higher
   learning rate.
5. **More in-season audio.** Off-season hours add only easy negatives; the
   test set needs more calling nights.

## Adding 8 kHz recordings (1 min every 20 min)

More positives are what this study needs most, but mixing recording formats
without controls would confound the results:

1. **Bandwidth.** 8 kHz audio has nothing above 4 kHz, so its embeddings differ
   systematically. The probe could learn the recording format as a shortcut.
2. **Bag size.** 3 clips per hour gives 36 windows. Max pooling then gets more
   chances to fire on noise.
3. **Prevalence.** 3 sampled minutes catch more calls than 2, so the base rate
   per bag differs.

Before mixing them in, downsample the 2019 audio to 8 kHz and rerun the sweep
to measure the cost of the lower bandwidth. Then band-limit all audio the same
way, record the regime per bag, split so each regime appears in train and test,
and report AP per regime.
