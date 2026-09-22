# Findings

Test set: 216 held-out hours (18 *Gastrotheca* and 62 *Oreobates* positives),
using audio from 2019-09-23 to 2019-12-03. Full tables are in
[`results/RESULTS.md`](../results/RESULTS.md) and figures in
[`viz/figures/`](../viz/figures/index.md). "Tied" means the paired 95% CI of
the difference includes 0. There is no multiple-comparison correction.

## Pooling comparison

- **Every trained model beats every baseline** on macro AP. The best baseline
  is the clock (hour × month) at 0.472.
- **Best: linear probe + max pooling**, 0.809 [0.707, 0.894] (`linear-ord2-s2/max`).
  Max at every ordinal weight is tied with it, and so is LME at every weight
  (−0.028 to −0.045, all CIs touching 0).
- **Pooler ranking**, the same at every ordinal weight:
  max ≥ LME > mean ≈ linear-softmax > attention.
  Macro AP ranges: max 0.71–0.81, LME 0.69–0.78, mean 0.70–0.73,
  linear-softmax 0.69–0.72, attention 0.62–0.66.
- ***Gastrotheca* drives the ranking.** On *Oreobates*, every model scores
  0.92–0.97 and most are tied.
- **The MLP-256 probe doesn't help.** It is significantly worse than linear in
  5 of 10 pairings (max and LME: −0.05 to −0.09), probably from overfitting the
  86 *Gastrotheca* training positives.
- **Attention comes last with either probe.** The MLP-256 fixed poolers have
  about the same parameter count and still score higher, so extra capacity
  alone doesn't explain it.
- **Perch zero-shot is tied with the best model on *Gastrotheca*:** congeneric
  logits reach 0.551, Δ −0.111 [−0.256, +0.093]. Perch already carries most
  of the genus signal.

## Ordinal loss (linear probe, w ∈ {0, 0.25, 0.5, 1, 2})

- **No weight gives a significant gain for any pooler.** Max and LME move by
  +0.013 to +0.021 at every w, but every CI includes 0. Mean and
  linear-softmax stay flat (|Δ| ≤ 0.006).
- Attention gets worse at high w: −0.037 [−0.067, −0.011] at w = 2. That is 1
  significant result out of 20 comparisons, so it is plausible but not
  established.
- The thresholds barely move from their initial values (b₃ shifts ≤ 0.42),
  so they stay mostly set by initialisation and weight decay.

## Model selection is unreliable

- **Validation and test disagree.** Across models, the rank correlation between
  val and test macro AP is ρ = 0.10. Validation prefers linear-softmax and
  mean (0.83–0.84) and smaller w. The model it would pick is significantly
  worse on test than the test-best (−0.087 [−0.148, −0.024]).
- There are 20 *Gastrotheca* positives in val and 18 in test, all from a few
  nights, so a single held-out split can't reliably pick a pooler.

## Recall at fixed precision (cutoff chosen on val)

- ***Gastrotheca*: not usable yet.** Cutoffs chosen on val give test precision
  of only 0.28–0.61 against targets of 0.8 and 0.9. Even with the cutoff
  chosen on test, recall at precision ≥ 0.8 is at most 0.33. The high-scoring
  test negatives are night hours inside the few positive nights, often with
  *Oreobates* calling. They are either calls the annotators missed or
  confusion with *Oreobates* and other night sounds.
- ***Oreobates*: the cutoffs carry over.** Test precision is 0.93–1.00. Best
  recall is 0.77 at P = 0.8 and 0.67 at P = 0.9 (`mlp256-bin-s2/mean`), with
  many models tied. Max and LME rank as well as any pooler, but their score
  scale shifts between val and test, so their fixed cutoffs lose the most
  recall.
- Table: [`results/recall_at_precision.csv`](../results/recall_at_precision.csv).

## Next steps

1. **Cross-validation over day-blocks**, so every positive is tested once. This
   addresses the val/test disagreement and the 18-positive test set.
2. **Listen to the top-weighted windows** in positive hours and in high-scoring
   silent hours (`frog_mil.audio.read_segment`), to separate missed labels
   from confusion.
3. **1 s instances** from Perch's pre-pooling embeddings `(5, 3, 1536)`: a finer
   bag from the same forward pass.
4. **Overlap ablation** (`--stride 1`).
5. **Ordinal thresholds:** exclude them from weight decay or give them a higher
   learning rate.

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
