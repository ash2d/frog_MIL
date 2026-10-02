# Methods

**Question.** Starting from frozen Perch v2 embeddings, which multiple-instance
learning (MIL) pooling function best recovers hourly frog presence from 5 s
windows? The candidates are mean, max, log-mean-exp, linear-softmax and
attention.

## Data

| | |
|---|---|
| labels | `data/df_helechos_with2020.csv`: one row per clock hour, 2018-09 → 2020-07, with a 0–3 calling index per species (0 silent, 1 isolated calls, 2 overlapping, 3 chorus) |
| audio | Helechos recorder, 1 min clips at :00 and :30, 44.1 kHz: 2019-02-27 → 2019-04-15 (1125 hours, all silent for both species) and 2019-09-23 → 2019-12-03 (1707 hours) |
| species | *Gastrotheca chrysosticta* and *Oreobates berdemenos*, both nocturnal. In the audio, *G.* calls Sep–Nov and *O.* Oct–Dec. Multi-label: 48 of the 2832 audio hours have both |

Annotators listened only to the recorded clips, so every positive hour contains
a call. The hourly index is the max over the clips in that hour. That is the
standard MIL assumption: *bag label = max(instance labels)*. Labels say whether
a call occurred, not where.

```
bag       = 1 hour  → presence (index > 0) and index (0–3) per species
 clip     = 1 min wav, 2 per hour
 instance = 5 s window (Perch v2 input), 12 per clip, 24 per bag
```

Windows are contiguous and non-overlapping (0–5 s, 5–10 s, …, 55–60 s), and
only these are embedded. Overlap would duplicate calls across neighbouring
windows, which would change what mean, LME and linear-softmax measure.

## Model

Perch v2 is frozen. Only the head is trained.

```
X [24 × 1536]  →  instance probe  →  logits [24 × 2]  →  pooling  →  bag logit s [2]  →  σ(s)
```

| probe | layers | params |
|---|---|---|
| `linear` | Dropout(0.2) → Linear(1536→2) | 3,074 |
| `mlp256` | Dropout(0.2) → Linear(1536→256) → GELU → Dropout(0.2) → Linear(256→2) | 393,986 |

| pooling | bag logit from window logits | extra params |
|---|---|---|
| mean | mean of logits | 0 |
| max | max of logits | 0 |
| lme | (1/r)·log mean exp(r·logit), r = 1 | 0 |
| linear_softmax | Σp² / Σp on probabilities, mapped back to a logit | 0 |
| attention | gated attention on the embeddings (Ilse et al. 2018), one head per species | 393,730 |

Each pooler is trained **with its own probe**. The architecture is the same, and
within a seed the initial weights are the same too. The pooler decides how the
bag loss's gradient reaches each window (all windows for mean, only the top one
for max), so the trained probes differ.

**Ordinal head** (optional, `--ordinal-weight w`):
P(index ≥ k) = σ(s − b_k), with b₁ = 0 ≤ b₂ ≤ b₃ learned. Because b₁ = 0,
σ(s) is still the presence score. The extra loss terms (k = 2, 3) ask chorus
hours to score above single-call hours.

## Training

| setting | value |
|---|---|
| input | embeddings z-scored with the train mean/std |
| loss | BCE on the bag logit, `pos_weight` = neg/pos per species (G 21.6, O 6.9). Ordinal runs add w × BCE on k = 2, 3 |
| optimiser | AdamW, lr 1e-3, weight decay 1e-2, grad clip 5, batch 32 bags |
| stopping | ≤ 60 epochs; stop after 10 epochs with no gain in val macro AP; restore the best epoch |
| seeds | 5 per model. The seed sets init, dropout and batch order, never the split |
| tuning | none; every model uses the same defaults |

## Splits

Days are grouped into 3-day blocks, and each block goes whole to one split so
neighbouring hours can't leak. Blocks are split 70/15/15. Of 500 shuffles, the
one closest to that ratio in bags and in per-species positives is kept.

| split | bags | G positive | O positive |
|---|---|---|---|
| train | 1968 | 87 | 249 |
| val | 432 | 18 | 72 |
| test | 432 | 19 | 58 |

The Feb–Apr blocks are split the same way, so every split holds both silent
off-season hours and in-season hours. Adding audio changes the block grid and
the shuffle, so it changes every split.

## Evaluation

- **Metric:** test average precision (AP) per species, averaged over seeds.
  Macro AP is the mean over the two species. Chance AP equals prevalence.
- **CIs:** percentile bootstrap over test bags (B = 2000). Every model uses the
  same resamples, so the difference between two models gets its own paired CI.
- **By index:** AP of the hours at each index against silent hours.
- **Recall at fixed precision:** the score cutoff is chosen on validation and
  applied unchanged to test (`scripts/recall_at_precision.py`).
- **Fit per split:** the saved checkpoints are rescored on train, val and test
  to compare AP across splits (`scripts/split_ap.py`). The script checks that
  its val and test numbers reproduce `metrics.json`.
- **Baselines** (no training on audio):
  - *clock*: the training presence rate for the bag's (hour of day × month).
  - *zero-shot congeneric*: max over the bag of Perch's logits for the same genus.
  - *zero-shot Frog*: max over the bag of Perch's generic "Frog" logit.
