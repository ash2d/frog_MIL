# Methods

**Question.** Starting from frozen Perch v2 embeddings, which multiple-instance
learning (MIL) pooling function best recovers hourly frog presence from 5 s
windows? The candidates are mean, max, log-mean-exp, linear-softmax and
attention.

## Data

| | |
|---|---|
| labels | `data/df_helechos_with2020.csv`: one row per clock hour, 2018-09 → 2020-07, with a 0–3 calling index per species (0 silent, 1 isolated calls, 2 overlapping, 3 chorus) |
| audio | Helechos recorder, 1 min WAV clips, in three recording regimes (below). 4914 labelled hours, 123,876 windows |
| species | *Gastrotheca chrysosticta* and *Oreobates berdemenos*, both nocturnal. Multi-label: some hours have both |

| regime | period | format | clips per hour | windows per bag | hours | G positive | O positive |
|---|---|---|---|---|---|---|---|
| `8k-3clip` | 2018-09-01 → 10-24 | 8 kHz stereo | 3 (:00, :20, :40) | 36 | 1267 | 291 (23%) | 101 (8%) |
| `44k-1clip` | 2018-11-14 → 12-16 | 44.1 kHz stereo | 1 (:00) | 12 | 762 | 14 (2%) | 101 (13%) |
| `44k-2clip` | 2019-02-27 → 04-17, 2019-09-23 → 12-03 | 44.1 kHz stereo | 2 (:00, :30) | 24 | 2885 | 124 (4%) | 379 (13%) |

The regime is the recording sample rate plus the period's usual number of clips
per hour, so an hour that lost a clip keeps its period's regime. **The regime is
confounded with the labels.** Most *Gastrotheca* positives come from the 8 kHz
period, and 8 kHz audio has no content above 4 kHz and runs about 10 dB louder,
so a model could learn the recording format as a shortcut. The splits are
therefore stratified by regime, and results are also reported within each regime.

Annotators listened only to the recorded clips, so every positive hour contains
a call. The hourly index is the max over the clips in that hour. That is the
standard MIL assumption: *bag label = max(instance labels)*. Labels say whether
a call occurred, not where.

```
bag       = 1 hour  → presence (index > 0) and index (0–3) per species
 clip     = 1 min WAV, 1–3 per hour depending on the regime
 instance = 5 s window (Perch v2 input), 12 per clip, so 12, 24 or 36 per bag
```

Windows are contiguous and non-overlapping (0–5 s, 5–10 s, …, 55–60 s), and
only these are embedded. Overlap would duplicate calls across neighbouring
windows, which would change what mean, LME and linear-softmax measure. Bags are
padded to 36 windows, and a mask keeps every pooler off the padding.

### Audio quality control

Every WAV passes `audio.qc_wav` before it enters the manifest. A file fails if its
header is unreadable, its data chunk is shorter than the header claims (an upload
in flight), or every channel holds exact digital zero for ≥ 1 s (a zero-filled
upload, as happened to one 2019 file). An hour with any failed clip is dropped
whole, because the annotators heard the intact original. Clipping is measured
(`clip_frac_max` in `bags.csv`) but not dropped. In the current audio no file
fails, and 108 files clip in more than 0.1% of samples.

## Model

Perch v2 (`google/bird-vocalization-classifier/tensorFlow2/perch_v2/2`, recorded
in the cache's `meta.json`) is frozen. Audio is resampled to 32 kHz mono
(`resample_poly`; ×4 for the 8 kHz files). Only the head is trained.

```
X [n × 1536]   →  instance probe  →  logits [n × 2]   →  pooling  →  bag logit s [2]  →  σ(s)
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
within a seed and fold the initial weights and batch order are the same too. The
pooler decides how the bag loss's gradient reaches each window (all windows for
mean, only the top one for max), so the trained probes differ.

**Ordinal head** (optional, `ordinal_weight = w`):
P(index ≥ k) = σ(s − b_k), with b₁ = 0 ≤ b₂ ≤ b₃ learned. Because b₁ = 0,
σ(s) is still the presence score. The extra loss terms (k = 2, 3) ask chorus
hours to score above single-call hours.

## Training

| setting | value |
|---|---|
| input | embeddings z-scored with the training folds' window mean/std |
| loss | BCE on the bag logit, `pos_weight` = neg/pos per species on the training folds. Ordinal runs add w × BCE on k = 2, 3 |
| optimiser | AdamW, lr 1e-3, weight decay 1e-2, grad clip 5, batch 32 bags |
| stopping | ≤ 60 epochs; stop after 10 epochs with no gain in validation macro AP; restore the best epoch |
| seeds | 5 per model. The seed sets init, dropout and batch order, never the folds |
| tuning | none; every model uses the same defaults |

The whole dataset is held on the GPU as one padded tensor, so a model (5 seeds ×
5 folds) trains in about a minute.

## Cross-validation

Days are grouped into 3-day blocks, and each block goes whole to one of K = 5
folds, so neighbouring hours can't leak between training and test. Blocks are
dealt out **separately within each regime** (shuffled, round-robin from a random
starting fold), so every regime is in every fold. Of 2000 deals, the one closest
to 1/K of the bags and of each species' positives (overall and per regime) is
kept, then polished by swapping same-regime blocks between folds. The folds hold
944–1008 bags, 85–87 *Gastrotheca* and 116–117 *Oreobates* positives each.

Model f is **tested on fold f**, early-stopped on **fold f + 1 (mod K)** and
trained on the other three. Every bag therefore gets exactly one out-of-fold test
score and one validation score per seed, from models that never trained on it.

## Evaluation

- **Metric:** average precision (AP) per species on the pooled out-of-fold test
  scores of all 4914 bags, averaged over seeds. Macro AP is the mean over the two
  species. Chance AP equals prevalence.
- **CIs:** block bootstrap (B = 2000). Each resample draws whole 3-day blocks with
  replacement, separately within each regime, so correlated hours stay together
  and the regime mix is fixed. Resampling single hours would treat the 24 hours
  of a night as independent and make every CI too narrow. Every model uses the
  same resamples, so the difference between two models gets its own paired CI.
- **Reference model:** the model with the best mean *validation* macro AP (over
  seeds and folds). Paired differences are reported against it. Choosing the
  reference on test would make every "worse than the best" claim optimistic.
- **By index:** AP of the hours at each index against silent hours.
- **By regime:** presence AP within each recording regime, with its own chance
  level. It is the first check against the 8 kHz shortcut; the controls below
  are the others.
- **Fit per split:** each fold model's AP on its train, validation and test
  folds (`metrics.json`).
- **Recall at fixed precision:** each fold model's score cutoff is the lowest one
  that reaches the target precision on its validation fold, applied unchanged to
  its test fold.
- **Baselines** (no training on audio):
  - *clock*: the training folds' presence rate for the bag's (hour of day × month),
    out-of-fold.
  - *zero-shot congeneric*: max over the bag of Perch's logits for the same genus.
  - *zero-shot Frog*: max over the bag of Perch's generic "Frog" logit.

## Controls for the recording regime

The 8 kHz regime is confounded with the labels, so two experiments test whether
the model detects the format rather than the frogs. Both use the reference model
(`mlp256-ord0.5-s2/lme`) with the main folds and seeds, and both keep their
state and runs in `STORE/experiments/` so they never enter the main tables or
the validation selection. Results are in
[findings](findings.md#controls-for-the-8-khz-confound).

- **Band-limit control** (`scripts/bandlimit_experiment.py`). Every 44.1 kHz
  clip is resampled to 8 kHz before the usual resampling to 32 kHz
  (`embed_perch.py --band-limit 8000`), the path the 8 kHz recordings take, so
  no hour holds anything above 4 kHz. The model is trained on these embeddings
  and on the recorded ones (a) on all hours and (b) on the 44.1 kHz hours only
  (a subset manifest, `frog-train --manifest`), where no format cue exists and
  the difference is the cost of losing > 4 kHz. A logistic regression on
  single window embeddings measures how well the format can still be told apart.
- **Cross-regime transfer** (`scripts/regime_transfer.py`). The model is trained
  and early-stopped on the 44.1 kHz hours only or on the 8 kHz hours only
  (`frog-train --train-regimes`), but the test fold is always every hour, so it
  scores the other regime's hours out of fold without having heard that format.
  This is done on the recorded and on the band-limited embeddings. The summary
  compares AP within each regime with the all-regimes model, and checks whether
  the transfer models rank the silent 8 kHz hours like the reference does, and
  whether the top-ranked silent hours sit next to labelled calls.

## Reproducibility

`frog run` rebuilds whatever is out of date: manifest (when the labels, audio or
manifest code change), embeddings, every run in `sweep.toml`, the tables and the
figures. The manifest is hashed into a `dataset_id` that the embedding cache,
runs, tables and figures all record. Earlier data versions (v1–v3, single
train/val/test split) are archived in `archive/` and in the group workspace.
