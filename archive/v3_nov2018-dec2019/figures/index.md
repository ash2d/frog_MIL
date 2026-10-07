# Figures

Generated 2026-10-03 19:39 by `viz/make_figures.py` from `runs/` (35 models: linear-bin-s2, linear-ord0.25-s2, linear-ord0.5-s2, linear-ord1-s2, linear-ord2-s2, mlp256-bin-s2, mlp256-ord0.5-s2; + baselines). Best = `linear-bin-s2/lme` (macro AP 0.829); per-pooler figures use `linear-bin-s2`. CIs: bootstrap over test bags, B=2000, paired. Do not edit by hand.

## 01_bag_structure

![01_bag_structure](01_bag_structure.png)

How an hour becomes a MIL bag: two 1-min clips per hour (2019), each tiled into 12 contiguous 5 s windows (instances). The 2018 hours have one clip, so 12 windows. Labels exist only per bag. The model predicts the bag label from its instance predictions.

## 02_pipeline

![02_pipeline](02_pipeline.png)

Model pipeline: frozen Perch v2 embeddings per 5 s window, a per-window probe, a pooling function that turns the window logits (24 per hour, 12 for the 2018 hours) into one bag logit per species, and an optional cumulative-link ordinal head on that same logit.

## 02_pipeline_wytham

![02_pipeline_wytham](02_pipeline_wytham.png)

Slide version of the pipeline for a single 1 min recording (12 windows): frozen Perch v2 embeddings per 5 s window, a per-window probe, and a pooling function that turns the window logits into one bag logit per species.

## 03_pooling_toy

![03_pooling_toy](03_pooling_toy.png)

How the fixed poolers turn window logits into a bag score, using the project's own pooling code on toy inputs (call windows logit +2, silent -3; C has 3 call windows). Mean needs calls to fill the hour. Max reacts to one confident window but gives gradient to that window only. LME and linear-softmax sit in between. Attention is learned, so it is shown on real data in figure 10.

## 04_calendar

![04_calendar](04_calendar.png)

Every labelled hour with audio, coloured by calling index, for each species. Each column is one night (noon to noon), so a night of calling is one contiguous band. Stretches of more than a week without audio are collapsed into a hatched break. The top strip shows which split each 3-day block belongs to. The audio covers Nov–Dec 2018 (one clip per hour) and Feb–Apr and Sep–Dec 2019 (two). Both frogs call at night and neither calls in the Feb–Apr recordings; Oreobates calls only in Oct–Dec.

## 05_forest_ap

![05_forest_ap](05_forest_ap.png)

Test average precision for every model and baseline, as seed-averaged AP with a 95% bootstrap CI over the test bags. The faint vertical line marks the best model. Colour and marker show the pooler, and hollow markers are the MLP probe.

## 06_effects

![06_effects](06_effects.png)

Controlled comparisons: each Δ compares two models that differ in one factor only (probe, ordinal loss weight, or window stride), with the same pooler, splits and seeds. Filled markers are CIs that exclude 0.

## 07_ordinal_sweep_linear

![07_ordinal_sweep_linear](07_ordinal_sweep_linear.png)

Test AP against the ordinal loss weight w for the linear probe. Points are dodged sideways so the CIs stay readable. The grey line is chance. Use `effects` for paired Δ vs w = 0.

## 08_per_index

![08_per_index](08_per_index.png)

AP of each calling index against silent hours, for every pooler in `linear-bin-s2` and the baselines (grey diamonds). Short dark bars are chance. Index 1 (isolated calls) is the hard, pooling-sensitive case.

## 09_val_vs_test

![09_val_vs_test](09_val_vs_test.png)

Validation vs test macro AP per model. Validation AP is the early-stopping optimum, so it is optimistic. A weak rank correlation means one validation split can't be trusted to pick the pooler.

## 10_pooling_example

![10_pooling_example](10_pooling_example.png)

A real test hour (`HELECHOS_20191001_00`, *G. chrysosticta* index 1), chosen automatically as the hour where the poolers disagree most. Left: each model's per-window probability. Right: the weight its pooler put on each window (max is one-hot, mean is uniform, attention is learned from the embedding).

## 10_pooling_example_wytham

![10_pooling_example_wytham](10_pooling_example_wytham.png)

A real test hour (`HELECHOS_20191003_01`, *G. chrysosticta* index 1), chosen automatically as the hour where the poolers disagree most. Left: each model's per-window probability. Right: the weight its pooler put on each window (max is one-hot, mean is uniform, attention is learned from the embedding). Only the first 12 windows (the first 1 min clip) are used: later windows are masked, so weights and bag P are for that clip alone. Bars and bag P are means over the training seeds; the bracket is the range of bag P across seeds.

## 11_forest_simple

![11_forest_simple](11_forest_simple.png)

Simplified forest plot for slides: test AP per species for each pooler in `linear-bin-s2` (colours as in figure 5) and the three baselines (grey), as seed-averaged AP with a 95% bootstrap CI over the test bags.
