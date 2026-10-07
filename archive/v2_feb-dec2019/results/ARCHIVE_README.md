# Archived: v2, Feb–Apr + Sep–Dec 2019 audio

Snapshot of the study before the Nov–Dec 2018 recordings were added (archived 2026-10-03).

- Audio: 5663 clips, 2019-02-27 → 2019-12-03 (2832 hour-bags, 67,956 windows)
- Test set: 432 bags (G. chrysosticta 19+, O. berdemenos 58+)
- Best model: `linear-ord2-s2/max`, macro AP 0.859 [0.784, 0.925]

The five v2 folders belong together: `results_`, `viz/figures_`, `runs_`, `outputs_`,
`embeddings_previous_v2_feb-dec2019/`. Paths inside the generated files (`runs/`, `outputs/`,
`embeddings/`) refer to these folders' original names. The embedding folder is a copy: the
live `embeddings/` cache was re-keyed and extended for v3, not rebuilt. The current results
(v3, Nov–Dec 2018 + Feb–Apr + Sep–Dec 2019) are in `results/`, `viz/figures/`, `runs/`,
`outputs/` and `embeddings/`. Test sets differ between v2 and v3, so compare them only
loosely, not as paired deltas.
