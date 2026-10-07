# Archive: results of earlier data versions

Each folder holds the generated tables (`results/`) and figures (`figures/`) of
one data version, as they were when that version was current. Their untracked
state (runs, manifest, embedding cache) is in the group workspace under the same
name: `/gws/ssde/j25b/iecdt/dash/frogs/frog_mil/archive/<version>/`. Paths inside
the archived files (`runs/`, `outputs/`, `embeddings/`) refer to the layout of
the time. Different versions have different test sets, so compare them only
loosely, never as paired deltas.

| version | audio | hours | evaluation | best model (macro AP) |
|---|---|---|---|---|
| `v1_sep-dec2019` | 2019-09-23 → 12-03 | 1707 | one split, 216 test hours | `linear-ord2-s2/max` 0.809 |
| `v2_feb-dec2019` | + 2019-02-27 → 04-15 | 2832 | one split, 432 test hours | `linear-ord2-s2/max` 0.859 |
| `v3_nov2018-dec2019` | + 2018-11-14 → 12-16 | 3594 | one split, 576 test hours | `linear-bin-s2/lme` 0.829 |

`v3_nov2018-dec2019/findings.md` is `docs/findings.md` as it stood for v3. A
manifest built on 2026-10-06 (3647 hours, adding 2019-04-15 → 17) was never
trained; it is kept in the workspace as `v3b_apr2019_manifest-only/`.

From the 5-fold cross-validated version on, `frog run` archives automatically:
before the report or figures are rebuilt for a new dataset, the old ones are
copied to `archive/<old dataset_id>/` (with `findings.md`), and that dataset's
runs and manifest stay in the workspace under `runs/<dataset_id>/` and
`manifests/<dataset_id>/`, so `frog-report --dataset <id>` can regenerate them.
