"""Build every figure for the MIL study from ``runs/``.

    .venv/bin/python viz/make_figures.py                       # all runs -> viz/figures/
    .venv/bin/python viz/make_figures.py --out viz/figures_mlp --include 'mlp256-*'
    .venv/bin/python viz/make_figures.py --focus-run linear-ord1-s2 --no-instances

Eleven figures, numbered in reading order (see ``common.FIGURES``): the MIL setup
(1-3), the data (4), results across all runs (5-7, 9), and pooler behaviour
within the focus run (8, 10), and a slide-ready forest plot of the focus run (11).

Run from the project root. Nothing is retrained: result figures come from each
model's ``predictions.npz`` (bootstrap identical in scheme to ``frog-report``),
instance figures re-run the saved checkpoints on the cached embeddings. The
bootstrap and inference outputs are cached in ``<out>/.cache`` and invalidated
when the underlying files change, so reruns are fast.

"Best" is the top test macro AP, as in ``frog-report``; the per-pooler figures
(AP by index, pooling example, simplified forest) use the best model's run unless
``--focus-run`` names another.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import data_figs  # noqa: E402
import method_figs  # noqa: E402
import result_figs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--bags-csv", type=Path, default=Path("outputs/bags.csv"))
    ap.add_argument("--out", type=Path, default=Path("viz/figures"))
    ap.add_argument("--include", nargs="*", default=None,
                    help="glob(s) on run_id to keep, e.g. 'linear-*' (baselines always kept)")
    ap.add_argument("--exclude", nargs="*", default=[], help="glob(s) on run_id to drop")
    ap.add_argument("--focus-run", default=None,
                    help="run_id for the per-pooler figures (default: run of the best model)")
    ap.add_argument("--example-bag", default=None,
                    help="bag_id for instances/pooling_example (default: auto)")
    ap.add_argument("--example-species", default=None, choices=["gastrotheca", "oreobates"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--formats", default="png,pdf", help="comma list: png,pdf,svg")
    ap.add_argument("--no-instances", action="store_true",
                    help="skip the pooling example (needs embeddings + checkpoints)")
    a = ap.parse_args()

    common.apply_style()

    infos = common.load_infos(a.runs)

    def keep(i):
        if not i.is_model:
            return True
        if a.include is not None and not any(fnmatch.fnmatch(i.run_id, g) for g in a.include):
            return False
        return not any(fnmatch.fnmatch(i.run_id, g) for g in a.exclude)

    infos = [i for i in infos if keep(i)]
    models = [i for i in infos if i.is_model]
    if not models:
        raise SystemExit("no models left after --include/--exclude")
    print(f"{len(models)} models from {len({i.run_id for i in models})} runs + "
          f"{len(infos) - len(models)} baselines, {len(infos[0].m.bag_ids)} test bags")

    cache = a.out / ".cache"
    st = common.compute_stats(infos, a.runs, a.n_boot, a.seed, a.workers, cache)
    best = max(models, key=lambda i: st.macro(i.model_id))
    focus = a.focus_run or best.run_id
    if focus not in {i.run_id for i in models}:
        raise SystemExit(f"--focus-run {focus} not among loaded runs")
    print(f"best = {best.model_id} (macro AP {st.macro(best.model_id):.3f}); "
          f"focus run = {focus}")

    save = common.Saver(a.out, tuple(a.formats.split(",")))
    bags = data_figs.load_bags(a.bags_csv)

    method_figs.bag_structure(save)
    method_figs.pipeline(save)
    method_figs.pipeline_wytham(save)
    method_figs.pooling_toy(save)
    data_figs.calendar(save, bags)
    result_figs.forest(save, infos, st, best.model_id)
    result_figs.effects(save, infos, st)
    result_figs.ordinal_sweep(save, infos, st)
    result_figs.per_index(save, infos, st, focus)
    result_figs.val_vs_test(save, infos, st)
    result_figs.forest_simple(save, infos, st, focus)
    if not a.no_instances:
        import instance_figs
        pools = sorted({i.pooling for i in models if i.run_id == focus})
        try:
            inst = instance_figs.infer_run(a.runs, focus, pools, cache)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"skipping pooling_example ({e.__class__.__name__}: {e})")
        else:
            instance_figs.pooling_example(save, inst, infos, focus, a.example_bag,
                                          a.example_species)
            inst12 = instance_figs.infer_run(a.runs, focus, pools, cache, max_windows=12)
            instance_figs.pooling_example(save, inst12, infos, focus, a.example_bag,
                                          a.example_species, name="pooling_example_wytham",
                                          width=12 * 0.7, title=False, max_windows=12,
                                          capitalise=True, font=1.8,
                                          short_titles=True, seed_range=True,
                                          labels={"linear_softmax": "Lin-softmax"})

    runs_used = sorted({i.run_id for i in models})
    save.write_index(
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `viz/make_figures.py` from "
        f"`{a.runs}/` ({len(models)} models: {', '.join(runs_used)}; + baselines). "
        f"Best = `{best.model_id}` (macro AP {st.macro(best.model_id):.3f}); per-pooler "
        f"figures use `{focus}`. CIs: bootstrap over test bags, B={a.n_boot}, paired. "
        f"Do not edit by hand.")
    print(f"written {len(save.items)} figures to {a.out}/ (index: {a.out}/index.md)")


if __name__ == "__main__":
    main()
