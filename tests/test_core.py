"""Unit tests for the pipeline's invariants, plus an end-to-end run on synthetic bags.

    .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from frog_mil import embcache, manifest, stats
from frog_mil.audio import qc_wav
from frog_mil.config import SPECIES
from frog_mil.evaluate import ap_score
from frog_mil.pooling import POOLERS, build_pooler


# --------------------------------------------------------------------------- pooling
@pytest.mark.parametrize("name", list(POOLERS))
def test_poolers_ignore_padded_windows(name):
    torch.manual_seed(0)
    B, N, C, D = 4, 6, 2, 16
    pool = build_pooler(name, dim=D, n_classes=C).eval()
    logits, emb = torch.randn(B, N, C), torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 4:] = False
    ref, w_ref = pool(logits, mask, emb)
    # append junk windows, masked out: nothing may change
    junk = 50 * torch.randn(B, 3, C)
    big_l = torch.cat([logits, junk], 1)
    big_e = torch.cat([emb, 50 * torch.randn(B, 3, D)], 1)
    big_m = torch.cat([mask, torch.zeros(B, 3, dtype=torch.bool)], 1)
    out, w = pool(big_l, big_m, big_e)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(w[:, N:], torch.zeros_like(w[:, N:]))
    assert torch.allclose(w[:, :N], w_ref, atol=1e-6)


# --------------------------------------------------------------------------- metrics
def test_ap_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.15
    s = np.round(rng.random(300) + 0.4 * y, 1)          # many ties
    assert ap_score(y, s) == pytest.approx(average_precision_score(y, s))
    W = rng.integers(0, 3, (10, 300))
    got = stats.ap_weighted(y, s, W)
    want = [average_precision_score(np.repeat(y, w), np.repeat(s, w)) for w in W]
    assert np.allclose(got, want)
    assert np.isnan(stats.ap_weighted(np.zeros(5, bool), np.arange(5.0), np.ones((1, 5))))[0]


def test_block_draws_resample_whole_blocks_within_strata():
    blocks = np.repeat(np.arange(12), 5)
    strata = np.where(blocks < 4, "a", "b")
    W = stats.block_draws(blocks, strata, 200, 0)
    assert (W[:, strata == "a"].sum(1) == 20).all()          # stratum size kept
    for b in range(12):                                      # blocks move as one
        assert (W[:, blocks == b] == W[:, blocks == b][:, :1]).all()


# --------------------------------------------------------------------------- manifest
def test_windows():
    assert len(manifest.windows(60.0, 5.0, False)) == 12
    assert manifest.windows(59.995, 5.0, False)[-1] == (55.0, 59.995)
    assert len(manifest.windows(59.995, 5.0, True)) == 11
    assert len(manifest.windows(62.0, 5.0, False)) == 12      # 2 s tail < half a window


def _blocks(n=40, seed=1):
    rng = np.random.default_rng(seed)
    st = {b: {"bags": 72, "gastrotheca": int(rng.poisson(3) * (b % 3 == 0)),
              "oreobates": int(rng.poisson(4) * (b % 2 == 0))} for b in range(n)}
    reg = {b: "8k-3clip" if b < 12 else ("44k-1clip" if b < 20 else "44k-2clip")
           for b in range(n)}
    return st, reg


def test_assign_folds_deterministic_whole_blocks_every_regime():
    st, reg = _blocks()
    f1 = manifest.assign_folds(st, reg, 5, seed=0, n_tries=200)
    f2 = manifest.assign_folds(st, reg, 5, seed=0, n_tries=200)
    assert f1 == f2
    assert set(f1) == set(st) and set(f1.values()) == set(range(5))
    for r in set(reg.values()):                               # each regime in each fold
        assert {f1[b] for b in st if reg[b] == r} == set(range(5))


def test_dataset_id_changes_with_content(tmp_path):
    p = tmp_path / "bags.csv"
    p.write_text("bag_id,fold\na,0\n")
    a = manifest.hash_dataset(p, "x")
    assert a == manifest.hash_dataset(p, "x") and a != manifest.hash_dataset(p, "y")
    p.write_text("bag_id,fold\na,1\n")
    assert manifest.hash_dataset(p, "x") != a


# --------------------------------------------------------------------------- audio QC
def _wav(path, x, sr=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.stack([x, x], 1).astype("<i2").tobytes())
    return path


def test_qc_flags_zero_fill_and_truncation(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.normal(0, 3000, 8000 * 4)).astype(np.int16)
    assert qc_wav(_wav(tmp_path / "ok.wav", x))["qc_ok"]
    z = x.copy()
    z[8000 * 2:] = 0
    q = qc_wav(_wav(tmp_path / "zero.wav", z))
    assert not q["qc_ok"] and q["zero_tail_s"] == pytest.approx(2.0)
    p = _wav(tmp_path / "cut.wav", x)
    p.write_bytes(p.read_bytes()[:-1000])
    assert qc_wav(p)["qc_reason"] == "truncated"


# --------------------------------------------------------------------------- cache
def _cache(d: Path, ids, sigs, values):
    d.mkdir(parents=True)
    rows = [{"instance_id": i, "bag_id": i[:1], "win_idx": 0} for i in ids]
    e = np.lib.format.open_memmap(d / "embeddings.f16.npy", "w+", np.float16,
                                  (len(ids), embcache.DIM))
    e[:] = np.asarray(values, np.float16)[:, None]
    e.flush()
    del e
    embcache.save_atomic(d / "done.npy", np.ones(len(ids), bool))
    embcache.write_index(d / "index.csv", rows, sigs)
    (d / "meta.json").write_text(json.dumps({"model": "m"}))


def test_rekey_carries_valid_rows_only(tmp_path):
    d = tmp_path / "emb"
    _cache(d, ["a1", "b1", "c1"], ["s", "s", "s"], [1, 2, 3])
    old = embcache.load_cache(d)
    # new order: c1 kept, a1's file changed, b1 dropped, d1 new
    rows = [{"instance_id": i, "bag_id": i[:1], "win_idx": 0} for i in ("c1", "a1", "d1")]
    done = embcache.rekey(d, rows, ["s", "CHANGED", "s"], old, 0, {"instances_id": "x"})
    assert done.tolist() == [True, False, False]
    e = np.load(d / "embeddings.f16.npy")
    assert e[0, 0] == 3 and e[1:].sum() == 0
    assert json.loads((d / "meta.json").read_text())["instances_id"] == "x"
    assert not (tmp_path / "emb.old").exists() and not (tmp_path / "emb.rekey").exists()


# --------------------------------------------------------------------------- end to end
@pytest.fixture
def synthetic(tmp_path):
    """60 bags in 5 folds over 3 regimes; a call shifts one window's embedding."""
    rng = np.random.default_rng(0)
    out, emb = tmp_path / "outputs", tmp_path / "emb"
    out.mkdir()
    bags, inst = [], []
    for i in range(60):
        n_win = (12, 24, 36)[i % 3]
        g, o = int(rng.random() < 0.3), int(rng.random() < 0.3)
        bid = f"H_{i:03d}"
        day = f"2019-10-{1 + i // 24:02d}"
        bags.append({"bag_id": bid, "datetime": f"{day} {i % 24:02d}:00:00", "date": day,
                     "hour": i % 24, "year": 2019,
                     "block": i // 4, "fold": (i // 4) % 5,
                     "regime": ("44k-1clip", "44k-2clip", "8k-3clip")[i % 3],
                     "gastrotheca_index": g * 2, "oreobates_index": o,
                     "gastrotheca_present": g, "oreobates_present": o})
        for w in range(n_win):
            inst.append((f"{bid}_{w:02d}", bid, w, g and w == 3, o and w == 7))
    with (out / "bags.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, list(bags[0]))
        wr.writeheader()
        wr.writerows(bags)
    iid = manifest.hash_instances([r[0] for r in inst])
    did = manifest.hash_dataset(out / "bags.csv", iid)
    (out / "summary.json").write_text(json.dumps(
        {"dataset_id": did, "instances_id": iid, "folds": 5}))
    x = rng.normal(0, 1, (len(inst), embcache.DIM)).astype(np.float16)
    x[[k for k, r in enumerate(inst) if r[3]], :8] += 3
    x[[k for k, r in enumerate(inst) if r[4]], 8:16] += 3
    emb.mkdir()
    np.save(emb / "embeddings.f16.npy", x)
    np.save(emb / "done.npy", np.ones(len(inst), bool))
    embcache.write_index(emb / "index.csv",
                         [{"instance_id": r[0], "bag_id": r[1], "win_idx": r[2]}
                          for r in inst],
                         ["s"] * len(inst))
    (emb / "meta.json").write_text(json.dumps(
        {"instances_id": iid, "dim": embcache.DIM, "window_s": 5.0, "hop_s": 5.0}))
    return tmp_path, did


def test_end_to_end_train_and_report(synthetic, monkeypatch):
    from frog_mil import data as D
    from frog_mil import report
    from frog_mil import runs as rs
    from frog_mil import train as T

    root, did = synthetic
    data = D.load_bags(root / "emb", root / "outputs" / "bags.csv",
                       root / "outputs" / "summary.json", dataset_id=did)
    assert data.x.shape == (60, 36, embcache.DIM)
    assert data.mask.sum(1).tolist()[:3] == [12, 24, 36]
    with pytest.raises(RuntimeError):
        D.load_bags(root / "emb", root / "outputs" / "bags.csv",
                    root / "outputs" / "summary.json", dataset_id="other")

    a = T.build_parser().parse_args(["--seeds", "2", "--epochs", "3", "--patience", "2"])
    cfg = T.config_from_args(a, did, data.n_folds)
    views = [D.FoldView(data, f) for f in range(data.n_folds)]
    for f, v in enumerate(views):              # roles partition the bags
        r = data.roles(f)
        assert (r["train"].astype(int) + r["val"] + r["test"] == 1).all()
    runs_root = root / "runs"
    run_dir = rs.dataset_dir(did, runs_root) / cfg["run_id"]
    run_dir.mkdir(parents=True)
    rs.write_json_atomic(run_dir / "config.json",
                         {**cfg, "poolings": ["max", "attention"], "status": "complete"})
    for p in ("max", "attention"):
        T.train_model(data, views, cfg, p, run_dir / p)
    monkeypatch.setattr(T, "zero_shot_baseline", lambda *a, **k: {})
    T.ensure_baselines(data, runs_root)

    d = np.load(run_dir / "max" / "predictions.npz")
    assert d["scores"].shape == (2, 60, len(SPECIES)) and not np.isnan(d["scores"]).any()
    assert str(d["dataset_id"]) == did
    w = np.load(run_dir / "max" / "windows.npz")
    assert (w["weights"][..., 0].sum(-1) > 0.99).all()       # max: one-hot over real windows
    m = rs.load_model(run_dir, "max", 0, 0, data.dim)          # checkpoint reproduces scores
    v = views[0]
    with torch.no_grad():
        p = torch.sigmoid(m(**{k: v.batch(v.idx["test"])[k] for k in ("x", "mask")})["logits"])
    assert np.allclose(p.numpy(), d["scores"][0, v.idx["test"].numpy()], atol=1e-5)

    models = rs.load_models(did, runs_root, verbose=False)
    assert {mm.model_id for mm in models} >= {f"{cfg['run_id']}/max", "baseline/clock"}
    out = root / "results"
    monkeypatch.setattr(sys, "argv", ["frog-report", "--dataset", did,
                                      "--runs", str(runs_root),
                                      "--bags-csv", str(root / "outputs" / "bags.csv"),
                                      "--out", str(out), "--n-boot", "50"])
    report.main()
    md = (out / "RESULTS.md").read_text()
    assert "validation-selected" in md and "## 6. AP by recording regime" in md
    assert json.loads((out / "dataset.json").read_text())["dataset_id"] == did
    for f in ("overview", "vs_selected", "per_level", "per_regime", "split_ap",
              "recall_at_precision"):
        assert (out / f"{f}.csv").exists()


# --------------------------------------------------------------------------- pipeline
def test_sweep_entries_parse_to_distinct_runs():
    from frog_mil import pipeline
    from frog_mil import runs as rs
    from frog_mil.train import build_parser

    ids = []
    for e in pipeline.sweep_entries():
        a = build_parser().parse_args(pipeline.entry_argv(e))
        ids.append(a.tag or rs.run_id_for(a.hidden, a.ordinal_weight))
    assert ids and len(ids) == len(set(ids))
    assert pipeline.entry_argv({"hidden": 256, "lme_learnable": True, "x": False}) == \
        ["--hidden", "256", "--lme-learnable"]
