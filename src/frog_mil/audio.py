"""Raw audio: per-file quality control, and reading windows to listen to.

    from frog_mil.audio import qc_wav, read_segment
    qc_wav(path)        # header, truncation, digital-zero runs, clipping
    x = read_segment(row["filepath"], row["clip_start_s"], row["clip_end_s"], 32000)

``row`` is a line of ``outputs/instances.csv``. Uses only numpy and the stdlib
``wave`` module, so it runs in either environment.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

# A field recording never holds exact digital zero for this long; uploads that
# were cut short and zero-filled do (HELECHOS_20190415_083000.wav, Oct 2026).
ZERO_RUN_FAIL_S = 1.0
# Fraction of samples at full scale above which a clip is flagged (not dropped).
CLIP_WARN_FRAC = 1e-3

QC_FIELDS = ["sample_rate", "channels", "sampwidth", "duration_s", "truncated",
             "zero_run_s", "zero_tail_s", "clip_frac", "rms_dbfs", "qc_ok", "qc_reason"]


def _decode(raw: bytes, sw: int, ch: int) -> np.ndarray:
    """PCM bytes -> int array [frames, channels]."""
    if sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x)
    else:
        x = np.frombuffer(raw, {1: np.uint8, 2: "<i2", 4: "<i4"}[sw]).astype(np.int32)
        if sw == 1:
            x = x - 128
    return x.reshape(-1, ch)


def _longest_true_run(b: np.ndarray) -> tuple[int, int]:
    """(longest run of True, length of the trailing run of True)."""
    if not b.any():
        return 0, 0
    d = np.diff(np.r_[0, b.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = ends - starts
    tail = int(runs[-1]) if ends[-1] == len(b) else 0
    return int(runs.max()), tail


def qc_wav(path: str | Path) -> dict:
    """Check one WAV file. ``qc_ok`` is False for files that must not be used.

    Hard failures: unreadable header, no frames, data chunk shorter than the
    header claims (upload in flight), or a run of exact digital zero on every
    channel of at least ``ZERO_RUN_FAIL_S`` (zero-filled upload). Clipping is
    measured and reported, never a failure.
    """
    path = Path(path)
    out = dict.fromkeys(QC_FIELDS, "")
    try:
        with wave.open(str(path)) as w:
            n, sr = w.getnframes(), w.getframerate()
            ch, sw = w.getnchannels(), w.getsampwidth()
            out.update(sample_rate=sr, channels=ch, sampwidth=sw,
                       duration_s=round(n / sr, 3) if sr else 0.0)
            if n == 0 or sr == 0:
                return {**out, "qc_ok": False, "qc_reason": "empty"}
            if path.stat().st_size < 44 + n * ch * sw:
                return {**out, "truncated": True, "qc_ok": False, "qc_reason": "truncated"}
            raw = w.readframes(n)
    except Exception as e:  # noqa: BLE001 - any parse failure means unusable
        return {**out, "qc_ok": False, "qc_reason": f"unreadable: {e.__class__.__name__}"}

    x = _decode(raw, sw, ch)
    if len(x) < n:
        return {**out, "truncated": True, "qc_ok": False, "qc_reason": "truncated"}
    full = (1 << (8 * sw - 1)) - 1
    zero = ~x.any(axis=1)
    run, tail = _longest_true_run(zero)
    clip = float((np.abs(x) >= full).mean())
    rms = float(np.sqrt(np.mean((x.astype(np.float64) / full) ** 2)))
    out.update(truncated=False, zero_run_s=round(run / sr, 3), zero_tail_s=round(tail / sr, 3),
               clip_frac=round(clip, 6),
               rms_dbfs=round(float(20 * np.log10(rms)), 2) if rms > 0 else float("-inf"))
    reasons = []
    if run / sr >= ZERO_RUN_FAIL_S:
        reasons.append(f"digital zero for {run / sr:.1f} s"
                       + (" at the end" if tail == run else ""))
    out["qc_ok"] = not reasons
    out["qc_reason"] = "; ".join(reasons) or ("clipping" if clip > CLIP_WARN_FRAC else "")
    return out


def read_segment(path: str, start_s: float, end_s: float, target_sr: int,
                 mono: bool = True) -> np.ndarray:
    """Read [start_s, end_s) from a PCM WAV as float32 in [-1, 1].

    Resamples to ``target_sr`` by linear interpolation, which is fine for
    listening but aliases the high band; embeddings use a proper resampler
    (``scripts/embed_perch.py``).
    """
    with wave.open(path) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        first = int(start_s * sr)
        n = max(0, int(end_s * sr) - first)
        w.setpos(min(first, w.getnframes()))
        raw = w.readframes(min(n, w.getnframes() - w.tell()))

    x = _decode(raw, sw, ch).astype(np.float32) / float(1 << (8 * sw - 1))
    x = x.mean(axis=1) if mono else x.T

    if sr != target_sr:
        want = int(round(x.shape[-1] * target_sr / sr))
        grid = np.linspace(0, 1, want, endpoint=False, dtype=np.float32)
        src = np.linspace(0, 1, x.shape[-1], endpoint=False, dtype=np.float32)
        x = np.interp(grid, src, x) if mono else \
            np.stack([np.interp(grid, src, c) for c in x])
    return np.ascontiguousarray(x, dtype=np.float32)
