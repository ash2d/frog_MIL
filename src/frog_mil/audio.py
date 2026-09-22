"""Read raw audio windows, e.g. to listen to the windows a model scored highest.

    from frog_mil.audio import read_segment
    x = read_segment(row["filepath"], row["clip_start_s"], row["clip_end_s"], 32000)

``row`` is a line of ``outputs/instances.csv``. Uses only the stdlib ``wave``
module, so it runs in either environment.
"""
from __future__ import annotations

import wave

import numpy as np


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

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if sw == 1 else x / float(1 << (8 * sw - 1))
    x = x.reshape(-1, ch)
    x = x.mean(axis=1) if mono else x.T

    if sr != target_sr:
        want = int(round(len(x) * target_sr / sr)) if mono else \
               int(round(x.shape[-1] * target_sr / sr))
        grid = np.linspace(0, 1, want, endpoint=False, dtype=np.float32)
        src = np.linspace(0, 1, x.shape[-1], endpoint=False, dtype=np.float32)
        x = np.interp(grid, src, x) if mono else \
            np.stack([np.interp(grid, src, c) for c in x])
    return np.ascontiguousarray(x, dtype=np.float32)
