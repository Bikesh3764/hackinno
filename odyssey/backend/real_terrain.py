"""Real Mars topography: a crop of a HiRISE DTM (Jezero crater) cached in data/.

Fetched once with tools/fetch_hirise.py (HTTP range requests, ~10 MB); afterwards offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def available() -> bool:
    return (DATA / "jezero_dtm.npy").exists()


def load_jezero(n: int = 160) -> tuple[np.ndarray, dict]:
    """Return an n x n elevation grid (metres) at ~2 m/cell and provenance metadata."""
    a = np.load(DATA / "jezero_dtm.npy").astype(float)
    meta = json.loads((DATA / "jezero_dtm.json").read_text())
    f = a.shape[0] // n
    a = a[:n * f, :n * f].reshape(n, f, n, f).mean(axis=(1, 3))  # block-average 1 m -> 2 m
    meta["cell_m"] = meta["m_per_px"] * f
    return a, meta
