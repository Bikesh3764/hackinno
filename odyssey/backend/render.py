"""Server-side rendering of map layers to small PNG data URIs (no matplotlib needed)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from terrain_sim import CLASS_COLORS

_STOPS = {
    "viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "magma": ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
    "ice": ["#0b0b10", "#16324f", "#2a6f97", "#61a5c2", "#d6f0ff"],
    "heat": ["#0b0b0b", "#3d1308", "#8a1c0b", "#e0461b", "#ffb347", "#fff2c2"],
    "mars": ["#2a1a14", "#5a2e1d", "#9c4a2a", "#c97a4a", "#e9c29a", "#fbefe0"],
    "gray": ["#000000", "#ffffff"],
}


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _lut(name):
    stops = np.array([_hex(h) for h in _STOPS[name]], float)
    x = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, stops[:, k]) for k in range(3)], 1).astype(np.uint8)


LUTS = {k: _lut(k) for k in _STOPS}


def to_png(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def scalar(arr, cmap="viridis", lo=None, hi=None, mask=None, mask_color=(0, 0, 0)):
    a = np.array(arr, float)
    fin = np.isfinite(a)
    lo = np.nanpercentile(a[fin], 1) if lo is None else lo
    hi = np.nanpercentile(a[fin], 99) if hi is None else hi
    t = np.clip((np.where(fin, a, lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = LUTS[cmap][(t * 255).astype(int)]
    bad = ~fin if mask is None else (mask | ~fin)
    rgb[bad] = mask_color
    return rgb


def shade(rgb, hs, k=0.55):
    return np.clip(rgb * ((1 - k) + k * hs[..., None] * 1.25), 0, 255)


def classes(pred, conf=None):
    pal = np.array([_hex(c) for c in CLASS_COLORS], float)
    rgb = pal[pred]
    if conf is not None:
        rgb = rgb * (0.35 + 0.65 * conf[..., None])
    return rgb


def false_color(bands):
    rgb = np.stack([bands[2], bands[1], bands[0]], -1).astype(float)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99.5)
    return np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8 * 255
