"""Fetch a small window of a real HiRISE DTM (Jezero crater, Mars 2020 site) using
HTTP range requests, so only ~10 MB of the 700 MB file is downloaded.

Usage:  .venv\Scripts\python tools\fetch_hirise.py
Writes data/jezero_dtm.npy (float32, metres, 1 m/px) + data/jezero_dtm.json (provenance).
The app then works fully offline.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = ("https://hirise-pds.lpl.arizona.edu/PDS/DTM/ESP/ORB_023500_023599/"
       "ESP_023524_1985_ESP_023379_1985/DTEEC_023524_1985_023379_1985_U01.IMG")
OUT = Path(__file__).resolve().parent.parent / "data"
WIN = 320  # 320 m x 320 m at ~1 m/px -> 160 x 160 cells at 2 m


def get(a, b):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}", "User-Agent": "odyssey-demo"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    head = get(0, 40000).decode("latin-1")
    val = lambda k: re.search(rf"(?<![A-Z_]){k}\s*=\s*([^\s<]+)", head).group(1)
    rec = int(val("RECORD_BYTES"))
    img_rec = int(val(r"\^IMAGE"))
    lines, samples = int(val("LINES")), int(val("LINE_SAMPLES"))
    scale = float(val("MAP_SCALE"))
    base = (img_rec - 1) * rec
    best = None
    for frac in (0.35, 0.5, 0.65):
        r0 = int(lines * frac)
        raw = get(base + r0 * rec, base + (r0 + WIN) * rec - 1)
        a = np.frombuffer(raw, dtype="<f4").reshape(WIN, samples).astype(np.float64)
        a[(a < -1e30) | ~np.isfinite(a)] = np.nan
        for c0 in range(0, samples - WIN, 160):
            w = a[:, c0:c0 + WIN]
            if np.isnan(w).mean() > 0.001:
                continue
            gy, gx = np.gradient(w, scale)
            slope = np.degrees(np.arctan(np.hypot(gx, gy)))
            steep = (slope > 25).mean()
            relief = np.nanmax(w) - np.nanmin(w)
            # want interesting but mostly drivable terrain
            score = relief - 400 * max(steep - 0.08, 0)
            if steep < 0.25 and (best is None or score > best[0]):
                best = (score, r0, c0, w.copy(), steep, relief)
        print(f"rows {r0}: checked", file=sys.stderr)
    if best is None:
        sys.exit("no fully valid window found")
    _, r0, c0, w, steep, relief = best
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "jezero_dtm.npy", w.astype(np.float32))
    meta = dict(source="HiRISE DTM DTEEC_023524_1985_023379_1985_U01 (USGS), Jezero crater, Mars",
                url=URL, row0=r0, col0=c0, size_px=WIN, m_per_px=scale,
                relief_m=float(relief), steep_frac=float(steep),
                min_elev=float(np.nanmin(w)), max_elev=float(np.nanmax(w)))
    (OUT / "jezero_dtm.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
