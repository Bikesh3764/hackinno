"""Render the console's hero image from REAL Mars elevation (HiRISE DTM, Jezero crater).

A ~1 km strip of the DTM is fetched by HTTP range request (cached in data/hero_dtm.npy),
then rendered as an oblique landscape with a simple height-field ray caster (numpy).
Output: frontend/hero.jpg.   Usage:  .venv\\Scripts\\python tools\\render_hero.py
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
URL = ("https://hirise-pds.lpl.arizona.edu/PDS/DTM/ESP/ORB_023500_023599/"
       "ESP_023524_1985_ESP_023379_1985/DTEEC_023524_1985_023379_1985_U01.IMG")
REC, IMG_REC, SAMPLES = 32940, 2, 8235      # from the PDS label (see fetch_hirise.py)
ROWS, R0 = 1100, 9900


def fetch():
    cache = ROOT / "data" / "hero_dtm.npy"
    if cache.exists():
        return np.load(cache)
    base = (IMG_REC - 1) * REC
    req = urllib.request.Request(URL, headers={"Range": f"bytes={base + R0 * REC}-{base + (R0 + ROWS) * REC - 1}",
                                               "User-Agent": "odyssey-demo"})
    with urllib.request.urlopen(req, timeout=300) as r:
        a = np.frombuffer(r.read(), dtype="<f4").reshape(ROWS, SAMPLES).astype(np.float32)
    a[(a < -1e30) | ~np.isfinite(a)] = np.nan
    # widest fully-valid column window
    valid = ~np.isnan(a).any(axis=0)
    best, cur, start, bs = 0, 0, 0, 0
    for i, v in enumerate(valid):
        cur = cur + 1 if v else 0
        if cur > best:
            best, bs = cur, i - cur + 1
    a = a[:, bs:bs + best]
    np.save(cache, a)
    return a


def render(z, W=2000, H=900):
    z = z.astype(np.float64)
    z = ndi.gaussian_filter(z, 0.6)
    n_r, n_c = z.shape
    # exaggerate relief for drama (documented on the page)
    zx = (z - np.nanmin(z)) * 2.2
    gy, gx = np.gradient(zx)
    lx, ly, lz = -0.55, 0.35, 0.45
    nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    shade = np.clip((-gx * lx - gy * ly + lz) / nrm / np.sqrt(lx ** 2 + ly ** 2 + lz ** 2), 0, 1)
    rng = np.random.default_rng(3)
    grain = ndi.gaussian_filter(rng.standard_normal(z.shape), 1.2) * 0.06
    base = np.array([[0.20, 0.09, 0.05], [0.55, 0.26, 0.13], [0.86, 0.55, 0.33], [0.97, 0.80, 0.60]])
    t = np.clip(shade ** 1.1 + grain, 0, 1) * (len(base) - 1)
    i0 = np.clip(t.astype(int), 0, len(base) - 2)
    f = (t - i0)[..., None]
    col = base[i0] * (1 - f) + base[i0 + 1] * f

    # camera looks along +row from row 0, centred in columns, above terrain
    cam_c = n_c / 2
    cam_h = np.nanmax(zx[:60]) + 55
    horizon = H * 0.36
    img = np.zeros((H, W, 3))
    ybuf = np.full(W, H, dtype=float)
    xs = np.arange(W)
    fov = 0.9
    dists = np.concatenate([np.arange(4, 200, 0.5), np.arange(200, n_r - 2, 1.0)])
    fog_col = np.array([0.70, 0.42, 0.26])
    for d in dists:
        r = int(d)
        cols = cam_c + (xs - W / 2) / (W / 2) * d * fov
        ok = (cols >= 0) & (cols < n_c - 1)
        c = np.clip(cols, 0, n_c - 1).astype(int)
        h = zx[r, c]
        y = horizon + (cam_h - h) / d * (H * 0.9)
        fog = min((d / n_r) ** 1.4, 1) * 0.85
        rgb = col[r, c] * (1 - fog) + fog_col * fog
        for x in np.nonzero(ok & (y < ybuf))[0]:
            y0 = int(max(y[x], 0))
            img[y0:int(ybuf[x]), x] = rgb[x]
            ybuf[x] = y0
    # sky: black zenith -> thin butterscotch haze at the horizon
    yy = np.arange(H)[:, None]
    sky_t = np.clip((yy - (horizon - H * 0.32)) / (H * 0.32), 0, 1) ** 2.2
    sky = (np.array([0.0, 0.0, 0.0]) * (1 - sky_t) + np.array([0.55, 0.30, 0.18]) * sky_t)[:, None, :]
    mask = (yy < ybuf[None, :])[..., None]
    img = np.where(mask, np.broadcast_to(sky, img.shape), img)
    # grade: darker overall, deeper left/top so white type lands without a scrim
    gx_ = np.linspace(0.55, 1.0, W)[None, :, None]
    gy_ = np.linspace(0.85, 1.0, H)[:, None, None]
    img = np.clip(img * gx_ * gy_ * 0.82, 0, 1) ** 1.08
    out = Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return out


def main():
    z = fetch()
    print("dtm strip", z.shape, "relief", float(np.nanmax(z) - np.nanmin(z)), file=sys.stderr)
    im = render(z)
    out = ROOT / "frontend" / "hero.jpg"
    im.save(out, quality=86, optimize=True, progressive=True)
    (ROOT / "data" / "hero.json").write_text(json.dumps(dict(source=URL, rows=[R0, R0 + ROWS], shape=list(z.shape),
                                                             vertical_exaggeration=2.2)))
    print("wrote", out)


if __name__ == "__main__":
    main()
