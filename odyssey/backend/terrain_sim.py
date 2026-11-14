"""
Synthetic Mars world + orbital sensor simulator for MARS-09.

Produces two things:
  * TRUTH  - what the terrain really is (only used to score the models and to
             drive the rover's local sensors during the traverse simulation).
  * OBSERVATIONS - what the rover/orbiter actually gets: a noisy DEM with holes,
             6-band multispectral image, thermal inertia, a coarse neutron
             (hydrogen) map and a near-field stereo point cloud with outliers.

Everything downstream (perception, ML, planning, science) only sees OBSERVATIONS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

CLASSES = ["Regolith", "Bedrock", "Sand / dunes", "Rocky field", "Hydrated clay", "Ice-rich ground"]
CLASS_COLORS = ["#c1693f", "#6e5a52", "#e0b56b", "#3b3330", "#4fa59b", "#d9ecf7"]
BANDS = ["B1 450nm", "B2 550nm", "B3 650nm", "B4 860nm", "B5 1.9um", "B6 2.3um"]

# Reflectance signatures per class (rows) and band (cols).
# Clay: absorptions at 1.9um (H2O) and 2.3um (Fe/Mg-OH). Ice: bright, deep 1.9um.
SIGNATURES = np.array([
    [0.100, 0.160, 0.280, 0.330, 0.310, 0.300],  # regolith (dusty, red)
    [0.080, 0.110, 0.170, 0.210, 0.205, 0.200],  # bedrock (dark basalt)
    [0.070, 0.090, 0.130, 0.160, 0.158, 0.155],  # sand (dark basaltic)
    [0.090, 0.130, 0.210, 0.250, 0.240, 0.232],  # rocky (spectrally ~bedrock+dust)
    [0.120, 0.170, 0.250, 0.310, 0.225, 0.240],  # hydrated clay
    [0.240, 0.270, 0.310, 0.330, 0.200, 0.270],  # ice-rich
])
THERMAL_INERTIA = np.array([250.0, 1200.0, 180.0, 700.0, 450.0, 1400.0])  # tiu

MARS_G = 3.71


@dataclass
class World:
    n: int
    cell: float  # metres per cell
    seed: int
    # --- truth ---
    true_dem: np.ndarray
    true_class: np.ndarray
    true_weh: np.ndarray          # water-equivalent hydrogen, %
    hidden_rocks: np.ndarray      # sub-resolution boulders (not in orbital DEM)
    # --- observations ---
    dem_obs: np.ndarray           # NaN = missing (shadow / stereo failure)
    dem_sigma: float
    bands: np.ndarray             # (6, n, n)
    thermal_obs: np.ndarray
    neutron_obs: np.ndarray       # WEH % estimate, coarse footprint
    neutron_sigma: float
    point_cloud: np.ndarray       # (k, 3) x,y,z metres, near the landing site
    start: tuple
    goal: tuple
    meta: dict = field(default_factory=dict)


def _fractal(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """1/f^beta Gaussian random field (spectral synthesis), zero mean, unit std."""
    white = rng.standard_normal((n, n))
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0
    f = np.real(np.fft.ifft2(np.fft.fft2(white) * k ** (-beta / 2.0)))
    f -= f.mean()
    return f / (f.std() + 1e-9)


def _hillshade(z: np.ndarray, cell: float, az=315.0, alt=35.0) -> np.ndarray:
    gy, gx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az_r, alt_r = np.radians(az), np.radians(alt)
    hs = np.sin(alt_r) * np.cos(slope) + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect)
    return np.clip(hs, 0, 1)


def generate_world(seed: int = 7, n: int = 160, cell: float = 2.0,
                   dem_noise: float = 0.35, dropout: float = 0.04,
                   spectral_noise: float = 0.008, base_dem: np.ndarray | None = None,
                   source: str = "procedural") -> World:
    """base_dem: optional REAL elevation (n x n, metres, e.g. a HiRISE DTM crop). When given,
    real topography replaces the procedural craters/cliff; surface materials and all sensor
    observations are still simulated on top of it (so accuracy can still be scored)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[0:n, 0:n].astype(float)
    cls = np.zeros((n, n), dtype=np.int8)  # regolith default
    if base_dem is not None:
        return _finish_world(rng, n, cell, seed, rr, cc, base_dem - np.median(base_dem), cls, [], None,
                             dem_noise, dropout, spectral_noise, source)

    # ---------------- elevation ----------------
    z = 7.0 * _fractal(n, 4.3, rng) + 0.6 * _fractal(n, 3.0, rng) + 0.04 * (rr - cc) * cell * 0.5

    # scarp (cliff) with two gentle ramp gaps -> forces interesting routing
    theta = rng.uniform(0.2, 1.2)
    nx, ny = np.cos(theta), np.sin(theta)
    off = rng.uniform(-15, 15)
    s = (rr - n / 2) * nx + (cc - n / 2) * ny + off
    along = -(rr - n / 2) * ny + (cc - n / 2) * nx
    gaps = [rng.uniform(-50, -20), rng.uniform(20, 50)]
    ramp = sum(np.exp(-((along - g) / 8.0) ** 2) for g in gaps)
    w = 1.2 + 14.0 * np.clip(ramp, 0, 1)
    z += 9.0 * 0.5 * (1 + np.tanh(s / w))
    scarp_face = (np.abs(s) < 2.5) & (w < 4)
    gap_pts = [(n / 2 - off * nx - g * ny, n / 2 - off * ny + g * nx) for g in gaps]

    # craters (kept clear of the scarp ramps so the map stays connected)
    craters = []
    tries = 0
    target_k = rng.integers(5, 8)
    while len(craters) < target_k and tries < 200:
        tries += 1
        R = rng.uniform(7, 18)
        cr, cc0 = rng.uniform(R, n - R, size=2)
        if any(np.hypot(cr - gr, cc0 - gc) < R + 14 for gr, gc in gap_pts):
            continue
        if any(np.hypot(cr - o[0], cc0 - o[1]) < R + o[2] + 4 for o in craters):
            continue
        d = np.hypot(rr - cr, cc - cc0) / R
        depth = 0.22 * R * cell
        bowl = np.where(d < 1, depth * (d ** 2 - 1), 0.0)
        rim = 0.10 * R * cell * np.exp(-((d - 1.0) / 0.18) ** 2)
        z += bowl + rim
        craters.append((cr, cc0, R, d))
    return _finish_world(rng, n, cell, seed, rr, cc, z, cls, craters, scarp_face,
                         dem_noise, dropout, spectral_noise, source)


def _finish_world(rng, n, cell, seed, rr, cc, z, cls, craters, scarp_face,
                  dem_noise, dropout, spectral_noise, source):
    if scarp_face is None:  # real terrain: steep natural slopes are exposed bedrock
        gy, gx = np.gradient(z, cell)
        scarp_face = np.degrees(np.arctan(np.hypot(gx, gy))) > 20

    # dune field
    dune_mask = ndi.gaussian_filter(rng.standard_normal((n, n)), 14) > 0
    dune_mask &= _fractal(n, 4.0, rng) > 0.55
    dune_mask = ndi.binary_opening(dune_mask, iterations=2)
    phi = rng.uniform(0, np.pi)
    wave = np.sin(2 * np.pi * (rr * np.cos(phi) + cc * np.sin(phi)) / 7.0 + 0.6 * _fractal(n, 3, rng))
    z += np.where(dune_mask, 0.9 * (wave + 1) ** 1.5 / 2.8, 0.0)

    # rocky field (boulder field) - high roughness
    rocky_mask = _fractal(n, 3.5, rng) > 1.0
    rocky_mask = ndi.binary_opening(rocky_mask, iterations=1) & ~dune_mask
    rocks = np.clip(rng.standard_normal((n, n)), 0, None) ** 2 * 0.55
    z += np.where(rocky_mask, rocks, 0.0)

    # bedrock exposures
    bedrock_mask = (_fractal(n, 3.0, rng) > 0.9) | scarp_face
    # ------------- classes (priority order) -------------
    cls[bedrock_mask] = 1
    cls[dune_mask] = 2
    cls[rocky_mask] = 3
    # crater floors: some clay, some ice
    order = rng.permutation(len(craters))
    for k, idx in enumerate(order[:4]):
        cr, cc0, R, d = craters[idx]
        floor = d < 0.65
        cls[floor] = 4 if k % 2 == 0 else 5
    if not craters:  # real terrain: clay / ice collect in the lowest local basins
        tpi = z - ndi.gaussian_filter(z, 10)
        lows, nl = ndi.label(tpi < np.percentile(tpi, 7))
        if nl:
            sizes = ndi.sum(np.ones_like(z), lows, range(1, nl + 1))
            for k, lab in enumerate(np.argsort(sizes)[::-1][:4] + 1):
                cls[(lows == lab) & (cls == 0)] = 4 if k % 2 == 0 else 5
    # extra layered clay deposit + polygonal ice terrain patches
    clay_patch = (_fractal(n, 3.6, rng) > 1.35) & (cls == 0)
    cls[clay_patch] = 4
    ice_patch = (_fractal(n, 3.6, rng) > 1.45) & (cls == 0)
    cls[ice_patch] = 5

    # ---------------- subsurface water (WEH %) ----------------
    weh = 2.5 + 0.8 * _fractal(n, 3, rng)
    weh += 22.0 * ndi.gaussian_filter((cls == 5).astype(float), 2)
    weh += 6.0 * ndi.gaussian_filter((cls == 4).astype(float), 2)
    # buried ice: invisible to cameras, only neutrons/thermal see it
    br, bc = rng.uniform(0.25 * n, 0.75 * n, size=2)
    buried = np.exp(-(((rr - br) ** 2 + (cc - bc) ** 2) / (2 * 9.0 ** 2)))
    weh += 16.0 * buried
    weh = np.clip(weh, 0.5, 60)

    # hidden boulders (below orbital resolution)
    hidden = (rng.random((n, n)) < 0.0035) & (cls != 2)
    hidden = ndi.binary_dilation(hidden, iterations=1)
    true_dem = z + np.where(hidden, 1.1, 0.0)

    # ---------------- observations ----------------
    dem_obs = ndi.gaussian_filter(z, 0.8) + rng.normal(0, dem_noise, (n, n))
    holes = np.zeros((n, n), bool)
    target = dropout * n * n
    while holes.sum() < target:
        h, w_ = rng.integers(4, 14, size=2)
        r0, c0 = rng.integers(0, n - h), rng.integers(0, n - w_)
        holes[r0:r0 + h, c0:c0 + w_] = True
    dem_obs[holes] = np.nan

    # multispectral: signature * illumination * natural variability + dust + noise
    illum = 0.55 + 0.55 * _hillshade(z, cell)
    variab = 1 + 0.06 * _fractal(n, 2.5, rng)
    dust = np.clip(_fractal(n, 3.2, rng) - 0.6, 0, 1.2) / 1.2  # dust mantling
    refl = SIGNATURES[cls]  # (n, n, 6)
    refl = (1 - 0.6 * dust[..., None]) * refl + 0.6 * dust[..., None] * SIGNATURES[0]
    bands = refl * (illum * variab)[..., None] + rng.normal(0, spectral_noise, (n, n, 6))
    bands = np.clip(bands, 0.005, 1).transpose(2, 0, 1).astype(np.float32)

    ti = THERMAL_INERTIA[cls] * (1 + 0.10 * rng.standard_normal((n, n))) + 22.0 * (weh - 2.5)
    thermal_obs = ndi.gaussian_filter(ti, 1.5).astype(np.float32)

    neutron_sigma = 1.6
    neutron_obs = ndi.gaussian_filter(weh, 6.0) + rng.normal(0, neutron_sigma, (n, n))
    neutron_obs = ndi.gaussian_filter(neutron_obs, 1.0)

    # ---------------- start / goal ----------------
    start = (n - 14, 12)
    goal = (14, n - 14)

    # near-field stereo point cloud around the landing site (with 1.5% outliers)
    k = 26000
    ang = rng.uniform(0, 2 * np.pi, k)
    rad = 26 * np.sqrt(rng.uniform(0, 1, k))
    pr = np.clip(start[0] + rad * np.sin(ang), 0, n - 1.001)
    pc = np.clip(start[1] + rad * np.cos(ang), 0, n - 1.001)
    pz = true_dem[pr.astype(int), pc.astype(int)] + rng.normal(0, 0.05 + 0.002 * rad, k)
    out = rng.random(k) < 0.015
    pz[out] += rng.normal(0, 6, out.sum())
    point_cloud = np.stack([pc * cell, pr * cell, pz], axis=1).astype(np.float32)

    return World(n=n, cell=cell, seed=seed, true_dem=true_dem, true_class=cls,
                 true_weh=weh, hidden_rocks=hidden, dem_obs=dem_obs, dem_sigma=dem_noise,
                 bands=bands, thermal_obs=thermal_obs, neutron_obs=neutron_obs,
                 neutron_sigma=neutron_sigma, point_cloud=point_cloud,
                 start=start, goal=goal,
                 meta=dict(dropout=float(holes.mean()), buried_ice=(float(br), float(bc)), source=source))
