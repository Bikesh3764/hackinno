"""
Perception: point-cloud rasterisation, uncertainty-aware elevation fusion,
geomorphometry (slope / roughness / curvature) and probabilistic hazard maps.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr  # standard normal CDF

SLOPE_LIMIT_DEG = 25.0     # rover static stability / traction limit
ROUGH_LIMIT_M = 0.70       # max step height the wheels can climb (~wheel radius)


def inpaint_harmonic(dem: np.ndarray, iters: int = 300):
    """Fill NaN holes by solving Laplace's equation (smooth membrane) inside them.
    Returns filled DEM and distance (cells) of each filled cell to the nearest real data."""
    mask = np.isnan(dem)
    if not mask.any():
        return dem.copy(), np.zeros_like(dem)
    idx = ndi.distance_transform_edt(mask, return_distances=False, return_indices=True)
    z = dem[tuple(idx)]                      # nearest-neighbour initial guess
    for _ in range(iters):                   # Jacobi relaxation only inside holes
        p = np.pad(z, 1, mode="edge")
        avg = 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
        z[mask] = avg[mask]
    dist = ndi.distance_transform_edt(mask)
    return z, dist


def rasterize_point_cloud(points: np.ndarray, n: int, cell: float):
    """Bin a 3D point cloud onto the grid with a robust estimator.
    Uses the per-cell median and MAD so stereo outliers do not corrupt the map."""
    col = np.clip((points[:, 0] / cell).astype(int), 0, n - 1)
    row = np.clip((points[:, 1] / cell).astype(int), 0, n - 1)
    lin = row * n + col
    order = np.argsort(lin)
    lin_s, z_s = lin[order], points[order, 2]
    uniq, start, counts = np.unique(lin_s, return_index=True, return_counts=True)
    med = np.full(n * n, np.nan)
    var = np.full(n * n, np.nan)
    cnt = np.zeros(n * n)
    for u, s, c in zip(uniq, start, counts):
        zz = z_s[s:s + c]
        m = np.median(zz)
        mad = np.median(np.abs(zz - m)) * 1.4826 + 0.03
        inl = np.abs(zz - m) < 3 * mad           # reject outliers
        zi = zz[inl] if inl.any() else zz
        med[u] = zi.mean()
        var[u] = (zi.std() ** 2 + 0.05 ** 2) / max(len(zi), 1) + 0.03 ** 2
        cnt[u] = len(zi)
    return med.reshape(n, n), var.reshape(n, n), cnt.reshape(n, n)


def fuse_elevation(world):
    """Inverse-variance (1-D Kalman) fusion of orbital DEM and rover point cloud."""
    n, cell = world.n, world.cell
    orb, hole_dist = inpaint_harmonic(world.dem_obs)
    # variance grows the further a filled cell is from real measurements
    orb_var = world.dem_sigma ** 2 + (0.35 * hole_dist) ** 2
    pc_mean, pc_var, pc_cnt = rasterize_point_cloud(world.point_cloud, n, cell)
    has_pc = ~np.isnan(pc_mean)
    fused = orb.copy()
    fvar = orb_var.copy()
    w_o = 1.0 / orb_var[has_pc]
    w_p = 1.0 / pc_var[has_pc]
    fused[has_pc] = (orb[has_pc] * w_o + pc_mean[has_pc] * w_p) / (w_o + w_p)
    fvar[has_pc] = 1.0 / (w_o + w_p)
    return dict(elev=fused, elev_std=np.sqrt(fvar), hole_mask=hole_dist > 0,
                pc_mask=has_pc, pc_count=pc_cnt, n_points=len(world.point_cloud))


def _slope_deg(z, cell):
    zs = ndi.gaussian_filter(z, 1.0)
    gy, gx = np.gradient(zs, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def geomorphometry(elev, elev_std, cell, n_mc: int = 24, seed: int = 0):
    """Slope / roughness / curvature with Monte-Carlo uncertainty propagation:
    sample plausible DEMs from N(elev, elev_std) and measure the spread of slope."""
    rng = np.random.default_rng(seed)
    slope = _slope_deg(elev, cell)
    samples = np.empty((n_mc,) + elev.shape, np.float32)
    for k in range(n_mc):
        eps = ndi.gaussian_filter(rng.standard_normal(elev.shape), 0.7)
        eps /= eps.std() + 1e-9
        samples[k] = _slope_deg(elev + eps * elev_std, cell)
    slope_std = samples.std(0)
    resid = elev - ndi.gaussian_filter(elev, 2.0)
    rough = np.sqrt(ndi.uniform_filter(resid ** 2, 5))
    # roughness bias from DEM noise removed in quadrature
    rough = np.sqrt(np.clip(rough ** 2 - (0.55 * elev_std) ** 2, 0, None))
    curv = ndi.laplace(ndi.gaussian_filter(elev, 1.5)) / cell ** 2
    rel = elev - ndi.gaussian_filter(elev, 15)
    return dict(slope=slope, slope_std=slope_std, rough=rough, curv=curv, rel_elev=rel)


def hazard_probability(geo, class_probs):
    """P(hazard) per cell = 1 - prod(1 - p_i) over independent hazard sources."""
    p_steep = ndtr((geo["slope"] - SLOPE_LIMIT_DEG) / np.maximum(geo["slope_std"], 0.75))
    p_rough = ndtr((geo["rough"] - ROUGH_LIMIT_M) / 0.08)
    p_rock = np.clip((class_probs[..., 3] - 0.35) / 0.45, 0, 1)   # boulder field
    p_haz = 1 - (1 - p_steep) * (1 - p_rough) * (1 - 0.85 * p_rock)
    return dict(p_hazard=p_haz, p_steep=p_steep, p_rough=p_rough, p_rock=p_rock)
