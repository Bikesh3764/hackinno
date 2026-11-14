"""
Risk-aware traversability cost map.

cost(cell) = E_class[terrain cost] * slope_term * roughness_term * (1 + lambda * U)
             + lambda * 8 * P(hazard) + clearance penalty
lethal     = P(hazard) > 0.5 (chance constraint), inflated by the rover footprint.

lambda (risk aversion) trades path length against uncertainty: lambda=0 trusts the
map blindly, high lambda avoids anything the models are unsure about.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from perception import SLOPE_LIMIT_DEG, ROUGH_LIMIT_M

# relative effort per terrain class (sand = slip / embedding risk, rocky = wheel damage)
TRAV_COST = np.array([1.0, 1.35, 3.2, 6.0, 1.5, 2.0])
# rolling-resistance coefficient per class (for the energy model)
ROLL_CRR = np.array([0.10, 0.07, 0.25, 0.18, 0.11, 0.12])
ROVER_RADIUS_CELLS = 1


def build_costmap(cls_out, geo, haz, lam: float = 1.0):
    probs = cls_out["probs"]
    terrain = probs @ TRAV_COST
    slope_term = 1 + 3.0 * (np.clip(geo["slope"], 0, 40) / SLOPE_LIMIT_DEG) ** 2
    rough_term = 1 + 1.5 * (geo["rough"] / ROUGH_LIMIT_M) ** 2
    unc = 0.6 * cls_out["entropy"] + 0.4 * np.clip(geo["slope_std"] / 6.0, 0, 1)
    p_haz = haz["p_hazard"]

    lethal_core = p_haz > 0.5
    dist = ndi.distance_transform_edt(~lethal_core)
    lethal = dist <= ROVER_RADIUS_CELLS
    clearance = np.where(dist <= ROVER_RADIUS_CELLS + 3, 2.0 * (ROVER_RADIUS_CELLS + 3 - dist + 1) / 4, 0)

    base = terrain * slope_term * rough_term
    risk = base * lam * unc + lam * 8.0 * p_haz
    cost = base + risk + clearance
    cost = np.where(lethal, np.inf, cost)
    crr = probs @ ROLL_CRR
    return dict(cost=cost, lethal=lethal, unc=unc, terrain=terrain, slope_term=slope_term,
                rough_term=rough_term, risk=risk, clearance=clearance, crr=crr, lam=lam,
                clearance_dist=dist)
