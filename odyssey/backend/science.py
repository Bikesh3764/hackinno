"""
Scientific discovery under uncertainty.

1. Subsurface water-ice probability: naive-Bayes evidence fusion (log-odds) of
   orbital neutron hydrogen (WEH), thermal inertia and the surface classifier.
   Uncertainty: Monte-Carlo over sensor noise + classifier tree spread.
2. Science value map  V = 1.0 P(ice) + 0.7 P(clay) + 0.25 P(bedrock) + contact bonus
3. Candidate targets = local maxima of V, each with mean, std, 90% CI and an
   evidence trail (which measurement contributed how many log-odds).
4. Ranking: utility / (1 + travel cost) with three policies:
     expected (mean), explore (UCB = mean + k std), conservative (LCB = mean - k std)
5. Mission tour: exact orienteering (max value s.t. energy budget) by bitmask DP
   (Held-Karp) over the top targets, legs from risk-aware Dijkstra paths.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.special import logit, expit

from terrain_sim import CLASSES

PRIOR_ICE = 0.08
# Gaussian likelihood models  (mean, std) for ice-bearing vs dry ground
NEUTRON_L = {"ice": (11.0, 4.5), "dry": (3.5, 2.2)}       # WEH %
THERMAL_L = {"ice": (1250.0, 380.0), "dry": (520.0, 380.0)}  # tiu


def _gauss_llr(x, L):
    (m1, s1), (m0, s0) = L["ice"], L["dry"]
    return (-0.5 * ((x - m1) / s1) ** 2 - np.log(s1)) - (-0.5 * ((x - m0) / s0) ** 2 - np.log(s0))


def ice_probability(world, cls_out, n_mc=40, seed=0):
    rng = np.random.default_rng(seed + 7)
    p_cls = cls_out["probs"][..., 5].astype(np.float64)
    p_std = cls_out["probs_std"][..., 5].astype(np.float64)
    class_prior = 1.0 / len(CLASSES)

    def posterior(neu, ti, pc):
        llr_n = np.clip(_gauss_llr(neu, NEUTRON_L), -6, 6)
        llr_t = np.clip(_gauss_llr(ti, THERMAL_L), -4, 4)
        pc = np.clip(pc, 0.01, 0.99)
        llr_s = np.clip(logit(pc) - logit(class_prior), -4, 4) * 0.6  # surface ice evidence
        lo = logit(PRIOR_ICE) + llr_n + llr_t + llr_s
        return expit(lo), (llr_n, llr_t, llr_s)

    mean_p, parts = posterior(world.neutron_obs, world.thermal_obs, p_cls)
    samples = np.empty((n_mc,) + p_cls.shape, np.float32)
    for k in range(n_mc):
        neu = world.neutron_obs + rng.normal(0, world.neutron_sigma, p_cls.shape)
        ti = world.thermal_obs * (1 + rng.normal(0, 0.08, p_cls.shape))
        pc = p_cls + rng.normal(0, 1, p_cls.shape) * p_std
        samples[k] = posterior(neu, ti, pc)[0]
    return dict(p_ice=samples.mean(0), p_ice_std=samples.std(0),
                p_ice_lo=np.percentile(samples, 5, axis=0), p_ice_hi=np.percentile(samples, 95, axis=0),
                llr_neutron=parts[0], llr_thermal=parts[1], llr_spectral=parts[2])


def science_value(cls_out, ice):
    P = cls_out["probs"].astype(np.float64)
    S = cls_out["probs_std"].astype(np.float64)
    # geological contacts (diverse neighbourhood) are scientifically interesting
    div = np.zeros(P.shape[:2])
    for c in range(P.shape[2]):
        pc = ndi.uniform_filter(P[..., c], 9)
        div -= pc * np.log(pc + 1e-9)
    div /= np.log(P.shape[2])
    v = 1.0 * ice["p_ice"] + 0.7 * P[..., 4] + 0.25 * P[..., 1] + 0.15 * div
    v_std = np.sqrt(ice["p_ice_std"] ** 2 + (0.7 * S[..., 4]) ** 2 + (0.25 * S[..., 1]) ** 2)
    return v, v_std, div


def find_targets(value, value_std, lethal, k=8, min_sep=14):
    sm = ndi.gaussian_filter(value, 2.0)
    peaks = (sm == ndi.maximum_filter(sm, size=min_sep)) & (sm > np.percentile(sm, 80))
    rr, cc = np.nonzero(peaks)
    order = np.argsort(sm[rr, cc])[::-1]
    out = []
    free = np.argwhere(~lethal)
    for i in order:
        r, c = int(rr[i]), int(cc[i])
        if lethal[r, c]:  # approach point: nearest safe cell
            d = (free[:, 0] - r) ** 2 + (free[:, 1] - c) ** 2
            j = d.argmin()
            if d[j] > 25:
                continue
            r, c = int(free[j][0]), int(free[j][1])
        if any((r - o[0]) ** 2 + (c - o[1]) ** 2 < min_sep ** 2 for o in out):
            continue
        out.append((r, c))
        if len(out) >= k:
            break
    return out


def describe_target(tid, pos, ctx):
    r, c = pos
    win = (slice(max(r - 2, 0), r + 3), slice(max(c - 2, 0), c + 3))
    P = ctx["probs"][win].reshape(-1, ctx["probs"].shape[-1]).mean(0)
    p_ice = float(ctx["p_ice"][win].mean())
    v = float(ctx["value"][win].mean())
    v_std = float(ctx["value_std"][win].mean())
    contrib = {"Water ice": 1.0 * p_ice, "Hydrated clay": 0.7 * P[4], "Bedrock outcrop": 0.25 * P[1]}
    kind = max(contrib, key=contrib.get)
    ev = [
        dict(src="Neutron spectrometer", value=f"WEH {ctx['neutron'][r, c]:.1f} %",
             llr=float(ctx["llr_neutron"][r, c])),
        dict(src="Thermal inertia", value=f"{ctx['thermal'][r, c]:.0f} tiu", llr=float(ctx["llr_thermal"][r, c])),
        dict(src="Surface classifier (ice)", value=f"{P[5] * 100:.0f} %", llr=float(ctx["llr_spectral"][r, c])),
        dict(src="1.9 um hydration band depth", value=f"{ctx['bd19'][r, c]:.3f}", llr=None),
        dict(src="Classifier: hydrated clay", value=f"{P[4] * 100:.0f} %", llr=None),
    ]
    hyp = {
        "Water ice": "Hydrogen-rich ground: candidate shallow subsurface water ice (ISRU resource).",
        "Hydrated clay": "Phyllosilicate signature: past aqueous alteration, high biosignature preservation potential.",
        "Bedrock outcrop": "Exposed bedrock: in-situ stratigraphy, good for sampling / dating.",
    }[kind]
    return dict(id=tid, r=r, c=c, kind=kind, hypothesis=hyp, value=v, value_std=v_std,
                ci90=[max(v - 1.645 * v_std, 0), v + 1.645 * v_std],
                p_ice=p_ice, p_ice_ci=[float(ctx["p_ice_lo"][r, c]), float(ctx["p_ice_hi"][r, c])],
                p_clay=float(P[4]), class_probs={CLASSES[i]: float(P[i]) for i in range(len(P))},
                evidence=ev, buried=bool(P[5] < 0.3 and p_ice > 0.5))


def rank_targets(targets, mode="expected", kappa=1.0):
    for t in targets:
        if mode == "explore":
            u = t["value"] + kappa * t["value_std"]
        elif mode == "conservative":
            u = t["value"] - kappa * t["value_std"]
        else:
            u = t["value"]
        reach = t.get("travel_cost")
        t["utility"] = float(u)
        t["score"] = float(u / (1 + (reach / 2500.0))) if reach is not None and np.isfinite(reach) else -1.0
    return sorted(targets, key=lambda t: -t["score"])


def orienteering(values, energy, budget):
    """Exact max-value path from node 0 (rover) visiting a subset of targets 1..k
    with total energy <= budget. Bitmask DP, O(2^k k^2)."""
    k = len(values)
    INF = float("inf")
    full = 1 << k
    dp = np.full((full, k), INF)
    par = -np.ones((full, k), int)
    for j in range(k):
        dp[1 << j, j] = energy[0][j + 1]
    for mask in range(full):
        for j in range(k):
            e = dp[mask, j]
            if not (mask >> j) & 1 or e == INF or e > budget:
                continue
            for nx in range(k):
                if (mask >> nx) & 1:
                    continue
                ne = e + energy[j + 1][nx + 1]
                m2 = mask | (1 << nx)
                if ne < dp[m2, nx]:
                    dp[m2, nx] = ne
                    par[m2, nx] = j
    best = (0.0, 0, -1, 0.0)
    for mask in range(1, full):
        for j in range(k):
            if dp[mask, j] <= budget:
                val = sum(values[i] for i in range(k) if (mask >> i) & 1)
                if val > best[0] + 1e-9 or (abs(val - best[0]) < 1e-9 and dp[mask, j] < best[3]):
                    best = (val, mask, j, dp[mask, j])
    val, mask, j, e = best
    order = []
    while j >= 0 and mask:
        order.append(j)
        pj = par[mask, j]
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return order, float(val), float(e) if order else 0.0
