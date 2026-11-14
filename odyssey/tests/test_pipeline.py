"""Correctness tests for the core algorithms.  Run:  .venv\\Scripts\\python -m pytest -q"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from planner import astar, build_graph, cost_to_go, rrt_star  # noqa: E402
from perception import inpaint_harmonic, rasterize_point_cloud  # noqa: E402
from science import orienteering  # noqa: E402
from mission import Mission  # noqa: E402


def _grid(seed=0, n=40):
    rng = np.random.default_rng(seed)
    cost = rng.uniform(1, 5, (n, n))
    cost[rng.random((n, n)) < 0.15] = np.inf
    cost[0, 0] = cost[-1, -1] = 1.0
    elev = rng.normal(0, 0.5, (n, n))
    return cost, elev


def test_astar_matches_dijkstra_and_scipy():
    for seed in range(5):
        cost, elev = _grid(seed)
        a = astar(cost, elev, 2.0, (0, 0), (39, 39), True)
        d = astar(cost, elev, 2.0, (0, 0), (39, 39), False)
        if not a["found"]:
            assert not d["found"]
            continue
        assert abs(a["cost"] - d["cost"]) < 1e-6           # admissible heuristic -> optimal
        assert a["expanded"] <= d["expanded"]                # heuristic never explores more
        dist, _ = cost_to_go(build_graph(cost, elev, 2.0), [(0, 0)], cost.shape)
        assert abs(dist[0][39 * 40 + 39] - a["cost"]) < 1e-6  # independent C implementation agrees


def test_paths_never_enter_lethal_cells():
    cost, elev = _grid(3)
    for res in (astar(cost, elev, 2.0, (0, 0), (39, 39)), rrt_star(cost, 2.0, (0, 0), (39, 39), n_iter=1500)):
        if res["found"]:
            assert all(np.isfinite(cost[r, c]) for r, c in res["path"])


def test_orienteering_is_exact():
    rng = np.random.default_rng(1)
    k = 6
    E = rng.uniform(5, 40, (k + 1, k + 1))
    vals = list(rng.uniform(0.2, 1.5, k))
    budget = 70.0
    order, val, used = orienteering(vals, E.tolist(), budget)
    best = 0.0
    for r in range(1, k + 1):
        for perm in itertools.permutations(range(k), r):
            e = E[0][perm[0] + 1] + sum(E[a + 1][b + 1] for a, b in zip(perm, perm[1:]))
            if e <= budget:
                best = max(best, sum(vals[i] for i in perm))
    assert abs(val - best) < 1e-9 and used <= budget


def test_inpainting_fills_holes_smoothly():
    yy, xx = np.mgrid[0:50, 0:50]
    z = 0.1 * xx + 0.05 * yy
    z_obs = z.copy()
    z_obs[20:30, 20:30] = np.nan
    filled, dist = inpaint_harmonic(z_obs)
    assert not np.isnan(filled).any()
    assert np.abs(filled - z).max() < 0.05                  # a plane is harmonic -> exact recovery
    assert dist[25, 25] > dist[20, 20]                      # uncertainty grows into the hole


def test_point_cloud_rejects_outliers():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 2, 500), rng.uniform(0, 2, 500), rng.normal(5, 0.02, 500)])
    pts[:20, 2] += 30                                        # 4% gross outliers
    med, var, cnt = rasterize_point_cloud(pts, 4, 2.0)
    assert abs(med[0, 0] - 5) < 0.05


def test_full_pipeline():
    m = Mission()
    s = m.build(seed=7)
    assert s["stats"]["accuracy"] > 0.85
    p = m.plan(s["start"], s["goal"])
    risk_aware = next(c for c in p["comparison"] if c["key"] == "astar")
    naive = next(c for c in p["comparison"] if c["key"] == "naive")
    assert risk_aware["found"] and naive["found"]
    assert risk_aware["stats"]["p_success"] >= naive["stats"]["p_success"]   # risk model buys safety
    tour = m.tour(s["start"], 250)
    assert tour["energy_wh"] <= 250
    sim = m.traverse(tour["waypoints"], ["Start"] + tour["order"])
    assert any(l["kind"] == "replan" for l in sim["log"])                      # injected hazard handled


def test_real_jezero_terrain_pipeline():
    import pytest
    import real_terrain
    if not real_terrain.available():
        pytest.skip("Jezero DTM not cached (run tools/fetch_hirise.py)")
    z, meta = real_terrain.load_jezero()
    assert z.shape == (160, 160) and np.isfinite(z).all()
    assert 10 < np.ptp(z) < 200            # real relief in metres
    m = Mission()
    s = m.build(seed=5, terrain="jezero")
    assert s["terrain"] == "jezero" and s["stats"]["accuracy"] > 0.8
    p = m.plan(s["start"], s["goal"])
    a = next(c for c in p["comparison"] if c["key"] == "astar")
    assert a["found"]


def test_pareto_risk_weight_increases_safety():
    m = Mission()
    s = m.build(seed=7)
    pts = m.pareto(s["start"], s["goal"])["points"]
    assert len(pts) >= 5
    # adding the risk term must never make the route less safe than lambda = 0 by a noticeable margin
    assert max(p["p_success"] for p in pts[1:]) >= pts[0]["p_success"] - 1e-6
    assert m.ctx["lam"] == 1.0  # lambda restored afterwards
    assert "# MARS-09 Odyssey mission report" in m.report()
