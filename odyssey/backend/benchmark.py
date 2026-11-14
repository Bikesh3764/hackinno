"""Robustness benchmark: run the whole mission on many independent worlds and report
mean / spread of every headline metric (so judges see it is not one lucky seed).

    .venv\Scripts\python backend\benchmark.py [n_seeds]
Writes data/benchmark.json, which the console's Model tab shows.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import real_terrain
import rover_sim
from mission import Mission

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"


def run_one(seed: int, terrain: str) -> dict:
    m = Mission()
    s = m.build(seed=seed, terrain=terrain)
    met = s["metrics"]
    p = m.plan(s["start"], s["goal"])
    comp = {c["key"]: c for c in p["comparison"]}
    ps = lambda k: comp[k]["stats"]["p_success"] if comp[k]["found"] else None
    tour = m.tour(s["start"], 250.0)
    tr = m.traverse(tour["waypoints"], ["Rover"] + tour["order"], inject=True)
    truth_haz = rover_sim.true_hazard_map(m.world)[0]
    hits = sum(bool(truth_haz[int(f["r"]), int(f["c"])]) for f in tr["frames"])
    checks = tr["science_checks"]
    return dict(
        seed=seed, terrain=terrain,
        accuracy=met["accuracy"], ece=met["ece"], macro_f1=met["macro_f1"],
        selective50=dict(met["selective"]).get(0.5),
        p_safe_astar=ps("astar"), p_safe_naive=ps("naive"), p_safe_rrt=ps("rrt"),
        len_astar=comp["astar"]["stats"]["length_m"] if comp["astar"]["found"] else None,
        len_naive=comp["naive"]["stats"]["length_m"] if comp["naive"]["found"] else None,
        targets_visited=len(tour["order"]), tour_value=tour["total_value"],
        decisions=tr["decisions"], hazard_cells_entered=hits,
        hazards_discovered=sum(len(x["cells"]) for x in tr["revealed"]),
        ice_checks=len(checks), ice_correct=sum(c["correct"] for c in checks),
        build_s=sum(s["timings"].values()) / 1000,
    )


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("seed", "terrain")]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r[k] is not None], float)
        if len(v):
            out[k] = dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()))
    out["ice_agreement"] = sum(r["ice_correct"] for r in rows) / max(sum(r["ice_checks"] for r in rows), 1)
    out["ice_checks_total"] = sum(r["ice_checks"] for r in rows)
    out["safe_traverses"] = sum(r["hazard_cells_entered"] == 0 for r in rows)
    out["astar_beats_naive"] = sum((r["p_safe_astar"] or 0) >= (r["p_safe_naive"] or 0) for r in rows)
    out["n"] = len(rows)
    return out


def main(n_seeds=10):
    t0 = time.time()
    terrains = ["procedural"] + (["jezero"] if real_terrain.available() else [])
    res = {"seeds": list(range(1, n_seeds + 1)), "terrains": {}}
    for terr in terrains:
        rows = []
        for sd in res["seeds"]:
            rows.append(run_one(sd, terr))
            print(terr, sd, {k: rows[-1][k] for k in ("accuracy", "p_safe_astar", "p_safe_naive",
                                                    "hazard_cells_entered", "ice_correct", "ice_checks")},
                  flush=True)
        res["terrains"][terr] = dict(rows=rows, summary=summarize(rows))
    res["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT} in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
