"""
Autonomous traverse simulation (closed loop, partial observability).

The rover starts with the ORBITAL belief map. While driving it senses the TRUE
terrain within a short radius (navcam / hazcam stereo), updates its belief and
replans whenever the remaining route becomes unsafe - no waiting for Earth.
It also learns online: measured wheel slip on sand raises the cost of sand.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from perception import _slope_deg, SLOPE_LIMIT_DEG, ROUGH_LIMIT_M
from costmap import TRAV_COST, ROLL_CRR, ROVER_RADIUS_CELLS
from planner import astar, nearest_free, ROVER_MASS, DRIVE_EFF, DRIVE_SPEED
from terrain_sim import MARS_G, CLASSES

ICE_WEH = 12.0  # water-equivalent hydrogen % above which ground counts as ice-bearing


def true_hazard_map(world):
    slope = _slope_deg(world.true_dem, world.cell)
    resid = world.true_dem - ndi.gaussian_filter(world.true_dem, 2.0)
    rough = np.sqrt(ndi.uniform_filter(resid ** 2, 5))
    core = (slope > SLOPE_LIMIT_DEG) | (rough > ROUGH_LIMIT_M) | world.hidden_rocks
    return core, slope, rough


def simulate(world, ctx, waypoints, names, inject=True, sense_r=6, comm_delay_min=12.5, seed=0):
    rng = np.random.default_rng(seed + 33)
    n, cell = world.n, world.cell
    core, t_slope, _ = true_hazard_map(world)
    core = core.copy()
    belief = ctx["cost"].copy()
    elev = ctx["elev"]
    sensed = np.zeros((n, n), bool)
    pred_cls = ctx["pred"]
    rr, cc = np.mgrid[-sense_r:sense_r + 1, -sense_r:sense_r + 1]
    disk = rr ** 2 + cc ** 2 <= sense_r ** 2
    offs = np.stack([rr[disk], cc[disk]], 1)

    log, frames, paths, revealed, cleared = [], [], [], [], []
    t_min = 0.0
    energy = 0.0
    sand_learned = False
    injected = False
    decisions = 0
    science_checks = []

    def L(kind, msg):
        log.append(dict(step=len(frames), t=round(t_min, 1), kind=kind, msg=msg))

    def true_lethal_near(r, c):
        r0, r1 = max(r - ROVER_RADIUS_CELLS, 0), r + ROVER_RADIUS_CELLS + 1
        c0, c1 = max(c - ROVER_RADIUS_CELLS, 0), c + ROVER_RADIUS_CELLS + 1
        return core[r0:r1, c0:c1].any()

    def sense(r, c):
        newly_haz, newly_clear = [], []
        for dr, dc in offs:
            a, b = r + dr, c + dc
            if a < 0 or b < 0 or a >= n or b >= n or sensed[a, b]:
                continue
            sensed[a, b] = True
            if true_lethal_near(a, b) and not known_safe[a, b]:
                if np.isfinite(belief[a, b]):
                    newly_haz.append((int(a), int(b)))
                belief[a, b] = np.inf
            else:
                tc = world.true_class[a, b]
                c_true = TRAV_COST[tc] * (1 + 3.0 * (min(t_slope[a, b], 40) / SLOPE_LIMIT_DEG) ** 2)
                if sand_learned and pred_cls[a, b] == 2:
                    c_true *= 1.4
                if not np.isfinite(belief[a, b]):
                    newly_clear.append((int(a), int(b)))
                belief[a, b] = c_true
        return newly_haz, newly_clear

    pos = tuple(int(x) for x in waypoints[0])
    # cells the rover has physically occupied (landing zone + its own tracks) are known traversable
    known_safe = np.zeros((n, n), bool)
    known_safe[max(pos[0] - 2, 0):pos[0] + 3, max(pos[1] - 2, 0):pos[1] + 3] = True
    belief[known_safe & ~np.isfinite(belief)] = 2.0
    L("info", f"Autonomy engaged at ({pos[0]},{pos[1]}). One-way light time {comm_delay_min:.1f} min: "
              f"rover plans on board, Earth gets a summary.")
    sense(*pos)
    frames.append(dict(r=pos[0], c=pos[1], e=0.0))

    for leg, goal in enumerate(waypoints[1:], 1):
        goal = nearest_free(~np.isfinite(belief), tuple(int(x) for x in goal))
        res = astar(belief, elev, cell, pos, goal)
        decisions += 1
        if not res["found"]:
            L("warn", f"No safe route to {names[leg]}. Skipping waypoint, flagging for Earth review.")
            continue
        path = res["path"]
        paths.append(dict(step=len(frames), path=path, reason=f"Leg {leg}: plan to {names[leg]}"))
        L("plan", f"Leg {leg} -> {names[leg]}: A* found {len(path)}-cell route "
                  f"({res['expanded']} nodes, {res['ms']:.0f} ms).")
        if inject and not injected and len(path) > 30:
            k = int(len(path) * 0.45)
            br, bc = path[k]
            yy, xx = np.ogrid[:n, :n]
            blob = (yy - br) ** 2 + (xx - bc) ** 2 <= 4
            core[blob] = True
            injected = True
            L("inject", f"[Fault injection] Unmapped boulder field placed on the route at ({br},{bc}) - "
                        f"invisible from orbit.")
        i = 0
        guard = 0
        while path and (path[i] != goal) and guard < 4 * n * n:
            guard += 1
            nxt = path[i + 1]
            # drive one cell
            a, b = nxt
            tc = world.true_class[a, b]
            dz = elev[a, b] - elev[pos]
            d = np.hypot(a - pos[0], b - pos[1]) * cell
            slip = 0.0
            if tc == 2:
                slip = float(rng.uniform(0.25, 0.45))
                if not sand_learned:
                    sand_learned = True
                    m = np.isfinite(belief) & (pred_cls == 2) & ~sensed
                    belief[m] *= 1.4
                    L("learn", f"Wheel slip {slip * 100:.0f}% measured on {CLASSES[2].lower()} (visual odometry vs "
                               f"wheel count). Online update: sand traverse cost x1.4 for the rest of the map.")
            seg = np.hypot(d, dz) / (1 - slip)
            energy += ROVER_MASS * MARS_G * (ROLL_CRR[tc] * seg + max(dz, 0)) / DRIVE_EFF / 3600
            t_min += seg / DRIVE_SPEED / 60
            pos = (int(a), int(b))
            known_safe[pos] = True
            i += 1
            haz, clr = sense(*pos)
            frames.append(dict(r=pos[0], c=pos[1], e=round(energy, 2)))
            if clr:
                cleared.append(dict(step=len(frames) - 1, cells=clr))
            if haz:
                revealed.append(dict(step=len(frames) - 1, cells=haz))
            ahead = path[i:]
            blocked = [p for p in ahead if not np.isfinite(belief[p])]
            if blocked:
                br, bc = blocked[0]
                dist_m = np.hypot(br - pos[0], bc - pos[1]) * cell
                old_len = len(ahead)
                if not np.isfinite(belief[goal]):
                    goal = nearest_free(~np.isfinite(belief), goal)
                    L("warn", f"Waypoint itself is unsafe on close inspection; shifted to nearest safe cell {goal}.")
                res = astar(belief, elev, cell, pos, goal)
                decisions += 1
                if not res["found"]:
                    L("warn", "Route blocked and no alternative found. Safing rover, waiting for Earth.")
                    path = []
                    break
                path, i = res["path"], 0
                paths.append(dict(step=len(frames) - 1, path=path, reason="replan: hazard ahead"))
                last = log[-1] if log else None
                if last and last["kind"] == "replan" and len(frames) - last["step"] <= 4:
                    last["refined"] = last.get("refined", 0) + 1
                    last["step"] = len(frames)
                    base = last["msg"].split(" [detour refined")[0]
                    last["msg"] = base + f" [detour refined {last['refined']}x as more of the obstacle came into view]"
                    continue
                L("replan", f"Hazard detected {dist_m:.0f} m ahead at ({br},{bc}) ({len(haz)} new hazard cells). "
                            f"Replanned on board in {res['ms']:.0f} ms ({res['expanded']} nodes): "
                            f"{old_len} -> {len(path)} cells. Earth-in-the-loop would have cost "
                            f"{2 * comm_delay_min:.0f} min of idle time.")
        if path and pos == goal:
            arrive = names[leg]
            if leg < len(waypoints) - 1 or arrive.startswith("T"):
                true_w = float(world.true_weh[pos] + rng.normal(0, 0.7))
                pred = ctx["p_ice"][pos]
                ok = (true_w > ICE_WEH) == (pred > 0.5)
                verdict = "orbital prediction CONFIRMED" if ok else "orbital prediction WRONG - model flagged for retraining"
                L("science", f"Arrived at {arrive}. In-situ neutron/drill reading WEH {true_w:.1f} % "
                             f"(ice if > {ICE_WEH:.0f} %; orbital P(ice)={pred:.2f}) -> {verdict}.")
                science_checks.append(dict(target=arrive, p_ice=float(pred), weh=true_w, correct=bool(ok)))
            else:
                L("info", f"Arrived at {arrive}.")
    L("done", f"Traverse complete: {len(frames) - 1} cells driven, {energy:.1f} Wh used, "
              f"{t_min / 60:.1f} h drive time, {decisions} autonomous planning decisions, "
              f"{sum(len(x['cells']) for x in revealed)} hazard cells discovered on the ground, "
              f"{sum(len(x['cells']) for x in cleared)} orbital false alarms cleared.")
    return dict(frames=frames, paths=paths, revealed=revealed, cleared=cleared, log=log,
                energy_wh=energy, drive_h=t_min / 60, decisions=decisions, science_checks=science_checks)
