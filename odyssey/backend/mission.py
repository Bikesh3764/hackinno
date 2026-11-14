"""Mission pipeline: world -> perception -> ML -> hazards -> cost map -> science -> plans."""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi

import render
from terrain_sim import generate_world, CLASSES, CLASS_COLORS
from perception import fuse_elevation, geomorphometry, hazard_probability, SLOPE_LIMIT_DEG
from classifier import train_and_predict
from costmap import build_costmap
from planner import (astar, rrt_star, build_graph, cost_to_go, backtrack, path_stats, nearest_free)
from science import ice_probability, science_value, find_targets, describe_target, rank_targets, orienteering
from terrain_sim import _hillshade
import rover_sim
import real_terrain


class Mission:
    def __init__(self):
        self.world = None
        self.ctx = None

    # ------------------------------------------------------------ build
    def build(self, seed=7, dem_noise=0.35, dropout=0.04, spectral_noise=0.008, lam=1.0, terrain="procedural"):
        timings = {}
        t = time.perf_counter()
        base, self.terrain_meta = None, None
        if terrain == "jezero" and real_terrain.available():
            base, self.terrain_meta = real_terrain.load_jezero()
        w = generate_world(seed=seed, dem_noise=dem_noise, dropout=dropout, spectral_noise=spectral_noise,
                           base_dem=base, source="jezero" if base is not None else "procedural")
        timings["simulate sensors"] = time.perf_counter() - t
        t = time.perf_counter()
        fused = fuse_elevation(w)
        geo = geomorphometry(fused["elev"], fused["elev_std"], w.cell, seed=seed)
        timings["elevation fusion + geomorphometry"] = time.perf_counter() - t
        t = time.perf_counter()
        cls = train_and_predict(w, geo, seed=seed)
        timings["terrain classifier (RF, 80 trees)"] = time.perf_counter() - t
        t = time.perf_counter()
        haz = hazard_probability(geo, cls["probs"])
        ice = ice_probability(w, cls)
        value, value_std, div = science_value(cls, ice)
        timings["hazards + ice fusion"] = time.perf_counter() - t
        b = w.bands.astype(float)
        self.world = w
        self.ctx = dict(
            n=w.n, cell=w.cell, elev=fused["elev"], elev_std=fused["elev_std"], hole_mask=fused["hole_mask"],
            pc_mask=fused["pc_mask"], n_points=fused["n_points"],
            slope=geo["slope"], slope_std=geo["slope_std"], rough=geo["rough"], curv=geo["curv"],
            probs=cls["probs"], probs_std=cls["probs_std"], pred=cls["pred"], entropy=cls["entropy"],
            aleatoric=cls["aleatoric"], epistemic=cls["epistemic"], train_mask=cls["train_mask"],
            metrics=cls["metrics"], p_hazard=haz["p_hazard"], p_steep=haz["p_steep"], p_rough=haz["p_rough"],
            p_rock=haz["p_rock"], p_ice=ice["p_ice"], p_ice_std=ice["p_ice_std"], p_ice_lo=ice["p_ice_lo"],
            p_ice_hi=ice["p_ice_hi"], llr_neutron=ice["llr_neutron"], llr_thermal=ice["llr_thermal"],
            llr_spectral=ice["llr_spectral"], value=value, value_std=value_std,
            neutron=w.neutron_obs, thermal=w.thermal_obs, bd19=1 - b[4] / (0.5 * (b[3] + b[5])),
            hillshade=_hillshade(fused["elev"], w.cell), timings=timings,
        )
        self.set_risk(lam)
        return self.summary()

    def set_risk(self, lam):
        t = time.perf_counter()
        cm = build_costmap(dict(probs=self.ctx["probs"], entropy=self.ctx["entropy"]),
                           dict(slope=self.ctx["slope"], slope_std=self.ctx["slope_std"], rough=self.ctx["rough"]),
                           dict(p_hazard=self.ctx["p_hazard"]), lam=lam)
        self.ctx.update(cost=cm["cost"], lethal=cm["lethal"], unc=cm["unc"], terrain_cost=cm["terrain"],
                        slope_term=cm["slope_term"], rough_term=cm["rough_term"], risk_cost=cm["risk"],
                        clearance=cm["clearance"], crr=cm["crr"], clearance_dist=cm["clearance_dist"], lam=lam)
        lab, _ = ndi.label(~cm["lethal"], structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        self.ctx["blocked"] = lab != sizes.argmax()   # outside the main connected safe region
        self.graph = build_graph(cm["cost"], self.ctx["elev"], self.world.cell)
        self.ctx["timings"]["cost map + graph"] = time.perf_counter() - t
        self._targets = None

    # ------------------------------------------------------------ outputs
    def summary(self):
        w, c = self.world, self.ctx
        m = c["metrics"]
        return dict(
            n=w.n, cell=w.cell, seed=w.seed, start=list(nearest_free(c["blocked"], w.start)),
            goal=list(nearest_free(c["blocked"], w.goal)),
            classes=CLASSES, class_colors=CLASS_COLORS,
            terrain=w.meta.get("source", "procedural"), terrain_meta=self.terrain_meta,
            jezero_available=real_terrain.available(),
            stats=dict(
                dem_missing=float(w.meta["dropout"]), n_points=int(c["n_points"]),
                pc_cells=int(c["pc_mask"].sum()), lethal_frac=float(c["lethal"].mean()),
                accuracy=m["accuracy"], macro_f1=m["macro_f1"], ece=m["ece"], oob=m["oob_accuracy"],
                train_fraction=m["train_fraction"], mean_entropy=float(c["entropy"].mean()),
                lam=c["lam"], slope_limit=SLOPE_LIMIT_DEG,
            ),
            metrics=m, timings={k: round(v * 1000) for k, v in c["timings"].items()},
            elevation=np.round(c["elev"][::2, ::2], 2).tolist(),
        )

    def layers(self):
        c, w = self.ctx, self.world
        hs = c["hillshade"]
        L = {}
        L["visible"] = render.false_color(w.bands)
        L["elevation"] = render.shade(render.scalar(c["elev"], "mars"), hs)
        L["dem_raw"] = render.scalar(w.dem_obs, "gray", mask_color=(255, 0, 80))
        L["class"] = render.shade(render.classes(c["pred"]), hs, 0.35)
        L["truth"] = render.shade(render.classes(w.true_class), hs, 0.35)
        L["uncertainty"] = render.scalar(c["entropy"], "magma", 0, 1)
        L["epistemic"] = render.scalar(c["epistemic"], "magma", 0, 0.6)
        L["slope"] = render.shade(render.scalar(c["slope"], "heat", 0, 35), hs, 0.3)
        L["slope_std"] = render.scalar(c["slope_std"], "magma", 0, 6)
        L["hazard"] = render.shade(render.scalar(c["p_hazard"], "heat", 0, 1), hs, 0.4)
        cost = np.log(np.where(np.isfinite(c["cost"]), c["cost"], np.nan))
        L["cost"] = render.scalar(cost, "viridis", mask=c["lethal"], mask_color=(120, 10, 20))
        L["ice"] = render.shade(render.scalar(c["p_ice"], "ice", 0, 1), hs, 0.3)
        L["ice_std"] = render.scalar(c["p_ice_std"], "magma", 0, 0.3)
        L["neutron"] = render.scalar(w.neutron_obs, "ice")
        L["science"] = render.shade(render.scalar(c["value"], "viridis"), hs, 0.3)
        return {k: render.to_png(v) for k, v in L.items()}

    # ------------------------------------------------------------ planning
    def plan(self, start, goal, algo="astar"):
        c = self.ctx
        start = nearest_free(c["blocked"], tuple(start))
        goal = nearest_free(c["blocked"], tuple(goal))
        out = dict(start=start, goal=goal)
        runs = {}
        runs["astar"] = astar(c["cost"], c["elev"], c["cell"], start, goal, True)
        runs["dijkstra"] = astar(c["cost"], c["elev"], c["cell"], start, goal, False)
        runs["rrt"] = rrt_star(c["cost"], c["cell"], start, goal)
        # baseline: shortest path that only avoids lethal cells (ignores risk / terrain)
        flat = np.where(c["lethal"], np.inf, 1.0)
        runs["naive"] = astar(flat, c["elev"], c["cell"], start, goal, True)
        comp = []
        for k, label in [("astar", "A* (risk-aware)"), ("dijkstra", "Dijkstra"), ("rrt", "RRT*"),
                         ("naive", "Shortest safe (no risk model)")]:
            r = runs[k]
            st = path_stats(r["path"], c) if r["found"] else None
            comp.append(dict(key=k, label=label, found=r["found"], expanded=r["expanded"], ms=r["ms"],
                             stats=st, path=r["path"]))
        out["comparison"] = comp
        out["selected"] = algo
        out["tree"] = runs["rrt"].get("tree", [])
        out["explain"] = self.explain_path(runs[algo]["path"]) if runs[algo]["found"] else None
        return out

    def pareto(self, start, goal, lams=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
        """Sweep the risk weight lambda: how much extra distance buys how much safety."""
        c = self.ctx
        keep = c["lam"]
        start = nearest_free(c["blocked"], tuple(start))
        goal = nearest_free(c["blocked"], tuple(goal))
        pts = []
        for lam in lams:
            self.set_risk(lam)
            r = astar(self.ctx["cost"], self.ctx["elev"], c["cell"], start, goal, True)
            if r["found"]:
                st = path_stats(r["path"], self.ctx)
                pts.append(dict(lam=lam, length_m=st["length_m"], p_success=st["p_success"],
                                energy_wh=st["energy_wh"], exposure=st["hazard_exposure"],
                                mean_uncertainty=st["mean_uncertainty"], path=r["path"]))
        self.set_risk(keep)
        return dict(points=pts, current=keep)

    def report(self):
        """Markdown mission report: every number needed to audit the run."""
        w, c = self.world, self.ctx
        m = c["metrics"]
        s = self.summary()
        st = s["start"]
        p = self.plan(st, s["goal"])
        tour = self.tour(st, 250.0)
        lines = [f"# MARS-09 Odyssey mission report",
                 f"", f"* Terrain: **{s['terrain']}**" + (f" ({self.terrain_meta['source']})" if self.terrain_meta else ""),
                 f"* World seed {w.seed}, {w.n}x{w.n} cells at {w.cell:.1f} m, risk weight lambda = {c['lam']}",
                 f"* DEM missing {s['stats']['dem_missing']*100:.1f} %, point cloud {s['stats']['n_points']} pts, "
                 f"lethal area {s['stats']['lethal_frac']*100:.1f} %",
                 "", "## Perception model (Random Forest, 4 % labels)",
                 f"* Accuracy {m['accuracy']*100:.1f} %, macro-F1 {m['macro_f1']:.3f}, ECE {m['ece']:.3f}, "
                 f"temperature {m.get('temperature', 1):.2f}",
                 "* Selective accuracy (most-confident fraction kept): " +
                 ", ".join(f"{int(a*100)} % -> {b*100:.1f} %" for a, b in m["selective"]),
                 "", "## Route comparison (start to goal)",
                 "| Planner | Found | Length m | Energy Wh | P(safe arrival) | Nodes | ms |",
                 "|---|---|---|---|---|---|---|"]
        for r in p["comparison"]:
            x = r["stats"] or {}
            lines.append(f"| {r['label']} | {r['found']} | {x.get('length_m', 0):.0f} | {x.get('energy_wh', 0):.1f} | "
                         f"{x.get('p_success', 0)*100:.1f} % | {r['expanded']} | {r['ms']:.0f} |")
        lines += ["", "## Science targets (expected-value ranking)",
                  "| ID | Kind | Value +- sd | P(ice) | Travel cost |", "|---|---|---|---|---|"]
        for t in self.targets(st):
            lines.append(f"| {t['id']} | {t['kind']} | {t.get('value', 0):.2f} +- {t.get('value_std', 0):.2f} | "
                         f"{t.get('p_ice', 0):.2f} | {t['travel_cost']:.0f} |")
        lines += ["", f"## Tour (budget 250 Wh)", f"* Order: {' -> '.join(tour['order'])}",
                  f"* Value {tour['total_value']:.2f}, energy {tour['energy_wh']:.1f} Wh. {tour['note']}",
                  "", "_Ground truth is used only for scoring; every decision above uses observations only._"]
        return "\n".join(lines)

    def explain_path(self, path):
        """Why this route? break the path cost down by factor."""
        c = self.ctx
        P = np.array(path)
        r, cc = P[:, 0], P[:, 1]
        comp = dict(
            terrain=float(c["terrain_cost"][r, cc].sum()),
            slope=float(((c["slope_term"] - 1) * c["terrain_cost"])[r, cc].sum()),
            roughness=float(((c["rough_term"] - 1) * c["terrain_cost"])[r, cc].sum()),
            uncertainty_risk=float(c["risk_cost"][r, cc].sum()),
            clearance=float(c["clearance"][r, cc].sum()),
        )
        cls_count = np.bincount(c["pred"][r, cc], minlength=len(CLASSES))
        return dict(cost_breakdown=comp,
                    terrain_mix={CLASSES[i]: int(cls_count[i]) for i in range(len(CLASSES)) if cls_count[i]})

    # ------------------------------------------------------------ science
    def targets(self, start, mode="expected", kappa=1.0):
        c = self.ctx
        start = nearest_free(c["blocked"], tuple(start))
        if self._targets is None:
            pos = find_targets(c["value"], c["value_std"], c["blocked"])
            self._targets = [describe_target(f"T{i + 1}", p, c) for i, p in enumerate(pos)]
        dist, pred = cost_to_go(self.graph, [start], (c["n"], c["n"]))
        out = []
        for t in self._targets:
            t = dict(t)
            p = backtrack(pred[0], dist[0], (t["r"], t["c"]), c["n"])
            t["travel_cost"] = float(dist[0][t["r"] * c["n"] + t["c"]])
            t["reachable"] = bool(p)
            t["path_stats"] = path_stats(p, c) if p else None
            out.append(t)
        return rank_targets(out, mode, kappa)

    def tour(self, start, budget_wh=120.0, mode="expected", kappa=1.0, k=7):
        c = self.ctx
        start = nearest_free(c["blocked"], tuple(start))
        ranked = [t for t in self.targets(start, mode, kappa) if t["reachable"]][:k]
        nodes = [start] + [(t["r"], t["c"]) for t in ranked]
        dist, pred = cost_to_go(self.graph, nodes, (c["n"], c["n"]))
        m = len(nodes)
        energy = np.full((m, m), np.inf)
        legs = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                p = backtrack(pred[i], dist[i], nodes[j], c["n"])
                if p:
                    st = path_stats(p, c)
                    energy[i, j] = st["energy_wh"]
                    legs[(i, j)] = (p, st)
        values = [max(t["utility"], 0) for t in ranked]
        t0 = time.perf_counter()
        order, val, used = orienteering(values, energy.tolist(), budget_wh)
        ms = (time.perf_counter() - t0) * 1000
        seq = [0] + [o + 1 for o in order]
        path, legs_out = [], []
        for a, b in zip(seq[:-1], seq[1:]):
            p, st = legs[(a, b)]
            path += p if not path else p[1:]
            legs_out.append(dict(frm="Rover" if a == 0 else ranked[a - 1]["id"], to=ranked[b - 1]["id"], stats=st))
        return dict(order=[ranked[o]["id"] for o in order], waypoints=[list(nodes[s]) for s in seq],
                    path=path, total_value=val, energy_wh=used, budget_wh=budget_wh, legs=legs_out,
                    candidates=len(ranked), solver_ms=ms,
                    note=f"Exact orienteering over {len(ranked)} targets: {2 ** len(ranked)} subsets evaluated by DP.")

    def traverse(self, waypoints, names, inject=True, comm_delay=12.5):
        return rover_sim.simulate(self.world, self.ctx, waypoints, names, inject=inject,
                                  comm_delay_min=comm_delay, seed=self.world.seed)

    # ------------------------------------------------------------ inspector
    def cell(self, r, cc):
        c, w = self.ctx, self.world
        r, cc = int(np.clip(r, 0, c["n"] - 1)), int(np.clip(cc, 0, c["n"] - 1))
        f = lambda a: None if not np.isfinite(a) else float(a)
        return dict(
            r=r, c=cc, x_m=cc * c["cell"], y_m=r * c["cell"],
            elevation=f(c["elev"][r, cc]), elevation_std=f(c["elev_std"][r, cc]),
            dem_missing=bool(c["hole_mask"][r, cc]), point_cloud=bool(c["pc_mask"][r, cc]),
            slope=f(c["slope"][r, cc]), slope_std=f(c["slope_std"][r, cc]), roughness=f(c["rough"][r, cc]),
            class_probs={CLASSES[i]: float(c["probs"][r, cc, i]) for i in range(len(CLASSES))},
            predicted=CLASSES[int(c["pred"][r, cc])], truth=CLASSES[int(w.true_class[r, cc])],
            entropy=f(c["entropy"][r, cc]), aleatoric=f(c["aleatoric"][r, cc]), epistemic=f(c["epistemic"][r, cc]),
            p_hazard=f(c["p_hazard"][r, cc]), p_steep=f(c["p_steep"][r, cc]), p_rough=f(c["p_rough"][r, cc]),
            p_rock=f(c["p_rock"][r, cc]), lethal=bool(c["lethal"][r, cc]), cost=f(c["cost"][r, cc]),
            cost_parts=dict(terrain=f(c["terrain_cost"][r, cc]), slope_x=f(c["slope_term"][r, cc]),
                            rough_x=f(c["rough_term"][r, cc]), risk=f(c["risk_cost"][r, cc]),
                            clearance=f(c["clearance"][r, cc])),
            p_ice=f(c["p_ice"][r, cc]), p_ice_ci=[f(c["p_ice_lo"][r, cc]), f(c["p_ice_hi"][r, cc])],
            llr=dict(neutron=f(c["llr_neutron"][r, cc]), thermal=f(c["llr_thermal"][r, cc]),
                     spectral=f(c["llr_spectral"][r, cc])),
            neutron=f(c["neutron"][r, cc]), thermal=f(c["thermal"][r, cc]), true_weh=f(w.true_weh[r, cc]),
            value=f(c["value"][r, cc]), value_std=f(c["value_std"][r, cc]),
            training_label=bool(c["train_mask"][r, cc]),
        )
