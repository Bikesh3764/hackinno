"""
Path planners on the risk-aware cost map:
  * A*        - optimal, admissible heuristic (straight-line distance x min cell cost)
  * Dijkstra  - A* with h = 0 (explores everything, same optimal path)
  * RRT*      - sampling-based, asymptotically optimal, good for continuous spaces
  * graph Dijkstra (scipy, C speed) - cost-to-go maps for science ranking & tour planning
Edge cost = 3-D step length (includes climbing) x mean cell cost of the two cells.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as cs_dijkstra

from terrain_sim import MARS_G

NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

ROVER_MASS = 900.0      # kg (Curiosity-class)
DRIVE_EFF = 0.45        # wheel-to-battery efficiency
DRIVE_SPEED = 0.045     # m/s average with autonav stops


def nearest_free(lethal, p):
    r, c = p
    if not lethal[r, c]:
        return (int(r), int(c))
    free = np.argwhere(~lethal)
    d = (free[:, 0] - r) ** 2 + (free[:, 1] - c) ** 2
    q = free[d.argmin()]
    return (int(q[0]), int(q[1]))


def astar(cost, elev, cell, start, goal, use_heuristic=True):
    n, m = cost.shape
    C = cost.ravel().tolist()
    Z = elev.ravel().tolist()
    finite = cost[np.isfinite(cost)]
    cmin = float(finite.min()) if finite.size else 1.0
    s, g = start[0] * m + start[1], goal[0] * m + goal[1]
    gr, gc = goal
    INF = float("inf")
    gscore = {s: 0.0}
    parent = {s: -1}
    closed = set()
    heap = [(0.0, s)]
    expanded = 0
    t0 = time.perf_counter()
    while heap:
        f, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == g:
            break
        ur, uc = divmod(u, m)
        gu = gscore[u]
        for dr, dc, d in NBRS:
            vr, vc = ur + dr, uc + dc
            if vr < 0 or vc < 0 or vr >= n or vc >= m:
                continue
            v = vr * m + vc
            cv = C[v]
            if cv == INF or v in closed:
                continue
            if dr and dc and (C[ur * m + vc] == INF or C[vr * m + uc] == INF):
                continue  # no corner cutting past obstacles
            horiz = d * cell
            dz = Z[v] - Z[u]
            step = math.sqrt(horiz * horiz + dz * dz) * 0.5 * (C[u] + cv)
            ng = gu + step
            if ng < gscore.get(v, INF):
                gscore[v] = ng
                parent[v] = u
                h = math.hypot(vr - gr, vc - gc) * cell * cmin if use_heuristic else 0.0
                heapq.heappush(heap, (ng + h, v))
    ms = (time.perf_counter() - t0) * 1000
    if g not in parent:
        return dict(path=[], cost=None, expanded=expanded, ms=ms, found=False)
    path = []
    u = g
    while u != -1:
        path.append(divmod(u, m))
        u = parent[u]
    path.reverse()
    return dict(path=[(int(r), int(c)) for r, c in path], cost=gscore[g], expanded=expanded,
                ms=ms, found=True)


# ---------------------------------------------------------------- RRT*
def _seg_cells(a, b):
    """All grid cells touched by segment a->b (dense sampling, shared by check and output)."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    k = max(int(L * 4), 2)
    t = np.linspace(0, 1, k)
    rr = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
    cc = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
    return rr, cc, L


def _seg_cost(cost, a, b, cell):
    rr, cc, L = _seg_cells(a, b)
    v = cost[rr, cc]
    if not np.all(np.isfinite(v)):
        return None
    return float(v.mean()) * L * cell


def rrt_star(cost, cell, start, goal, n_iter=2200, step=6.0, radius=11.0, seed=1):
    rng = np.random.default_rng(seed)
    n, m = cost.shape
    t0 = time.perf_counter()
    nodes = np.zeros((n_iter + 2, 2))
    nodes[0] = start
    par = np.full(n_iter + 2, -1)
    cst = np.zeros(n_iter + 2)
    cnt = 1
    goal_idx = -1
    for _ in range(n_iter):
        q = np.array(goal, float) if rng.random() < 0.08 else rng.uniform([0, 0], [n - 1, m - 1])
        d = np.hypot(*(nodes[:cnt] - q).T)
        i = int(d.argmin())
        if d[i] < 1e-6:
            continue
        new = nodes[i] + (q - nodes[i]) * min(1.0, step / d[i])
        new = np.clip(new, 0, [n - 1, m - 1])
        if not np.isfinite(cost[int(round(new[0])), int(round(new[1]))]):
            continue
        dn = np.hypot(*(nodes[:cnt] - new).T)
        near = np.nonzero(dn <= radius)[0]
        best, best_c = -1, math.inf
        for j in near:
            sc = _seg_cost(cost, nodes[j], new, cell)
            if sc is not None and cst[j] + sc < best_c:
                best, best_c = j, cst[j] + sc
        if best < 0:
            continue
        nodes[cnt], par[cnt], cst[cnt] = new, best, best_c
        k = cnt
        cnt += 1
        for j in near:  # rewire
            sc = _seg_cost(cost, new, nodes[j], cell)
            if sc is not None and best_c + sc < cst[j]:
                par[j], cst[j] = k, best_c + sc
        if math.hypot(new[0] - goal[0], new[1] - goal[1]) < 3:
            sc = _seg_cost(cost, new, np.array(goal, float), cell)
            if sc is not None and (goal_idx < 0 or best_c + sc < cst[goal_idx]):
                if goal_idx < 0:
                    goal_idx = cnt
                    cnt += 1
                nodes[goal_idx], par[goal_idx], cst[goal_idx] = goal, k, best_c + sc
    ms = (time.perf_counter() - t0) * 1000
    tree = [[float(nodes[i][0]), float(nodes[i][1]), float(nodes[par[i]][0]), float(nodes[par[i]][1])]
            for i in range(1, cnt) if par[i] >= 0]
    if goal_idx < 0:
        return dict(path=[], cost=None, expanded=cnt, ms=ms, found=False, tree=tree[:1500])
    pts = []
    i = goal_idx
    while i != -1:
        pts.append(nodes[i])
        i = par[i]
    pts.reverse()
    # densify to grid cells with exactly the cells the collision check validated
    path = []
    for a, b in zip(pts[:-1], pts[1:]):
        rr, cc, _ = _seg_cells(a, b)
        for p in zip(rr.tolist(), cc.tolist()):
            if not path or path[-1] != p:
                path.append(p)
    if path[-1] != (int(goal[0]), int(goal[1])):
        path.append((int(goal[0]), int(goal[1])))
    return dict(path=path, cost=float(cst[goal_idx]), expanded=cnt, ms=ms, found=True, tree=tree[:1500])


# ------------------------------------------------- graph Dijkstra (scipy)
def build_graph(cost, elev, cell):
    n, m = cost.shape
    idx = np.arange(n * m).reshape(n, m)
    fin = np.isfinite(cost)
    rows, cols, w = [], [], []
    for dr, dc, d in NBRS:
        r0, r1 = max(0, -dr), n - max(0, dr)
        c0, c1 = max(0, -dc), m - max(0, dc)
        a = (slice(r0, r1), slice(c0, c1))
        b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        ok = fin[a] & fin[b]
        if dr and dc:
            ok &= fin[r0:r1, c0 + dc:c1 + dc] & fin[r0 + dr:r1 + dr, c0:c1]
        dz = elev[b] - elev[a]
        L = np.sqrt((d * cell) ** 2 + dz ** 2)
        wt = L * 0.5 * (np.where(fin[a], cost[a], 0) + np.where(fin[b], cost[b], 0))
        rows.append(idx[a][ok]); cols.append(idx[b][ok]); w.append(wt[ok])
    return csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n * m, n * m))


def cost_to_go(graph, sources, shape):
    m = shape[1]
    src = [r * m + c for r, c in sources]
    dist, pred = cs_dijkstra(graph, directed=True, indices=src, return_predecessors=True)
    return dist, pred


def backtrack(pred_row, dist_row, target, m):
    t = target[0] * m + target[1]
    if not np.isfinite(dist_row[t]):
        return []
    path = [t]
    while pred_row[path[-1]] >= 0:
        path.append(pred_row[path[-1]])
    path.reverse()
    return [divmod(int(p), m) for p in path]


# ------------------------------------------------- path statistics
def path_stats(path, ctx):
    if not path:
        return None
    cell = ctx["cell"]
    P = np.array(path)
    r, c = P[:, 0], P[:, 1]
    elev = ctx["elev"][r, c]
    dxy = np.hypot(np.diff(r), np.diff(c)) * cell
    dz = np.diff(elev)
    seg = np.sqrt(dxy ** 2 + dz ** 2)
    length = float(seg.sum())
    crr = ctx["crr"][r, c]
    crr_seg = 0.5 * (crr[1:] + crr[:-1])
    energy_j = ROVER_MASS * MARS_G * (crr_seg * seg + np.clip(dz, 0, None)).sum() / DRIVE_EFF
    p_haz = ctx["p_hazard"][r, c]
    unc = ctx["unc"][r, c]
    slope = ctx["slope"][r, c]
    finite_cost = ctx["cost"][r, c]
    finite_cost = finite_cost[np.isfinite(finite_cost)]
    exposure = float(p_haz.sum())
    return dict(
        length_m=length,
        energy_wh=float(energy_j / 3600),
        drive_time_h=float(length / DRIVE_SPEED / 3600),
        climb_m=float(np.clip(dz, 0, None).sum()),
        max_slope=float(slope.max()),
        mean_slope=float(slope.mean()),
        max_p_hazard=float(p_haz.max()),
        hazard_exposure=exposure,
        p_success=float(np.exp(-exposure)),
        mean_uncertainty=float(unc.mean()),
        min_clearance_m=float(ctx["clearance_dist"][r, c].min() * cell),
        total_cost=float(finite_cost.sum()),
        n_cells=len(path),
    )
