"""MARS-09 Odyssey autonomy console - FastAPI backend (local or Render)."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mission import Mission

FRONT = Path(__file__).resolve().parent.parent / "frontend"
app = FastAPI(title="MARS-09 Odyssey")
# the GitHub Pages frontend calls this API from another origin
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
# JSON responses (maps, routes, logs) shrink 3-10x over the wire
app.add_middleware(GZipMiddleware, minimum_size=1000)
M = Mission()
LOCK = threading.Lock()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def J(o):
    return Response(json.dumps(_clean(o), separators=(",", ":")), media_type="application/json")


class BuildReq(BaseModel):
    seed: int = 7
    dem_noise: float = 0.35
    dropout: float = 0.04
    spectral_noise: float = 0.008
    lam: float = 1.0
    terrain: str = "procedural"


class RiskReq(BaseModel):
    lam: float = 1.0


class PlanReq(BaseModel):
    start: list[int]
    goal: list[int]
    algo: str = "astar"
    rrt: bool = True   # False: skip RRT* here; the console fetches it from /api/rrt afterwards


class TargetReq(BaseModel):
    start: list[int]
    mode: str = "expected"
    kappa: float = 1.0


class TourReq(TargetReq):
    budget: float = 250.0


class TraverseReq(BaseModel):
    waypoints: list[list[int]]
    names: list[str]
    inject: bool = True
    comm_delay: float = 12.5


def _ensure():
    if M.world is None:
        M.build()


@app.post("/api/build")
def build(req: BuildReq):
    with LOCK:
        s = M.build(req.seed, req.dem_noise, req.dropout, req.spectral_noise, req.lam, req.terrain)
        s["layers"] = M.layers()
        return J(s)


@app.post("/api/risk")
def risk(req: RiskReq):
    with LOCK:
        _ensure()
        M.set_risk(req.lam)
        s = M.summary()
        s.pop("elevation")
        s["layers"] = M.layers(only="cost")   # only the cost map depends on the Caution setting
        return J(s)


@app.post("/api/plan")
def plan(req: PlanReq):
    with LOCK:
        _ensure()
        return J(M.plan(req.start, req.goal, req.algo, req.rrt))


@app.post("/api/rrt")
def plan_rrt(req: PlanReq):
    with LOCK:
        _ensure()
        return J(M.plan_rrt(req.start, req.goal))


@app.post("/api/targets")
def targets(req: TargetReq):
    with LOCK:
        _ensure()
        return J(M.targets(req.start, req.mode, req.kappa))


@app.post("/api/tour")
def tour(req: TourReq):
    with LOCK:
        _ensure()
        return J(M.tour(req.start, req.budget, req.mode, req.kappa))


@app.post("/api/traverse")
def traverse(req: TraverseReq):
    with LOCK:
        _ensure()
        return J(M.traverse(req.waypoints, req.names, req.inject, req.comm_delay))


class ParetoReq(BaseModel):
    start: list[int]
    goal: list[int]


@app.post("/api/pareto")
def pareto(req: ParetoReq):
    with LOCK:
        _ensure()
        return J(M.pareto(req.start, req.goal))


@app.get("/api/report")
def report():
    with LOCK:
        _ensure()
        md = M.report()
    return Response(md, media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="odyssey_report_seed{M.world.seed}.md"'})


@app.get("/api/benchmark")
def benchmark():
    p = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"
    if not p.exists():
        return J(dict(available=False))
    d = json.loads(p.read_text())
    d["available"] = True
    return J(d)


@app.get("/api/cell")
def cell(r: int, c: int):
    with LOCK:
        _ensure()
        return J(M.cell(r, c))


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


app.mount("/static", StaticFiles(directory=FRONT), name="static")


@app.get("/api/health")
def health():
    return {"ok": True}
