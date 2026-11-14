# MARS-09 Odyssey: uncertainty-aware autonomous reconnaissance

**HackInnovision 2026, Track A (Planetary Reconnaissance & Autonomous Navigation)**

The rover receives raw orbital and onboard sensor data. It builds a 3-D map, classifies the terrain with an ML model, and estimates **how sure** it is about every cell. From that it plans a safe, science-rich route under an energy budget and drives it in a closed loop. When its own sensors reveal a hazard the orbiter missed, it replans on board instead of waiting 25 minutes for Earth.

```
SENSE ─► FUSE ─► PERCEIVE ─► ASSESS RISK ─► COST ─► PLAN ─► DISCOVER ─► ACT (closed loop)
```

## Run it (fully offline, no internet needed)

```bat
run.bat
```
Then open http://localhost:8000. The first run creates a virtual environment in `.venv` and installs `requirements.txt` (numpy, scipy, scikit-learn, fastapi, uvicorn, pillow). Plotly is bundled locally in `frontend/`.

Tests: `.venv\Scripts\python -m pytest -q tests`

## What the demo shows

| Deliverable asked for | Where in the app |
|---|---|
| Terrain / cost map | Layers: *Terrain AI, Slope, Hazard P, Cost map* |
| Planned path with hazards avoided | Route + planner comparison table (A*, Dijkstra, RRT*, no-risk baseline) |
| Ranked science targets **with uncertainty** | *Science targets* panel: value ± σ, 90 % CI bars, P(ice) CI, evidence trail |
| 3-D elevation map | *3D terrain* tab |
| Uncertainty quantification | *Uncertainty / Epistemic / Slope σ / Ice σ* layers, model calibration tab |
| Traceability | Cell inspector (every number behind a decision), route cost breakdown, autonomy decision log |
| Real Mars data | *Real Mars · Jezero* terrain source: HiRISE DTM of Jezero crater |
| Robustness | Model tab → benchmark over 24 worlds |

## Architecture

```
backend/
  terrain_sim.py  synthetic Mars + sensor simulator (truth vs. noisy observations)
  perception.py   point-cloud rasterisation, DEM in-painting, Kalman elevation fusion,
                  Monte-Carlo slope uncertainty, probabilistic hazard map
  classifier.py   Random-Forest terrain classifier, entropy / aleatoric / epistemic
                  uncertainty, patch-grouped CV temperature calibration, metrics
  costmap.py      risk-aware cost map, chance-constrained lethal mask, footprint inflation
  planner.py      A*, Dijkstra, RRT*, scipy graph-Dijkstra cost-to-go, path statistics, energy model
  science.py      Bayesian ice fusion (log-odds), science value with MC CI, UCB/LCB ranking,
                  exact orienteering tour (bitmask DP)
  rover_sim.py    closed-loop traverse: local sensing, replanning, online slip learning, in-situ checks
  mission.py      pipeline orchestration;  app.py  FastAPI endpoints
frontend/         single-page console (vanilla JS + canvas + Plotly, no build step)
tests/            correctness tests (A* optimality, exact orienteering, no lethal cells, ...)
```

## Data

**Real topography (optional mode):** a 320 m × 320 m crop of the HiRISE DTM `DTEEC_023524_1985_023379_1985_U01` (Jezero crater, USGS / University of Arizona, 1 m/px averaged to 2 m). `tools/fetch_hirise.py` downloads only ~10 MB of the 700 MB product using HTTP range requests and stores it in `data/` so the app stays offline. On this terrain, surface materials and sensor noise are simulated on top of the real elevation.

**Procedural worlds (default):**

A procedural 320 m × 320 m site (160 × 160 cells at 2 m) with craters, a cliff with two ramps, dune fields, boulder fields, bedrock, hydrated-clay and ice-rich ground, a **buried ice deposit invisible to cameras**, and **sub-resolution boulders invisible from orbit**. Observations are degraded on purpose: DEM noise σ = 0.35 m, 4 % missing DEM data, 6-band spectra with dust mantling and noise, coarse, noisy neutron data, and a stereo point cloud with 1.5 % gross outliers. Ground truth is used only for scoring, never for decisions.

## Key results (seed 7, defaults)

* Terrain classification: **~94.8 % accuracy** on unseen pixels from **4 % labels**, and **99.7 %** on the 50 % of pixels the model is most confident about. The uncertainty really does separate right answers from wrong ones.
* Risk-aware A*: **P(safe arrival) ≈ 56 %** vs **≈ 15 %** for a shortest path that ignores the risk model (that baseline only avoids cells marked lethal).
* Over 7 random worlds, on-site checks agreed with the orbital ice predictions at **27 of 29** sites visited.
* Full perception pipeline about 1.3 s on a laptop CPU; replanning 1–40 ms.

**Robustness benchmark** (`.venv\Scripts\python backendenchmark.py 12`, 24 full missions, ~90 s):

| | Procedural (12) | Real Jezero DTM (12) |
|---|---|---|
| Classifier accuracy | 92.3 % ± 2.0 (min 87.4 %) | 93.3 % ± 1.8 (min 89.2 %) |
| Accuracy, confident 50 % | 99.5 % | 99.7 % |
| P(safe) A* vs no-risk baseline | 50 % vs 9 % | 92 % vs 44 % |
| A* ≥ baseline | 12 / 12 | 12 / 12 |
| True hazard cells entered | 0 in 12 / 12 | 0 in 12 / 12 |
| In-situ ice agreement | 95 % (44 sites) | 92 % (71 sites) |

## Assumptions & known limitations

* Surface materials and instrument data are simulated (topography can be real HiRISE). Real CRISM spectra and AI4Mars labels would plug into the same interfaces.
* The classifier is per-pixel (with context features), not a CNN. We chose that for flight-CPU budget and interpretability; a U-Net trained on AI4Mars is the upgrade path.
* Hazard sources are treated as independent, so P(safe arrival) = exp(−Σp) is a Poisson approximation, best used to compare routes.
* The model is slightly under-confident against clean truth because the training labels carry 4 % noise. Under-confidence is the safe direction for a rover.
* Energy model: rolling resistance + climbing only (no thermal / comms load).
* Single-user demo server, so state is kept in memory.

## Development phases (12-hour plan)

1. **Core:** simulator, perception, cost map, A*.
2. **Intelligence:** ML classifier with uncertainty, Bayesian ice fusion, science ranking, tour optimisation, closed-loop rover.
3. **Show:** console UI, 3-D view, model evaluation, tests, docs.
4. **Hardening:** real HiRISE Jezero topography, 24-world robustness benchmark, one-click demo.

See [docs/JUDGE_GUIDE.md](docs/JUDGE_GUIDE.md) for the technical explanation and Q&A prep.
Design reference: [../design/DESIGN.md](../design/DESIGN.md) (opendesignmd.com, MIT). The console follows its black canvas, uppercase DIN type, hairline borders and ghost pill buttons. It uses no third-party branding.
