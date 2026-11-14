# HackInno · MARS-09 Odyssey

**HackInnovision 2026 · Track A: Planetary Reconnaissance & Autonomous Navigation**

An autonomous Mars-rover brain that maps the terrain, knows *how sure* it is about every cell,
plans a safe route, ranks science targets (including buried water ice) with confidence ranges,
and replans on board when it discovers hazards the orbiter missed.

* **Live app (frontend, GitHub Pages):** https://bikesh3764.github.io/hackinno/
* **Live backend (Render):** https://hackinno-4vza.onrender.com
  (free tier: the first request after idle can take ~1 minute while the server wakes up)

## What's inside

| Folder | Contents |
|---|---|
| `odyssey/backend` | FastAPI API + the whole pipeline: sensor simulator, elevation fusion, Random Forest terrain AI with uncertainty, hazard & cost maps, A* / Dijkstra / RRT*, Bayesian ice detection, science ranking, closed-loop rover |
| `odyssey/frontend` | Single-page console (vanilla JS, canvas, Plotly) |
| `odyssey/tests` | Correctness tests (A* optimality, exact tour, no lethal cells, …) |
| `odyssey/docs` | `PROJECT_EXPLAINED.md` (full technical explanation) and `JUDGE_GUIDE.md` (demo script + Q&A) |
| `odyssey/tools` | Fetch a real HiRISE Jezero-crater elevation map; render the hero image |
| `design/DESIGN.md` | UI design reference (opendesignmd.com, MIT) |

## Run locally

```bat
cd odyssey
run.bat
```
Open http://localhost:8000. Tests: `.venv\Scripts\python -m pytest -q tests`.

## Key results

* Terrain AI: ~92–95 % accuracy from 4 % labelled pixels; 99.7 % on its most confident half.
* Risk-aware A*: 56 % safe arrival vs 15 % for a route that ignores risk (seed 7).
* Benchmark over 24 worlds (12 simulated, 12 on real Jezero topography): A* at least as safe as the
  baseline in 24/24, zero true hazards entered.

Data: sensors and surface materials are simulated (so accuracy can be scored against a known truth);
topography can be real NASA HiRISE data of Jezero crater.
