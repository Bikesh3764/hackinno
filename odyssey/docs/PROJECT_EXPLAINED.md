# MARS-09 Odyssey: Complete Project Explanation

> **How to use this file:** paste it into ChatGPT (or any AI assistant) and ask questions such as
> "explain section 5.4 like I'm a beginner" or "what would a judge ask about the cost map?".
> It describes everything in the project: the problem, the website, every algorithm, the data, the ML model,
> the numbers we measured, the files, and the honest limitations.

---

## 1. The hackathon and the problem

* **Event:** HackInnovision 2026, 12-hour hackathon (deadline 14 Nov 2026, 12:00 noon). Top 10 teams present for 8 minutes plus 4 minutes of Q&A.
* **Track A: Planetary Reconnaissance & Autonomous Navigation (problem "Odyssey", rover MARS-09).**
* **What the problem statement asks for:**
  1. **Terrain perception** from multispectral images, DEMs (digital elevation models) and point clouds.
  2. **Hazard detection** (steep slopes, rough ground, rocks).
  3. **3-D elevation maps.**
  4. **Path planning** with A*, Dijkstra or RRT over a **cost map**.
  5. **Science target ranking with uncertainty**, including **subsurface water ice**.
* **Judging criteria:** technical depth, reasoning, handling of uncertainty and robustness, explainability, innovation and usability.

**One-sentence pitch:** a rover on Mars is 4–24 light-minutes from Earth, so it must decide alone. Our system turns raw, noisy sensor data into a map that knows *how sure it is* about every cell. It plans safe routes, ranks science targets (including buried ice) with confidence intervals, and replans on board when it discovers hazards the orbiter missed.

---

## 2. Pipeline overview

```
SENSE → FUSE → PERCEIVE → ASSESS RISK → COST → PLAN → DISCOVER → ACT (closed loop)
```

| Stage | What happens | Main technique |
|---|---|---|
| Sense | Receive orbital DEM, 6-band multispectral image, thermal inertia, neutron (hydrogen) map, rover stereo point cloud | Simulated sensors with realistic noise (or real HiRISE topography) |
| Fuse | Build one clean elevation map with an error bar per cell | Robust point-cloud rasterisation (median + MAD), harmonic in-painting, inverse-variance (Kalman) fusion |
| Perceive | Slope, roughness, curvature and terrain class per cell | Monte-Carlo slope uncertainty; Random Forest classifier |
| Assess risk | Probability that each cell is dangerous | P(hazard) = 1 − Π(1 − pᵢ) over steep / rough / boulder sources |
| Cost | Price of driving through each cell | Terrain × slope × roughness × (1 + λ·uncertainty) + λ·risk + clearance; lethal if P(hazard) > 0.5 |
| Plan | Best route from start to goal | A*, Dijkstra, RRT*, plus a no-risk baseline for comparison |
| Discover | Where is the science? Is there ice? | Naive-Bayes log-odds ice fusion; science value with confidence interval; ranking; best tour |
| Act | Drive in a closed loop | Local sensing, on-board replanning, slip learning, in-situ checking of predictions |

---

## 3. Technology stack

* **Backend:** Python 3.11, FastAPI (web API), uvicorn (server), numpy, scipy, scikit-learn, Pillow.
* **Frontend:** a single HTML page with vanilla JavaScript (no framework, no build step), HTML canvas for the map, and Plotly (bundled locally) for charts and the 3-D view.
* **Design:** follows the SpaceX `DESIGN.md` from opendesignmd.com (MIT licence), saved in `design/DESIGN.md`: pure black canvas, uppercase **D-DIN** font, thin hairline borders, ghost pill buttons and a full-screen photographic hero. No SpaceX branding is used. Type system: **D-DIN Bold** for the big headlines and KPI numbers; everything else uses Apple's system font (**SF Pro**) on Mac/iPhone via `-apple-system`, and **Inter** (bundled in `frontend/fonts/`, the closest open-source match to SF Pro) on Windows/Android. SF Pro itself is not bundled because Apple's licence does not allow embedding it in websites. Numbers use tabular figures so columns line up.
* **Runs fully offline.** Everything, including fonts, Plotly and the real Mars elevation data, is stored locally. It is **not deployed** anywhere; it runs on `localhost:8000`.

**How to run:** double-click `odyssey/run.bat`, then open http://localhost:8000. Tests: `.venv\Scripts\python -m pytest -q tests`.

---

## 4. The data

### 4.1 Procedural worlds (default)
`backend/terrain_sim.py` generates a 320 m × 320 m site: 160 × 160 cells, each 2 m wide. It has two parts:

* **Truth** (used only for scoring and for the rover's local sensors during the drive simulation): real elevation, real terrain class, real water content (WEH, water-equivalent hydrogen %), hidden boulders.
* **Observations** (the only thing the algorithms see):
  * DEM with noise σ = 0.35 m and **4 % missing data** (holes, like stereo-matching failures).
  * 6-band multispectral image (450, 550, 650, 860 nm, 1.9 µm, 2.3 µm). Clay absorbs at 1.9 µm (water) and 2.3 µm (Fe/Mg–OH); ice is bright with a deep 1.9 µm absorption. **Dust** partly covers the spectra.
  * Thermal inertia: how fast the ground heats and cools. Rock and ice are high, sand is low.
  * Neutron spectrometer hydrogen map: very coarse and noisy. It senses water **below** the surface.
  * Rover stereo **point cloud** of 26,000 points near the landing site, with **1.5 % gross outliers**.
* **Terrain features:** fractal hills, 5–7 craters, a cliff (scarp) with two ramps, dune fields, boulder fields, bedrock outcrops, hydrated clay and ice-rich ground.
* **Hidden traps (on purpose):**
  * A **buried ice deposit**, invisible to cameras and visible only to neutron and thermal sensors.
  * **Sub-resolution boulders** that the orbital DEM cannot see; the rover only discovers them when it gets close.
* Six terrain classes: Regolith, Bedrock, Sand/dunes, Rocky field, Hydrated clay, Ice-rich ground.

### 4.2 Real Mars topography (Jezero crater)
* Source: NASA HiRISE digital terrain model **DTEEC_023524_1985_023379_1985_U01** (USGS / University of Arizona) of **Jezero crater**, the Perseverance rover's landing site. It has 1 m per pixel.
* The full file is 700 MB. `tools/fetch_hirise.py` reads the file header, then uses **HTTP range requests** to download only ~10 MB: a 320 × 320 m window, averaged to 2 m cells. The result is cached in `data/jezero_dtm.npy`, so the demo works offline.
* In this mode the **elevation is real**. Surface materials and sensor noise are still simulated on top, because we need ground truth to measure accuracy. Clay and ice are placed in the lowest local basins, and steep natural slopes become bedrock.
* The website's **hero image** is also rendered from a larger (~1.1 km) strip of the same real DTM (`tools/render_hero.py`). It uses a height-field ray caster written in numpy, with 2.2× vertical exaggeration, which is stated on the page.

---

## 5. Algorithms in detail

### 5.1 Elevation fusion (`perception.py`)
1. **Point-cloud rasterisation:** points are binned into cells. Each cell takes the **median** height, and outliers are rejected with **MAD** (median absolute deviation, a robust 3σ test). A unit test injects 4 % gross outliers and the result stays within 5 cm.
2. **Harmonic in-painting:** DEM holes are filled by solving Laplace's equation (repeated neighbour averaging, i.e. Jacobi iteration), which gives the smoothest possible fill. Filled cells get an uncertainty that grows with distance from real data.
3. **Inverse-variance fusion** (a 1-D Kalman update): the fused value is `(z₁/σ₁² + z₂/σ₂²) / (1/σ₁² + 1/σ₂²)`. The rover point cloud (σ ≈ 5 cm) dominates where it exists; the orbital DEM (σ ≈ 35 cm) fills in elsewhere.
4. Output: **elevation ± σ for every cell.**

### 5.2 Geomorphometry with uncertainty
* **Slope** from the elevation gradient. Rover limit: **25°**.
* **Slope uncertainty by Monte-Carlo:** 24 random elevation maps are sampled using each cell's σ; slope is computed for each, and their standard deviation is the slope σ.
* **Roughness:** RMS of the elevation minus a smoothed version, with the noise bias removed. Limit: **0.70 m**.
* **Curvature:** a Laplacian of elevation.

### 5.3 Terrain classifier: the ML part (`classifier.py`)
* **Model:** scikit-learn **Random Forest**, 80 trees, `class_weight="balanced"`, with out-of-bag score.
* **19 features per pixel:** 6 normalised bands, brightness, 1.9 µm band depth (hydration), 2.3 µm band depth (clay OH), red slope (B3/B1), thermal inertia, slope, roughness, curvature, relative elevation, texture (local std), plus 3 *context* features: neighbourhood averages of the 1.9 µm depth, thermal inertia and roughness.
* **Training data:** about **4 % of pixels** (~1,020 of 25,600), as 5×5 patches, 7 patches per class, with **4 % deliberately wrong labels** to imitate human labelling errors. The model is **trained from scratch every time** you click *Generate & perceive* (about 1 second).
* **Evaluation:** on all ~24,500 pixels it never trained on, against the hidden truth.
* **Uncertainty decomposition** (from the disagreement between the 80 trees):
  * **Total uncertainty** = entropy of the averaged probabilities.
  * **Epistemic** (model doesn't know; more data would help) = mutual information = total − average per-tree entropy.
  * **Aleatoric** (the data itself is ambiguous, e.g. dust) = total − epistemic.
* **Calibration:** temperature scaling. The temperature T is fitted by minimising log-loss with **patch-grouped 4-fold cross-validation**. Patches are kept together so neighbouring pixels don't leak between folds. The fitted T ≈ 1.05, and the model is **slightly under-confident**, which is the safe direction for a rover.
* **Metrics shown:** accuracy, macro-F1, out-of-bag accuracy, ECE (expected calibration error), reliability diagram, confusion matrix, feature importances, and **selective accuracy** (accuracy when you keep only the most-confident X % of pixels).
* **No external dataset is used.** Honest statement: the numbers are on simulated data; real-world performance on Mars imagery is unknown. The upgrade path is a U-Net trained on NASA's **AI4Mars** labels.

### 5.4 Hazard probability
For each cell:
* `p_steep = Φ((slope − 25°) / σ_slope)` (Φ = normal CDF). An uncertain slope near the limit gives a moderate probability, not a hard yes/no.
* `p_rough = Φ((roughness − 0.70) / 0.08)`
* `p_rock = clip((P(rocky field) − 0.35) / 0.45, 0, 1)` (from the classifier)
* **Combined:** `P(hazard) = 1 − (1 − p_steep)(1 − p_rough)(1 − 0.85·p_rock)`, i.e. the probability that at least one hazard source is real.

### 5.5 Cost map (`costmap.py`)
* `terrain = Σ_class P(class) × traversal_cost(class)`, the *expected* cost over the classifier's probabilities. Class costs: Regolith 1.0, Bedrock 1.35, Sand 3.2, Rocky 6.0, Clay 1.5, Ice 2.0.
* `slope_term = 1 + 3·(slope/25)²`, `rough_term = 1 + 1.5·(rough/0.70)²`
* `uncertainty = 0.6·entropy + 0.4·clip(slope_σ/6, 0, 1)`
* `cost = terrain·slope_term·rough_term × (1 + λ·uncertainty) + λ·8·P(hazard) + clearance`
* **Chance constraint:** cells with P(hazard) > 0.5 are **lethal** (infinite cost). They are inflated by the rover's footprint (1 cell = 2 m) using a distance transform. A small clearance penalty keeps routes away from lethal edges.
* **λ (lambda, risk aversion)** is a slider. λ = 0 ignores uncertainty; higher λ avoids poorly understood ground at the price of distance.
* Cells outside the largest connected safe region are marked "blocked", so start, goal and targets always snap to reachable ground.

### 5.6 Path planning (`planner.py`)
* **Grid graph:** 8-connected (diagonals allowed, no corner cutting). Edge cost = mean of the two cells' costs × step length (2 m or 2√2 m).
* **A\*:** uses an admissible heuristic (straight-line distance × minimum cell cost, which never overestimates), so it is **optimal** but explores fewer nodes than Dijkstra.
* **Dijkstra:** the same algorithm without a heuristic. It is used to verify A*.
* **RRT\*:** a sampling-based planner (random tree with rewiring, 2,200 iterations). Each edge is checked by dense sampling so it never crosses a lethal cell. It suits continuous or high-dimensional spaces; on a 2-D grid A* beats it, which the comparison table shows.
* **No-risk baseline:** the shortest route that only avoids lethal cells. It shows what our risk model buys.
* **Route statistics:** length, energy, drive time, climb, max slope, hazard exposure Σp, **P(safe arrival) = exp(−Σp)** (Poisson approximation, best used for *comparing* routes), mean uncertainty, minimum clearance, nodes expanded, milliseconds.
* **Energy model:** `E = m·g·(C_rr·distance + climb) / η`, with m = 900 kg, g = 3.71 m/s² (Mars), η = 0.45, and rolling resistance C_rr per terrain class (sand 0.25, bedrock 0.07, ...).
* **Cost-to-go:** scipy's compiled Dijkstra over a sparse graph computes the distance from one point to every cell at once (used for ranking and the tour).

### 5.7 Subsurface ice detection (`science.py`)
* **Naive-Bayes log-odds fusion:** start from a prior P(ice) = 0.08, i.e. log-odds = logit(0.08). Add one log-likelihood ratio (LLR) per instrument:
  * **Neutron hydrogen:** Gaussian likelihoods, ice ~ N(11 %, 4.5) vs dry ~ N(3.5 %, 2.2), clipped to ±6.
  * **Thermal inertia:** ice ~ N(1250, 380) vs dry ~ N(520, 380), clipped to ±4.
  * **Surface classifier:** P(ice-rich class) vs the class prior, ×0.6, clipped to ±4.
* `P(ice) = sigmoid(sum)`. Log-odds are **additive**, so the UI shows each instrument's contribution as a bar (explainability).
* **Monte-Carlo confidence interval:** inputs are resampled with their noise, giving a mean, σ and 90 % CI of P(ice).
* A target is flagged **"buried ice – invisible to cameras"** when P(ice) > 0.5 but the surface classifier sees no ice.
* The likelihood numbers are set by hand (not learned); this is an honest limitation.

### 5.8 Science value and ranking
* `value = P(ice) + 0.7·P(clay) + 0.25·P(bedrock) + 0.15·diversity` (diversity = local entropy of classes, because geological contacts are interesting).
* Value σ comes from propagating the classifier and ice uncertainties, which gives a 90 % CI bar on each target.
* **Target finding:** local maxima of the smoothed value map, at least 14 cells apart, top 8, only on reachable ground.
* **Three ranking policies:**
  * **Expected value** (mean).
  * **Explore / UCB** (mean + σ): prefer uncertain sites that would teach the most.
  * **Conservative / LCB** (mean − σ): prefer guaranteed value.
  * Score = utility / (1 + travel cost / 2500).
* **Science tour (orienteering problem):** choose the subset and order of targets that maximises total value under an energy budget (fixed at 250 Wh in the UI). It is solved **exactly** with bitmask dynamic programming over up to 7 targets (2⁷ subsets); a test checks it against brute force.

### 5.9 Closed-loop autonomous traverse (`rover_sim.py`)
* The rover drives cell by cell along the plan. Each step it "sees" true terrain within a **6-cell (12 m) radius**.
* **Fault injection:** an unmapped boulder field is placed on the route (checkbox "Inject unmapped boulder field").
* When newly seen hazards block the remaining route, the rover **replans with A\* on board** in milliseconds and logs it. Consecutive replans are merged into "detour refined N×".
* **False alarms cleared:** cells that orbit thought lethal but the rover sees are safe are released (green on the map).
* **Online learning:** when wheel slip is measured on sand (25–45 %), the sand cost is multiplied by 1.4 for the rest of the drive.
* **Safety:** the landing area and already-driven cells are known-safe. If no route exists, the rover stops in a safe state and asks Earth.
* **In-situ verification:** at each science target it takes a simulated drill/neutron reading of the true water content. Ice is called true if WEH > 12 %, and this is compared with the orbital P(ice) > 0.5 → "CONFIRMED" or "WRONG – flag for retraining".
* The log has timestamps in mission minutes and kinds: info, replan, learn, science, warn, done.

---

## 6. The website, screen by screen

> UI labels were simplified for readability: tabs are **Map / 3D / AI model / How it works**; control sections are **1 Terrain, 2 Route, 3 Science, 4 Drive**; λ is labelled **Caution**; the terrain switch is **Simulated / Real Mars**; sensor-noise sliders sit in a collapsible **Sensor noise** box; the traverse button is **Drive** and the fault-injection checkbox is **Surprise boulders**; map layers are grouped as **See / AI / Risk / Science**.

### 6.1 Top navigation
Fixed bar that is transparent over the hero and turns black after scrolling. Brand "MARS-09 // ODYSSEY", four tabs (Mission map, 3D terrain, Perception model, How it works) and a status chip with a pulsing dot (READY / THINKING / TRAVERSE / COMPLETE). On phones the tabs scroll sideways in one row.

### 6.2 Hero (first screen)
* Full-screen background: a Mars landscape **rendered from the real Jezero HiRISE elevation data**, slowly zooming (Ken Burns effect).
* Headline in 80 px D-DIN: "SEE. DECIDE. DRIVE. WITHOUT WAITING FOR EARTH.", plus a one-paragraph explanation.
* **Launch mission ›** button scrolls to the console and runs the full demo automatically. **Explore the console ↓** just scrolls.
* **KPI tiles** that count up: terrain accuracy, lethal terrain %, DEM missing %, point-cloud size.
* **Pipeline strip** SENSE → ACT, which lights up as each stage completes (an orange animated stage is running).
* A line showing the terrain source, and a vertical caption crediting the HiRISE data.

### 6.3 Band header
A large heading that changes per tab: "Perceive. Plan. Drive." / "The ground truth of the route." / "Measured, not claimed." / "Sense to act."

### 6.4 Mission map tab
**Left control panel**
* **▶ Run full mission:** build → plan → science tour → traverse in one click.
* **01 World & sensors:** terrain source (Procedural / Real Mars · Jezero), seed, DEM noise σ, DEM dropout %, spectral noise, and a *Generate & perceive* button. The sliders let judges stress-test robustness.
* **02 Risk & route:** λ slider (rebuilds the cost map), planner choice, click mode (Inspect / Set start / Set goal), "Compare planners" and "RRT* tree" checkboxes, *Plan route*.
* **03 Science mission:** ranking policy (Expected / UCB / LCB), *Plan science tour*.
* **04 Autonomous traverse:** "Inject unmapped boulder field", *Execute traverse*, play/pause and a scrub slider.

**Centre map (canvas)**
* 15 layer chips (keys 1–9 and 0 switch the first ten): Camera, Raw DEM, Elevation, Terrain AI, Ground truth, Uncertainty, Epistemic, Slope, Slope σ, Hazard P, Cost map, Neutron, Ice P, Ice σ, Science value. Each has a legend and a one-line explanation.
* Overlays: selected route (white), other planners (dashed colours), no-risk baseline (red), science tour (orange), targets (dots with a dashed ring sized by uncertainty), and S/G markers.
* HUD: layer label (top-left), north arrow (top-right), 50 m scale bar (bottom-right), hover coordinates.
* **During the traverse:** orange driven trail, sensing circle with a **radar sweep**, red revealed hazards with a **pulse** when first discovered, green cleared false alarms, faded old plans, and a **telemetry panel** (rover position, odometry, energy, replans, step). Space pauses or resumes.

**Right panel**
* **Cell inspector:** click any cell to see elevation ± σ, slope ± σ, roughness, class probabilities, total/aleatoric/epistemic uncertainty, each hazard source, every cost term, ice log-odds per instrument, P(ice) with 90 % CI, and the hidden truth for comparison. Every number behind a decision is shown.
* **Science targets:** ranked list with value ± σ, a 90 % CI bar, P(ice) with CI, distance / energy / P(safe) to reach it, a "buried ice" tag, and an expandable evidence trail and hypothesis. Hovering highlights the target on the map; targets in the tour get an orange marker.

**Below the map**
* **Route comparison table:** A*, Dijkstra, RRT*, no-risk baseline × length, energy, drive time, **P(safe arrival)**, hazard exposure, max slope, mean uncertainty, clearance, nodes expanded, ms.
* **"Why this route?"** stacked bar: share of cost from terrain, slope, roughness, uncertainty-risk and clearance, plus the terrain types crossed.
* **Autonomy decision log:** every on-board decision with mission time; future entries are faded until the replay reaches them. It shows the in-situ check score.

### 6.5 3D terrain tab
Plotly 3-D surface of the fused elevation with the planned route, no-risk baseline, tour or driven path, and labelled targets draped on it. Drag to orbit.

### 6.6 Perception model tab
* KPIs: accuracy on unseen pixels, macro-F1, out-of-bag accuracy, ECE, temperature, number of labelled pixels, accuracy on the 50 % most confident.
* Charts: confusion matrix, **accuracy vs coverage** (proves the uncertainty is meaningful), reliability diagram (calibration), top feature importances, predicted map vs ground-truth map.
* **Robustness benchmark:** 24 full missions (12 seeds × procedural and Jezero). KPI tiles, an accuracy box plot per terrain type, a scatter of A* P(safe) vs baseline P(safe) (every point above the diagonal means A* is safer), and a per-run table.

### 6.7 How it works tab
The 8 pipeline stages explained, plus a data provenance table (procedural terrain, real HiRISE source, what is simulated on real terrain, training labels).

### 6.8 Small UX details
Toast messages instead of browser alerts, a busy overlay with a scanning line, fade-up animations on scroll, motion turned off for users who prefer reduced motion, tabular numerals, and responsive layout down to phone width with no horizontal scroll.

---

## 7. Results we measured

### Seed 7, procedural terrain, default settings
| Metric | Value |
|---|---|
| Classifier accuracy (unseen pixels, from 4 % labels) | **94.8 %** |
| Accuracy on the 50 % most-confident pixels | **99.7 %** |
| P(safe arrival): A* / RRT* / no-risk baseline | **56 %** / 41 % / **15 %** |
| A* route vs baseline length | 388 m vs 370 m (5 % longer, ~3.7× safer) |
| λ sweep | λ = 0 → 417 m at 49 %; λ ≥ 0.25 → 388 m at 56 % (shorter **and** safer) |
| Pipeline time | ≈ 1.2–1.3 s; A* ≈ 60 ms; tour DP ≈ 1 ms |

### Robustness benchmark (24 missions, `backend/benchmark.py`)
| | Procedural (12) | Real Jezero DTM (12) |
|---|---|---|
| Accuracy | 92.3 % ± 2.0 (worst 87.4 %) | 93.3 % ± 1.8 (worst 89.2 %) |
| Accuracy, confident 50 % | 99.5 % | 99.7 % |
| P(safe) A* vs baseline (mean) | 50 % vs 9 % | 92 % vs 44 % |
| A* at least as safe as baseline | 12 / 12 | 12 / 12 |
| True hazard cells entered while driving | 0 in 12 / 12 | 0 in 12 / 12 |
| In-situ ice predictions confirmed | 95 % (44 sites) | 92 % (71 sites) |

### Tests (`tests/test_pipeline.py`, 8 tests, all passing)
1. A* path cost equals Dijkstra's and scipy's on random grids (optimality).
2. A* and RRT* paths never enter lethal cells (this caught a real RRT* bug, now fixed).
3. Orienteering DP equals brute force.
4. Harmonic in-painting reconstructs a plane exactly.
5. Point-cloud rasterisation rejects 4 % outliers (error < 5 cm).
6. The full pipeline runs end to end.
7. The pipeline works on the real Jezero terrain.
8. The λ sweep restores state, and the mission report is generated.

---

## 8. Files

```
HackInno/
  design/DESIGN.md                 SpaceX design system (opendesignmd.com, MIT)
  odyssey/
    run.bat                        create venv, install, start server on :8000
    requirements.txt               numpy, scipy, scikit-learn, fastapi, uvicorn, pillow
    backend/
      app.py                       FastAPI endpoints (see below)
      mission.py                   orchestrates the whole pipeline; plan / targets / tour / traverse / cell / pareto / report
      terrain_sim.py               world + sensor simulator; accepts a real DEM as base
      real_terrain.py              loads the cached Jezero DTM
      perception.py                fusion, in-painting, slope MC, roughness, hazard probability
      classifier.py                Random Forest, uncertainty, calibration, metrics
      costmap.py                   cost map, lethal mask, clearance
      planner.py                   A*, Dijkstra, RRT*, graph cost-to-go, path stats, energy
      science.py                   ice fusion, science value, targets, ranking, orienteering
      rover_sim.py                 closed-loop traverse simulation
      render.py                    colour maps → PNG layers
      benchmark.py                 24-world robustness benchmark → data/benchmark.json
    tools/
      fetch_hirise.py              downloads the Jezero DTM window via HTTP range requests
      render_hero.py               renders the hero landscape from a real DTM strip
    data/                          jezero_dtm.npy/.json, benchmark.json, hero.json (cached, offline)
    frontend/
      index.html, style.css, app.js, plotly.min.js, hero.jpg, fonts/D-DIN*.woff
    tests/test_pipeline.py         8 correctness tests
    docs/JUDGE_GUIDE.md            Q&A preparation, demo script, numbers to remember
    docs/PROJECT_EXPLAINED.md      this file
```

**API endpoints:** `POST /api/build` (seed, noise settings, λ, terrain), `POST /api/risk` (λ), `POST /api/plan` (start, goal, algo), `POST /api/targets` (start, mode), `POST /api/tour` (start, budget, mode), `POST /api/traverse` (waypoints, names, inject), `GET /api/cell?r=&c=`, `GET /api/benchmark`. `POST /api/pareto` and `GET /api/report` exist in the backend but have no button in the UI any more; they were removed to keep the site focused on the problem statement.

---

## 9. What changed during development (history)

1. **Phase 1 – Core:** simulator, fusion, cost map, A*.
2. **Phase 2 – Intelligence:** Random Forest with uncertainty, Bayesian ice fusion, science ranking, exact tour, closed-loop rover.
3. **Phase 3 – Show:** console UI in the SpaceX style, 3-D view, model evaluation tab, tests, README and judge guide.
4. **Phase 4 – Hardening:** real HiRISE Jezero terrain, 24-world benchmark, one-click demo, pipeline strip, map HUD.
5. **Cleanup:** removed features outside the problem statement (light-time slider, energy chips, energy-budget slider, report export button, λ trade-off chart, pipeline timings).
6. **Cinematic UI:** full-screen hero rendered from real Mars data, D-DIN font, count-up KPIs, scroll animations, radar sweep, hazard pulses, telemetry panel, mobile nav.

**Bugs fixed along the way:** a numpy/pandas conflict (fixed with a project virtual environment); 86 % of the map lethal (terrain smoothed, roughness limit raised to 0.70 m); no path found because the cliff split the map (two ramps added, craters kept away from them); the rover trapped at the start (known-safe landing zone); ice check counting clay as ice (threshold raised to 12 % WEH); replan log spam (merged); RRT* cutting through lethal cells (dense edge checking); calibration not helping (kept patch-grouped CV and documented under-confidence).

---

## 10. Honest limitations

* Surface materials and instrument readings are **simulated**, even on the real Jezero topography. Accuracy on real Mars images is unknown.
* Training and testing pixels come from the **same map** (different pixels), so it is not a cross-site generalisation test. The 24-world benchmark partly addresses this.
* The classifier is per-pixel with context features, not a deep CNN. This was chosen for low compute, small labels and interpretability.
* P(safe arrival) assumes independent cells (a Poisson approximation); it is good for comparing routes, not an absolute guarantee.
* The ice model's likelihoods are hand-set, and naive Bayes double-counts correlated evidence a little (we clip the LLRs).
* The energy model covers rolling resistance and climbing only (no heating or communications load).
* It is a single-user demo server with state in memory.

**Future work:** U-Net on AI4Mars labels, real CRISM spectra, spatially correlated risk (Gaussian process), D* Lite incremental replanning, multi-day planning with a power/thermal model.

---

## 11. Glossary

* **DEM / DTM:** digital elevation / terrain model, a grid of heights.
* **HiRISE:** High Resolution Imaging Science Experiment, a camera on NASA's Mars Reconnaissance Orbiter; its stereo pairs give 1 m DTMs.
* **Point cloud:** 3-D points from the rover's stereo cameras.
* **WEH:** water-equivalent hydrogen, % of water by mass implied by neutron data.
* **Thermal inertia:** resistance to temperature change; high for rock and ice.
* **Aleatoric uncertainty:** noise in the data that won't go away with more data.
* **Epistemic uncertainty:** the model's lack of knowledge; more data helps.
* **Entropy:** a measure of how spread out a probability distribution is (0 = certain).
* **Calibration / ECE:** whether "80 % confident" really means right 80 % of the time; ECE measures the gap.
* **Temperature scaling:** softening or sharpening probabilities with one parameter T.
* **Selective accuracy:** accuracy when only the most confident predictions are kept.
* **Admissible heuristic:** an estimate that never overestimates the true remaining cost, which keeps A* optimal.
* **RRT\*:** Rapidly-exploring Random Tree with rewiring; asymptotically optimal sampling planner.
* **Chance constraint:** forbid any cell whose hazard probability exceeds a threshold (0.5 here).
* **Orienteering problem:** choose which places to visit, and in what order, to maximise reward under a budget (NP-hard).
* **UCB / LCB:** upper / lower confidence bound (mean ± σ), used for explore vs exploit.
* **Log-odds / LLR:** log(p/(1−p)); evidence from independent sources adds up in log-odds.
* **Monte-Carlo:** estimating uncertainty by running the calculation many times on random samples.
