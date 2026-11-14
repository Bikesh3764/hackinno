# Judge Guide: everything you need to explain MARS-09 Odyssey

Read this once fully, then re-read sections 4–6 before presenting.

---

## 1. The problem in one sentence

A Mars rover is 4–24 light-minutes from Earth, so it cannot ask "is this rock safe?" for every metre. It must **turn raw, noisy sensor data into decisions by itself**, and **know when it doesn't know**.

Our one-line pitch:
> "MARS-09 doesn't just find a path. It knows how confident it is about every square metre, trades risk against science against energy, and replans on its own when reality differs from the orbital map."

The judging criteria map directly onto features:

| Criterion | Our answer |
|---|---|
| Technical depth | Kalman fusion, Monte-Carlo uncertainty, RF ensemble with uncertainty decomposition, A*/Dijkstra/RRT*, exact orienteering DP, Bayesian evidence fusion |
| Investigation & reasoning | Science targets come with a hypothesis and an evidence trail ("+2.1 log-odds from neutrons…") |
| Uncertainty & robustness | Noise, holes, outliers, dust, label noise are all injected on purpose and handled; every output carries ± or a CI |
| Execution & explainability | Cell inspector, route cost breakdown, decision log, chance-constrained safety |
| Innovation & usability | Risk slider (λ) changes the route live; UCB vs LCB science policy; fault injection; on-site verification |

---

## 2. The pipeline step by step (with the "why")

### 2.1 Sense: what data does the rover get?
| Sensor | Real-world analogue | What it tells us | Imperfection we simulate |
|---|---|---|---|
| Orbital DEM (elevation) | HiRISE stereo DTM (~1 m) | height → slope, roughness | Gaussian noise σ = 0.35 m, **4 % missing** (shadows, failed stereo) |
| 6-band multispectral | HiRISE colour + CRISM bands | mineralogy | illumination, dust mantling, noise |
| Thermal inertia | THEMIS | grain size; ice and rock are "thermally heavy" | 10 % noise, blurred |
| Neutron spectrometer | **Mars Odyssey** GRS/NS, which actually discovered subsurface ice in 2002 | hydrogen = water, even buried | very coarse footprint, noisy |
| Rover stereo point cloud | Navcam / Hazcam | precise near-field 3-D | **1.5 % gross outliers** |

**Ground truth is hidden.** It is only used to score our models and to simulate what the rover's cameras see while driving.

### 2.2 Fuse: building the 3-D elevation map
1. **Point cloud → grid**: each 2 m cell takes the **median** of its points, and points more than 3 MAD from it are rejected. (Median Absolute Deviation is the robust cousin of standard deviation; the median isn't dragged by outliers, unlike the mean.)
2. **Fill DEM holes**: *harmonic in-painting* solves Laplace's equation ∇²z = 0 inside the hole, which gives the smoothest surface that matches the edges, like stretching a rubber sheet. Solved by Jacobi iteration (each cell repeatedly becomes the average of its 4 neighbours).
3. **Uncertainty of filled cells** grows with distance from real data: σ² = σ_sensor² + (0.35·d)².
4. **Fusion**: where both orbital and rover data exist, we take the **inverse-variance weighted average**: z = (z₁/σ₁² + z₂/σ₂²)/(1/σ₁² + 1/σ₂²), σ² = 1/(1/σ₁² + 1/σ₂²). This is the one-dimensional **Kalman filter** update. It is the optimal linear estimator, and the more precise sensor automatically gets more weight.

### 2.3 Perceive: slope, roughness, terrain class
* **Slope** = arctan(|∇z|) using central differences after a light Gaussian smoothing.
* **Slope uncertainty** by **Monte-Carlo propagation**: draw 24 plausible DEMs from N(z, σ_z), compute slope for each, and take the standard deviation. Inside data holes the slope σ is large, so the planner is careful there.
* **Roughness** = RMS of the elevation residual after removing the smooth trend (σ = 2 cells), measured in a 5×5 window. DEM noise bias is subtracted in quadrature.
* **Terrain classifier = Random Forest (80 trees)** on 19 features per pixel:
  * band ratios (each band ÷ mean brightness) cancel shadows and illumination
  * band depths at 1.9 µm (water) and 2.3 µm (clay OH)
  * thermal inertia, slope, roughness, curvature, relative elevation, texture
  * "context" features (Gaussian-smoothed) give the model neighbourhood information
* Trained on **only ~4 % labelled pixels** (a few 5×5 patches per class, with **4 % deliberately wrong labels**), simulating a small expert annotation campaign.

**Why Random Forest and not a CNN?**
1. A rover CPU is weak (Perseverance's RAD750 runs at ~200 MHz), and an RF is cheap at inference.
2. It works with very little labelled data.
3. It is interpretable: feature importances show *why*.
4. **The tree ensemble gives uncertainty for free.**
The upgrade path is a U-Net segmentation network trained on the real AI4Mars dataset (NASA, ~35 k labelled Curiosity images), with MC-dropout or deep ensembles.

### 2.4 Uncertainty: the core idea of the project
For each pixel we have 80 trees, each giving a probability vector pₜ.
* **Total uncertainty** = entropy of the average: H[mean pₜ]
* **Aleatoric** = average entropy of individual trees: mean H[pₜ]. This is *data* ambiguity (e.g. dust-covered rock looks like soil). More data **won't** fix it.
* **Epistemic** = total − aleatoric = **mutual information**. The trees *disagree*, so the model has not seen this kind of terrain. More data **would** fix it.

Proof that it works (Model tab): accuracy is ~94.8 % overall but **99.7 % on the 50 % of pixels the model is most confident about**. The uncertainty is meaningful, and we can use it to decide where to be careful.

**Calibration:** a "70 % confident" prediction should be right 70 % of the time. The reliability diagram shows this. We fit **temperature scaling** with **patch-grouped cross-validation**: whole patches are held out, because neighbouring pixels are nearly identical and ordinary CV would leak. The model is slightly **under-confident** against clean truth because we trained on noisy labels. For a rover that is the safe direction.

### 2.5 Assess risk: hazard probability
Three independent hazard sources per cell:
* P(too steep) = Φ((slope − 25°) / σ_slope). Φ is the normal CDF, so the slope uncertainty enters directly.
* P(too rough) = Φ((roughness − 0.70 m) / 0.08)
* P(boulder field) from the classifier's "Rocky field" probability
* Combined: **P(hazard) = 1 − Π(1 − pᵢ)**, the probability that at least one hazard occurs.

**Chance constraint:** a cell is *lethal* if P(hazard) > 0.5. Lethal cells are **inflated by the rover's radius** (a distance transform), so the rover's *body* never touches them, not just its centre point.

### 2.6 Cost map
```
cost = E_class[terrain cost] × slope factor × roughness factor × (1 + λ·U) + λ·8·P(hazard) + clearance
```
* **E_class[terrain cost]** = Σ_c P(class c) × cost_c. We use the *expected* cost over the class posterior instead of trusting the top class. A pixel that is 50 % sand / 50 % soil costs in between.
* **U** = uncertainty (classifier entropy + slope σ).
* **λ (risk aversion)**: λ = 0 trusts the map blindly; high λ pays to stay in well-understood terrain. This is the **slider in the UI**, and the route visibly changes.

### 2.7 Plan
| Algorithm | Idea | Pros / cons |
|---|---|---|
| **Dijkstra** | expand nodes in order of cost-so-far g | optimal, but explores in every direction |
| **A\*** | expand by f = g + h; h = straight-line distance × cheapest cell cost | **same optimal path, fewer nodes** if h is *admissible* (never overestimates) |
| **RRT\*** | grow a random tree in continuous space, rewire neighbours to cheaper parents | handles continuous / high-dimensional spaces, *asymptotically* optimal, but random and slower on a grid |
| **No-risk baseline** | shortest path that only avoids lethal cells | shows what the risk model buys |

Edge cost = 3-D step length √(horizontal² + Δz²) × average cell cost, so climbing costs more. Diagonal moves cannot cut corners past obstacles.

*If asked "A* and Dijkstra expanded about the same number of nodes":* our heuristic uses the **minimum** cell cost to stay admissible, and the cost field varies a lot, so h is a weak lower bound. A weighted A* (ε·h) would be faster but gives up exact optimality, trading guarantees for speed.

**Path metrics:** length, energy, drive time, max slope, hazard exposure Σp, **P(safe arrival) ≈ exp(−Σp)** (Poisson approximation), mean uncertainty, minimum clearance.

**Energy model:** E = m·g_Mars·(C_rr·distance + climb) / efficiency, with m = 900 kg (Curiosity-class), g = 3.71 m/s², terrain-specific rolling resistance (sand 0.25, bedrock 0.07), efficiency 45 %.

### 2.8 Discover: science and subsurface ice
**Ice probability: naive-Bayes evidence fusion in log-odds**
```
log-odds(ice) = log-odds(prior 0.08) + LLR_neutron + LLR_thermal + LLR_spectral
```
Each LLR (log-likelihood ratio) = log P(observation | ice) / P(observation | dry), from Gaussian likelihood models. **Log-odds add**, so the evidence trail shows exactly how much each instrument contributed (that's the bar chart in the inspector). Each term is clipped so one sensor can't dominate.

This finds **buried ice that cameras cannot see**: neutrons and thermal inertia detect it while the spectral term is neutral. Targets with this pattern get the tag "buried ice, invisible to cameras".

**Uncertainty:** 40 Monte-Carlo samples perturb the neutron reading, thermal reading and classifier probability within their noise, giving the mean, σ, and 90 % interval of P(ice).

**Science value** V = 1.0·P(ice) + 0.7·P(clay) + 0.25·P(bedrock) + 0.15·geological diversity. The diversity term is the entropy of nearby classes: contacts between rock units are scientifically interesting. Targets are local maxima of V, snapped to safe approach points.

**Ranking policies** (what to do with uncertainty is itself a decision):
* *Expected*: rank by mean value.
* *Explore (UCB)*: mean + σ. Prefer uncertain sites because visiting them teaches us the most (information gain, the same idea as multi-armed bandits).
* *Conservative (LCB)*: mean − σ. Only go where we're sure it is valuable.
Score = utility / (1 + travel cost / 2500), so reachability matters.

**Mission tour = Orienteering Problem:** choose *which* targets to visit and in what order to maximise total value without exceeding the energy budget. It is NP-hard in general, but for ≤ 7 targets we solve it **exactly** with a bitmask DP (Held-Karp style), O(2ⁿ·n²) ≈ 6,000 states, in < 10 ms. The legs come from a multi-source Dijkstra (scipy, C speed) on the risk-aware cost graph.

### 2.9 Act: closed-loop autonomy
* The rover starts with the **orbital belief map**. Every step it senses the true terrain within ~12 m (hazcams).
* If a newly seen hazard blocks the remaining route, it **replans with A* on board in milliseconds**. The log states that Earth-in-the-loop would have cost 2 × light time of idle waiting.
* **Fault injection** (checkbox) places an unmapped boulder field on the route to prove the replanning is real.
* **Online learning:** measured wheel slip on sand (visual odometry vs. wheel turns) raises the cost of sand-classified cells everywhere it hasn't been yet.
* **False alarms cleared:** orbital "lethal" cells that look fine up close are re-opened (green).
* **Safe failure:** if no safe route exists, it stops and flags for Earth review instead of gambling.
* **In-situ verification:** at each target it "drills" and compares the reading to the orbital prediction, then reports confirmed / wrong, closing the scientific loop.

---

## 3. Numbers to remember (seed 7, defaults)
* Map: 160 × 160 cells × 2 m = 320 m × 320 m.
* Classifier: ~94.8 % accuracy from 4 % labels; 99.7 % on the most-confident half; 80 trees.
* Risk-aware A*: P(safe arrival) ≈ 56 % vs ≈ 15 % for the no-risk shortest path.
* On-site ice checks: 27 / 29 correct across 7 random worlds.
* Perception pipeline ≈ 1.3 s; A* 10–50 ms; tour DP < 10 ms.
* Slope limit 25°; roughness limit 0.70 m RMS; rover footprint radius 1 cell (2 m).
* Light time Earth–Mars: 4 to 24 minutes one way.

**Robustness benchmark (12 seeds × 2 terrain types = 24 full missions, `backend/benchmark.py`):**
* Procedural worlds: accuracy **92.3 % ± 2.0** (worst 87.4 %), 99.5 % on the confident half; P(safe) A* **50 %** vs baseline **9 %** on average.
* Real Jezero topography: accuracy **93.3 % ± 1.8**; P(safe) A* **92 %** vs baseline **44 %**.
* A* was at least as safe as the baseline in **24 / 24** worlds; the rover drove into a true hazard cell in **0 / 24** traverses.
* In-situ ice checks: **95 %** agreement (44 sites, procedural), **92 %** (71 sites, Jezero).
* λ sweep (seed 7): λ = 0 gives a 417 m route at 49 % P(safe); λ ≥ 0.25 gives 388 m at 56 %, so the risk term finds a route that is **both** shorter and safer than ignoring uncertainty.

---

## 4. 8-minute demo script

| Time | Do | Say |
|---|---|---|
| 0:00 | Hero screen | "A rover 20 light-minutes away must decide alone. The danger isn't just hazards, it's being *confidently wrong*." |
| 0:45 | Layers: Camera → Raw DEM | "Raw data is noisy: magenta is missing elevation, dust confuses the spectra." |
| 1:30 | Elevation → Terrain AI → Ground truth | "We fuse orbit and rover point clouds, and classify terrain from 4 % labels at 95 % accuracy." |
| 2:15 | Uncertainty → Epistemic | "Here's what makes us different: it knows where it's unsure, and on its confident half it's 99.7 % right." |
| 3:00 | Hazard P → Cost map, click a cell | "Every decision is traceable: this cell is lethal because P(steep) = 0.9." |
| 3:45 | Route comparison table, move λ slider | "The no-risk route is shorter but only ~15 % safe; ours is ~56 %. Raise λ and the rover avoids uncertain ground." |
| 4:45 | Ice P layer, open a target | "Ice probability fuses three instruments. Here, buried ice the cameras can't see, with the evidence and confidence interval." |
| 5:30 | Plan science tour | "We solve the orienteering problem exactly: best science under 250 Wh." |
| 6:15 | Execute traverse | "Now the rover drives. A boulder field invisible from orbit appears, and it replans in milliseconds. It also learns sand is slippery." |
| 7:15 | Log + 3D tab | "Each arrival checks the orbital prediction against a drill reading. Everything is logged for Earth." |
| 7:30 | Switch to *Real Mars · Jezero*, then Model tab → Robustness benchmark | "Same pipeline on real HiRISE topography of Jezero, and across 24 worlds: never drove into a hazard." |
| 7:50 | Close | "Honest uncertainty, traceable decisions, safe autonomy." |

**Before presenting:** run `run.bat`, open the page, use seed 7, and do one dry run. Backup plan: the **▶ Run full mission** button runs build → tour → traverse in one click. Keys 1–9 / 0 switch map layers; Space pauses the traverse. If the projector is small, zoom the browser to 80 %.

---

## 5. Likely judge questions and good answers

**Q: Is the data real?**
A: Two modes. (1) **Real Mars topography:** the *Real Mars · Jezero* button loads a 320 m crop of a NASA HiRISE digital terrain model of Jezero crater, Perseverance's landing site (DTEEC_023524_1985_023379_1985_U01, 1 m/px, averaged to 2 m). We downloaded only ~10 MB of the 700 MB file using HTTP range requests and cached it, so the demo stays offline. (2) **Procedural worlds.** In both modes the surface materials and instrument noise are simulated, because that gives us *ground truth* to measure accuracy, calibration and in-situ agreement, which real orbital data cannot. The next step is real labels (AI4Mars) for the classifier.

**Q: Is 95 % just one lucky seed?**
A: No. The Model tab's robustness benchmark runs the full mission on 24 independent worlds (12 procedural, 12 on real Jezero topography): accuracy 92–93 % mean, worst case 87 %, A* at least as safe as the baseline in 24/24, and zero true hazard cells entered in 24/24 traverses.

**Q: Where is the AI/ML?**
A: (1) Random-Forest terrain classifier with uncertainty decomposition and calibration. (2) Bayesian evidence fusion for ice. (3) Online learning of slip cost during the drive. (4) Planning and search (A*, RRT*) and combinatorial optimisation (orienteering) are classical AI.

**Q: Why not deep learning?**
A: Flight computers are radiation-hardened and slow, labelled Mars data is scarce, and we need interpretability. The RF gets ~95 % with 4 % labels and gives uncertainty for free. A U-Net on AI4Mars with deep ensembles is our upgrade path; the uncertainty machinery stays the same.

**Q: How do you know the uncertainty is correct?**
A: Two tests in the Model tab. *Selective accuracy*: keeping only the most-confident 50 % raises accuracy from ~95 % to 99.7 %. *Reliability diagram*: we're slightly under-confident (because of the 4 % label noise), which is the safe direction.

**Q: Aleatoric vs epistemic?**
A: Aleatoric is noise in the world (dust makes rock look like soil); more data won't help. Epistemic is a gap in the model's knowledge; more data helps. We separate them using tree disagreement (mutual information). Epistemic hot-spots are where you'd send the rover's cameras or ask for more labels.

**Q: Why A* and not just Dijkstra? Is A* optimal?**
A: Both are optimal. A* uses an admissible heuristic (straight-line distance × minimum cell cost, which never overestimates) to explore fewer nodes. Our tests verify A*'s path cost equals Dijkstra's and an independent scipy implementation on random maps.

**Q: When would you use RRT*?**
A: In continuous or high-dimensional spaces (arm motion, rover heading plus kinematic constraints) where gridding is expensive. On a 2-D grid, A* is better, and our comparison table shows it.

**Q: What does λ do?**
A: (Move the **Caution** slider, which is λ, and watch the route table.) It prices uncertainty. Cost includes λ × uncertainty and λ × hazard probability. λ = 0 is a blind shortest-cost path; higher λ avoids poorly understood terrain at the price of distance. Mission control would set it per sol depending on the remaining mission margin.

**Q: How is "P(safe arrival)" computed? Is it real?**
A: Sum of per-cell hazard probabilities along the path, then exp(−Σp), the Poisson approximation assuming independent cells. It's best for *comparing* routes. A more rigorous version would model spatial correlation.

**Q: How do you handle missing data?**
A: Harmonic in-painting fills holes smoothly, and filled cells get variance that grows with distance from real data. That variance flows into slope σ, then into P(steep), then into cost. Missing data is **expensive**, not invisible.

**Q: How do you handle outliers in the point cloud?**
A: Per-cell median plus MAD rejection (3σ-equivalent). A unit test injects 4 % gross outliers and the estimate stays within 5 cm.

**Q: How do you combine orbital and rover elevation?**
A: Inverse-variance weighting, the 1-D Kalman update. The rover's point cloud (σ ≈ 5 cm) dominates where available; the orbital DEM (σ ≈ 35 cm) fills the rest.

**Q: How do you detect subsurface ice?**
A: Naive-Bayes log-odds fusion of neutron hydrogen, thermal inertia and the spectral classifier. Log-odds are additive, so we show each instrument's contribution. Buried ice shows up in neutrons and thermal data but not in spectra; we flag those targets.

**Q: Isn't naive Bayes's independence assumption wrong?**
A: Partly. Neutron and thermal both respond to ice, so the evidence is double-counted a little. We clip each LLR to limit over-confidence. The fix would be a learned joint model (e.g. logistic regression on ground-truth sites).

**Q: How do you choose which targets to visit?**
A: The Orienteering Problem: maximise total value subject to an energy budget. It is NP-hard in general; we solve it exactly for ≤ 7 candidates with bitmask DP, and a test checks it against brute force. For more targets you'd switch to a heuristic (greedy plus 2-opt, or an ILP solver).

**Q: UCB vs LCB, why both?**
A: That's the exploration–exploitation trade-off. Early in a mission you explore (UCB: uncertain sites teach the most); near the end, or with low power, you exploit (LCB: guaranteed value).

**Q: What happens if the rover gets stuck / no path exists?**
A: It stops in a safe state and flags for Earth review (shown in the log as a warning), rather than taking an unsafe route.

**Q: How does it react to things the orbital map missed?**
A: Closed loop: sense about 12 m around, update the belief, and if the remaining route crosses a newly lethal cell, replan with A*. The fault-injection checkbox demonstrates it live.

**Q: What would you do with more time?**
A: A U-Net on real AI4Mars imagery; spatially correlated risk (Gaussian process); D* Lite incremental replanning; real CRISM spectra on top of the HiRISE DTM we already use; a multi-sol planner with a power/thermal model; human-approval gates for high-risk actions.

**Q: What's novel here?**
A: Uncertainty is treated as a first-class signal from sensing to science: it changes cost, ranking policy (UCB/LCB), safety constraints and the explanation. Plus a closed loop that verifies its own science predictions in situ.

**Q: Did you build it during the hackathon?**
A: Yes. Only open-source libraries are used (numpy, scipy, scikit-learn, FastAPI, Plotly). The UI style follows an MIT-licensed DESIGN.md from opendesignmd.com.

---

## 6. Glossary (one line each)
* **DEM**: Digital Elevation Model, a height grid.
* **Point cloud**: a set of 3-D points from stereo cameras or LiDAR.
* **Inverse-variance weighting / Kalman update**: combine estimates, trusting precise ones more.
* **Harmonic in-painting**: fill holes with the smoothest surface (Laplace equation).
* **Monte-Carlo propagation**: sample inputs from their noise and look at the spread of outputs.
* **Entropy**: how spread out a probability distribution is (0 = certain).
* **Mutual information**: how much the ensemble members disagree, i.e. epistemic uncertainty.
* **Calibration / ECE**: whether confidence matches accuracy; Expected Calibration Error averages the gap.
* **Temperature scaling**: sharpen or soften probabilities with one parameter T.
* **Chance constraint**: "P(failure) must stay below α", instead of a hard yes/no.
* **Admissible heuristic**: never overestimates the remaining cost, so A* stays optimal.
* **RRT\***: Rapidly-exploring Random Tree with rewiring; asymptotically optimal.
* **Orienteering problem**: pick and order targets to maximise reward under a budget.
* **Log-odds / LLR**: log(p/(1−p)); evidence from independent sources adds up.
* **UCB / LCB**: upper / lower confidence bound; optimistic vs pessimistic decisions.
* **WEH**: Water-Equivalent Hydrogen, % of soil mass that would be water.
* **Thermal inertia**: how slowly ground heats and cools. Rock and ice are high; dust and sand are low.
