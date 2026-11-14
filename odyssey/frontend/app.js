/* MARS-09 Odyssey console - vanilla JS, no build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const pct = (v, d = 0) => (v === null || v === undefined) ? "–" : (v * 100).toFixed(d) + "%";

const CMAPS = {
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
  magma: ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
  ice: ["#0b0b10", "#16324f", "#2a6f97", "#61a5c2", "#d6f0ff"],
  heat: ["#0b0b0b", "#3d1308", "#8a1c0b", "#e0461b", "#ffb347", "#fff2c2"],
  mars: ["#2a1a14", "#5a2e1d", "#9c4a2a", "#c97a4a", "#e9c29a", "#fbefe0"],
  gray: ["#000000", "#ffffff"],
};

const LAYERS = [
  { key: "visible", grp: "See", label: "Camera", desc: "Orbital colour image. Dust and shadows make it ambiguous." },
  { key: "dem_raw", grp: "See", label: "Raw height", cmap: "gray", ends: ["low", "high"], desc: "Raw orbital elevation. Pink = missing data." },
  { key: "elevation", grp: "See", label: "Elevation", cmap: "mars", ends: "elev", desc: "Gaps filled and rover 3D points merged." },
  { key: "class", grp: "AI", label: "Terrain AI", classes: true, desc: "Terrain type predicted by the Random Forest." },
  { key: "truth", grp: "AI", label: "Truth", classes: true, desc: "Hidden truth, for scoring only. The rover never sees it." },
  { key: "uncertainty", grp: "AI", label: "Uncertainty", cmap: "magma", ends: ["sure", "unsure"], desc: "Bright = the AI is unsure here." },
  { key: "epistemic", grp: "AI", label: "Knowledge gap", cmap: "magma", ends: ["0", "0.6"], desc: "Where more training data would help." },
  { key: "slope", grp: "Risk", label: "Slope", cmap: "heat", ends: ["0°", "35°"], desc: "Ground steepness. Rover limit 25°." },
  { key: "slope_std", grp: "Risk", label: "Slope ±", cmap: "magma", ends: ["0°", "6°"], desc: "How unsure the slope is. High inside data gaps." },
  { key: "hazard", grp: "Risk", label: "Hazard", cmap: "heat", ends: ["0", "1"], desc: "Chance a cell is too steep, rough or rocky." },
  { key: "cost", grp: "Risk", label: "Cost", cmap: "viridis", ends: ["easy", "hard"], lethal: true, desc: "Driving cost. Dark red = off-limits." },
  { key: "neutron", grp: "Science", label: "Neutron", cmap: "ice", ends: ["dry", "wet"], desc: "Raw hydrogen signal from orbit. Coarse and noisy." },
  { key: "ice", grp: "Science", label: "Ice", cmap: "ice", ends: ["0", "1"], desc: "Chance of buried water ice, from 3 instruments." },
  { key: "ice_std", grp: "Science", label: "Ice ±", cmap: "magma", ends: ["0", "0.3"], desc: "How unsure the ice estimate is." },
  { key: "science", grp: "Science", label: "Value", cmap: "viridis", ends: ["low", "high"], desc: "Science value: ice, clay, bedrock and variety." },
];

const ALGO_STYLE = {
  astar: { color: "#ffffff", dash: [], w: 2.5 },
  dijkstra: { color: "#9aa0ff", dash: [2, 4], w: 1.5 },
  rrt: { color: "#ffb347", dash: [6, 4], w: 1.6 },
  naive: { color: "#ff4d4d", dash: [3, 3], w: 1.6 },
};

const S = {
  n: 160, cell: 2, layer: "visible", layers: {}, images: {}, summary: null,
  start: null, goal: null, plan: null, targets: [], tour: null, sim: null, step: 0,
  playing: false, clickMode: "inspect", inspect: null, highlight: null, elevRange: [0, 1],
  terrain: "procedural", bench: null,
};
const STAGES = ["sense", "fuse", "perceive", "risk", "cost", "plan", "discover", "act"];
function stages(on, run = []) {
  document.querySelectorAll("#stages li").forEach((li) => {
    li.classList.toggle("on", on.includes(li.dataset.s));
    li.classList.toggle("run", run.includes(li.dataset.s));
  });
}
let toastT = null;
function toast(msg, err = false) {
  const t = $("toast");
  t.textContent = msg; t.classList.toggle("err", err); t.classList.add("on");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("on"), 4200);
}
function status(txt, thinking = false) {
  $("chipStatus").innerHTML = `<i class="dot"></i>${txt}`;
  $("chipStatus").classList.toggle("thinking", thinking);
}

const canvas = $("map");
const ctx = canvas.getContext("2d");

// ------------------------------------------------------------------ API
// Plotly is 4.8 MB, so it loads only when a chart is first needed (keeps the first paint fast).
let plotlyLoading = null;
function loadPlotly() {
  if (window.Plotly) return Promise.resolve();
  if (!plotlyLoading) plotlyLoading = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "static/plotly.min.js"; s.async = true; s.onload = res; s.onerror = rej;
    document.head.appendChild(s);
  });
  return plotlyLoading;
}
async function preact(...args) { await loadPlotly(); return Plotly.react(...args); }

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch((window.ODYSSEY_API || "") + path, opt);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
function busy(on, msg = "Computing…") {
  $("busy").classList.toggle("on", on);
  $("busy").querySelector("span").textContent = msg;
  document.querySelectorAll(".controls .pill, #btnNext").forEach((b) => (b.disabled = on));
  status(on ? "THINKING" : (S.sim ? "TRAVERSE" : "READY"), on);
}

// ------------------------------------------------------------------ build
async function build() {
  busy(true, S.terrain === "jezero" ? "Loading real Mars terrain…" : "Building world · training AI…");
  stages([], STAGES.slice(0, 5));
  try {
    const s = await api("/api/build", {
      seed: +$("seed").value, dem_noise: +$("demNoise").value, dropout: +$("dropout").value,
      spectral_noise: +$("spec").value, lam: +$("lam").value, terrain: S.terrain,
    });
    stages(STAGES.slice(0, 5));
    $("sourceLine").innerHTML = s.terrain === "jezero"
      ? `<b>Real Mars · Jezero</b> · ${fmt(s.terrain_meta.relief_m, 0)} m relief`
      : `<b>Simulated · seed ${s.seed}</b> · ${s.n * s.cell} × ${s.n * s.cell} m`;
    if (S.terrain === "jezero" && s.terrain !== "jezero") toast("Real terrain not found. Using simulated.", true);
    S.summary = s; S.n = s.n; S.cell = s.cell;
    S.start = s.start; S.goal = s.goal;
    S.plan = null; S.tour = null; S.sim = null; S.step = 0; S.inspect = null; S.done = false;
    const flat = s.elevation.flat();
    S.elevRange = [Math.min(...flat), Math.max(...flat)];
    await loadLayers(s.layers);
    renderKpis(); renderModel(); renderLayerbar(); flowUpdate();
    $("log").innerHTML = '<p class="empty">Press Drive to see the rover’s decisions.</p>';
    await plan();
    await loadTargets();
    draw();
  } catch (e) { console.error(e); toast(e.message, true); }
  busy(false);
}

function loadLayers(layers) {
  S.layers = { ...(S.layers || {}), ...layers };   // /api/risk sends only the cost layer
  return Promise.all(Object.entries(layers).map(([k, src]) => new Promise((res) => {
    const im = new Image(); im.onload = () => { S.images[k] = im; res(); }; im.src = src;
  })));
}

async function setRisk() {
  busy(true, "Updating costs…");
  try {
    const s = await api("/api/risk", { lam: +$("lam").value });
    S.summary.stats = s.stats;
    await loadLayers(s.layers);
    renderKpis();
    S.tour = null;
    await plan();
    await loadTargets();
  } catch (e) { console.error(e); }
  busy(false);
}

// ------------------------------------------------------------------ planning
async function plan() {
  busy(true, "Planning routes…");
  try {
    const algo = $("algo").value;
    // fast planners first; RRT* (slow, pure Python) follows in the background unless it is selected
    S.plan = await api("/api/plan", { start: S.start, goal: S.goal, algo, rrt: algo === "rrt" });
    S.start = S.plan.start; S.goal = S.plan.goal;
    stages(STAGES.slice(0, 6));
    renderCompare(); flowUpdate();
    draw();
  } catch (e) { console.error(e); }
  busy(false);
  // small delay so quick follow-up calls (targets) reach the server before the slow RRT* job
  const p = S.plan;
  if (p && p.comparison.some((c) => c.pending)) setTimeout(() => fetchRrt(p), 250);
}

async function fetchRrt(p) {
  try {
    const r = await api("/api/rrt", { start: p.start, goal: p.goal });
    if (S.plan !== p) return;   // a newer plan replaced this one
    p.comparison = p.comparison.map((c) => (c.key === "rrt" ? r.entry : c));
    p.tree = r.tree;
    renderCompare(); draw();
  } catch (e) { console.error(e); }
}

async function loadTargets() {
  S.targets = await api("/api/targets", { start: S.start, mode: $("mode").value, kappa: 1.0 });
  renderTargets();
  draw();
}

async function tour() {
  busy(true, "Planning tour…");
  try {
    await loadTargets();
    S.tour = await api("/api/tour", { start: S.start, mode: $("mode").value, kappa: 1.0, budget: 250 });
    $("targetsNote").textContent = S.tour.order.length
      ? `${S.tour.order.join(" → ")} · ${fmt(S.tour.energy_wh, 0)} Wh`
      : "none in budget";
    stages(STAGES.slice(0, 7));
    renderTargets(); flowUpdate(); sideTab("targets");
    draw();
  } catch (e) { console.error(e); }
  busy(false);
}

async function drive() {
  busy(true, "Driving…");
  try {
    let wps, names;
    if (S.tour && S.tour.order.length) { wps = S.tour.waypoints; names = ["Start", ...S.tour.order]; }
    else { wps = [S.start, S.goal]; names = ["Start", "Goal"]; }
    S.sim = await api("/api/traverse", { waypoints: wps, names, inject: $("inject").checked });
    S.step = 0;
    $("scrub").max = S.sim.frames.length - 1;
    $("scrub").value = 0;
    S.playing = true; $("btnPlay").textContent = "Pause";
    $("logNote").textContent = `${S.sim.decisions} decisions · ${fmt(S.sim.energy_wh, 0)} Wh · ${fmt(S.sim.drive_h, 1)} h`;
    renderLog();
    busy(false);
    status("TRAVERSE");
    stages(STAGES.slice(0, 7), ["act"]);
    flowUpdate(); sideTab("log");
    play();
    return;
  } catch (e) { console.error(e); }
  busy(false);
}

let raf = null, lastT = 0;
function play() {
  cancelAnimationFrame(raf);
  const tick = (t) => {
    if (!S.sim || !S.playing) return;
    if (t - lastT > 45) {
      lastT = t;
      if (S.step < S.sim.frames.length - 1) { S.step++; $("scrub").value = S.step; onStep(); }
      else {
        S.playing = false; S.done = true; $("btnPlay").textContent = "Replay"; status("COMPLETE"); stages(STAGES); flowUpdate();
        const ch = S.sim.science_checks || [];
        toast(`Done · ${S.sim.decisions} decisions · ${fmt(S.sim.energy_wh, 0)} Wh` +
          (ch.length ? ` · ice ${ch.filter((c) => c.correct).length}/${ch.length} confirmed` : ""));
        if (S.demoResolve) { S.demoResolve(); S.demoResolve = null; }
        return;
      }
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
function onStep() {
  fillRange($("scrub"));
  const f = S.sim.frames[S.step];
  document.querySelectorAll("#log .e").forEach((el) => el.classList.toggle("future", +el.dataset.step > S.step));
  draw();
}

// ------------------------------------------------------------------ drawing
function cellToPx(r, c) { const k = canvas.width / S.n; return [(c + 0.5) * k, (r + 0.5) * k]; }

function draw() {
  const W = canvas.width, k = W / S.n;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, W);
  const im = S.images[S.layer];
  if (im) ctx.drawImage(im, 0, 0, W, W);

  // RRT* tree
  if ($("showTree").checked && S.plan && S.plan.tree) {
    ctx.strokeStyle = "rgba(255,179,71,0.18)"; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.beginPath();
    for (const [r1, c1, r0, c0] of S.plan.tree) {
      ctx.moveTo((c0 + 0.5) * k, (r0 + 0.5) * k); ctx.lineTo((c1 + 0.5) * k, (r1 + 0.5) * k);
    }
    ctx.stroke();
  }
  // planner comparison
  if (S.plan && !S.sim) {
    const sel = $("algo").value;
    for (const c of S.plan.comparison) {
      if (!c.found) continue;
      if (c.key !== sel && !$("showAlts").checked) continue;
      const st = ALGO_STYLE[c.key];
      polyline(c.path, c.key === sel ? "#ffffff" : st.color, c.key === sel ? 3 : st.w, c.key === sel ? [] : st.dash, c.key === sel ? 1 : 0.85);
    }
  }
  // science tour
  if (S.tour && S.tour.path.length && !S.sim) polyline(S.tour.path, "#ffb347", 2.5, [8, 5], 0.95);

  // traverse
  if (S.sim) drawSim(k);

  // targets
  for (const t of S.targets) {
    const [x, y] = cellToPx(t.r, t.c);
    const rad = 5 + 6 * Math.min(t.value, 1.5);
    const unc = 4 + 40 * t.value_std;
    ctx.setLineDash([2, 3]); ctx.strokeStyle = "rgba(143,211,255,0.8)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x, y, rad + unc, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = t.kind === "Water ice" ? "#8fd3ff" : (t.kind === "Hydrated clay" ? "#4fa59b" : "#cfcfcf");
    ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, rad * 0.55, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (S.highlight === t.id) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, rad + unc + 4, 0, Math.PI * 2); ctx.stroke(); }
    label(t.id, x + rad + 2, y - rad - 2);
  }
  // start / goal
  if (S.start) marker(S.start, "S", "#ffffff");
  if (S.goal) marker(S.goal, "G", "#ffffff");
  // inspected cell
  if (S.inspect) {
    const [x, y] = cellToPx(S.inspect.r, S.inspect.c);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.strokeRect(x - k * 1.5, y - k * 1.5, k * 3, k * 3);
  }
  hud(W, k);
}

function hud(W, k) {
  // scale bar (50 m) bottom-right + north arrow top-right
  const m = 50, px = (m / S.cell) * k, x0 = W - px - 18, y0 = W - 22;
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(x0 - 8, y0 - 20, px + 16, 30);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath();
  ctx.moveTo(x0, y0 - 5); ctx.lineTo(x0, y0); ctx.lineTo(x0 + px, y0); ctx.lineTo(x0 + px, y0 - 5); ctx.stroke();
  ctx.font = "600 11px -apple-system, BlinkMacSystemFont, Inter, sans-serif"; ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillText(`${m} M`, x0 + px / 2, y0 - 5);
  const nx = W - 26, ny = 34;
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.beginPath(); ctx.arc(nx, ny, 16, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.moveTo(nx, ny - 11); ctx.lineTo(nx + 5, ny + 3); ctx.lineTo(nx - 5, ny + 3); ctx.closePath(); ctx.fill();
  ctx.font = "700 9px -apple-system, BlinkMacSystemFont, Inter, sans-serif"; ctx.textBaseline = "top"; ctx.fillText("N", nx, ny + 4);
  const L = LAYERS.find((l) => l.key === S.layer);
  ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "700 13px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
  const tag = `${L ? L.label.toUpperCase() : ""}${S.summary && S.summary.terrain === "jezero" ? " · JEZERO (HIRISE)" : ""}`;
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(10, 10, ctx.measureText(tag).width + 16, 22);
  ctx.fillStyle = "#fff"; ctx.fillText(tag, 18, 15);
}

function drawSim(k) {
  const sim = S.sim, step = S.step;
  // cleared false alarms and revealed hazards up to now
  ctx.fillStyle = "rgba(111,227,176,0.35)";
  for (const ev of sim.cleared) if (ev.step <= step) for (const [r, c] of ev.cells) ctx.fillRect(c * k, r * k, k, k);
  ctx.fillStyle = "rgba(255,77,77,0.85)";
  for (const ev of sim.revealed) if (ev.step <= step) for (const [r, c] of ev.cells) ctx.fillRect(c * k, r * k, k, k);
  // planned paths: earlier ones faint, current bright
  let cur = null;
  for (const p of sim.paths) if (p.step <= step) cur = p;
  for (const p of sim.paths) if (p.step <= step && p !== cur) polyline(p.path, "rgba(255,255,255,0.25)", 1, [3, 3], 1);
  if (cur) polyline(cur.path, "#ffffff", 2, [6, 4], 1);
  // driven trail
  const trail = sim.frames.slice(0, step + 1).map((f) => [f.r, f.c]);
  polyline(trail, "#ffb347", 3.5, [], 1);
  // rover + sensor footprint
  const f = sim.frames[step];
  const [x, y] = cellToPx(f.r, f.c);
  ctx.strokeStyle = "rgba(255,179,71,0.7)"; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
  ctx.beginPath(); ctx.arc(x, y, 6 * k, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  // radar sweep inside the sensing radius
  const R = 6 * k, ang = (S.step * 0.35) % (Math.PI * 2);
  const g = ctx.createRadialGradient(x, y, 0, x, y, R);
  g.addColorStop(0, "rgba(255,179,71,0.05)"); g.addColorStop(1, "rgba(255,179,71,0.22)");
  ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, R, ang - 0.9, ang); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(255,179,71,0.9)"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x + R * Math.cos(ang), y + R * Math.sin(ang)); ctx.stroke();
  // fresh hazard discoveries pulse
  for (const ev of sim.revealed) {
    const age = step - ev.step;
    if (age >= 0 && age < 12) {
      ctx.strokeStyle = `rgba(255,77,77,${1 - age / 12})`; ctx.lineWidth = 2;
      for (const [r, c] of ev.cells.slice(0, 1)) { ctx.beginPath(); ctx.arc((c + 0.5) * k, (r + 0.5) * k, 6 + age * 3, 0, Math.PI * 2); ctx.stroke(); }
    }
  }
  ctx.fillStyle = "#ffb347"; ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  // telemetry readout
  const nRep = sim.paths.filter((p) => p.step <= step).length - 1;
  const lines = [["ROVER", `R${f.r} C${f.c}`], ["ODOMETRY", `${step * S.cell} M`], ["ENERGY", `${fmt(f.e, 1)} WH`],
    ["REPLANS", `${Math.max(nRep, 0)}`], ["STEP", `${step}/${sim.frames.length - 1}`]];
  const W = canvas.width, bw = 170, bh = 18 * lines.length + 16, bx = 10, by = W - bh - 10;
  ctx.fillStyle = "rgba(0,0,0,.72)"; ctx.fillRect(bx, by, bw, bh);
  ctx.strokeStyle = "#3a3a3f"; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, bw, bh);
  ctx.textBaseline = "top"; ctx.font = "600 11px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
  lines.forEach(([a, b], i) => {
    ctx.textAlign = "left"; ctx.fillStyle = "#8a8a92"; ctx.fillText(a, bx + 10, by + 9 + i * 18);
    ctx.textAlign = "right"; ctx.fillStyle = i === 3 && nRep > 0 ? "#ffb347" : "#fff"; ctx.fillText(b, bx + bw - 10, by + 9 + i * 18);
  });
}

function polyline(path, color, w, dash, alpha) {
  if (!path || path.length < 2) return;
  const k = canvas.width / S.n;
  ctx.globalAlpha = alpha; ctx.strokeStyle = "#000"; ctx.lineWidth = w + 2.5; ctx.setLineDash(dash); ctx.lineJoin = "round";
  ctx.beginPath();
  path.forEach(([r, c], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, (c + 0.5) * k, (r + 0.5) * k));
  ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}
function marker(p, txt, color) {
  const [x, y] = cellToPx(p[0], p[1]);
  ctx.fillStyle = "#000"; ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.fillRect(x - 9, y - 9, 18, 18); ctx.strokeRect(x - 9, y - 9, 18, 18);
  ctx.fillStyle = color; ctx.font = "700 12px -apple-system, BlinkMacSystemFont, Inter, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(txt, x, y + 1);
}
function label(txt, x, y) {
  ctx.font = "700 13px -apple-system, BlinkMacSystemFont, Inter, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.lineWidth = 3; ctx.strokeStyle = "#000"; ctx.strokeText(txt, x, y); ctx.fillStyle = "#fff"; ctx.fillText(txt, x, y);
}

// ------------------------------------------------------------------ map interaction
function evtCell(e) {
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor(((e.clientX - rect.left) / rect.width) * S.n);
  const r = Math.floor(((e.clientY - rect.top) / rect.height) * S.n);
  return [Math.max(0, Math.min(S.n - 1, r)), Math.max(0, Math.min(S.n - 1, c))];
}
canvas.addEventListener("mousemove", (e) => {
  const [r, c] = evtCell(e);
  $("hover").textContent = `row ${r} · col ${c} · x ${c * S.cell} m · y ${r * S.cell} m`;
});
canvas.addEventListener("click", async (e) => {
  $("mapHint").classList.add("gone");
  const [r, c] = evtCell(e);
  if (S.clickMode === "start") { S.start = [r, c]; S.sim = null; S.tour = null; await plan(); await loadTargets(); return; }
  if (S.clickMode === "goal") { S.goal = [r, c]; S.sim = null; await plan(); return; }
  S.inspect = await api(`/api/cell?r=${r}&c=${c}`);
  sideTab("inspect");
  renderInspector(); draw();
});

// ------------------------------------------------------------------ panels
function renderKpis() {
  const st = S.summary.stats;
  $("kpis").innerHTML = [
    ["AI accuracy", pct(st.accuracy, 1), `${pct(st.train_fraction, 0)} labels`],
    ["Off-limits", pct(st.lethal_frac, 0), "of the map"],
    ["Missing data", pct(st.dem_missing, 1), "filled"],
    ["3D points", (st.n_points / 1000).toFixed(0) + "k", "rover stereo"],
  ].map(([a, b, c]) => `<div class="kpi"><span>${a}</span><b data-to="${b}">${b}</b><small>${c}</small></div>`).join("");
  $("kpis").querySelectorAll("b").forEach(countUp);
}

// animate a numeric label from 0 to its value (keeps suffix like "%" or "k")
function countUp(el) {
  const m = el.dataset.to.match(/^([\d.]+)(.*)$/);
  if (!m || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const target = parseFloat(m[1]), dec = (m[1].split(".")[1] || "").length, suf = m[2], t0 = performance.now();
  const step = (t) => {
    const k = Math.min((t - t0) / 900, 1), e = 1 - Math.pow(1 - k, 3);
    el.textContent = (target * e).toFixed(dec) + suf;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderLayerbar() {
  const grps = [...new Set(LAYERS.map((l) => l.grp))];
  $("layerbar").innerHTML = grps.map((g) => `<div class="grp"><span class="gl">${g.toUpperCase()}</span>${LAYERS.filter((l) => l.grp === g)
    .map((l) => `<button data-k="${l.key}" class="${l.key === S.layer ? "on" : ""}" title="${l.desc}">${l.label}</button>`).join("")}</div>`).join("");
  $("layerbar").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { S.layer = b.dataset.k; renderLayerbar(); draw(); }));
  const L = LAYERS.find((l) => l.key === S.layer);
  $("layerdesc").textContent = L.desc;
  let html = "";
  if (L.classes) {
    html = `<div class="classes">${S.summary.classes.map((c, i) => `<span><i class="sw" style="background:${S.summary.class_colors[i]}"></i>${c}</span>`).join("")}</div>`;
  } else if (L.cmap) {
    const ends = L.ends === "elev" ? [`${fmt(S.elevRange[0], 1)} m`, `${fmt(S.elevRange[1], 1)} m`] : L.ends;
    html = `<div class="bar" style="background:linear-gradient(90deg,${CMAPS[L.cmap].join(",")})"></div><div class="ends"><span>${ends[0]}</span>${L.lethal ? '<span><i class="sw" style="background:#780a14"></i>lethal</span>' : ""}<span>${ends[1]}</span></div>`;
  } else {
    html = `<div class="classes"><span style="color:#fff">━ Route</span><span style="color:#ff4d4d">┅ No-risk route</span><span style="color:#ffb347">┅ Tour / rover</span><span style="color:#8fd3ff">● Target (ring = uncertainty)</span></div>`;
  }
  $("legend").innerHTML = html;
}

function bars(obj, max = 1, digits = 0) {
  return `<div class="bars">${Object.entries(obj).map(([k, v]) =>
    `<div class="row"><span>${k}</span><div class="track"><div class="fill" style="width:${Math.max(0, Math.min(1, (v ?? 0) / max)) * 100}%"></div></div><span>${digits ? fmt(v, digits) : pct(v)}</span></div>`).join("")}</div>`;
}
function llrBars(obj) {
  return `<div class="bars llr">${Object.entries(obj).map(([k, v]) => {
    const w = Math.min(Math.abs(v ?? 0) / 6, 1) * 50;
    const left = v >= 0 ? 50 : 50 - w;
    return `<div class="row"><span>${k}</span><div class="track"><div class="fill" style="left:${left}%;width:${w}%;background:${v >= 0 ? "#8fd3ff" : "#ff4d4d"}"></div><div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#666"></div></div><span>${v >= 0 ? "+" : ""}${fmt(v, 1)}</span></div>`;
  }).join("")}</div>`;
}

function renderInspector() {
  const d = S.inspect;
  if (!d) return;
  const tags = [];
  if (d.lethal) tags.push('<span class="tag danger">off-limits</span>');
  if (d.dem_missing) tags.push('<span class="tag signal">gap filled</span>');
  if (d.point_cloud) tags.push('<span class="tag ok">rover 3D</span>');
  if (d.training_label) tags.push('<span class="tag">training</span>');
  tags.push(`<span class="tag ${d.predicted === d.truth ? "ok" : "danger"}">truth: ${d.truth}</span>`);
  $("inspector").innerHTML = `
    <div class="kv"><span>Position</span><b>${d.x_m}, ${d.y_m} m</b>
      <span>Height</span><b>${fmt(d.elevation, 2)} ± ${fmt(d.elevation_std, 2)} m</b>
      <span>Slope</span><b>${fmt(d.slope, 1)}° ± ${fmt(d.slope_std, 1)}°</b>
      <span>Roughness</span><b>${fmt(d.roughness, 2)} m</b>
      <span>AI says</span><b>${d.predicted}</b></div>
    <div style="margin:8px 0">${tags.join("")}</div>
    <h4>Terrain type</h4>${bars(d.class_probs)}
    <h4>Uncertainty</h4>${bars({ "Total": d.entropy, "Noise": d.aleatoric, "Knowledge gap": d.epistemic })}
    <h4>Hazard</h4>${bars({ "Steep": d.p_steep, "Rough": d.p_rough, "Boulders": d.p_rock, "Overall": d.p_hazard })}
    <h4>Cost <small>${d.cost === null ? "off-limits" : fmt(d.cost, 2)}</small></h4>
    <div class="kv"><span>Terrain</span><b>${fmt(d.cost_parts.terrain, 2)}</b>
      <span>× Slope</span><b>${fmt(d.cost_parts.slope_x, 2)}</b>
      <span>× Roughness</span><b>${fmt(d.cost_parts.rough_x, 2)}</b>
      <span>+ Risk</span><b>${fmt(d.cost_parts.risk, 2)}</b>
      <span>+ Clearance</span><b>${fmt(d.cost_parts.clearance, 2)}</b></div>
    <h4>Ice evidence</h4>
    ${llrBars({ "Neutron": d.llr.neutron, "Heat": d.llr.thermal, "Camera": d.llr.spectral })}
    <div class="kv"><span>Ice chance</span><b>${pct(d.p_ice)} [${pct(d.p_ice_ci[0])}–${pct(d.p_ice_ci[1])}]</b>
      <span>True water</span><b>${fmt(d.true_weh, 1)} %</b></div>`;
}

function renderTargets() {
  const inTour = new Set(S.tour ? S.tour.order : []);
  const maxV = Math.max(1.6, ...S.targets.map((t) => t.ci90[1]));
  $("targets").innerHTML = S.targets.map((t, i) => {
    const ps = t.path_stats;
    const lo = (t.ci90[0] / maxV) * 100, hi = (t.ci90[1] / maxV) * 100, mu = (t.value / maxV) * 100;
    return `<div class="tgt ${inTour.has(t.id) ? "in-tour" : ""}" data-id="${t.id}">
      <div class="top"><div><span class="id">#${i + 1} ${t.id}</span><span class="kind">${t.kind}</span></div>
        <span class="score">${t.reachable ? fmt(t.score, 2) : "—"}</span></div>
      <div class="ci" title="science value with 90% confidence interval"><div class="rng" style="left:${lo}%;width:${hi - lo}%"></div><div class="mu" style="left:${mu}%"></div></div>
      <div class="sub"><span>Ice ${pct(t.p_ice)}</span>${ps ? `<span>${fmt(ps.length_m, 0)} m</span><span>Safe ${pct(ps.p_success)}</span>` : ""}<span>± ${fmt(t.value_std)}</span></div>
      ${t.buried ? '<span class="tag signal" style="margin-top:6px">hidden ice</span>' : ""}
      <div class="ev"><p style="margin:4px 0 8px">${t.hypothesis}</p>
        ${t.evidence.map((e) => `<div class="kv"><span>${e.src}</span><b>${e.value}${e.llr !== null ? ` · ${e.llr >= 0 ? "+" : ""}${fmt(e.llr, 1)} log-odds` : ""}</b></div>`).join("")}
      </div></div>`;
  }).join("");
  $("tgtCount").textContent = S.targets.length ? S.targets.length : "";
  document.querySelectorAll(".tgt").forEach((el) => {
    el.addEventListener("click", () => el.classList.toggle("open"));
    el.addEventListener("mouseenter", () => { S.highlight = el.dataset.id; draw(); });
    el.addEventListener("mouseleave", () => { S.highlight = null; draw(); });
  });
}

function renderCompare() {
  const sel = $("algo").value;
  const rows = S.plan.comparison.map((c) => {
    const s = c.stats, st = ALGO_STYLE[c.key];
    if (!s) return `<tr><td>${c.label}</td><td colspan="10">${c.pending ? "computing…" : "no path found"}</td></tr>`;
    return `<tr class="${c.key === sel ? "sel" : ""}"><td><i class="swatch" style="background:${st.color}"></i>${c.label}</td>
      <td>${fmt(s.length_m, 0)} m</td><td>${fmt(s.energy_wh, 0)} Wh</td><td>${fmt(s.drive_time_h, 1)} h</td>
      <td><b>${pct(s.p_success, 1)}</b></td><td>${fmt(s.hazard_exposure, 2)}</td><td>${fmt(s.max_slope, 1)}°</td>
      <td>${fmt(s.mean_uncertainty, 2)}</td><td>${fmt(s.min_clearance_m, 0)} m</td><td>${c.expanded}</td><td>${fmt(c.ms, 0)}</td></tr>`;
  }).join("");
  $("compare").innerHTML = `<div class="tablewrap"><table><thead><tr><th>Planner</th><th>Length</th><th>Energy</th><th>Time</th>
    <th>Safe arrival</th><th>Risk Σp</th><th>Max slope</th><th>Uncertainty</th><th>Clearance</th><th>Nodes</th><th>ms</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
  const best = Math.max(...S.plan.comparison.filter((c) => c.stats).map((c) => c.stats.p_success));
  $("routeCards").innerHTML = S.plan.comparison.map((c) => {
    const s = c.stats, col = ALGO_STYLE[c.key].color;
    if (!s) return `<div class="rcard" style="--c:${col}"><div class="nm">${c.label}</div><div class="big">—</div><div class="facts">${c.pending ? "computing…" : "no path"}</div></div>`;
    return `<div class="rcard ${c.key === sel ? "sel" : ""}" data-k="${c.key}" style="--c:${col}">
      ${s.p_success === best ? '<span class="badge">safest</span>' : ""}
      <div class="nm">${c.label}</div>
      <div class="big">${pct(s.p_success, 0)}<small>safe arrival</small></div>
      <div class="meter"><div style="width:${(s.p_success * 100).toFixed(1)}%"></div></div>
      <div class="facts"><span><b>${fmt(s.length_m, 0)}</b> m</span><span><b>${fmt(s.energy_wh, 0)}</b> Wh</span><span><b>${fmt(s.drive_time_h, 1)}</b> h</span><span><b>${fmt(c.ms, 0)}</b> ms</span></div>
    </div>`;
  }).join("");
  document.querySelectorAll(".rcard[data-k]").forEach((el) => el.addEventListener("click", () => {
    $("algo").value = el.dataset.k; $("algo").dispatchEvent(new Event("change"));
  }));
  const ex = S.plan.explain;
  if (!ex) { $("breakdown").innerHTML = ""; return; }
  const cols = { terrain: "#c1693f", slope: "#e0461b", roughness: "#8c2981", uncertainty_risk: "#ffb347", clearance: "#61a5c2" };
  const tot = Object.values(ex.cost_breakdown).reduce((a, b) => a + b, 0);
  $("breakdown").innerHTML = `<h4>Why this route?</h4>
    <div class="stack">${Object.entries(ex.cost_breakdown).map(([k, v]) => `<div title="${k}" style="width:${(v / tot) * 100}%;background:${cols[k]}"></div>`).join("")}</div>
    <div class="stacklegend">${Object.entries(ex.cost_breakdown).map(([k, v]) => `<span><i class="sw" style="background:${cols[k]}"></i>${k.replace("_", " ")} ${pct(v / tot)}</span>`).join("")}
      <span style="margin-left:auto">terrain crossed: ${Object.entries(ex.terrain_mix).map(([k, v]) => `${k} ${v}`).join(" · ")}</span></div>`;
}

function renderLog() {
  $("log").innerHTML = S.sim.log.map((l) => `<div class="e future" data-step="${l.step}"><span class="t">T+${fmt(l.t, 0)} min</span><span class="k k-${l.kind}">${l.kind}</span><span>${l.msg}</span></div>`).join("");
  $("logCount").textContent = S.sim.log.length;
  const checks = S.sim.science_checks || [];
  if (checks.length) {
    const ok = checks.filter((c) => c.correct).length;
    $("logNote").textContent += ` · ice ${ok}/${checks.length} confirmed`;
  }
}

// ------------------------------------------------------------------ model view
function renderModel() {
  const m = S.summary.metrics, st = S.summary.stats;
  $("modelKpis").innerHTML = [
    ["Accuracy", pct(m.accuracy, 1)], ["Top-50 % confident", pct(m.selective.find((s) => s[0] === 0.5)[1], 1)],
    ["Macro F1", fmt(m.macro_f1, 3)], ["Calibration error", fmt(m.ece, 3)],
    ["Training pixels", `${m.n_train_pixels}`],
  ].map(([a, b]) => `<div class="kpi"><span>${a}</span><b>${b}</b></div>`).join("");
  const base = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#f0f0fa", family: "-apple-system, BlinkMacSystemFont, Inter, sans-serif", size: 11 }, margin: { l: 110, r: 10, t: 10, b: 70 } };
  const cfg = { displayModeBar: false, responsive: true };
  const cm = m.confusion.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => v / s); });
  preact("plotCM", [{ z: cm, x: S.summary.classes, y: S.summary.classes, type: "heatmap", colorscale: [[0, "#0a0a0d"], [0.35, "#3a2410"], [1, "#c9681f"]], showscale: false,
    text: m.confusion, texttemplate: "%{z:.0%}", textfont: { color: "#ffffff" } }], { ...base, yaxis: { autorange: "reversed" } }, cfg);
  preact("plotSel", [{ x: m.selective.map((s) => s[0] * 100), y: m.selective.map((s) => s[1] * 100), type: "scatter", mode: "lines+markers", line: { color: "#ffb347", width: 2.5 }, marker: { color: "#fff", size: 7 } }],
    { ...base, margin: { l: 50, r: 10, t: 10, b: 50 }, xaxis: { title: "coverage: % most-confident pixels kept", gridcolor: "#222" }, yaxis: { title: "accuracy %", gridcolor: "#222" } }, cfg);
  preact("plotRel", [
    { x: [0, 1], y: [0, 1], mode: "lines", line: { color: "#555", dash: "dot" }, name: "perfect" },
    { x: m.reliability.map((r) => r[0]), y: m.reliability.map((r) => r[1]), mode: "lines+markers", line: { color: "#ffb347", width: 2.5 }, marker: { color: "#fff", size: 7 }, name: "model" }],
    { ...base, margin: { l: 50, r: 10, t: 10, b: 50 }, showlegend: false, xaxis: { title: "predicted confidence", gridcolor: "#222" }, yaxis: { title: "observed accuracy", gridcolor: "#222" } }, cfg);
  const fi = m.feature_importance.slice().reverse();
  preact("plotFI", [{ x: fi.map((f) => f[1]), y: fi.map((f) => f[0]), type: "bar", orientation: "h", marker: { color: "#ffb347" } }],
    { ...base, margin: { l: 160, r: 10, t: 10, b: 40 }, xaxis: { gridcolor: "#222" } }, cfg);
  $("imgPred").src = S.layers.class;
  $("imgTruth").src = S.layers.truth;
}

function render3d() {
  if (!S.summary) return;
  const z = S.summary.elevation;
  const n2 = z.length, k = S.n / n2;
  const xs = [...Array(n2).keys()].map((i) => i * k * S.cell);
  const data = [{ z, x: xs, y: xs, type: "surface", colorscale: CMAPS.mars.map((c, i, a) => [i / (a.length - 1), c]), showscale: false,
    lighting: { ambient: 0.55, diffuse: 0.8, roughness: 0.9 }, contours: { z: { show: false } } }];
  const zAt = (r, c) => z[Math.min(n2 - 1, Math.floor(r / k))][Math.min(n2 - 1, Math.floor(c / k))] + 1.2;
  const path3 = (p, color, name, w) => ({ type: "scatter3d", mode: "lines", name, x: p.map((q) => q[1] * S.cell), y: p.map((q) => q[0] * S.cell), z: p.map((q) => zAt(q[0], q[1])), line: { color, width: w } });
  if (S.plan) {
    const sel = S.plan.comparison.find((c) => c.key === $("algo").value);
    if (sel && sel.found) data.push(path3(sel.path, "#ffffff", "planned route", 7));
    const naive = S.plan.comparison.find((c) => c.key === "naive");
    if (naive && naive.found) data.push(path3(naive.path, "#ff4d4d", "no-risk baseline", 4));
  }
  if (S.sim) data.push(path3(S.sim.frames.map((f) => [f.r, f.c]), "#ffb347", "driven", 6));
  else if (S.tour && S.tour.path.length) data.push(path3(S.tour.path, "#ffb347", "science tour", 6));
  data.push({ type: "scatter3d", mode: "markers+text", name: "targets", x: S.targets.map((t) => t.c * S.cell), y: S.targets.map((t) => t.r * S.cell),
    z: S.targets.map((t) => zAt(t.r, t.c) + 3), text: S.targets.map((t) => t.id), textfont: { color: "#fff" }, marker: { size: 5, color: "#8fd3ff" } });
  preact("plot3d", data, {
    paper_bgcolor: "#0a0a0a", font: { color: "#f0f0fa", family: "-apple-system, BlinkMacSystemFont, Inter, sans-serif" }, margin: { l: 0, r: 0, t: 0, b: 0 },
    legend: { x: 0.01, y: 0.98 },
    scene: { aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.28 }, xaxis: { title: "x (m)", color: "#888" }, yaxis: { title: "y (m)", color: "#888", autorange: "reversed" }, zaxis: { title: "z (m)", color: "#888" }, bgcolor: "#0a0a0a",
      camera: { eye: { x: 1.2, y: 1.4, z: 0.9 } } },
  }, { displayModeBar: false, responsive: true });
}

// ------------------------------------------------------------------ trade-off, benchmark, demo
const PBASE = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#f0f0fa", family: "-apple-system, BlinkMacSystemFont, Inter, sans-serif", size: 11 } };
const PCFG = { displayModeBar: false, responsive: true };

async function loadBench() {
  const d = await api("/api/benchmark");
  if (!d.available) { $("benchNote").textContent = "not generated"; return; }
  S.bench = d;
  const T = d.terrains, names = { procedural: "Simulated", jezero: "Real Mars" };
  const sm = (t) => T[t].summary;
  const ks = Object.keys(T);
  $("benchNote").textContent = `${d.seeds.length * ks.length} worlds`;
  const kp = [];
  for (const t of ks) {
    const s = sm(t);
    kp.push([`${names[t]} · accuracy`, `${pct(s.accuracy.mean, 1)}`, `± ${pct(s.accuracy.std, 1)}`]);
    kp.push([`${names[t]} · safe arrival`, `${pct(s.p_safe_astar.mean, 0)} / ${pct(s.p_safe_naive.mean, 0)}`, "A* / no-risk"]);
    kp.push([`${names[t]} · ice confirmed`, pct(s.ice_agreement, 0), `${s.ice_checks_total} sites`]);
    kp.push([`${names[t]} · hazards hit`, `${fmt(s.hazard_cells_entered.max, 0)}`, `${s.safe_traverses}/${s.n} clean`]);
  }
  $("benchKpis").innerHTML = kp.map(([a, b, c]) => `<div class="kpi"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join("");
  const col = { procedural: "#ffffff", jezero: "#ffb347" };
  preact("plotBenchAcc", ks.map((t) => ({ type: "box", y: T[t].rows.map((r) => r.accuracy * 100), name: names[t], boxpoints: "all", jitter: 0.4,
    pointpos: 0, marker: { color: col[t], size: 6 }, line: { color: col[t] }, fillcolor: "rgba(255,255,255,0.04)" })),
    { ...PBASE, margin: { l: 50, r: 10, t: 10, b: 40 }, showlegend: false, yaxis: { title: "accuracy %", gridcolor: "#222" } }, PCFG);
  const tr = [];
  for (const t of ks) {
    const rows = T[t].rows;
    tr.push({ x: rows.map((r) => r.p_safe_naive * 100), y: rows.map((r) => r.p_safe_astar * 100), mode: "markers", name: names[t],
      marker: { color: col[t], size: 9, line: { color: "#000", width: 1 } }, text: rows.map((r) => `seed ${r.seed}`),
      hovertemplate: "%{text}<br>baseline %{x:.0f}%<br>A* %{y:.0f}%<extra></extra>" });
  }
  tr.push({ x: [0, 100], y: [0, 100], mode: "lines", line: { color: "#555", dash: "dot" }, name: "equal", hoverinfo: "skip" });
  preact("plotBenchSafe", tr, { ...PBASE, margin: { l: 55, r: 10, t: 10, b: 50 }, legend: { x: 0.62, y: 0.2 },
    xaxis: { title: "no-risk route · safe %", range: [0, 100], gridcolor: "#1c1c20" },
    yaxis: { title: "A* · safe %", range: [0, 100], gridcolor: "#1c1c20" },
    annotations: [{ x: 72, y: 34, text: "above line = A* safer", showarrow: false, font: { color: "#8a8a92" } }] }, PCFG);
  const rows = ks.flatMap((t) => T[t].rows.map((r) => `<tr><td>${names[t]}</td><td>${r.seed}</td><td>${pct(r.accuracy, 1)}</td><td>${pct(r.selective50, 1)}</td>
    <td><b>${pct(r.p_safe_astar, 0)}</b></td><td>${pct(r.p_safe_naive, 0)}</td><td>${fmt(r.len_astar, 0)} / ${fmt(r.len_naive, 0)} m</td>
    <td>${r.targets_visited}</td><td>${r.decisions}</td><td>${r.hazards_discovered}</td>
    <td class="${r.hazard_cells_entered ? "bad" : "good"}">${r.hazard_cells_entered}</td><td>${r.ice_correct}/${r.ice_checks}</td></tr>`)).join("");
  $("benchTable").innerHTML = `<table><thead><tr><th>Terrain</th><th>Seed</th><th>Accuracy</th><th>Top-50 %</th><th>Safe A*</th><th>Safe no-risk</th>
    <th>Length</th><th>Targets</th><th>Decisions</th><th>Hazards seen</th><th>Hazards hit</th><th>Ice</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function demo() {
  $("btnDemo").disabled = true;
  try {
    await build();
    toast("Map ready. Planning…");
    await tour();
    toast(`Tour ${S.tour.order.join(" → ") || "—"}. Driving…`);
    const done = new Promise((res) => (S.demoResolve = res));
    await drive();
    if (!S.sim && S.demoResolve) S.demoResolve();
    await done;
  } finally { $("btnDemo").disabled = false; }
}

function setTerrain(t) {
  S.terrain = t;
  document.querySelectorAll("#terrainSeg button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  $("terrainNote").textContent = t === "jezero"
    ? "Real HiRISE heights · Jezero crater"
    : "Generated world";
}

// ------------------------------------------------------------------ side tabs + mission flow
function sideTab(name) {
  document.querySelectorAll(".sidetabs button").forEach((b) => b.classList.toggle("on", b.dataset.st === name));
  document.querySelectorAll(".stab").forEach((t) => t.classList.toggle("on", t.id === "st-" + name));
}
document.querySelectorAll(".sidetabs button").forEach((b) => b.addEventListener("click", () => sideTab(b.dataset.st)));

function flowUpdate() {
  const st = S.summary ? S.summary.stats : null;
  const a = S.plan && S.plan.comparison.find((c) => c.key === $("algo").value);
  const done = { perceive: !!st, plan: !!(a && a.stats), science: !!(S.tour && S.tour.order.length), drive: !!S.done };
  $("fs-perceive").textContent = st ? `AI accuracy ${pct(st.accuracy, 1)}` : "building map…";
  $("fs-plan").textContent = a && a.stats ? `${pct(a.stats.p_success, 0)} safe · ${fmt(a.stats.length_m, 0)} m` : "—";
  $("fs-science").textContent = S.tour ? (S.tour.order.length ? `${S.tour.order.length} targets · ${fmt(S.tour.energy_wh, 0)} Wh` : "none in budget") : "plan a tour";
  $("fs-drive").textContent = S.sim ? (S.done ? `${S.sim.decisions} decisions · done` : "driving…") : "not started";
  const order = ["perceive", "plan", "science", "drive"];
  const next = order.find((k) => !done[k]);
  document.querySelectorAll(".fstep").forEach((el) => {
    el.classList.toggle("done", done[el.dataset.f]);
    el.classList.toggle("active", el.dataset.f === next);
  });
  const label = { perceive: "Generate", plan: "Plan route", science: "Next: plan tour", drive: "Next: drive" };
  const driving = S.sim && !S.done;
  $("btnNext").textContent = driving ? "Driving…" : next ? label[next] + " →" : "Run again ↻";
  $("btnNext").dataset.next = driving ? "wait" : next || "again";
}
$("btnNext").addEventListener("click", () => {
  const n = $("btnNext").dataset.next;
  if (n === "perceive") build();
  else if (n === "plan") plan();
  else if (n === "science") { S.sim = null; tour(); }
  else if (n === "drive") drive();
  else if (n === "again") demo();
});

// ------------------------------------------------------------------ wiring
document.querySelectorAll("#terrainSeg button").forEach((b) => b.addEventListener("click", () => { setTerrain(b.dataset.t); build(); }));
$("btnDemo").addEventListener("click", demo);
$("btnLaunch").addEventListener("click", () => { $("console").scrollIntoView({ behavior: "smooth" }); demo(); });
// nav turns solid once the hero scrolls away
const onScroll = () => $("nav").classList.toggle("solid", scrollY > 40);
addEventListener("scroll", onScroll, { passive: true }); onScroll();
// panels fade up as they enter the viewport
const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => es.forEach((e) => {
  if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}), { threshold: 0.08 }) : null;
document.querySelectorAll(".panel, .bandhead").forEach((el) => { if (io) { el.classList.add("reveal"); io.observe(el); } });
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  const i = "1234567890".indexOf(e.key);
  if (i >= 0 && LAYERS[i] && S.summary) { S.layer = LAYERS[i].key; renderLayerbar(); draw(); }
  if (e.key === " " && S.sim) { e.preventDefault(); $("btnPlay").click(); }
});
const BANDS = {
  map: ["01 · Map", "Perceive. Plan. Drive.", "Click anywhere on the map to see why."],
  terrain3d: ["02 · 3D", "Terrain in 3D.", "Route and targets on the surface."],
  model: ["03 · AI model", "Measured, not claimed.", "Tested on pixels it never saw, across 24 worlds."],
  about: ["04 · How it works", "Sense to act.", "Eight steps from raw data to a decision."],
};
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  const b = BANDS[t.dataset.view];
  $("bandEyebrow").textContent = b[0]; $("bandTitle").textContent = b[1]; $("bandLead").textContent = b[2];
  if ($("hero").getBoundingClientRect().bottom > 80) $("console").scrollIntoView({ behavior: "smooth" });
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + t.dataset.view));
  if (t.dataset.view === "terrain3d") render3d();
  if (t.dataset.view === "model") window.dispatchEvent(new Event("resize"));
}));
document.querySelectorAll("#clickMode button").forEach((b) => b.addEventListener("click", () => {
  S.clickMode = b.dataset.mode;
  document.querySelectorAll("#clickMode button").forEach((x) => x.classList.toggle("on", x === b));
}));
// premium sliders: paint the filled part of every range track (CSS reads --p)
function fillRange(el) {
  const min = +el.min || 0, max = +el.max || 100;
  el.style.setProperty("--p", `${((+el.value - min) / (max - min || 1)) * 100}%`);
}
document.querySelectorAll('input[type="range"]').forEach((el) => { el.addEventListener("input", () => fillRange(el)); fillRange(el); });
const bindOut = (id, out, f) => { const u = () => ($(out).textContent = f(+$(id).value)); $(id).addEventListener("input", u); u(); };
bindOut("demNoise", "demNoiseV", (v) => v.toFixed(2) + " m");
bindOut("dropout", "dropoutV", (v) => Math.round(v * 100) + " %");
bindOut("spec", "specV", (v) => v.toFixed(3));
bindOut("lam", "lamV", (v) => v.toFixed(2));
$("lam").addEventListener("change", setRisk);
$("algo").addEventListener("change", () => { renderCompare(); S.plan.explain = null; plan(); });
$("mode").addEventListener("change", async () => { S.tour = null; await loadTargets(); });
$("showAlts").addEventListener("change", draw);
$("showTree").addEventListener("change", draw);
$("btnBuild").addEventListener("click", build);
$("btnPlan").addEventListener("click", () => { S.sim = null; plan(); });
$("btnTour").addEventListener("click", () => { S.sim = null; tour(); });
$("btnDrive").addEventListener("click", drive);
$("btnPlay").addEventListener("click", () => {
  if (!S.sim) return;
  if (S.step >= S.sim.frames.length - 1) { S.step = 0; }
  S.playing = !S.playing; $("btnPlay").textContent = S.playing ? "Pause" : "Play";
  if (S.playing) play();
});
$("scrub").addEventListener("input", () => { if (!S.sim) return; S.playing = false; $("btnPlay").textContent = "Play"; S.step = +$("scrub").value; onStep(); });

// world first; benchmark charts (and the Plotly library) load after it, so they never delay the map
build().then(() => loadBench()).catch((e) => console.error(e));
