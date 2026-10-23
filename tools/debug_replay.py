# ruff: noqa: E501
"""Record one flight and write an interactive HTML replay.

The replay shows the planned waypoints, the gate approach/center/exit points,
the pole positions, the actual drone path, and a time slider. Use it to inspect
exactly where the drone deviates from the harmonic plan.

    pixi run -e tests python tools/debug_replay.py \
        --config level0.toml --controller harmonic_controller.py --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import json
from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config, load_controller

REPO_ROOT = Path(__file__).resolve().parents[1]
HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { background:#11151c; color:#d7dde8; font:14px/1.4 system-ui, sans-serif; margin:18px; }
  h2 { margin:0 0 4px; font-weight:600; }
  .muted { color:#8b97a8; }
  canvas { background:#18202b; border:1px solid #2b3646; border-radius:6px; display:block; }
  .row { display:flex; gap:14px; align-items:center; margin:10px 0; flex-wrap:wrap; }
  input[type=range] { width:520px; accent-color:#4ea1ff; }
  button { background:#26364a; color:#e6edf7; border:1px solid #3c5068; border-radius:5px; padding:6px 14px; cursor:pointer; }
  button:hover { background:#31465f; }
  .pill { background:#1d2836; border:1px solid #2c3b4e; border-radius:999px; padding:3px 10px; }
</style>
</head>
<body>
<h2>__TITLE__</h2>
<div class="muted">Dense plan (faint), sparse waypoints (white), active goal (red cross), actual flight, and sensor range.</div>
<div class="row"><canvas id="top" width="980" height="540"></canvas></div>
<div class="row"><canvas id="alt" width="980" height="170"></canvas></div>
<div class="row">
  <button id="play">Play</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1">
  <span id="label" class="pill"></span>
  <span id="status" class="muted"></span>
</div>
<script>
const D = __DATA__;
const topCanvas = document.getElementById('top'), tctx = topCanvas.getContext('2d');
const alt = document.getElementById('alt'), actx = alt.getContext('2d');
const slider = document.getElementById('slider'), label = document.getElementById('label');
const statusEl = document.getElementById('status'), play = document.getElementById('play');
const W = topCanvas.width, H = topCanvas.height, PAD = 34;
const [xmin, ymin, xmax, ymax] = D.bounds;
const scale = Math.min((W - 2*PAD) / (xmax - xmin), (H - 2*PAD) / (ymax - ymin));
const cx = PAD + (W - 2*PAD - (xmax - xmin)*scale)/2;
const cy = PAD + (H - 2*PAD - (ymax - ymin)*scale)/2;
function px(x) { return cx + (x - xmin) * scale; }
function py(y) { return cy + (ymax - y) * scale; }

function rot(q, v) {
  const [x,y,z,w] = q;
  const ix =  w*v[0] + y*v[2] - z*v[1];
  const iy =  w*v[1] + z*v[0] - x*v[2];
  const iz =  w*v[2] + x*v[1] - y*v[0];
  const iw = -x*v[0] - y*v[1] - z*v[2];
  return [ix*w + iw*-x + iy*-z - iz*-y,
          iy*w + iw*-y + iz*-x - ix*-z,
          iz*w + iw*-z + ix*-y - iy*-x];
}
function yawOf(q) {
  const [x,y,z,w] = q;
  return Math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z));
}
function line(a, b, color, width, dash=[]) {
  tctx.save(); tctx.strokeStyle = color; tctx.lineWidth = width; tctx.setLineDash(dash);
  tctx.beginPath(); tctx.moveTo(px(a[0]), py(a[1])); tctx.lineTo(px(b[0]), py(b[1])); tctx.stroke(); tctx.restore();
}
function dot(p, color, r) {
  tctx.fillStyle = color; tctx.beginPath(); tctx.arc(px(p[0]), py(p[1]), r, 0, 2*Math.PI); tctx.fill();
}
function drawGates() {
  for (const g of D.gates) {
    const yaxis = rot(g.quat, [0,1,0]);
    const xaxis = rot(g.quat, [1,0,0]);
    const c = g.center;
    const outerA = [c[0]-0.36*yaxis[0], c[1]-0.36*yaxis[1]];
    const outerB = [c[0]+0.36*yaxis[0], c[1]+0.36*yaxis[1]];
    const innerA = [c[0]-0.20*yaxis[0], c[1]-0.20*yaxis[1]];
    const innerB = [c[0]+0.20*yaxis[0], c[1]+0.20*yaxis[1]];
    line(outerA, outerB, '#e8a33d', 5);
    line(innerA, innerB, '#18202b', 3);
    line(c, [c[0]+0.25*xaxis[0], c[1]+0.25*xaxis[1]], '#e8a33d', 1.5, [4,4]);
    tctx.fillStyle = '#e8a33d'; tctx.font = '11px system-ui';
    tctx.fillText('g' + g.gate_id, px(c[0]) + 6, py(c[1]) - 8);
    for (const p of [g.approach, g.center, g.exit]) dot(p, '#e8a33d', 3);
  }
}
function drawPoles() {
  for (const p of D.poles) {
    tctx.fillStyle = '#a06bd6'; tctx.beginPath();
    tctx.arc(px(p[0]), py(p[1]), 4, 0, 2*Math.PI); tctx.fill();
    tctx.fillStyle = '#8b6aa8'; tctx.font = '10px system-ui';
    tctx.fillText('pole', px(p[0]) + 6, py(p[1]) + 12);
  }
}


function drawGoals() {
  if (!D.goals) return;
  for (let k = 0; k < D.goals.length; k++) {
    dot(D.goals[k], '#ffffff', 4.5);
    tctx.fillStyle = '#ffffff'; tctx.font = '11px system-ui';
    tctx.fillText(String(k), px(D.goals[k][0]) + 7, py(D.goals[k][1]) - 7);
  }
}function drawTop(i) {
  tctx.clearRect(0, 0, W, H);
  tctx.fillStyle = '#18202b'; tctx.fillRect(0, 0, W, H);
  tctx.strokeStyle = '#222d3a'; tctx.lineWidth = 1;
  for (let x = Math.ceil(xmin); x <= xmax; x++) line([x,ymin],[x,ymax],'#222d3a',1);
  for (let y = Math.ceil(ymin); y <= ymax; y++) line([xmin,y],[xmax,y],'#222d3a',1);

  tctx.strokeStyle = '#405268'; tctx.lineWidth = 1.4;
  tctx.beginPath();
  D.route.forEach((p, k) => { if (k === 0) tctx.moveTo(px(p[0]), py(p[1])); else tctx.lineTo(px(p[0]), py(p[1])); });
  tctx.stroke();

  for (const p of D.route) dot(p, '#5a6c82', 1.0);
  drawGates(); drawPoles(); drawGoals();

  tctx.strokeStyle = '#53d18a'; tctx.lineWidth = 1.5;
  tctx.beginPath();
  for (let k = 0; k <= i; k++) {
    const p = D.pos[k];
    if (k === 0) tctx.moveTo(px(p[0]), py(p[1])); else tctx.lineTo(px(p[0]), py(p[1]));
  }
  tctx.stroke();

  const p = D.pos[i], d = D.des[i];
  // sensor range: the actual gate pose is revealed when the drone is inside this circle
  tctx.save();
  tctx.fillStyle = 'rgba(80, 200, 255, 0.10)';
  tctx.strokeStyle = 'rgba(80, 200, 255, 0.45)';
  tctx.setLineDash([4, 4]);
  tctx.beginPath();
  tctx.arc(px(p[0]), py(p[1]), D.sensor_range * scale, 0, 2 * Math.PI);
  tctx.fill();
  tctx.stroke();
  tctx.restore();
  dot(d, '#ff5b6e', 4);
  tctx.strokeStyle = '#ff5b6e'; tctx.lineWidth = 1;
  tctx.beginPath(); tctx.moveTo(px(d[0])-5, py(d[1])); tctx.lineTo(px(d[0])+5, py(d[1]));
  tctx.moveTo(px(d[0]), py(d[1])-5); tctx.lineTo(px(d[0]), py(d[1])+5); tctx.stroke();

  const yaw = yawOf(D.quat[i]);
  tctx.save(); tctx.translate(px(p[0]), py(p[1])); tctx.rotate(-yaw);
  tctx.fillStyle = '#ffd166';
  tctx.beginPath(); tctx.moveTo(9,0); tctx.lineTo(-6,5); tctx.lineTo(-6,-5); tctx.closePath(); tctx.fill();
  tctx.restore();

  tctx.fillStyle = '#8b97a8'; tctx.font = '13px system-ui';
  tctx.fillText('dense plan', 12, 20);
  tctx.fillStyle = '#6f8fb5'; tctx.fillText('• waypoint', 12, 38);
  tctx.fillStyle = '#e8a33d'; tctx.fillText('gate A / C / E', 120, 38);
  tctx.fillStyle = '#53d18a'; tctx.fillText('flown path', 250, 38);
  tctx.fillStyle = '#ff5b6e'; tctx.fillText('active goal', 350, 38);
  tctx.fillStyle = '#a06bd6'; tctx.fillText('pole', 425, 38);
  tctx.fillStyle = '#ffffff'; tctx.fillText('sparse waypoint', 610, 38);
  tctx.fillStyle = '#50c8ff'; tctx.fillText('sensor ' + D.sensor_range.toFixed(1) + ' m', 500, 38);
}

function drawAlt(i) {
  const w = alt.width, h = alt.height, pad = 30;
  actx.clearRect(0,0,w,h); actx.fillStyle = '#18202b'; actx.fillRect(0,0,w,h);
  const tmax = D.t[D.t.length-1] || 1;
  const zmin = Math.min(0, ...D.pos.map(p=>p[2])) - 0.05;
  const zmax = Math.max(...D.pos.map(p=>p[2]), ...D.route.map(p=>p[2])) + 0.1;
  const sx = (t) => pad + (w-2*pad) * t / tmax;
  const sy = (z) => h-pad - (h-2*pad) * (z-zmin) / (zmax-zmin);
  actx.strokeStyle = '#222d3a'; actx.lineWidth = 1;
  for (let k=0;k<=4;k++) { const z=zmin+(zmax-zmin)*k/4; actx.beginPath(); actx.moveTo(pad, sy(z)); actx.lineTo(w-pad, sy(z)); actx.stroke(); }
  actx.strokeStyle = '#4ea1ff'; actx.lineWidth = 1;
  actx.beginPath(); D.route.forEach((p,k)=>{ if(k===0) actx.moveTo(sx(D.t[0]),sy(p[2])); else actx.lineTo(sx(D.t[0] + (D.t[D.t.length-1]-D.t[0])*k/(D.route.length-1)), sy(p[2])); }); actx.stroke();
  actx.strokeStyle = '#53d18a'; actx.lineWidth = 1.5; actx.beginPath();
  D.pos.forEach((p,k)=>{ if(k===0) actx.moveTo(sx(D.t[k]),sy(p[2])); else actx.lineTo(sx(D.t[k]),sy(p[2])); }); actx.stroke();
  actx.strokeStyle = '#ff5b6e'; actx.lineWidth = 1; actx.beginPath();
  D.des.forEach((p,k)=>{ if(k===0) actx.moveTo(sx(D.t[k]),sy(p[2])); else actx.lineTo(sx(D.t[k]),sy(p[2])); }); actx.stroke();
  const t = D.t[i];
  actx.strokeStyle = '#ffd166'; actx.setLineDash([4,4]); actx.beginPath(); actx.moveTo(sx(t),pad); actx.lineTo(sx(t),h-pad); actx.stroke(); actx.setLineDash([]);
  actx.fillStyle = '#d7dde8'; actx.font='12px system-ui';
  actx.fillText('altitude [m] vs time [s]', pad, 16);
  actx.fillStyle='#8b97a8'; actx.fillText('z', 8, h/2);
}

function update() {
  const i = Number(slider.value);
  drawTop(i); drawAlt(i);
  const p = D.pos[i];
  label.textContent = `t=${D.t[i].toFixed(2)} s  frame ${i+1}/${D.pos.length}  p=(${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)})`;
  statusEl.textContent = `passes ${D.n[i]}/${D.n_required}  speed ${D.speed[i].toFixed(2)} m/s  tracking ${D.error[i].toFixed(3)} m`;
}

slider.max = String(D.pos.length - 1);
slider.addEventListener('input', update);
let timer = null;
play.addEventListener('click', () => {
  if (timer) { clearInterval(timer); timer = null; play.textContent='Play'; return; }
  play.textContent='Pause';
  timer = setInterval(() => {
    let i = Number(slider.value);
    if (i >= D.pos.length - 1) { clearInterval(timer); timer = null; play.textContent='Play'; return; }
    slider.value = String(i + 1); update();
  }, 40);
});
update();
</script>
</body>
</html>
"""


def _bounds(config: dict) -> list[float]:
    limits = config.env.track.safety_limits
    low = list(limits.pos_limit_low)
    high = list(limits.pos_limit_high)
    return [float(low[0]), float(low[1]), float(high[0]), float(high[1])]


def _gates_from_obs(obs: dict) -> list[dict]:
    """Build gate visualization data when the controller has no Plan."""
    gates = []
    for i, (center, quat) in enumerate(zip(obs["gates_pos"], obs["gates_quat"])):
        center = np.asarray(center, dtype=float)
        quat = np.asarray(quat, dtype=float)
        normal = R.from_quat(quat).apply([1.0, 0.0, 0.0])
        gates.append(
            {
                "gate_id": int(i),
                "center": center.tolist(),
                "quat": quat.tolist(),
                "approach": (center - 0.25 * normal).tolist(),
                "exit": (center + 0.25 * normal).tolist(),
            }
        )
    return gates


def main(
    config: str = "level0.toml",
    controller: str = "harmonic_controller.py",
    seed: int = 0,
    out: str | None = None,
    device: str = "cpu",
) -> None:
    """Run one episode and write an interactive replay HTML."""
    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = False
    ctrl_cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / controller)
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
        device=device,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset(seed=seed)
    ctrl = ctrl_cls(obs, info, cfg)
    plan = getattr(ctrl, "_plan", None)
    if plan is not None:
        tsrs = plan.tsrs
        route = np.asarray(plan.points, dtype=float)
        gates_data = [
            {
                "gate_id": int(t.gate_id),
                "center": np.asarray(t.center, dtype=float).tolist(),
                "quat": np.asarray(t.quat, dtype=float).tolist(),
                "approach": np.asarray(t.approach, dtype=float).tolist(),
                "exit": np.asarray(t.exit, dtype=float).tolist(),
            }
            for t in tsrs
        ]
    else:
        tsrs = []
        path = getattr(ctrl, "path", None)
        if path is not None:
            ref_times = np.linspace(0.0, path.T, 120)
            route = np.asarray([path.sample(t).pos for t in ref_times], dtype=float)
        else:
            route = np.asarray([obs["pos"]], dtype=float)
        gates_data = _gates_from_obs(obs)
    goals = [np.asarray(g.position, dtype=float) for g in getattr(ctrl, "_goals", [])]
    if not goals:
        path = getattr(ctrl, "path", None)
        if path is not None:
            goals = [np.asarray(path.goal, dtype=float)]
    n_required = len(obs["gate_sequence"])

    times: list[float] = []
    pos: list[np.ndarray] = []
    des: list[np.ndarray] = []
    vel: list[np.ndarray] = []
    quat: list[np.ndarray] = []
    passed: list[int] = []
    speed: list[float] = []
    error: list[float] = []

    while True:
        p = np.asarray(obs["pos"], dtype=float)
        v = np.asarray(obs["vel"], dtype=float)
        action = np.asarray(ctrl.compute_control(obs, info), dtype=float)
        times.append(len(times) / cfg.env.freq)
        pos.append(p)
        des.append(action[:3].copy())
        vel.append(v)
        quat.append(np.asarray(obs["quat"], dtype=float))
        passed.append(int(obs["n_gates_passed"]))
        speed.append(float(np.linalg.norm(v)))
        error.append(float(np.linalg.norm(action[:3] - p)))
        obs, reward, terminated, truncated, info = env.step(action)
        finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or finished:
            break
    ctrl.episode_callback()
    env.close()

    data = {
        "title": f"{config}  {controller}  seed {seed}",
        "route": np.asarray(route, dtype=float).tolist(),
        "gates": gates_data,
        "poles": np.asarray(obs["obstacles_pos"], dtype=float).tolist(),
        "goals": [g.tolist() for g in goals],
        "t": [round(t, 4) for t in times],
        "pos": [p.tolist() for p in pos],
        "des": [p.tolist() for p in des],
        "vel": [v.tolist() for v in vel],
        "quat": [q.tolist() for q in quat],
        "n": passed,
        "speed": [round(s, 4) for s in speed],
        "error": [round(e, 4) for e in error],
        "n_required": int(n_required),
        "sensor_range": float(cfg.env.sensor_range),
        "bounds": _bounds(cfg),
    }
    out_path = Path(out) if out else REPO_ROOT / "replays" / f"{Path(config).stem}_{seed}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__TITLE__", data["title"]).replace("__DATA__", json.dumps(data))
    out_path.write_text(html)
    print(f"replay -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
