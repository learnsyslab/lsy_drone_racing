"""Headless debugging harness for drone racing controllers.

This script runs a controller for one or more episodes, records the flight, classifies *why* an
episode ended, and plots the trajectory. It deliberately does not render anything, so it works over
SSH and on machines without a display.

Run as:

    $ python scripts/debug_run.py --config level0.toml --controller state_controller.py

inside the repository's pixi environment, or from outside of it with

    $ pixi run python scripts/debug_run.py --config level0.toml --controller my_controller.py

For GPU execution use the ``gpu`` environment and pass ``--device gpu`` explicitly. The environment
defaults to CPU and the TOML config cannot change that, so the ``gpu`` environment alone would still
run on CPU:

    $ pixi run -e gpu python scripts/debug_run.py --config level0.toml --device gpu

Output:

* A one-line verdict per episode printed to stdout.
* ``plots/<config>_<controller>_seed<seed>.png`` with three panels: top-down track, altitude
  and gate progress over time, and speed over time.

Note:
    When an episode terminates because the drone was disabled, the environment warps the drone
    below the ground (``pos = [-1, -1, -1]``) *before* reporting the observation. This script
    therefore records the state at the *start* of every step, so the last recorded pose is the last
    live pose of the drone, not the warped one.
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import logging
from pathlib import Path
from typing import Any

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_OPENING = 0.4  # m, outer frame is 0.72 m
GATE_OUTER = 0.72
GATE_PASS_BOX = 0.45  # m, the box used by the environment's pass check


def gate_geometry(config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal gate positions and xyzw quaternions from a config."""
    gates = config.env.track.gates
    pos = np.array([g["pos"] for g in gates], dtype=float)
    quat = R.from_euler("xyz", np.array([g["rpy"] for g in gates])).as_quat()
    return pos, quat


def crossing_margin(
    prev_pos: np.ndarray, pos: np.ndarray, gate_pos: np.ndarray, gate_quat: np.ndarray
) -> dict[str, float]:
    """Reproduce the environment's gate test for one step and report the margin.

    The environment crosses gates along the gate frame's x-axis and accepts the pass if the
    interpolated crossing point lies inside a ``GATE_PASS_BOX`` sized square in the gate's y/z
    plane. This returns the local crossing coordinates and the remaining margin in metres.
    """
    rot = R.from_quat(gate_quat)
    last_local = rot.apply(prev_pos - gate_pos, inverse=True)
    local = rot.apply(pos - gate_pos, inverse=True)
    denom = local[0] - last_local[0]
    alpha = -last_local[0] / denom if abs(denom) > 1e-12 else 0.0
    y_int = alpha * local[1] + (1 - alpha) * last_local[1]
    z_int = alpha * local[2] + (1 - alpha) * last_local[2]
    half = GATE_PASS_BOX / 2
    return {
        "y": float(y_int),
        "z": float(z_int),
        "margin": float(half - max(abs(y_int), abs(z_int))),
    }


def classify(
    n_passed: int,
    n_required: int,
    truncated: bool,
    last_pos: np.ndarray,
    last_vel: np.ndarray,
    gates_pos: np.ndarray,
    obstacles_pos: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
) -> tuple[str, str]:
    """Guess why an episode ended and explain the evidence.

    Args:
        n_passed: Number of gate-order entries passed at the end of the episode.
        n_required: Total number of entries that must be passed.
        truncated: Whether the episode hit the step limit.
        last_pos: Last live drone position (before any warping).
        last_vel: Last live drone velocity.
        gates_pos: Nominal gate positions.
        obstacles_pos: Nominal obstacle positions.
        bounds_low: Lower position bound at which the environment disables the drone.
        bounds_high: Upper position bound at which the environment disables the drone.

    Returns:
        A ``(verdict, detail)`` tuple.
    """
    if n_passed >= n_required:
        return "FINISHED", f"all {n_required} gate passes completed"
    if truncated:
        return "TIMEOUT", f"hit the step limit with {n_passed}/{n_required} passes"
    outside = last_pos < bounds_low
    outside |= last_pos > bounds_high
    if np.any(outside):
        axes = np.array(["x", "y", "z"])[outside]
        return "OUT_OF_BOUNDS", f"last live pos {np.round(last_pos, 2)} left bounds on {axes}"
    d_obs = np.linalg.norm(obstacles_pos - last_pos, axis=1)
    d_gate = np.linalg.norm(gates_pos - last_pos, axis=1)
    if d_obs.min() < 0.15:
        return "HIT_OBSTACLE", f"closest obstacle {d_obs.argmin()} at {d_obs.min():.3f} m"
    if d_gate.min() < 0.45:
        return "HIT_GATE", f"closest gate {d_gate.argmin()} centre at {d_gate.min():.3f} m"
    predicted = last_pos + last_vel / 50
    return (
        "CONTACT",
        f"pos {np.round(last_pos, 2)} vel {np.round(last_vel, 2)} -> {np.round(predicted, 2)}",
    )


def run_episode(
    config: Any, controller_cls: type, seed: int, verbose: bool = False, device: str = "cpu"
) -> dict[str, Any]:
    """Run a single episode and record it.

    Args:
        config: Race configuration.
        controller_cls: Controller class to instantiate.
        seed: Random seed for the episode.
        verbose: Print per-gate-pass margins.
        device: ``"cpu"`` or ``"gpu"``. Note that the environment defaults to ``"cpu"``; the
            ``device`` cannot be selected from the TOML config.
    """
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=seed,
        device=device,
    )
    env = JaxToNumpy(env)

    obs, info = env.reset(seed=seed)
    controller = controller_cls(obs, info, config)
    gates_pos, gates_quat = gate_geometry(config)
    obstacles_pos = np.array([o["pos"] for o in config.env.track.obstacles], dtype=float)
    gate_sequence = np.abs(np.asarray(config.env.track.gate_order)) - 1
    bounds_low = np.asarray(env.unwrapped.data.pos_limit_low, dtype=float)
    bounds_high = np.asarray(env.unwrapped.data.pos_limit_high, dtype=float)

    rec: dict[str, list] = {"t": [], "pos": [], "vel": [], "des": [], "n": [], "speed": []}
    n_steps = 0
    terminated = truncated = stopped = False
    try:
        while True:
            pos = np.asarray(obs["pos"], dtype=float)
            vel = np.asarray(obs["vel"], dtype=float)
            rec["t"].append(n_steps / config.env.freq)
            rec["pos"].append(pos)
            rec["vel"].append(vel)
            rec["n"].append(int(obs["n_gates_passed"]))
            rec["speed"].append(float(np.linalg.norm(vel)))

            action = np.asarray(controller.compute_control(obs, info), dtype=float)
            rec["des"].append(action[:3].copy())
            obs, reward, terminated, truncated, info = env.step(action)
            finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            n_steps += 1
            # Mirror sim.py: a truthy step_callback return ends the episode. Note that the base
            # class default returns True, so a controller that forgets to override it stops after
            # the first step.
            if terminated or truncated or finished:
                stopped = bool(finished) and not (terminated or truncated)
                break
    finally:
        controller.episode_callback()
        controller.episode_reset()
        env.close()

    rec["pos"] = np.array(rec["pos"])
    rec["vel"] = np.array(rec["vel"])
    rec["des"] = np.array(rec["des"])
    rec["n"] = np.array(rec["n"])
    rec["speed"] = np.array(rec["speed"])

    n_final = int(np.asarray(obs["n_gates_passed"]))
    if stopped and n_final < len(gate_sequence):
        verdict, detail = (
            "CONTROLLER_STOPPED",
            "step_callback returned True before the race finished "
            "(did you forget to override it? the base class returns True)",
        )
    else:
        verdict, detail = classify(
            n_final,
            len(gate_sequence),
            bool(truncated),
            rec["pos"][-1],
            rec["vel"][-1],
            gates_pos,
            obstacles_pos,
            bounds_low,
            bounds_high,
        )
    # Gate crossing quality: for every increment of n_gates_passed find the step pair that crossed.
    crossings = []
    for i in range(len(rec["n"]) - 1):
        for k in range(rec["n"][i], rec["n"][i + 1]):
            gate_id = int(gate_sequence[k])
            info_ = crossing_margin(
                rec["pos"][i], rec["pos"][i + 1], gates_pos[gate_id], gates_quat[gate_id]
            )
            crossings.append({"pass": k, "gate": gate_id, "t": rec["t"][i + 1], **info_})

    if verbose:
        for c in crossings:
            print(
                f"  pass {c['pass']} (gate {c['gate']}) at t={c['t']:.2f}s "
                f"offset y={c['y']:+.3f} z={c['z']:+.3f} m, margin {c['margin'] * 100:+.1f} cm"
            )

    return {
        "verdict": verdict,
        "detail": detail,
        "n_final": n_final,
        "n_required": len(gate_sequence),
        "steps": n_steps,
        "sim_time": n_steps / config.env.freq,
        "crossings": crossings,
        "record": rec,
        "gates_pos": gates_pos,
        "gates_quat": gates_quat,
        "obstacles_pos": obstacles_pos,
        "gate_sequence": gate_sequence,
        "bounds_low": bounds_low,
        "bounds_high": bounds_high,
    }


def plot_episode(ep: dict[str, Any], out_path: Path, title: str) -> None:
    """Plot the top-down track, altitude/progress and speed of one episode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = ep["record"]
    pos = rec["pos"]
    gates_pos, gates_quat = ep["gates_pos"], ep["gates_quat"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # --- Panel 1: top-down track -------------------------------------------------------------
    ax = axes[0]
    pass_ids = rec["n"]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=pass_ids, cmap="viridis", s=7, zorder=3)
    ax.plot(pos[:, 0], pos[:, 1], color="0.6", lw=0.7, zorder=2)
    ax.plot(pos[0, 0], pos[0, 1], "o", color="tab:green", ms=9, label="start", zorder=5)
    ax.plot(pos[-1, 0], pos[-1, 1], "X", color="tab:red", ms=11, label="last live pose", zorder=5)
    for gid, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(gq)
        y_axis = rot.apply([0, 1, 0])
        x_axis = rot.apply([1, 0, 0])
        half = GATE_OPENING / 2
        ends = np.stack([gp - half * y_axis, gp + half * y_axis])
        ax.plot(ends[:, 0], ends[:, 1], color="tab:orange", lw=4, solid_capstyle="butt", zorder=4)
        ax.annotate(
            f"g{gid}",
            xy=(gp[0], gp[1]),
            xytext=(gp[0] + 0.12, gp[1] + 0.12),
            color="tab:orange",
            fontsize=9,
            zorder=6,
        )
        ax.arrow(
            gp[0],
            gp[1],
            0.28 * x_axis[0],
            0.28 * x_axis[1],
            color="tab:orange",
            width=0.008,
            head_width=0.06,
            zorder=4,
        )
    for oid, op in enumerate(ep["obstacles_pos"]):
        ax.plot(op[0], op[1], "s", color="tab:purple", ms=5, zorder=4)
        ax.annotate(
            f"o{oid}",
            xy=(op[0], op[1]),
            xytext=(op[0] + 0.1, op[1] - 0.14),
            color="tab:purple",
            fontsize=8,
            zorder=6,
        )
    ax.add_patch(
        plt.Rectangle(
            (ep["bounds_low"][0], ep["bounds_low"][1]),
            ep["bounds_high"][0] - ep["bounds_low"][0],
            ep["bounds_high"][1] - ep["bounds_low"][1],
            fill=False,
            ec="0.7",
            ls="--",
            lw=1,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down track (colour = passes completed)\narrows = gate crossing direction")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="n_gates_passed", shrink=0.8)

    # --- Panel 2: altitude and gate progress -------------------------------------------------
    ax = axes[1]
    ax.plot(rec["t"], pos[:, 2], color="tab:blue", lw=1.3)
    ax.axhline(0.0, color="tab:red", ls=":", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [m]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.step(rec["t"], rec["n"], where="post", color="tab:green", lw=1.6)
    ax2.set_ylabel("n_gates_passed", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(-0.2, ep["n_required"] + 0.2)
    ax.set_title("Altitude (red dotted = floor) and gate progress")

    # --- Panel 3: reference vs achieved ------------------------------------------------------
    ax = axes[2]
    for i, name in enumerate(["x", "y", "z"]):
        ax.plot(rec["t"], rec["pos"][:, i], lw=1.2, label=f"p{name}")
        ax.plot(rec["t"], rec["des"][:, i], lw=0.9, ls="--", alpha=0.8, label=f"p{name} setpoint")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("position [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Position vs. commanded setpoint\n(gap = inner-loop tracking error)")

    fig.suptitle(f"{title}\n{ep['verdict']}: {ep['detail']}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def resolve_device(device: str) -> str:
    """Validate the requested device and report what JAX actually sees.

    Args:
        device: Requested device, ``"cpu"`` or ``"gpu"``.

    Returns:
        The device string to hand to the environment.

    Raises:
        SystemExit: If a GPU is requested but JAX cannot see one, with a diagnosis.
    """
    import jax

    if device == "cpu":
        return device
    try:
        devices = jax.devices("gpu")
    except Exception as e:  # JAX raises RuntimeError when no GPU backend is present
        raise SystemExit(
            f"Requested --device gpu, but JAX sees no GPU.\n"
            f"  reason: {e}\n"
            f"  active backend: {jax.default_backend()}\n"
            f'Check with: pixi run -e gpu python -c "import jax; print(jax.devices())"\n'
            "Common causes: no NVIDIA driver, /dev/nvidia* not exposed to this process "
            "(containers/sandboxes), or the wrong pixi environment."
        ) from e
    print(f"gpu: {len(devices)} device(s) -> {devices[0]}")
    return device


def main(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    seed: int = 0,
    plot: bool = True,
    out_dir: str = "plots",
    verbose: bool = True,
    device: str = "cpu",
    control_mode: str | None = None,
):
    """Run a controller headless and report why each episode ended.

    Args:
        config: Race config file name in ``config/``.
        controller: Controller file name in ``lsy_drone_racing/control/``. Defaults to the one set
            in the config.
        n_runs: Number of episodes to run.
        seed: Base random seed. Episode ``i`` uses ``seed + i``.
        plot: Whether to write a PNG per episode.
        out_dir: Directory for plots (``plots/`` is gitignored).
        verbose: Print per-gate-pass margins.
        device: ``"cpu"`` or ``"gpu"``. The environment defaults to CPU and the TOML config cannot
            change this, so it must be requested explicitly.
        control_mode: Override the config's ``env.control_mode`` (``"state"`` or ``"attitude"``).
            Needed when a config ships with a mode that does not match the controller's action
            space, e.g. running the attitude-based MPCC on a ``"state"`` config.
    """
    logging.basicConfig(level=logging.WARNING)
    device = resolve_device(device)
    config_path = REPO_ROOT / "config" / config
    cfg = load_config(config_path)
    cfg.sim.render = False  # Never open a window in this harness
    if control_mode is not None:
        cfg.env.control_mode = control_mode
    controller_file = controller or cfg.controller.file
    controller_cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / controller_file)

    print(
        f"config={config}  controller={controller_file}  runs={n_runs}  seed0={seed}  "
        f"device={device}  control_mode={cfg.env.control_mode}"
    )
    results = []
    for i in range(n_runs):
        ep_seed = seed + i
        ep = run_episode(cfg, controller_cls, ep_seed, verbose=verbose, device=device)
        results.append(ep)
        print(
            f"[seed {ep_seed}] {ep['verdict']:<13} {ep['n_final']}/{ep['n_required']} passes  "
            f"t={ep['sim_time']:.2f}s  | {ep['detail']}"
        )
        if plot:
            stem = f"{config_path.stem}_{Path(controller_file).stem}_seed{ep_seed}"
            out = Path(out_dir) / f"{stem}.png"
            plot_episode(ep, out, f"{config_path.stem} / {controller_file} / seed {ep_seed}")
            print(f"          plot -> {out}")

    n_ok = sum(ep["verdict"] == "FINISHED" for ep in results)
    times = [ep["sim_time"] for ep in results if ep["verdict"] == "FINISHED"]
    print(f"\nfinished {n_ok}/{n_runs}" + (f", mean time {np.mean(times):.2f}s" if times else ""))


if __name__ == "__main__":
    fire.Fire(main)
