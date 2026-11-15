r"""Deterministic Level 2 evaluation following the protocol of Pasumarti et al. (2026).

Runs N episodes (default 50) in parallel worlds from official starts (no segment
starts) with the Gaussian mean action, and reports success rate, mean time to
complete, mean speed, crashes split by cause, and mean gates passed.

Usage::

    python -m lsy_drone_racing.control.rl_single_l2.eval_paper \\
        --model=<checkpoint step dir or exported actor dir> [--episodes=50] [--seed=0] \\
        [--csv_path=episodes.csv] [--json_path=summary.json]

Crash causes are attributed exactly: before each step, the same jitted action
and physics substeps are replayed on the identical pre-step state (including its
RNG key), and the raw contacts and bounds are inspected before the official
step warps disabled drones below the ground.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ActionConfig,
    Level2SBXEnv,
    ObservationConfig,
    decode_action,
)
from lsy_drone_racing.control.rl_single_l2.policy import network_from_config
from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "level2.toml"
CAUSES = ("gate", "obstacle", "out_of_bounds", "other")


def load_bundle(model: str | Path) -> dict[str, Any]:
    """Read a checkpoint step directory or an exported actor directory."""
    path = Path(model).expanduser().resolve()
    name = "metadata.json" if (path / "metadata.json").is_file() else "config.json"
    bundle = json.loads((path / name).read_text())
    bundle["params"] = serialization.msgpack_restore((path / "actor.msgpack").read_bytes())
    return bundle


def build_cause_classifier(env: Any) -> Any:
    """Return jit(data, commands) -> (n_worlds, 4) one-hot of potential crash causes.

    Mirrors steps 2-4 of race_core's step function without the warp, so the
    result is the cause the official step would see for a newly disabled drone.
    """
    from crazyflow.sim.sim import sync_sim2mjx

    apply_action = env.build_apply_action_fn()
    sim_step = env.sim.build_step_fn()
    gate_ids, obstacle_ids = env.mocap_ids
    model, mjx_template = env.sim.mj_model, env.sim.mjx_data
    contact = mjx_template._impl.contact
    geom1, geom2 = np.asarray(contact.geom1[0]), np.asarray(contact.geom2[0])
    names = [
        (model.body(model.geom_bodyid[a]).name, model.body(model.geom_bodyid[b]).name)
        for a, b in zip(geom1, geom2, strict=True)
    ]
    involves_drone = np.array(["drone:0" in pair for pair in names])
    gate_mask = jnp.asarray(
        involves_drone & np.array([any("gate:" in n for n in p) for p in names])
    )
    obstacle_mask = jnp.asarray(
        involves_drone & np.array([any("obstacle:" in n for n in p) for p in names])
    )

    @jax.jit
    def classify(data: Any, commands: jax.Array) -> jax.Array:
        data = apply_action(commands, data)
        n_substeps = data.sim_data.core.freq // env.settings.freq
        data = data.replace(sim_data=sim_step(data.sim_data, n_substeps))
        mocap_pos, mocap_quat = mjx_template.mocap_pos, mjx_template.mocap_quat
        mocap_pos = mocap_pos.at[..., gate_ids, :].set(data.gates_pos)
        mocap_quat = mocap_quat.at[..., gate_ids, :].set(jnp.roll(data.gates_quat, 1, axis=-1))
        mocap_pos = mocap_pos.at[..., obstacle_ids, :].set(data.obstacles_pos)
        mjx_data = mjx_template.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)
        _, mjx_data = sync_sim2mjx(data.sim_data, mjx_data, env.sim.mjx_model)
        touching = mjx_data._impl.contact.dist < 0
        gate = jnp.any(touching & gate_mask, axis=-1)
        obstacle = jnp.any(touching & obstacle_mask, axis=-1)
        pos = data.sim_data.states.pos[:, 0]
        takeoff = data.takeoff_pos[:, 0]
        off_platform = jnp.any(jnp.abs(pos[:, :2] - takeoff[:, :2]) > 0.02, axis=-1)
        low = jnp.any(pos < data.pos_limit_low, axis=-1) & off_platform
        bounds = low | jnp.any(pos > data.pos_limit_high, axis=-1)
        bounds = bounds & ~gate & ~obstacle
        obstacle = obstacle & ~gate
        other = ~(gate | obstacle | bounds)
        return jnp.stack([gate, obstacle, bounds, other], axis=-1)

    return classify


def evaluate(
    model: str,
    episodes: int = 50,
    seed: int = 0,
    config: str | None = None,
    max_episode_steps: int = 1500,
    device: str = "cpu",
    csv_path: str | None = None,
    json_path: str | None = None,
) -> dict[str, Any]:
    """Run the protocol and return the summary dictionary (also printed as a table)."""
    bundle = load_bundle(model)
    environment = bundle["environment"]
    observation = ObservationConfig(**environment["observation"])
    action = ActionConfig(**environment["action"])
    network = network_from_config(bundle["network"])
    actor = network.actor()
    params = bundle["params"]
    env = Level2SBXEnv(
        Path(config).expanduser() if config else DEFAULT_CONFIG,
        n_envs=episodes,
        seed=seed,
        device=device,
        max_episode_steps=max_episode_steps,
        observation_config=observation,
        action_config=action,
        reset_config=ResetConfig(segment_init_prob=0.0),
    )
    if env.config.env.freq != environment["freq"]:
        raise ValueError("Evaluation control frequency differs from training")
    env.set_normalizer(bundle["normalizer"])
    policy = jax.jit(lambda o: actor.apply(params, o).mode())
    classify = build_cause_classifier(env.jax_env)
    results: list[dict[str, Any] | None] = [None] * episodes
    speed_sum = np.zeros(episodes)
    try:
        env.seed(seed)
        obs = env.reset()
        for _ in range(max_episode_steps + 1):
            raw = np.asarray(policy(jnp.asarray(obs)))
            commands = decode_action(
                raw, env._current_obs["quat"], env.thrust_min, env.thrust_max, action
            )
            causes = np.asarray(classify(env.jax_env.data, commands))
            speed_sum += np.linalg.norm(np.asarray(env._current_obs["vel"]), axis=-1)
            obs, _, done, infos = env.step(raw)
            for index in np.flatnonzero(done):
                if results[index] is not None:
                    continue
                info = infos[index]
                steps = info["episode"]["l"]
                cause = "timeout" if info["timeout"] else None
                if info["crash"]:
                    cause = CAUSES[int(np.argmax(causes[index]))]
                results[index] = {
                    "episode": int(index),
                    "success": info["finished"],
                    "time": steps / env.config.env.freq,
                    "mean_speed": speed_sum[index] / steps,
                    "gates_passed": info["n_gates_passed"],
                    "end": "finished" if info["finished"] else cause,
                }
            if all(row is not None for row in results):
                break
    finally:
        env.close()
    rows = [row for row in results if row is not None]
    summary = summarize(rows, episodes)
    print(format_table(summary, model))
    if csv_path:
        with Path(csv_path).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if json_path:
        Path(json_path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize(rows: list[dict[str, Any]], episodes: int) -> dict[str, Any]:
    """Aggregate per-episode rows into the paper's metrics."""
    successes = [row for row in rows if row["success"]]
    ends = [row["end"] for row in rows]
    return {
        "episodes": episodes,
        "success_rate": len(successes) / episodes,
        "mean_time_to_complete": float(np.mean([r["time"] for r in successes]))
        if successes
        else None,
        "mean_speed_successful": float(np.mean([r["mean_speed"] for r in successes]))
        if successes
        else None,
        "mean_speed_all": float(np.mean([r["mean_speed"] for r in rows])),
        "mean_gates_passed": float(np.mean([r["gates_passed"] for r in rows])),
        "crashes": {
            cause: ends.count(cause) for cause in ("gate", "obstacle", "out_of_bounds", "other")
        },
        "timeouts": ends.count("timeout"),
    }


def format_table(summary: dict[str, Any], model: str) -> str:
    """Render the summary as a Markdown table."""

    def fmt(value: float | None, unit: str = "") -> str:
        return "n/a" if value is None else f"{value:.3f}{unit}"

    crashes = summary["crashes"]
    lines = [
        f"Model: {model}",
        "| metric | value |",
        "| --- | --- |",
        f"| episodes | {summary['episodes']} |",
        f"| success rate | {summary['success_rate']:.0%} |",
        f"| mean time to complete | {fmt(summary['mean_time_to_complete'], ' s')} |",
        f"| mean speed (successful) | {fmt(summary['mean_speed_successful'], ' m/s')} |",
        f"| mean speed (all) | {fmt(summary['mean_speed_all'], ' m/s')} |",
        f"| mean gates passed | {summary['mean_gates_passed']:.2f} |",
        f"| crashes: gate contact | {crashes['gate']} |",
        f"| crashes: obstacle contact | {crashes['obstacle']} |",
        f"| crashes: out of bounds / ground | {crashes['out_of_bounds']} |",
        f"| crashes: unattributed | {crashes['other']} |",
        f"| timeouts | {summary['timeouts']} |",
    ]
    return "\n".join(lines)


def main(
    model: str,
    episodes: int = 50,
    seed: int = 0,
    config: str | None = None,
    max_episode_steps: int = 1500,
    device: str = "cpu",
    csv_path: str | None = None,
    json_path: str | None = None,
) -> None:
    """Fire entry point; see the module docstring."""
    with jax.default_device(jax.devices(device)[0]):
        evaluate(model, episodes, seed, config, max_episode_steps, device, csv_path, json_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
