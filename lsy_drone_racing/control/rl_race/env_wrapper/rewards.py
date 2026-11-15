"""Reward functions: semi-sparse ``race`` and its pre-training variant ``dense_path``.

    r = gate * gates_passed + lap * finished - time - ang_vel * |omega_measured|
        - crash * crashed + progress * delta(arc - path_lateral * distance to the path)
        + lap_bonus * finished * (lap_ref - lap seconds)

``lap_bonus`` (per second under ``lap_ref``) pays only on laps flown from the official start; the
env passes ``lap_ref`` as the lap time of other starts, so their bonus is zero.

``race`` has progress = 0. ``dense_path`` adds the progress term for pre-training; the switch to
``race`` happens once the policy finishes the course often enough (see train.py). Gate and lap
events come from the simulator's ``n_gates_passed``; a crash step pays no gate reward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
from jax import Array

COMPONENTS = ("gate", "lap", "time", "ang_vel", "crash", "progress", "lap_bonus")


@dataclass(frozen=True)
class RewardConfig:
    """Reward coefficients (per event, per step or per metre); static jit argument."""

    gate: float = 10.0
    lap: float = 50.0
    time: float = 0.05
    ang_vel: float = 0.005
    crash: float = 20.0
    progress: float = 0.0
    path_lateral: float = 1.0  # Weight of the distance-to-path term in the path potential.
    lap_bonus: float = 0.0  # Per second the lap is faster than lap_ref (negative if slower).
    lap_ref: float = 4.0  # Reference lap time of the lap bonus, in seconds.


REWARD_PRESETS = {"race": RewardConfig(), "dense_path": RewardConfig(progress=15.0)}


def preset(name: str, **overrides: float) -> RewardConfig:
    """Return a named preset with optional coefficient overrides."""
    if name not in REWARD_PRESETS:
        raise ValueError(f"reward must be one of {sorted(REWARD_PRESETS)}")
    return replace(REWARD_PRESETS[name], **overrides)


def events(prev_passed: Array, passed: Array, terminated: Array, n_seq: int) -> dict[str, Array]:
    """Gate passes, lap completion and crashes of one transition, each (n,)."""
    finished = (prev_passed < n_seq) & (passed >= n_seq)
    crashed = terminated & ~finished & (prev_passed < n_seq)
    gates = jnp.where(crashed, 0, jnp.maximum(passed - prev_passed, 0))
    return {"gates": gates, "finished": finished, "crashed": crashed}


def compute(
    cfg: RewardConfig,
    event: dict[str, Array],
    ang_vel: Array,
    progress: Array,
    lap_seconds: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Total reward and its components, each (n,).

    Args:
        cfg: Coefficients.
        event: Output of ``events``.
        ang_vel: (n, 3) measured body rates after the step.
        progress: (n,) change of the path potential (see ``path.arc_progress``); zeroed on
            crash steps.
        lap_seconds: (n,) episode time in seconds, used on the finishing step for the lap bonus;
            ``None`` means no bonus.
    """
    if lap_seconds is None:
        lap_seconds = jnp.full(event["gates"].shape, cfg.lap_ref, dtype=jnp.float32)
    components = {
        "gate": cfg.gate * event["gates"].astype(jnp.float32),
        "lap": cfg.lap * event["finished"].astype(jnp.float32),
        "time": jnp.full(event["gates"].shape, -cfg.time, dtype=jnp.float32),
        "ang_vel": -cfg.ang_vel * jnp.linalg.norm(ang_vel, axis=-1),
        "crash": -cfg.crash * event["crashed"].astype(jnp.float32),
        "progress": cfg.progress * jnp.where(event["crashed"], 0.0, progress),
        "lap_bonus": cfg.lap_bonus
        * event["finished"].astype(jnp.float32)
        * (cfg.lap_ref - lap_seconds),
    }
    total = sum(components.values())
    return total.astype(jnp.float32), components
