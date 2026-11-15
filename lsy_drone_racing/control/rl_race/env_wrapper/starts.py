"""Path-based start states between gates (applied after an official reset)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from lsy_drone_racing.control.rl_race.env_wrapper import path

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array


@dataclass(frozen=True)
class StartConfig:
    """Share of official starts and the distribution of path starts."""

    official_prob: float = 0.5
    s_min: float = 0.1  # Curve parameter range; 0 and 1 are gate centers.
    s_max: float = 0.9
    pos_noise_m: float = 0.15
    speed_min_mps: float = 0.0
    speed_max_mps: float = 4.0
    direction_noise: float = 0.15  # Std of the noise added to the unit tangent.
    min_height_m: float = 0.15

    def __post_init__(self) -> None:
        """Check ranges."""
        if not 0 <= self.official_prob <= 1:
            raise ValueError("official_prob must be in [0, 1]")
        if not 0 <= self.s_min <= self.s_max <= 1:
            raise ValueError("Need 0 <= s_min <= s_max <= 1")
        if not 0 <= self.speed_min_mps <= self.speed_max_mps:
            raise ValueError("Need 0 <= speed_min_mps <= speed_max_mps")


def build_path_start(
    cfg: StartConfig, contact_check: Callable[[Any], Array], start_pos: Array
) -> Callable[[Any, Array, Array], tuple[Any, Array]]:
    """Compile ``path_start(data, mask, key) -> (data, accepted)``.

    For worlds in ``mask``: pick a leg k uniformly, a point on its Hermite curve with position
    noise, a velocity along the tangent (toward gate k's entry side), level attitude with yaw 0,
    and ``n_gates_passed = k``. Proposals out of bounds, too low or in contact keep the official
    reset state already present in ``data``.
    """
    from lsy_drone_racing.envs.race_core import _reset_env_data

    def path_start(data: Any, mask: Array, key: Array) -> tuple[Any, Array]:
        count = data.steps.shape[0]
        n_seq = data.gate_sequence.shape[0]
        k_leg, k_s, k_pos, k_speed, k_dir = jax.random.split(key, 5)
        leg = jax.random.randint(k_leg, (count,), 0, n_seq)
        s = jax.random.uniform(k_s, (count,), minval=cfg.s_min, maxval=cfg.s_max)
        controls = path.leg_control_points(
            data.gates_pos,
            data.gates_quat,
            data.gate_sequence,
            data.gate_sequence_direction,
            leg,
            start_pos,
        )
        point, tangent = path.hermite(*controls, s)
        position = point + jax.random.uniform(
            k_pos, (count, 3), minval=-cfg.pos_noise_m, maxval=cfg.pos_noise_m
        )
        direction = tangent / jnp.maximum(jnp.linalg.norm(tangent, axis=-1, keepdims=True), 1e-6)
        direction = direction + cfg.direction_noise * jax.random.normal(k_dir, (count, 3))
        direction = direction / jnp.maximum(
            jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-6
        )
        speed = jax.random.uniform(
            k_speed, (count, 1), minval=cfg.speed_min_mps, maxval=cfg.speed_max_mps
        )
        velocity = speed * direction

        states = data.sim_data.states
        sel = mask[:, None, None]
        identity = jnp.array([0.0, 0.0, 0.0, 1.0])
        proposed = states.replace(
            pos=jnp.where(sel, position[:, None], states.pos),
            vel=jnp.where(sel, velocity[:, None], states.vel),
            quat=jnp.where(sel, identity, states.quat),
            ang_vel=jnp.where(sel, 0.0, states.ang_vel),
        )
        core = data.sim_data.core.replace(mjx_synced=False)
        candidate = data.replace(sim_data=data.sim_data.replace(states=proposed, core=core))
        in_bounds = jnp.all(
            (position > candidate.pos_limit_low) & (position < candidate.pos_limit_high), axis=-1
        )
        high_enough = position[:, 2] > cfg.min_height_m
        collision = contact_check(candidate).reshape(count, -1).any(axis=1)
        accepted = mask & in_bounds & high_enough & ~collision
        keep = accepted[:, None, None]
        states = states.replace(
            pos=jnp.where(keep, proposed.pos, states.pos),
            vel=jnp.where(keep, proposed.vel, states.vel),
            quat=jnp.where(keep, proposed.quat, states.quat),
            ang_vel=jnp.where(keep, proposed.ang_vel, states.ang_vel),
        )
        result = data.replace(sim_data=data.sim_data.replace(states=states, core=core))
        # Reset step counters, sensor masks and the gate-passing reference position.
        result = _reset_env_data(result, mask=accepted)
        result = result.replace(
            n_gates_passed=jnp.where(accepted[:, None], leg[:, None], result.n_gates_passed)
        )
        return result, accepted

    return path_start
