"""Four normalized actions [roll, pitch, yaw, thrust] to absolute attitude commands."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

ACTION_SIZE = 4
ACTION_ORDER = ("roll", "pitch", "yaw", "thrust")


@dataclass(frozen=True)
class ActionConfig:
    """Attitude limits; frozen so it can be a static jit argument."""

    roll_limit_deg: float = 70.0
    pitch_limit_deg: float = 70.0
    yaw_limit_deg: float = 70.0
    thrust_min_fraction: float = 0.0
    thrust_max_fraction: float = 1.0
    thrust_min: float = 0.0
    thrust_max: float = 1.0

    def __post_init__(self) -> None:
        """Check limits: the official attitude interface clips at +/- 90 degrees."""
        for limit in (self.roll_limit_deg, self.pitch_limit_deg, self.yaw_limit_deg):
            if not 0 < limit <= 90:
                raise ValueError("Attitude limits must be in (0, 90] degrees")
        if not 0 <= self.thrust_min_fraction < self.thrust_max_fraction <= 1:
            raise ValueError("Thrust fractions must satisfy 0 <= min < max <= 1")
        if not self.thrust_max > self.thrust_min >= 0:
            raise ValueError("Need thrust_max > thrust_min >= 0")


def to_command(action: Array, cfg: ActionConfig) -> Array:
    """Map (n, 4) actions in [-1, 1] to (n, 4) commands [roll, pitch, yaw, total_thrust]."""
    action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
    if action.shape[-1] != ACTION_SIZE:
        raise ValueError("Expected four actions in roll, pitch, yaw, thrust order")
    angles = action[..., :3] * jnp.deg2rad(
        jnp.array([cfg.roll_limit_deg, cfg.pitch_limit_deg, cfg.yaw_limit_deg])
    )
    thrust = cfg.thrust_min + 0.5 * (action[..., 3] + 1.0) * (cfg.thrust_max - cfg.thrust_min)
    return jnp.concatenate([angles, thrust[..., None]], axis=-1)
