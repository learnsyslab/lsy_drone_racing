"""Policy features computed from the course simulator's own observation fields.

Only fields returned by ``race_core.obs`` are used (sensor-masked gate and obstacle poses). The
configured next gates are given by center and passing normal, and the nearest obstacles
by position; all relative to the drone and expressed in its body frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ObsConfig:
    """Feature layout; frozen so it can be a static jit argument."""

    n_gates: int = 2
    n_obstacles: int = 4

    @property
    def size(self) -> int:
        """Flattened feature dimension for a ``action_dim``-free layout plus 4 previous actions."""
        # pos 3, rotation 9, body vel 3, ang vel 3, previous action 4, progress 1,
        # per gate: center 3 + normal 3 + valid 1, per obstacle: relative position 3.
        return 23 + 7 * self.n_gates + 3 * self.n_obstacles


def squeeze_drone_axis(raw: dict[str, Array]) -> dict[str, Array]:
    """Select drone 0 of each world from ``race_core.obs`` (single-agent training)."""
    return {key: value[:, 0] for key, value in raw.items()}


def encode(raw: dict[str, Array], previous_action: Array, cfg: ObsConfig) -> Array:
    """Encode one batch of squeezed observations into features of shape (n, cfg.size).

    Args:
        raw: Output of ``race_core.obs`` after ``squeeze_drone_axis``.
        previous_action: (n, 4) last policy action in [-1, 1].
        cfg: Feature layout.
    """
    pos = raw["pos"]
    count = pos.shape[0]
    rows = jnp.arange(count)
    rotation = Rotation.from_quat(raw["quat"]).as_matrix()
    world_to_body = jnp.swapaxes(rotation, 1, 2)

    def body(vectors: Array) -> Array:
        return jnp.einsum("bij,b...j->b...i", world_to_body, vectors)

    sequence = raw["gate_sequence"].astype(jnp.int32)
    n_seq = sequence.shape[1]
    passed = raw["n_gates_passed"].astype(jnp.int32).reshape(count)
    features = [
        pos,
        rotation.reshape(count, 9),
        body(raw["vel"]),
        raw["ang_vel"],
        previous_action,
        (passed / n_seq)[:, None],
    ]
    for offset in range(cfg.n_gates):
        slot = passed + offset
        valid = slot < n_seq
        index = jnp.minimum(slot, n_seq - 1)
        gate_ids = sequence[rows, index]
        center = raw["gates_pos"][rows, gate_ids] - pos
        normal = Rotation.from_quat(raw["gates_quat"][rows, gate_ids]).apply(
            jnp.array([1.0, 0.0, 0.0])
        )
        normal = normal * raw["gate_sequence_direction"][rows, index][:, None]
        gate = jnp.concatenate([body(center), body(normal), jnp.ones((count, 1))], axis=1)
        features.append(jnp.where(valid[:, None], gate, 0.0))
    if cfg.n_obstacles:
        relative = raw["obstacles_pos"] - pos[:, None]
        # Obstacles are vertical poles: rank them by horizontal distance.
        order = jnp.argsort(jnp.linalg.norm(relative[..., :2], axis=-1), axis=1)
        nearest = jnp.take_along_axis(relative, order[..., None], axis=1)[:, : cfg.n_obstacles]
        nearest = body(nearest)
        missing = cfg.n_obstacles - nearest.shape[1]
        if missing > 0:
            nearest = jnp.concatenate([nearest, jnp.zeros((count, missing, 3))], axis=1)
        features.append(nearest.reshape(count, -1))
    return jnp.concatenate(features, axis=1).astype(jnp.float32)
