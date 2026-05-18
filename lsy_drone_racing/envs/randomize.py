"""Randomization functions for the simulation.

The functions in this module are inserted (compiled) into the reset function of the simulation for
efficiency. Because of this, they have to be functionally pure to work with JAX (see
https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jp
from crazyflow.utils import leaf_replace
from jax import Array
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

    from lsy_drone_racing.envs.race_core import EnvData


def randomize_drone_pos_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone position."""

    def randomize_drone_pos(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        drone_pos = data.states.pos + randomize_fn(subkey, shape=data.states.pos.shape)
        states = leaf_replace(data.states, mask, pos=drone_pos)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_pos


def randomize_drone_quat_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone quaternion."""

    def randomize_drone_quat(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        rpy = R.from_quat(data.states.quat).as_euler("xyz")
        quat = R.from_euler("xyz", rpy + randomize_fn(subkey, shape=rpy.shape)).as_quat()
        states = leaf_replace(data.states, mask, quat=quat)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_quat


def randomize_drone_mass_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone mass."""

    def randomize_drone_mass(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        mass = data.params.mass + randomize_fn(subkey, shape=data.params.mass.shape)
        params = leaf_replace(data.params, mask, mass=mass)
        return data.replace(core=data.core.replace(rng_key=key), params=params)

    return randomize_drone_mass


def randomize_drone_inertia_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone inertia."""

    def randomize_drone_inertia(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        J = data.params.J + randomize_fn(subkey, shape=data.params.J.shape)
        J_inv = jp.linalg.inv(J)
        params = leaf_replace(data.params, mask, J=J, J_inv=J_inv)
        return data.replace(core=data.core.replace(rng_key=key), params=params)

    return randomize_drone_inertia


def randomize_gate_pos_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int, ...]], jax.Array],
) -> Callable[[EnvData, Array | None, jax.random.PRNGKey], EnvData]:
    """Create a function that randomizes the gate position."""

    def randomize_gate_pos(data: EnvData, mask: Array | None, key: jax.random.PRNGKey) -> EnvData:
        gates_pos = data.gates_pos + randomize_fn(key, shape=data.gates_pos.shape)
        return leaf_replace(data, mask, gates_pos=gates_pos)

    return randomize_gate_pos


def randomize_gate_rpy_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[EnvData, Array | None, jax.random.PRNGKey], EnvData]:
    """Create a function that randomizes the gate rotation."""

    def randomize_gate_rpy(data: EnvData, mask: Array | None, key: jax.random.PRNGKey) -> EnvData:
        gate_rpy = R.from_quat(data.gates_quat).as_euler("xyz")
        gate_rpy = gate_rpy + randomize_fn(key, shape=gate_rpy.shape)
        return leaf_replace(data, mask, gates_quat=R.from_euler("xyz", gate_rpy).as_quat())

    return randomize_gate_rpy


def randomize_obstacle_pos_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[EnvData, Array | None, jax.random.PRNGKey], EnvData]:
    """Create a function that randomizes the obstacle position."""

    def randomize_obstacle_pos(
        data: EnvData, mask: Array | None, key: jax.random.PRNGKey
    ) -> EnvData:
        obstacles_pos = data.obstacles_pos + randomize_fn(key, shape=data.obstacles_pos.shape)
        return leaf_replace(data, mask, obstacles_pos=obstacles_pos)

    return randomize_obstacle_pos


def build_random_track_fn(
    gates_z: Array,
    obstacles_z: Array,
    pos_limit_low: Array,
    pos_limit_high: Array,
    *,
    border_margin: float = 0.5,
    start_excl_r: float = 1.0,
    gate_excl_r: float = 1.0,
    obstacle_excl_r: float = 1.0,
    gate_corridor_width: float = 0.4,
    obstacle_corridor_width: float = 0.4,
    yaw_range: float = 0.75,
    grid_h: int = 40,
    grid_w: int = 40,
) -> Callable[[Array], tuple[Array, Array, Array]]:
    """Build a JIT- and vmap-compatible function that generates a complete random track layout.

    Gates and obstacles are placed on a 2-D grid using iterative exclusion zones and corridor
    masks. Gate z-heights and obstacle z-heights are fixed at the values provided.

    Args:
        n_objects: Number of gates (= number of obstacles).
        gates_z: Z-height for each gate, shape ``(n_objects,)``.
        obstacles_z: Z-height for each obstacle, shape ``(n_objects,)``.
        pos_limit_low: XY lower bounds of the arena ``[xmin, ymin]``.
        pos_limit_high: XY upper bounds of the arena ``[xmax, ymax]``.
        border_margin: Min distance [m] of all objects from the arena boundary.
        start_excl_r: Exclusion radius [m] around the drone start position.
        gate_excl_r: Min distance [m] between consecutive gates.
        obstacle_excl_r: Min distance [m] from gates to obstacles and between obstacles.
        gate_corridor_width: Half-width [m] of the flight corridor masked out for gate placement.
        obstacle_corridor_width: Half-width [m] of the corridor used for obstacle placement.
        yaw_range: Maximum yaw offset [rad] from the travel direction for gate orientation.
        grid_h: Grid height (number of rows).
        grid_w: Grid width (number of columns).

    Returns:
        ``generate(key) -> (gates_pos, gates_quat, obstacles_pos)`` — a pure JAX function that
        produces one random track per call. Shapes: ``(N, 3)``, ``(N, 4)`` xyzw, ``(N, 3)``.
    """
    gates_z = jp.array(gates_z, dtype=jp.float32)
    obstacles_z = jp.array(obstacles_z, dtype=jp.float32)
    N = gates_z.shape[0]
    assert obstacles_z.shape[0] == N, "Number of gates and obstacles must be the same."

    xmin, ymin = jp.array(pos_limit_low[:2], dtype=jp.float32) + border_margin
    xmax, ymax = jp.array(pos_limit_high[:2], dtype=jp.float32) - border_margin

    # Precompute the placement grid (static).
    xs = jp.linspace(xmin, xmax, grid_w)
    ys = jp.linspace(ymin, ymax, grid_h)
    grid = jp.stack(jp.meshgrid(xs, ys), axis=-1)  # (H, W, 2)
    grid_flat = grid.reshape(-1, 2)
    cell_dxy = jp.array([(xmax - xmin) / grid_w, (ymax - ymin) / grid_h])

    def _sample(weight: Array, key: Array) -> Array:
        """Weighted sample from the placement grid with sub-cell jitter."""
        flat = weight.reshape(-1)
        total = flat.sum()
        p = jp.where(total > 0, flat / total, jp.ones_like(flat) / flat.size)
        k_choice, k_jitter = jax.random.split(key)
        pos = grid_flat[jax.random.choice(k_choice, flat.shape[0], p=p)]
        return pos + (jax.random.uniform(k_jitter, (2,)) - 0.5) * cell_dxy

    def _excl_circle(center: Array, radius: float) -> Array:
        """Float mask: 1 where grid point is farther than `radius` from `center`."""
        return (jp.sum((grid - center) ** 2, axis=-1) > radius**2).astype(jp.float32)

    def _corridor(from_xy: Array, to_xy: Array, width: float) -> Array:
        """Float mask: 1 for grid points inside the corridor of given `width`."""
        v = to_xy - from_xy
        n = jp.linalg.norm(v) + 1e-8
        u = v / n
        to_cell = grid - from_xy
        proj = jp.sum(to_cell * u, axis=-1)
        closest = from_xy + proj[..., None] * u
        perp = jp.linalg.norm(grid - closest, axis=-1)
        return ((perp < width) & (proj >= 0) & (proj <= n)).astype(jp.float32)

    def generate(key: Array) -> tuple[Array, Array, Array]:
        """Generate one random track.

        Args:
            key: JAX PRNG key.

        Returns:
            ``(gates_pos, gates_quat, obstacles_pos)`` with shapes ``(N, 3)``, ``(N, 4)`` (xyzw),
            ``(N, 3)``.
        """
        k_start, *sub_keys = jax.random.split(key, 1 + 3 * N)
        k_gates = jp.array(sub_keys[:N])
        k_yaws = jp.array(sub_keys[N : 2 * N])
        k_obs = jp.array(sub_keys[2 * N :])

        start_xy = jax.random.uniform(
            k_start,
            (2,),
            minval=jp.array([xmin - border_margin, ymin - border_margin]),
            maxval=jp.array([xmax + border_margin, ymax + border_margin]),
        )
        start_excl = _excl_circle(start_xy, start_excl_r)

        ones = jp.ones((grid_h, grid_w), jp.float32)
        init = (
            start_excl,  # gate placement weight
            ones,  # cumulative gate exclusion (gate-to-gate)
            ones,  # cumulative gate exclusion (gate-to-obstacle)
            ones,  # cumulative obstacle exclusion
            jp.zeros((N, 3), jp.float32),  # placed gates: [x, y, yaw]
            jp.zeros((N, 2), jp.float32),  # placed obstacles: [x, y]
        )

        def place_one(
            carry: tuple[Array, Array, Array, Array, Array, Array], i: int
        ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
            gate_w, gate_excl, gate_excl_obs, obs_excl, gates, obstacles = carry

            # Place gate
            gate_xy = _sample(gate_w * gate_excl, k_gates[i])
            prev_xy = jax.lax.cond(
                i == 0, lambda _: start_xy, lambda _: gates[i - 1, :2], operand=None
            )
            travel_dir = gate_xy - prev_xy
            yaw_offset = jax.random.uniform(k_yaws[i], minval=-yaw_range, maxval=yaw_range)
            yaw = (yaw_offset + jp.arctan2(travel_dir[1], travel_dir[0])) % (2 * jp.pi)
            gates = gates.at[i].set(jp.array([gate_xy[0], gate_xy[1], yaw]))

            # Place obstacle inside the travel corridor
            in_corridor = _corridor(prev_xy, gate_xy, obstacle_corridor_width)
            obs_weight = in_corridor * gate_excl_obs * obs_excl * start_excl
            obs_xy = _sample(obs_weight, k_obs[i])
            obstacles = obstacles.at[i].set(obs_xy)

            # Update exclusion zones
            gate_excl_new = gate_excl * _excl_circle(gate_xy, gate_excl_r)
            gate_excl_obs_new = gate_excl_obs * _excl_circle(gate_xy, obstacle_excl_r)
            obs_excl_new = obs_excl * _excl_circle(obs_xy, obstacle_excl_r)
            gate_corr = _corridor(prev_xy, gate_xy, gate_corridor_width)
            gate_w_new = gate_w * gate_excl_new * (1.0 - gate_corr)

            return (
                gate_w_new,
                gate_excl_new,
                gate_excl_obs_new,
                obs_excl_new,
                gates,
                obstacles,
            ), None

        (_, _, _, _, gates, obstacles), _ = jax.lax.scan(place_one, init, jp.arange(N))

        # Assemble output arrays with correct z-heights.
        gates_pos = jp.concatenate([gates[:, :2], gates_z[:, None]], axis=-1)
        half_yaw = gates[:, 2] / 2.0
        # Pure-yaw quaternion (xyzw): roll=pitch=0 → [0, 0, sin(yaw/2), cos(yaw/2)]
        gates_quat = jp.stack(
            [jp.zeros_like(half_yaw), jp.zeros_like(half_yaw), jp.sin(half_yaw), jp.cos(half_yaw)],
            axis=-1,
        )
        obstacles_pos = jp.concatenate([obstacles, obstacles_z[:, None]], axis=-1)

        return gates_pos, gates_quat, obstacles_pos

    return generate


def build_full_track_randomization_fn(
    gates_z: Array, obstacles_z: Array, pos_limit_low: Array, pos_limit_high: Array
) -> Callable[[EnvData, Array, Array], EnvData]:
    """Build a track randomization function that fully regenerates the track per world.

    Unlike the perturbation-based approach, this generates an entirely new gate and obstacle layout
    for every environment world that is being reset. Z-heights are fixed at the provided values.

    Args:
        n_objects: Number of gates (= number of obstacles).
        gates_z: Z-height for each gate, shape ``(n_objects,)``.
        obstacles_z: Z-height for each obstacle, shape ``(n_objects,)``.
        pos_limit_low: XY lower bounds of the arena ``[xmin, ymin]``.
        pos_limit_high: XY upper bounds of the arena ``[xmax, ymax]``.

    Returns:
        ``randomize_track(data, mask, key) -> data`` compatible with the reset pipeline.
    """
    batched_generate = jax.vmap(
        build_random_track_fn(gates_z, obstacles_z, pos_limit_low, pos_limit_high)
    )

    def randomize_track(data: EnvData, mask: Array, key: Array) -> EnvData:
        n_envs = data.gates_pos.shape[0]
        keys = jax.random.split(key, n_envs)
        gates_pos, gates_quat, obstacles_pos = batched_generate(keys)
        return leaf_replace(
            data,
            mask,
            gates_pos=gates_pos,
            gates_quat=gates_quat,
            obstacles_pos=obstacles_pos,
            nominal_gates_pos=gates_pos,
            nominal_gates_quat=gates_quat,
            nominal_obstacles_pos=obstacles_pos,
        )

    return randomize_track
