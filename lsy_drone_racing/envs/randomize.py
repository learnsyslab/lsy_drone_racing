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
    drone_excl_r: float = 1.0,
    gate_excl_r: float = 1.0,
    obstacle_excl_r: float = 0.8,
    obstacle_obstacle_excl_r: float = 0.3,
    obstacle_corridor_width: float = 0.5,
    yaw_range: float = 0.75,
    grid_res: int = 40,
) -> Callable[[Array, Array], tuple[Array, Array, Array]]:
    """Build a JIT- and vmap-compatible function that generates a complete random track layout.

    The track is built around a given drone start position. Gates and obstacles are placed one after
    another on a static 2-D grid. Each object type keeps a running per-cell clearance field (signed
    distance to the nearest placed object minus its radius) and samples a cell that still keeps
    every exclusion radius. If the grid saturates, the least-violating cell is used. Gate and
    obstacle heights are fixed at the values provided.

    Args:
        gates_z: Z-height for each gate, shape ``(n_objects,)``.
        obstacles_z: Z-height for each obstacle, shape ``(n_objects,)``.
        pos_limit_low: XY lower bounds of the arena ``[xmin, ymin]``.
        pos_limit_high: XY upper bounds of the arena ``[xmax, ymax]``.
        border_margin: Min distance [m] of all objects from the arena boundary.
        drone_excl_r: Min distance [m] of gates and obstacles from the drone start position.
        gate_excl_r: Min distance [m] between consecutive gates.
        obstacle_excl_r: Min distance [m] from gates to obstacles.
        obstacle_obstacle_excl_r: Min distance [m] between obstacles.
        obstacle_corridor_width: Half-width [m] of the corridor used for obstacle placement.
        yaw_range: Maximum yaw offset [rad] from the travel direction for gate orientation.
        grid_res: Number of grid nodes along the x-axis. The y-axis count is derived from the arena
            aspect ratio so the grid spacing is equal in both axes (square cells).

    Returns:
        ``sample_track(drone_pos, key) -> (gates_pos, gates_quat, obstacles_pos)``, a pure JAX
        function that produces one random track per call.
    """
    gates_z = jp.array(gates_z, dtype=jp.float32)
    obstacles_z = jp.array(obstacles_z, dtype=jp.float32)
    N = gates_z.shape[0]
    assert obstacles_z.shape[0] == N, "Number of gates and obstacles must be the same."

    xmin, ymin = jp.array(pos_limit_low[:2], dtype=jp.float32) + border_margin
    xmax, ymax = jp.array(pos_limit_high[:2], dtype=jp.float32) - border_margin

    width = float(pos_limit_high[0] - pos_limit_low[0]) - 2 * border_margin
    height = float(pos_limit_high[1] - pos_limit_low[1]) - 2 * border_margin
    grid_w = grid_res
    grid_h = max(1, round(grid_res * height / width))

    xs = jp.linspace(xmin, xmax, grid_w)
    ys = jp.linspace(ymin, ymax, grid_h)
    grid = jp.stack(jp.meshgrid(xs, ys), axis=-1)  # (H, W, 2)
    grid_flat = grid.reshape(-1, 2)

    def _sample(clearance: Array, preference: Array, key: Array) -> Array:
        """Sample a grid cell.

        Args:
            clearance: The signed margin to every placed object.
            preference: A soft 0/1 mask marking where the cell would ideally lie.
            key: JAX PRNG key.
        """
        valid = (clearance > 0).reshape(-1).astype(jp.float32)
        preferred = valid * preference.reshape(-1)
        # Case 1: cells that keep every exclusion radius *and* lie in the soft preference region.
        # Case 2: preference leaves nothing -> any cell that still keeps every exclusion radius.
        weight = jp.where(preferred.sum() > 0, preferred, valid)
        # Case 3: grid saturated, no cell keeps all radii -> the single least-violating cell.
        best = (jp.arange(valid.size) == jp.argmax(clearance)).astype(jp.float32)
        weight = jp.where(weight.sum() > 0, weight, best)
        return grid_flat[jax.random.choice(key, weight.shape[0], p=weight / weight.sum())]

    def _clearance(center: Array, radius: float) -> Array:
        """Calculate the signed clearance per cell.

        Args:
            center: The XY position of the object.
            radius: The exclusion radius of the object.
        """
        return jp.sqrt(jp.sum((grid - center) ** 2, axis=-1)) - radius

    def _corridor(from_xy: Array, to_xy: Array, width: float) -> Array:
        """Create a grid mask for cells within `width` of the segment `from_xy -> to_xy`."""
        v = to_xy - from_xy
        n = jp.linalg.norm(v) + 1e-8
        proj = jp.sum((grid - from_xy) * (v / n), axis=-1)
        perp = jp.linalg.norm(grid - (from_xy + proj[..., None] * v / n), axis=-1)
        return ((perp < width) & (proj >= 0) & (proj <= n)).astype(jp.float32)

    def sample_track(drone_pos: Array, key: Array) -> tuple[Array, Array, Array]:
        """Sample one random track around the given drone start position.

        Args:
            drone_pos: Drone start position ``(3,)``.
            key: JAX PRNG key.

        Returns:
            ``(gates_pos (N, 3), gates_quat (N, 4), obstacles_pos (N, 3))``.
        """
        keys = jax.random.split(key, 3 * N)
        k_gates, k_yaws, k_obs = keys[:N], keys[N : 2 * N], keys[2 * N :]
        prev_xy = drone_pos[:2]
        gate_clear = obs_clear = _clearance(prev_xy, drone_excl_r)
        no_preference = jp.ones((grid_h, grid_w), jp.float32)

        gates, obstacles = [], []
        for i in range(N):  # N is usually small, so this unrolled loop instead of scan is fine.
            gate_xy = _sample(gate_clear, no_preference, k_gates[i])
            travel = gate_xy - prev_xy
            yaw = jax.random.uniform(k_yaws[i], minval=-yaw_range, maxval=yaw_range)
            yaw = (yaw + jp.arctan2(travel[1], travel[0])) % (2 * jp.pi)
            gates.append(jp.array([gate_xy[0], gate_xy[1], yaw]))
            gate_clear = jp.minimum(gate_clear, _clearance(gate_xy, gate_excl_r))
            obs_clear = jp.minimum(obs_clear, _clearance(gate_xy, obstacle_excl_r))

            corridor = _corridor(prev_xy, gate_xy, obstacle_corridor_width)
            obs_xy = _sample(obs_clear, corridor, k_obs[i])
            obstacles.append(obs_xy)
            gate_clear = jp.minimum(gate_clear, _clearance(obs_xy, obstacle_excl_r))
            obs_clear = jp.minimum(obs_clear, _clearance(obs_xy, obstacle_obstacle_excl_r))
            prev_xy = gate_xy

        gates, obstacles = jp.stack(gates), jp.stack(obstacles)
        gates_pos = jp.concatenate([gates[:, :2], gates_z[:, None]], axis=-1)
        half_yaw = gates[:, 2] / 2.0
        zeros = jp.zeros_like(half_yaw)
        gates_quat = jp.stack([zeros, zeros, jp.sin(half_yaw), jp.cos(half_yaw)], axis=-1)
        obstacles_pos = jp.concatenate([obstacles, obstacles_z[:, None]], axis=-1)
        return gates_pos, gates_quat, obstacles_pos

    return sample_track


def build_full_track_randomization_fn(
    gates_z: Array, obstacles_z: Array, pos_limit_low: Array, pos_limit_high: Array
) -> Callable[[EnvData, Array, Array], EnvData]:
    """Build a track randomization function that fully regenerates the track per world.

    Args:
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
        drones_pos = data.sim_data.states.pos[:, 0]  # build each track around the first drone
        gates_pos, gates_quat, obstacles_pos = batched_generate(drones_pos, keys)
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
