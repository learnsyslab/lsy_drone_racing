"""Integration test: level3 produces collision-free tracks through the real reset pipeline."""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path

import gymnasium
import numpy as np
from jax import Array
from ml_collections import ConfigDict

import lsy_drone_racing  # noqa: F401  (registers the gymnasium environments)
from lsy_drone_racing.utils import load_config

CONFIG_PATH = Path(__file__).parents[3] / "config"
N_WORLDS = 1000


def min_distance_between(a: Array, b: Array) -> float:
    """Smallest XY distance between sets `a` and `b`, minimized over all worlds."""
    return np.linalg.norm(a[:, :, None] - b[:, None], axis=-1).min()


def min_distance_within(points: Array) -> float:
    """Smallest XY distance between two distinct points of the same set, over all worlds."""
    distances = np.linalg.norm(points[:, :, None] - points[:, None], axis=-1)
    self_pairs = np.arange(points.shape[1])
    distances[:, self_pairs, self_pairs] = np.inf
    return distances.min()


def max_xy_shift(randomization: ConfigDict) -> float:
    """Largest XY displacement a uniform position randomization can apply to an object."""
    kwargs = randomization.kwargs
    return float(np.hypot(*np.maximum(np.abs(kwargs.minval[:2]), np.abs(kwargs.maxval[:2]))))


def test_level3_tracks_are_collision_free():
    config = load_config(CONFIG_PATH / "level3.toml")
    env = gymnasium.make_vec(
        "DroneRacing-v0",
        num_envs=N_WORLDS,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = gymnasium.wrappers.vector.JaxToNumpy(env)
    env.reset()
    data = env.unwrapped.data  # ground-truth state, not the sensor-masked observation
    env.close()

    gates = np.asarray(data.gates_pos)[..., :2]
    obstacles = np.asarray(data.obstacles_pos)[..., :2]
    drone = np.asarray(data.sim_data.states.pos)[:, 0, :2][:, None, :]

    # Gates and obstacles are perturbed after track generation. Relax each exclusion radius by the
    # maximum displacement those randomizations can apply.
    gate_displacement = max_xy_shift(config.env.randomizations.gate_pos)
    obstacle_displacement = max_xy_shift(config.env.randomizations.obstacle_pos)
    assert min_distance_within(gates) >= 1.0 - 2 * gate_displacement
    assert min_distance_within(obstacles) >= 0.3 - 2 * obstacle_displacement
    assert min_distance_between(gates, obstacles) >= 0.8 - gate_displacement - obstacle_displacement
    assert min_distance_between(drone, gates) >= 1.0 - gate_displacement
    assert min_distance_between(drone, obstacles) >= 1.0 - obstacle_displacement
