"""Single drone racing environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gymnasium
from gymnasium import Env
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from packaging.version import Version

from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space, build_observation_space

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict

AutoresetMode = None
if Version(gymnasium.__version__) >= Version("1.1"):
    from gymnasium.vector import AutoresetMode


class DroneRaceEnv(RaceCoreEnv, Env):
    """Single-agent drone racing environment."""

    def __init__(
        self,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int | None = None,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the single-agent drone racing environment.

        Args:
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            track: Track configuration.
            sensor_range: Sensor range.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: None / -1 for a generated seed or the random seed directly.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
        """
        super().__init__(
            n_envs=1,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = build_action_space(control_mode, sim_config.drone_model)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.observation_space = build_observation_space(
            n_gates, n_obstacles, self.data.gate_sequence.shape[0]
        )
        self.settings = self.settings.replace(autoreset=False)
        self._step = self.build_step_fn()  # Apply new settings to capture autoreset effect

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            The initial observation and info.
        """
        Env.reset(self, seed=seed, options=options)
        self.data, (obs, info) = self._reset(self.data, seed=seed)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        return obs, info

    def step(self, action: Array) -> tuple[dict, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action for the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        self.data, (obs, reward, terminated, truncated, info) = self._step(self.data, action)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        return obs, float(reward[0, 0]), bool(terminated[0, 0]), bool(truncated[0, 0]), info


class VecDroneRaceEnv(RaceCoreEnv, VectorEnv):
    """Vectorized single-agent drone racing environment."""

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP if AutoresetMode is not None else None}

    def __init__(
        self,
        num_envs: int,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the vectorized single-agent drone racing environment.

        Args:
            num_envs: Number of worlds in the vectorized environment.
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            track: Track configuration.
            sensor_range: Sensor range.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: Random seed.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
        """
        super().__init__(
            n_envs=num_envs,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.num_envs = num_envs
        self.single_action_space = build_action_space(control_mode, sim_config.drone_model)
        self.action_space = batch_space(self.single_action_space, num_envs)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        self.single_observation_space = build_observation_space(
            n_gates, n_obstacles, self.data.gate_sequence.shape[0]
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment in all worlds.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            The initial observation and info.
        """
        VectorEnv.reset(self, seed=seed, options=options)
        self.data, (obs, info) = self._reset(self.data, seed=seed)
        obs = {k: v[:, 0] for k, v in obs.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        return obs, info

    def step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step the environment in all worlds.

        Args:
            action: Action for all worlds, i.e., a batch of (n_envs, action_dim) arrays.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        self.data, (obs, reward, terminated, truncated, info) = self._step(self.data, action)
        obs = {k: v[:, 0] for k, v in obs.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        return obs, reward[:, 0], terminated[:, 0], truncated[:, 0], info
