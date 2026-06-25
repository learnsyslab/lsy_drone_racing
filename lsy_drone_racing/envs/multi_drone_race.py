"""Multi-agent drone racing environments."""

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


class MultiDroneRaceEnv(RaceCoreEnv, Env):
    """Multi-agent drone racing environment.

    This environment enables multiple agents to simultaneously compete with each other on the same
    track.
    """

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
        """Initialize the multi-agent drone racing environment.

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
        n_gates, n_obstacles, n_drones = len(track.gates), len(track.obstacles), len(track.drones)
        super().__init__(
            n_envs=1,
            n_drones=n_drones,
            freq=freq,
            sim_config=sim_config,
            sensor_range=sensor_range,
            track=track,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = batch_space(
            build_action_space(control_mode, sim_config.drone_model), n_drones
        )
        self.observation_space = batch_space(
            build_observation_space(n_gates, n_obstacles), n_drones
        )
        self.settings = self.settings.replace(autoreset=False)
        self._step = self.build_step_fn()  # Apply new settings to capture autoreset effect

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment for all drones.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            Observation and info for all drones.
        """
        Env.reset(self, seed=seed, options=options)
        self.data, (obs, info) = self._reset(self.data, seed=seed)
        obs = {k: v[0] for k, v in obs.items()}
        info = {k: v[0] for k, v in info.items()}
        return obs, info

    def step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step the environment for all drones.

        Args:
            action: Action for all drones, i.e., a batch of (n_drones, action_dim) arrays.

        Returns:
            Observation, reward, terminated, truncated, and info for all drones.
        """
        self.data, (obs, reward, terminated, truncated, info) = self._step(self.data, action)
        obs = {k: v[0] for k, v in obs.items()}
        info = {k: v[0] for k, v in info.items()}
        # TODO: Fix by moving towards pettingzoo API
        # https://pettingzoo.farama.org/api/parallel/
        return obs, reward[0, 0], terminated[0].all(), truncated[0].all(), info


class VecMultiDroneRaceEnv(RaceCoreEnv, VectorEnv):
    """Vectorized multi-agent drone racing environment.

    This environment enables vectorized training of multi-agent drone racing agents.
    """

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
        """Vectorized multi-agent drone racing environment.

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
        n_gates, n_obstacles, n_drones = len(track.gates), len(track.obstacles), len(track.drones)
        super().__init__(
            n_envs=num_envs,
            n_drones=n_drones,
            freq=freq,
            sim_config=sim_config,
            sensor_range=sensor_range,
            track=track,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.num_envs = num_envs
        self.single_action_space = batch_space(
            build_action_space(control_mode, sim_config.drone_model), n_drones
        )
        self.action_space = batch_space(batch_space(self.single_action_space), num_envs)
        self.single_observation_space = batch_space(
            build_observation_space(n_gates, n_obstacles), n_drones
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment for all drones.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.

        Returns:
            Observation and info for all drones.
        """
        VectorEnv.reset(self, seed=seed, options=options)
        self.data, (obs, info) = self._reset(self.data, seed=seed)
        return obs, info

    def step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step the environment for all drones.

        Args:
            action: Action for all drones, i.e., a batch of (n_drones, action_dim) arrays.
        """
        self.data, (obs, reward, terminated, truncated, info) = self._step(self.data, action)
        return obs, reward, terminated, truncated, info
