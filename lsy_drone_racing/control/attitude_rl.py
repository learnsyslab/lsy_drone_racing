"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeRL(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        # For more info on the models, check out https://github.com/learnsyslab/drone-models
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4  # min total thrust
        self.thrust_max = drone_params["thrust_max"] * 4  # max total thrust

        # Set num of stacked obs
        self.n_obs = 2
        # Set trajectory parameters
        self.n_samples = 10
        self.samples_dt = 0.1
        self.trajectory_time = 15.0
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )
        self._tick = 0

        # Same waypoints as in the state controller. Determined by trial and error.
        start_pos = obs["pos"]
        waypoints = np.array(
            [
                start_pos,
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
        # Generate spline trajectory
        ts = np.linspace(0, self.trajectory_time, int(self.freq * self.trajectory_time))
        spline = CubicSpline(np.linspace(0, self.trajectory_time, waypoints.shape[0]), waypoints)
        self.trajectory = spline(ts)  # (n_steps, 3)

        # Load RL policy
        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")
        model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.last_action = np.array([0.0, 0.0, 0.0, self.drone_mass * 9.81], dtype=np.float32)
        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))

        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        i = min(self._tick, self.trajectory.shape[0] - 1)
        if i == self.trajectory.shape[0] - 1:  # Maximum duration reached
            self._finished = True

        obs_rl = self._obs_rl(obs)
        obs_rl = torch.tensor(obs_rl, dtype=torch.float32).unsqueeze(0).to("cpu")
        with torch.no_grad():
            act, _, _, _ = self.agent.get_action_and_value(obs_rl, deterministic=True)
            self.last_action = np.asarray(torch.asarray(act.squeeze(0))).copy()
            act[..., 2] = 0.0

        act = self._scale_actions(act.squeeze(0).numpy()).astype(np.float32)

        return act

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Extract the relevant parts of the observation for the RL policy."""
        obs_rl = {}
        obs_rl["basic_obs"] = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        dpos = self.trajectory[idx] - obs["pos"]  # (n_samples, 3)
        obs_rl["local_samples"] = dpos.reshape(-1)  # (n_samples*3,)
        obs_rl["prev_obs"] = self.prev_obs.reshape(-1)  # (n_obs*13,)
        obs_rl["last_action"] = self.last_action  # (4,)
        self.prev_obs = np.concatenate([self.prev_obs[1:, :], obs_rl["basic_obs"][None, :]], axis=0)

        return np.concatenate([v for v in obs_rl.values()], axis=-1).astype(np.float32)

    def _scale_actions(self, actions: NDArray) -> NDArray:
        """Rescale and clip actions from [-1, 1] to [action_sim_low, action_sim_high]."""
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        mean = np.array(
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0], dtype=np.float32
        )
        return np.clip(actions, -1.0, 1.0) * scale + mean

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
