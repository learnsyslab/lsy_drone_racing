"""This module implements a Controller Input on attitude level.

To run it, you need to use the gamepad environment. You can connect any gamepad
(here for an XBOX controller) and fly. Note that there is cover support to simplify the
controlls. You might also want to set the camera to fpv in the toml file.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import pygame
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController(Controller):
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
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4

        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._tick = 0

        pygame.init()
        pygame.joystick.init()

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self._roll = 0
        self._pitch = 0
        self._yaw = 0

        print("Controller:", self._joystick.get_name())

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
        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        pygame.event.pump()

        for i in range(self._joystick.get_numbuttons()):
            if self._joystick.get_button(i):
                print("Button", i, "pressed")

        deadzone = 0.1
        stick_l_x = self._apply_deadzone(self._joystick.get_axis(0), deadzone)
        # stick_l_y = self._apply_deadzone(self._joystick.get_axis(1), deadzone)
        lt = self._apply_deadzone(self._joystick.get_axis(2), deadzone)
        stick_r_x = self._apply_deadzone(self._joystick.get_axis(3), deadzone)
        stick_r_y = self._apply_deadzone(self._joystick.get_axis(4), deadzone)
        rt = self._apply_deadzone(self._joystick.get_axis(5), deadzone)

        max_angle = np.pi / 4
        max_angle_rate = np.pi / self.freq

        self._roll = 0.9 * (self._roll + stick_r_x * max_angle_rate)
        self._pitch = 0.9 * (self._pitch + -stick_r_y * max_angle_rate)
        self._yaw = self._yaw + -stick_l_x * max_angle_rate
        euler = [self._roll, self._pitch, self._yaw]
        euler[:2] = np.clip(euler[:2], -max_angle, max_angle)

        thrust_desired += self.thrust_max / 8 * (rt + 1) / 2
        thrust_desired -= self.thrust_max / 8 * (lt + 1) / 2

        return np.concat([euler, [thrust_desired]], dtype=np.float32)

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
        self.i_error[:] = 0
        self._tick = 0

    def _apply_deadzone(self, value: float, deadzone: float = 0.1) -> float:
        if abs(value) < deadzone:
            return 0.0
        return value
