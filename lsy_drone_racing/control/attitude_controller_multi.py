"""This module wraps the AttitudeController to handle batched multi-agent environments.

In multi-agent simulations, observations are batched across all drones.
The rank index is used to select the state of the current drone.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.attitude_controller import (
    AttitudeController as SingleAttitudeController,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController(SingleAttitudeController):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        self.rank = info["rank"]
        super().__init__({k: v[self.rank] for k, v in obs.items()}, info, config)
        # We don't want the example controllers to crash, so we speed up this one to get ahead
        self._t_total = 10
        waypoints = self._des_pos_spline._c[-1]
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        return super().compute_control({k: v[self.rank] for k, v in obs.items()}, info)
