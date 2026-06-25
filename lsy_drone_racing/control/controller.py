"""Base class for controller implementations.

Your task is to implement your own controller. This class must be the parent class of your
implementation. You have to use the same function signatures as defined by the base class. Apart
from that, you are free to add any additional methods, attributes, or classes to your controller.

As an example, you could load the weights of a neural network in the constructor and use it to
compute the control commands in the
[compute_control][lsy_drone_racing.control.controller.Controller.compute_control] method. You could
also use the [step_callback][lsy_drone_racing.control.controller.Controller.step_callback] method to
update the controller state at runtime.

Note:
    You can only define one controller class in a single file. Otherwise we will not be able to
    determine which class to use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from crazyflow import Sim
    from numpy.typing import NDArray


class Controller(ABC):
    """Base class for controller implementations."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Instructions:
            The controller's constructor has access the initial observation `obs`, the a priori
            information contained in dictionary `info`, and the config of the race track. Use this
            method to initialize constants, counters, pre-plan trajectories, etc.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """

    @abstractmethod
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Instructions:
            Implement this method to return the target state to be sent to the Crazyflie.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            A drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] in
            absolute coordinates or an attitude command [thrust, roll, pitch, yaw] as a numpy array.
        """

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback function called once after the control step.

        You can use this function to update your controller's internal state, save training data,
        update your models, and to terminate the episode.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.

        Returns:
            A flag to signal if the controller has finished.
        """
        return True  # Does not finish by default

    def episode_callback(self):
        """Callback function called once after each episode.

        You can use this function to reset your controller's internal state, save training data,
        train your models, compute additional statistics, etc.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.
        """

    def render_callback(self, sim: Sim):
        """Callback function called before the environment's rendering.

        You can use this function to render additional information on the screen, such as the
        planned trajectory, the drone's target state, etc.
        """

    def reset(self):
        """Reset internal variables if necessary."""

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""
