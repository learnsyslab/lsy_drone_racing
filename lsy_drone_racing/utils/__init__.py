"""Utility module.

We separate utility functions that require ROS into a separate module to avoid ROS as a
dependency for sim-only scripts.
"""

from lsy_drone_racing.utils.utils import load_config, load_controller

__all__ = ["load_config", "load_controller"]
