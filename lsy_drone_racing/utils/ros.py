"""Utility functions that require ROS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_estimators.ros import ROSConnector

if TYPE_CHECKING:
    from numpy.typing import NDArray


def track_poses(n_gates: int, n_obstacles: int) -> tuple[NDArray, NDArray, NDArray]:
    """Get the track's gate positions, orientations and obstacle positions from the Mocap system.

    Args:
        n_gates: Number of gates in the track.
        n_obstacles: Number of obstacles in the track.

    Returns:
        A tuple containing the (n_gates, 3) gate positions, the (n_gates, 4) gate orientations, and
        the (n_obstacles, 3) obstacle positions.
    """
    gate_pos = np.zeros((n_gates, 3), dtype=np.float32)
    gate_quat = np.zeros((n_gates, 4), dtype=np.float32)
    obstacle_pos = np.zeros((n_obstacles, 3), dtype=np.float32)

    tf_names = [f"gate{i}" for i in range(1, n_gates + 1)]
    tf_names += [f"obstacle{i}" for i in range(1, n_obstacles + 1)]
    ros_connector = None
    try:
        ros_connector = ROSConnector(tf_names=tf_names, timeout=10.0)
        pos, quat = ros_connector.tf_pos, ros_connector.tf_quat
        for i in range(n_gates):
            gate_pos[i, ...] = pos[f"gate{i + 1}"]
            gate_quat[i, ...] = quat[f"gate{i + 1}"]
        for i in range(n_obstacles):
            obstacle_pos[i, ...] = pos[f"obstacle{i + 1}"]
    except KeyError as e:
        raise KeyError(
            f"Could not find all track objects in the ROS TF tree: {e}. Have you enabled the track "
            "objects in Vicon and started the motion capture tracking node?"
        ) from e
    finally:
        if ros_connector is not None:
            ros_connector.close()
    return gate_pos, gate_quat, obstacle_pos


def drone_poses(drone_names: list[str]) -> tuple[NDArray, NDArray]:
    """Get the drone positions from the Mocap system.

    Args:
        drone_names: List of drone estimator names.

    Returns:
        A tuple containing:
            - drone_pos: An (n_drones, 3) array of drone positions.
            - drone_quat: An (n_drones, 4) array of drone orientations as quaternions.
    """
    drone_pos = np.zeros((len(drone_names), 3), dtype=np.float32)
    drone_quat = np.zeros((len(drone_names), 4), dtype=np.float32)
    try:
        ros_connector = ROSConnector(estimator_names=drone_names, timeout=10.0)
        for i, drone_name in enumerate(drone_names):
            drone_pos[i, ...] = ros_connector.est_pos[drone_name]
            drone_quat[i, ...] = ros_connector.est_quat[drone_name]
    except KeyError as e:
        raise KeyError(
            f"Could not find drone in the ROS estimator messages: {e}. Have you enabled the drone "
            "in Vicon and started the estimator node?"
        ) from e
    finally:
        ros_connector.close()
    return drone_pos, drone_quat
