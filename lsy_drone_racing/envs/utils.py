"""Utility functions for the drone racing environments."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.numpy import vectorize
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from jax import Array


def load_track(track: ConfigDict) -> tuple[ConfigDict, ConfigDict, ConfigDict]:
    """Load the track from a config dict.

    Gates and obstacles are loaded as a config dicts with keys `pos`, `quat`, `nominal_pos`, and
    `nominal_quat`. Drones are loaded as a config dicts with keys `pos`, `rpy`, `quat`, `vel` and
    `ang_vel`.

    Args:
        track: The track config dict.

    Returns:
        The gates, obstacles, and drones as config dicts.
    """
    assert "gates" in track, "Track must contain gates field."
    assert "obstacles" in track, "Track must contain obstacles field."
    assert "drones" in track, "Track must contain drones field."
    gate_pos = np.array([g["pos"] for g in track.gates], dtype=np.float32)
    gate_quat = (
        R.from_euler("xyz", np.array([g["rpy"] for g in track.gates])).as_quat().astype(np.float32)
    )
    gates = {
        "pos": gate_pos,
        "quat": gate_quat,
        "nominal_pos": gate_pos.copy(),
        "nominal_quat": gate_quat.copy(),
    }
    obstacle_pos = np.array([o["pos"] for o in track.obstacles], dtype=np.float32)
    obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos.copy()}
    drones = {
        k: np.array([drone.get(k) for drone in track.drones], dtype=np.float32)
        for k in track.drones[0].keys()
    }
    drones["quat"] = R.from_euler("xyz", drones["rpy"]).as_quat().astype(np.float32)
    return ConfigDict(gates), ConfigDict(obstacles), ConfigDict(drones)


@jax.jit
@partial(vectorize, signature="(3),(3),(3),(4)->()", excluded=[4])
def gate_passed(
    drone_pos: Array,
    last_drone_pos: Array,
    gate_pos: Array,
    gate_quat: Array,
    gate_size: tuple[float, float],
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the x-Axis (pointing from -x to +x). Therefore, we check if x
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the y
    and z box boundaries. First, we linearly interpolate to get the y and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
        gate_pos: The position of the gate in the world frame.
        gate_quat: The rotation of the gate as an xyzw quaternion.
        gate_size: The size of the gate box in meters.
    """
    # Transform last and current drone position into current gate frame.
    gate_rot = R.from_quat(gate_quat)
    last_pos_local = gate_rot.apply(last_drone_pos - gate_pos, inverse=True)
    pos_local = gate_rot.apply(drone_pos - gate_pos, inverse=True)
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    passed_plane = (last_pos_local[0] < 0) & (pos_local[0] > 0)
    alpha = -last_pos_local[0] / (pos_local[0] - last_pos_local[0])
    y_intersect = alpha * (pos_local[1]) + (1 - alpha) * last_pos_local[1]
    z_intersect = alpha * (pos_local[2]) + (1 - alpha) * last_pos_local[2]
    # Divide gate size by 2 to get the distance from the center to the edges
    in_box = (abs(y_intersect) < gate_size[0] / 2) & (abs(z_intersect) < gate_size[1] / 2)
    return passed_plane & in_box
