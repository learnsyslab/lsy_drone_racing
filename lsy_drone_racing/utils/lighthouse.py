"""Lighthouse positioning helpers.

When using the Lighthouse positioning system instead of a motion capture system, the drone
estimates its own pose onboard. We read that estimate back from the drone via cflib2 log blocks
instead of receiving it from an external tracker. This module contains the pure, hardware-free
parts of that path (the log variable layout and the decoding into an observation) so they can be
unit tested without a radio or the ``cflib2`` dependency.

The :class:`~lsy_drone_racing.utils.crazyflie.Crazyflie` wrapper imports these helpers and adds
the actual cflib2 log streaming on top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["LIGHTHOUSE_DECK_PARAM", "POS_VEL_VARS", "ORI_RATE_VARS", "decode_state"]

# Firmware parameter that is 1 when a Lighthouse deck is attached and detected.
LIGHTHOUSE_DECK_PARAM = "deck.bcLighthouse4"

# A single CRTP log block carries at most ~26 bytes of payload, so the twelve floats we need do
# not fit into one block. We split them across two blocks of six floats (24 bytes) each.
POS_VEL_VARS = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
)
ORI_RATE_VARS = (
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stateEstimate.yaw",
    "gyro.x",
    "gyro.y",
    "gyro.z",
)


def decode_state(
    pos_vel: dict[str, float], ori_rate: dict[str, float]
) -> dict[str, NDArray[np.floating]]:
    """Decode two onboard log samples into a simulation-compatible observation.

    The returned dictionary matches the per-drone fields of the simulation observation space (see
    :func:`lsy_drone_racing.envs.race_core.build_observation_space`), so the real environment can
    drop it in without any further conversion.

    Args:
        pos_vel: A sample containing ``stateEstimate.x/y/z`` (world-frame position, m) and
            ``stateEstimate.vx/vy/vz`` (world-frame linear velocity, m/s).
        ori_rate: A sample containing ``stateEstimate.roll/pitch/yaw`` (world-frame orientation,
            degrees) and ``gyro.x/y/z`` (body-frame angular velocity, degrees/s).

    Returns:
        A dictionary with ``float32`` ``pos``, ``quat`` (xyzw), ``vel`` and ``ang_vel`` (rad/s).
    """
    pos = np.array(
        [pos_vel["stateEstimate.x"], pos_vel["stateEstimate.y"], pos_vel["stateEstimate.z"]],
        dtype=np.float32,
    )
    vel = np.array(
        [pos_vel["stateEstimate.vx"], pos_vel["stateEstimate.vy"], pos_vel["stateEstimate.vz"]],
        dtype=np.float32,
    )
    rpy = np.deg2rad(
        [
            ori_rate["stateEstimate.roll"],
            ori_rate["stateEstimate.pitch"],
            ori_rate["stateEstimate.yaw"],
        ]
    )
    quat = R.from_euler("xyz", rpy).as_quat().astype(np.float32)
    ang_vel = np.deg2rad([ori_rate["gyro.x"], ori_rate["gyro.y"], ori_rate["gyro.z"]]).astype(
        np.float32
    )
    return {"pos": pos, "quat": quat, "vel": vel, "ang_vel": ang_vel}
