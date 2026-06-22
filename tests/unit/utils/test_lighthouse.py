"""Unit tests for the hardware-free lighthouse helpers."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils.lighthouse import ORI_RATE_VARS, POS_VEL_VARS, decode_state


@pytest.mark.unit
def test_decode_state_values_and_types():
    pos_vel = {
        "stateEstimate.x": 1.0,
        "stateEstimate.y": 2.0,
        "stateEstimate.z": 3.0,
        "stateEstimate.vx": 0.1,
        "stateEstimate.vy": 0.2,
        "stateEstimate.vz": 0.3,
    }
    ori_rate = {
        "stateEstimate.roll": 0.0,
        "stateEstimate.pitch": 0.0,
        "stateEstimate.yaw": 90.0,
        "gyro.x": 10.0,
        "gyro.y": -20.0,
        "gyro.z": 30.0,
    }
    obs = decode_state(pos_vel, ori_rate)

    assert set(obs) == {"pos", "quat", "vel", "ang_vel"}
    for key, value in obs.items():
        assert value.dtype == np.float32, f"{key} must be float32"
    assert obs["pos"].shape == (3,)
    assert obs["quat"].shape == (4,)
    assert obs["vel"].shape == (3,)
    assert obs["ang_vel"].shape == (3,)

    np.testing.assert_allclose(obs["pos"], [1.0, 2.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(obs["vel"], [0.1, 0.2, 0.3], rtol=1e-6)
    # gyro is reported in deg/s and must be converted to rad/s.
    np.testing.assert_allclose(obs["ang_vel"], np.deg2rad([10.0, -20.0, 30.0]), rtol=1e-6)
    # 90 deg yaw about z, returned in xyzw convention to match the simulation observation.
    expected_quat = R.from_euler("z", 90, degrees=True).as_quat()
    np.testing.assert_allclose(obs["quat"], expected_quat, atol=1e-6)


@pytest.mark.unit
def test_decode_state_identity_quat():
    pos_vel = dict.fromkeys(POS_VEL_VARS, 0.0)
    ori_rate = dict.fromkeys(ORI_RATE_VARS, 0.0)
    obs = decode_state(pos_vel, ori_rate)
    # Zero roll/pitch/yaw -> identity quaternion (0, 0, 0, 1) in xyzw.
    np.testing.assert_allclose(obs["quat"], [0.0, 0.0, 0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(obs["ang_vel"], [0.0, 0.0, 0.0], atol=1e-7)


@pytest.mark.unit
def test_log_var_blocks_fit_crtp_limit():
    # A single CRTP log block carries at most ~26 bytes; six 4-byte floats = 24 bytes per block.
    assert len(POS_VEL_VARS) == 6
    assert len(ORI_RATE_VARS) == 6
    assert len(POS_VEL_VARS) * 4 <= 26
    assert len(ORI_RATE_VARS) * 4 <= 26
