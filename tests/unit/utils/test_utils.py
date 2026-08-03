from pathlib import Path

import numpy as np
import pytest
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.envs.utils import gate_passed, load_gate_order
from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.unit
def test_load_config():
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    assert isinstance(config, ConfigDict), f"Config file is not a ConfigDict: {type(config)}"


@pytest.mark.unit
def test_load_controller():
    c = load_controller(Path(__file__).parents[3] / "lsy_drone_racing/control/state_controller.py")
    assert issubclass(c, Controller), f"Controller {c} is not a subclass of `Controller`"


@pytest.mark.unit
def test_gate_pass():
    # TODO: Check accelerated function in RaceCore instead
    gate_pos = np.array([0, 0, 0])
    gate_quat = R.identity().as_quat()
    gate_size = np.array([1, 1])
    # Test passing through the gate
    drone_pos, last_drone_pos = np.array([1, 0, 0]), np.array([-1, 0, 0])
    assert gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, False, gate_size)
    # Test passing outside the gate boundaries
    drone_pos, last_drone_pos = np.array([-2, 2, 0]), np.array([-1, 2, 0])
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, False, gate_size)
    # Test passing close to the gate
    drone_pos, last_drone_pos = np.array([1, 0.51, 0]), np.array([-1, 0.51, 0])
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, False, gate_size)
    # Test passing opposite direction
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, False, gate_size)
    # Test with rotated gate
    rotated_gate_quat = R.from_euler("xyz", [0, np.pi / 4, 0]).as_quat()
    drone_pos, last_drone_pos = np.array([0.5, 0.5, 0]), np.array([-0.5, -0.5, 0])
    assert gate_passed(drone_pos, last_drone_pos, gate_pos, rotated_gate_quat, False, gate_size)
    # Test with moved gate
    moved_gate_pos = np.array([1, 1, 1])
    drone_pos, last_drone_pos = np.array([2, 1, 1]), np.array([0, 1, 1])
    assert gate_passed(drone_pos, last_drone_pos, moved_gate_pos, gate_quat, False, gate_size)
    # Test not crossing the plane
    drone_pos, last_drone_pos = np.array([-0.5, 0, 0]), np.array([-1, 0, 0])
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, False, gate_size)

    # Test reverse passing when requested
    drone_pos, last_drone_pos = np.array([-1, 0, 0]), np.array([1, 0, 0])
    assert gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, True, gate_size)
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, False, gate_size)

    # Test vectorized reverse flags
    drone_pos = np.array([[1, 0, 0], [-1, 0, 0]])
    last_drone_pos = np.array([[-1, 0, 0], [1, 0, 0]])
    gate_pos = np.zeros((2, 3))
    gate_quat = np.tile(R.identity().as_quat(), (2, 1))
    reverse = np.array([False, True])
    passed = gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, reverse, gate_size)
    np.testing.assert_array_equal(np.asarray(passed), np.array([True, True]))


@pytest.mark.unit
def test_load_gate_order():
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    config.env.track.gate_order = [1, -2, 2, -1]
    gate_sequence, gate_sequence_direction = load_gate_order(
        config.env.track, len(config.env.track.gates)
    )
    np.testing.assert_array_equal(gate_sequence, np.array([0, 1, 1, 0]))
    np.testing.assert_array_equal(gate_sequence_direction, np.array([1, -1, 1, -1]))


@pytest.mark.unit
def test_load_gate_correctness():
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    config.env.track.gate_order = [0]  # 0 is not allowed, since the sign is ambiguous
    with pytest.raises(AssertionError):
        _, _ = load_gate_order(config.env.track, len(config.env.track.gates))

    config.env.track.gate_order = [10]  # 10 is out of bounds
    with pytest.raises(AssertionError):
        _, _ = load_gate_order(config.env.track, len(config.env.track.gates))
