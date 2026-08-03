from pathlib import Path

import gymnasium
import numpy as np
import pytest
from drone_models import available_models
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.integration
@pytest.mark.parametrize("controller_file", ["state_controller.py"])
def test_controllers(controller_file: str):
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    config.sim.physics = "first_principles"
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / f"lsy_drone_racing/control/{controller_file}"
    )
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=1337,
    )
    env = JaxToNumpy(env)

    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info, config)
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    # No assertion for finishing the race


@pytest.mark.integration
@pytest.mark.parametrize("controller", ["controller", "mpc", "rl"])  # TODO add rl when available
@pytest.mark.parametrize("physics", available_models.keys())
def test_attitude_controller(physics: str, controller: str):
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    config.sim.physics = physics
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / f"lsy_drone_racing/control/attitude_{controller}.py"
    )
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=1337,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info, config)
    while True:
        action = ctrl.compute_control(obs, info).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    env.close()
    assert obs["n_gates_passed"] == obs["gate_sequence"].shape[0], (
        "Attitude controller failed to complete the track"
    )


@pytest.mark.integration
@pytest.mark.parametrize("yaw", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
@pytest.mark.parametrize("physics", ["first_principles"])
def test_trajectory_controller_finish(yaw: float, physics: str):
    """Test if the trajectory controller can finish the track.

    To catch bugs that only occur with orientations other than the unit quaternion, we test if the
    controller can finish the track with different desired yaws.

    Does not work for sys_id physics mode, since it assumes a 0 yaw angle.
    """
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.physics = physics
    config.sim.gui = False
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / "lsy_drone_racing/control/state_controller.py"
    )
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=1337,
    )
    env = JaxToNumpy(env)

    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info, config)
    while True:
        action = ctrl.compute_control(obs, info)
        action[9] = yaw  # Quadrotor should be able to finish the track regardless of yaw
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    assert obs["n_gates_passed"] == obs["gate_sequence"].shape[0], (
        "Trajectory controller failed to complete the track"
    )
