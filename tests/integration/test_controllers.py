import copy
from pathlib import Path

import gymnasium
import numpy as np
import pytest
from crazyflow.dynamics import Dynamics
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.integration
@pytest.mark.parametrize("controller_file", ["state_controller.py"])
def test_controllers(controller_file: str):
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    config.sim.dynamics = Dynamics.first_principles.value
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
@pytest.mark.parametrize("dynamics", Dynamics)
def test_attitude_controller(dynamics: Dynamics, controller: str):
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    config.sim.dynamics = dynamics.value
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
@pytest.mark.parametrize("dynamics", [Dynamics.first_principles])
def test_trajectory_controller_finish(yaw: float, dynamics: Dynamics):
    """Test if the trajectory controller can finish the track.

    To catch bugs that only occur with orientations other than the unit quaternion, we test if the
    controller can finish the track with different desired yaws.

    Does not work for sys_id dynamics mode, since it assumes a 0 yaw angle.
    """
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.dynamics = dynamics.value
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
        # Quadrotor should be able to finish the track regardless of yaw
        action[9:13] = R.from_euler("z", yaw).as_quat()
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    assert obs["n_gates_passed"] == obs["gate_sequence"].shape[0], (
        "Trajectory controller failed to complete the track"
    )


@pytest.mark.integration
def test_multi_drone_controllers():
    """Test if the multi-drone example controllers complete the track together."""
    config = load_config(Path(__file__).parents[2] / "config/multi_level0.toml")
    control_path = Path(__file__).parents[2] / "lsy_drone_racing/control"
    ctrl_classes = [load_controller(control_path / ctrl["file"]) for ctrl in config.controller]
    ctrl_freqs = np.array([kwargs["freq"] for kwargs in config.env.kwargs])
    env_freq = int(ctrl_freqs.max())
    periods = env_freq // ctrl_freqs
    env = gymnasium.make(
        "MultiDroneRacing-v0",
        freq=env_freq,
        sim_config=config.sim,
        sensor_range=config.env.kwargs[0]["sensor_range"],
        control_mode=config.env.kwargs[0]["control_mode"],
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=1337,
    )
    env = JaxToNumpy(env)

    obs, info = env.reset()
    ctrls = []
    for rank, ctrl_cls in enumerate(ctrl_classes):
        ctrl_config = copy.deepcopy(config)
        ctrl_config.env.freq = config.env.kwargs[rank]["freq"]
        ctrls.append(ctrl_cls(obs, info | {"rank": rank}, ctrl_config))
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    step = 0
    while True:
        active = (step % periods == 0) & ~np.asarray(env.unwrapped.data.disabled_drones[0])
        ctrl_infos = [info | {"rank": rank} for rank in range(len(ctrls))]
        for rank in np.flatnonzero(active):
            actions[rank] = ctrls[rank].compute_control(obs, ctrl_infos[rank])
        obs, reward, terminated, truncated, info = env.step(actions)
        for rank in np.flatnonzero(active):
            ctrls[rank].step_callback(
                actions[rank], obs, reward, terminated, truncated, ctrl_infos[rank]
            )
        step += 1
        if terminated or truncated:
            break
    env.close()
    assert np.all(obs["n_gates_passed"] == obs["gate_sequence"].shape[-1]), (
        "Multi-drone controllers failed to complete the track"
    )
