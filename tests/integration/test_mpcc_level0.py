"""Integration test for the MPCC controller on level 0.

Building the acados solver takes a while and needs the acados code generation toolchain, so this
test is opt-in:

    RUN_MPCC_BUILD=1 pytest tests/integration/test_mpcc_level0.py
"""

import os
from pathlib import Path

import gymnasium
import pytest
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MPCC_BUILD") != "1",
    reason="set RUN_MPCC_BUILD=1 to build the acados solver and fly the MPCC",
)


@pytest.mark.integration
def test_mpcc_finishes_level0():
    """The MPCC controller has to complete the level-0 track."""
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.render = False
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / "lsy_drone_racing/control/my_controller.py"
    )
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=0,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset(seed=0)
    ctrl = ctrl_cls(obs, info, config)
    steps = 0
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    assert int(obs["n_gates_passed"]) == int(obs["gate_sequence"].shape[0])
    assert steps / config.env.freq < 20.0
