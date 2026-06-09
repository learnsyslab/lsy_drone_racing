"""Unit tests for the RaceCoreEnv class.

These tests focus on the race-core logic (obs, reward, terminated, truncated, close, gate-pass
detection) rather than physics or rendering. They exercise both the public properties and a few
"hidden" helpers (``_step_env``, ``_disabled_drones``) by crafting the env data directly, following
the same pattern used in the render integration tests.
"""

import os
from pathlib import Path
from typing import Any

os.environ["SCIPY_ARRAY_API"] = "1"

import gymnasium
import jax.numpy as jp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.race_core import _update_disabled_drones, _update_target_gates
from lsy_drone_racing.utils import load_config

CONFIG_PATH = Path(__file__).parents[3] / "config"


def make_env(config_name: str = "level0.toml", **overrides: Any) -> gymnasium.Env:
    """Build a single-drone env with sensible defaults that individual tests can override."""
    config = load_config(CONFIG_PATH / config_name)
    kwargs = dict(
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    kwargs.update(overrides)
    return gymnasium.make("DroneRacing-v0", **kwargs)


@pytest.mark.unit
def test_close_after_reset():
    """close() after a normal reset must not raise."""
    env = make_env()
    env.reset()
    env.close()


@pytest.mark.unit
def test_close_without_reset():
    """close() without ever calling reset() must not raise."""
    env = make_env()
    env.close()


@pytest.mark.unit
def test_obs_structure_and_initial_values():
    """obs() returns the expected keys, lives in observation_space, and starts at gate 0."""
    env = make_env()
    obs, _ = env.reset()
    expected_keys = {
        "pos",
        "quat",
        "vel",
        "ang_vel",
        "target_gate",
        "gates_pos",
        "gates_quat",
        "gates_visited",
        "obstacles_pos",
        "obstacles_visited",
    }
    assert set(obs.keys()) == expected_keys
    # Single-drone make() squeezes leading (world, drone) dims: pos is (3,), gates_pos is
    # (n_gates, 3), target_gate is a 0-d scalar.
    assert np.asarray(obs["pos"]).shape == (3,)
    assert np.asarray(obs["gates_pos"]).ndim == 2
    assert np.asarray(obs["gates_pos"]).shape[1] == 3
    assert int(np.asarray(obs["target_gate"]).item()) == 0
    env.close()


@pytest.mark.unit
def test_obs_returns_nominal_when_out_of_sensor_range():
    """With sensor_range=0, gates/obstacles are not visited and obs returns nominal poses."""
    env = make_env(sensor_range=0.0)
    obs, _ = env.reset()
    assert not bool(jp.any(obs["gates_visited"])), "no gate should be visited"
    assert not bool(jp.any(obs["obstacles_visited"])), "no obstacle should be visited"

    nominal_gate_pos = np.asarray(env.unwrapped.data.nominal_gates_pos[0])
    nominal_obstacle_pos = np.asarray(env.unwrapped.data.nominal_obstacles_pos[0])
    np.testing.assert_allclose(np.asarray(obs["gates_pos"]), nominal_gate_pos, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(obs["obstacles_pos"]), nominal_obstacle_pos, rtol=1e-5)
    env.close()


@pytest.mark.unit
def test_obs_returns_real_pose_when_in_sensor_range():
    """With a huge sensor range, obs returns the actual mocap pose for each gate."""
    env = make_env(sensor_range=100.0)
    obs, _ = env.reset()
    assert bool(jp.all(obs["gates_visited"])), "all gates should be visited"
    assert bool(jp.all(obs["obstacles_visited"])), "all obstacles should be visited"

    real_gates_pos = env.unwrapped.data.gates_pos[0]
    real_obstacles_pos = env.unwrapped.data.obstacles_pos[0]
    np.testing.assert_allclose(np.asarray(obs["gates_pos"]), real_gates_pos, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(obs["obstacles_pos"]), real_obstacles_pos, rtol=1e-5)
    env.close()


@pytest.mark.unit
def test_level3_nominal_positions_set_after_randomization():
    """Full track randomization must populate nominal positions, not leave them zeroed from TOML.

    In level 3 the gate/obstacle x,y coords in the TOML are all 0.0. After reset the nominal
    positions must reflect the randomly generated layout, not those zeros.
    """
    env = make_env("level3.toml", sensor_range=0.0)
    env.reset()
    nominal_gates_pos = np.asarray(env.unwrapped.data.nominal_gates_pos)
    nominal_obstacles_pos = np.asarray(env.unwrapped.data.nominal_obstacles_pos)
    # After full-track randomization every gate/obstacle should have been moved
    assert not np.all(nominal_gates_pos[..., :2] == 0.0), "nominal gate pos not updated"
    assert not np.all(nominal_obstacles_pos[..., :2] == 0.0), "nominal obstacle pos not updated"
    env.close()


@pytest.mark.unit
def test_level3_initial_obs_differs_from_real_positions():
    """Level 3 initial obs must report nominal positions, which differ from actual (perturbed) ones.

    With sensor_range=0 every gate is out of range, so obs reports nominal positions. Level 3 also
    applies per-gate perturbation randomization on top of the full-track layout, so the real gate
    positions differ from the nominal ones.
    """
    env = make_env("level3.toml", sensor_range=0.0)
    env.reset()
    obs, _ = env.reset()
    assert not bool(np.any(obs["gates_visited"])), "no gate should be visited with sensor_range=0"
    obs_gates_pos = np.asarray(obs["gates_pos"])
    real_gates_pos = np.asarray(env.unwrapped.data.gates_pos[0])
    assert not np.allclose(obs_gates_pos, real_gates_pos, atol=1e-6), "obs pos is using real pos"
    env.close()


@pytest.mark.unit
def test_terminated_false_after_reset():
    """Fresh reset: no drone is disabled, so terminated is False."""
    env = make_env()
    env.reset()
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert not terminated, "terminated should be False after reset"
    env.close()


@pytest.mark.unit
def test_terminated_true_when_target_gate_negative():
    """Setting target_gate = -1 marks the drone disabled on the next step."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(target_gate=data.target_gate.at[...].set(-1))
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert terminated, "terminated should be True when target_gate is -1"
    env.close()


@pytest.mark.unit
def test_disabled_drones_out_of_bounds():
    """``_disabled_drones`` flags a drone positioned above pos_limit_high."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    # Place the drone well above the z upper limit (2.5). Shape: (n_worlds, n_drones, 3).
    pos = env.unwrapped.data.sim_data.states.pos.at[..., 2].set(5.0)
    data = data.replace(
        sim_data=data.sim_data.replace(states=data.sim_data.states.replace(pos=pos))
    )
    contact_check_fn = env.unwrapped.build_contact_check_fn()
    data = _update_disabled_drones(data, contact_check_fn(data))
    assert bool(jp.all(data.disabled_drones)), "drone above pos_limit_high should be disabled"
    env.close()


@pytest.mark.unit
def test_disabled_drones_nominal_not_disabled():
    """``_disabled_drones`` does not flag a drone at its nominal starting pose without contacts."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    contact_check_fn = env.unwrapped.build_contact_check_fn()
    data = _update_disabled_drones(data, contact_check_fn(data))
    assert not bool(jp.any(data.disabled_drones)), "drones with no contacts should not be disabled"
    env.close()


@pytest.mark.unit
def test_disabled_drones_on_contact():
    """A masked contact (e.g. hitting an obstacle) disables the drone."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    # Set the drones into contact with obstacles by placing them at the same position
    pos = env.unwrapped.sim.data.states.pos.at[...].set(data.obstacles_pos[:, 0, :])
    data = data.replace(
        sim_data=data.sim_data.replace(states=data.sim_data.states.replace(pos=pos))
    )
    contact_check_fn = env.unwrapped.build_contact_check_fn()
    data = _update_disabled_drones(data, contact_check_fn(data))
    assert bool(jp.all(data.disabled_drones)), "drone with collisions should be disabled"
    env.close()


@pytest.mark.unit
def test_truncated_false_after_reset():
    """Fresh reset: steps=0, so truncated is False."""
    env = make_env()
    env.reset()
    _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert not bool(jp.any(truncated)), "truncated should be False after reset"
    env.close()


@pytest.mark.unit
def test_truncated_on_timeout_does_not_terminate():
    """Bumping steps to max_episode_steps flips truncated True while terminated stays False.

    This pins down two things at once: (1) ``truncated()`` fires on timeout, and (2) timeout is
    a separate signal from termination — despite the intuition that "for a single drone
    truncated == terminated", the two branches in race_core are genuinely independent.
    """
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(steps=data.steps.at[...].set(data.max_episode_steps))
    _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert bool(jp.all(truncated)), "truncated should be True on timeout"
    env.close()


@pytest.mark.unit
def test_gate_pass_increments_target_gate():
    """Crossing the target gate's plane makes ``_update_target_gates`` increment target_gate."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data

    # Current target gate (index 0 after reset). gates_quat is stored in xyzw order.
    gate_idx = 0
    gate_pos = np.asarray(data.gates_pos[0, gate_idx])
    gate_quat_xyzw = np.asarray(data.gates_quat[0, gate_idx])
    # Gates are crossed from -x to +x in the local gate frame (see gate_passed docstring).
    forward = R.from_quat(gate_quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))

    behind = gate_pos - 0.05 * forward  # last drone position: just before the gate plane
    front = gate_pos + 0.05 * forward  # current drone position: just past it

    # Craft data so that last_drone_pos is "behind" and current sim pos is "front".
    new_last = data.last_drone_pos.at[0, 0, :].set(jp.asarray(behind))
    new_pos = data.sim_data.states.pos.at[0, 0, :].set(jp.asarray(front))
    new_sim_data = data.sim_data.replace(states=data.sim_data.states.replace(pos=new_pos))
    env.unwrapped.data = data.replace(last_drone_pos=new_last, sim_data=new_sim_data)

    # Call _update_target_gates directly so physics doesn't overwrite our crafted positions.
    new_data = _update_target_gates(env.unwrapped.data)
    assert int(np.asarray(new_data.target_gate[0, 0])) == 1
    env.close()


@pytest.mark.unit
def test_gate_not_passed_without_crossing():
    """Moving around without crossing the gate plane leaves target_gate unchanged."""
    env = make_env()
    env.reset()
    data = env.unwrapped.data
    assert int(np.asarray(data.target_gate[0, 0])) == 0
    # Nominal step without any crafted crossing: drone still on the takeoff pad.
    new_data = _update_target_gates(data)
    assert int(np.asarray(new_data.target_gate[0, 0])) == 0
    env.close()


@pytest.mark.unit
def test_gate_pass_non_target_gate_does_not_increment():
    """Crossing a gate that is not the current target must not increment target_gate."""
    from scipy.spatial.transform import Rotation as R

    env = make_env()
    env.reset()
    data = env.unwrapped.data
    n_gates = data.gates_pos.shape[1]
    assert n_gates >= 2, "need at least 2 gates for this test"
    assert int(np.asarray(data.target_gate[0, 0])) == 0

    # Straddle gate 1 (the *next* gate) while target_gate is still 0.
    non_target_idx = 1
    gate_pos = np.asarray(data.gates_pos[0, non_target_idx])
    gate_quat_xyzw = np.asarray(data.gates_quat[0, non_target_idx])
    forward = R.from_quat(gate_quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))

    behind = gate_pos - 0.05 * forward
    front = gate_pos + 0.05 * forward

    new_last = data.last_drone_pos.at[0, 0, :].set(jp.asarray(behind))
    new_pos = data.sim_data.states.pos.at[0, 0, :].set(jp.asarray(front))
    new_sim_data = data.sim_data.replace(states=data.sim_data.states.replace(pos=new_pos))
    env.unwrapped.data = data.replace(last_drone_pos=new_last, sim_data=new_sim_data)

    new_data = _update_target_gates(env.unwrapped.data)
    assert int(np.asarray(new_data.target_gate[0, 0])) == 0
    env.close()


@pytest.mark.unit
def test_gate_not_passed_in_reverse():
    """Flying through the gate from +x to -x (reverse) must not count as a pass."""
    from scipy.spatial.transform import Rotation as R

    env = make_env()
    env.reset()
    data = env.unwrapped.data

    gate_pos = np.asarray(data.gates_pos[0, 0])
    gate_quat_xyzw = np.asarray(data.gates_quat[0, 0])
    forward = R.from_quat(gate_quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))

    # Reverse crossing: last position is in front of the gate, current is behind.
    front = gate_pos + 0.05 * forward
    behind = gate_pos - 0.05 * forward

    new_last = data.last_drone_pos.at[0, 0, :].set(jp.asarray(front))
    new_pos = data.sim_data.states.pos.at[0, 0, :].set(jp.asarray(behind))
    new_sim_data = data.sim_data.replace(states=data.sim_data.states.replace(pos=new_pos))
    env.unwrapped.data = data.replace(last_drone_pos=new_last, sim_data=new_sim_data)

    new_data = _update_target_gates(env.unwrapped.data)
    assert int(np.asarray(new_data.target_gate[0, 0])) == 0
    env.close()


@pytest.mark.unit
def test_gate_not_passed_when_outside_gate_box():
    """Crossing the gate plane but far outside the gate opening must not count as a pass."""
    from scipy.spatial.transform import Rotation as R

    env = make_env()
    env.reset()
    data = env.unwrapped.data

    gate_pos = np.asarray(data.gates_pos[0, 0])
    gate_quat_xyzw = np.asarray(data.gates_quat[0, 0])
    rot = R.from_quat(gate_quat_xyzw)
    forward = rot.apply(np.array([1.0, 0.0, 0.0]))
    # Offset along the gate's local y-axis, well outside the gate box (half-width is 0.225 m).
    sideways = rot.apply(np.array([0.0, 1.0, 0.0]))

    # Cross the plane in the correct direction, but 2 m to the side of the opening.
    behind = gate_pos - 0.05 * forward + 2.0 * sideways
    front = gate_pos + 0.05 * forward + 2.0 * sideways

    new_last = data.last_drone_pos.at[0, 0, :].set(jp.asarray(behind))
    new_pos = data.sim_data.states.pos.at[0, 0, :].set(jp.asarray(front))
    new_sim_data = data.sim_data.replace(states=data.sim_data.states.replace(pos=new_pos))
    env.unwrapped.data = data.replace(last_drone_pos=new_last, sim_data=new_sim_data)

    new_data = _update_target_gates(env.unwrapped.data)
    assert int(np.asarray(new_data.target_gate[0, 0])) == 0
    env.close()


@pytest.mark.unit
def test_gate_pass_at_last_gate_clamps_to_negative_one():
    """Passing the final gate must set target_gate to -1 (course finished sentinel)."""
    from scipy.spatial.transform import Rotation as R

    env = make_env()
    env.reset()
    data = env.unwrapped.data
    n_gates = data.gates_pos.shape[1]

    # Pre-advance target_gate to the last gate so _update_target_gates will check against it.
    last_idx = n_gates - 1
    env.unwrapped.data = data.replace(target_gate=data.target_gate.at[0, 0].set(last_idx))
    data = env.unwrapped.data

    # Craft a forward crossing of the last gate.
    gate_pos = np.asarray(data.gates_pos[0, last_idx])
    gate_quat_xyzw = np.asarray(data.gates_quat[0, last_idx])
    forward = R.from_quat(gate_quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))

    behind = gate_pos - 0.05 * forward
    front = gate_pos + 0.05 * forward

    new_last = data.last_drone_pos.at[0, 0, :].set(jp.asarray(behind))
    new_pos = data.sim_data.states.pos.at[0, 0, :].set(jp.asarray(front))
    new_sim_data = data.sim_data.replace(states=data.sim_data.states.replace(pos=new_pos))
    env.unwrapped.data = data.replace(last_drone_pos=new_last, sim_data=new_sim_data)

    new_data = _update_target_gates(env.unwrapped.data)
    assert int(np.asarray(new_data.target_gate[0, 0])) == -1
    env.close()


@pytest.mark.unit
def test_single_compile():
    env = make_env()
    env.reset()
    env.step(env.action_space.sample())
    reset_cache_size = env.unwrapped._reset._cache_size()
    step_cache_size = env.unwrapped._step._cache_size()
    env.reset()  # This reset should hit the cache and not cause a second compile.
    env.step(env.action_space.sample())
    assert env.unwrapped._reset._cache_size() == reset_cache_size, "unexpected reset recompilation"
    assert env.unwrapped._step._cache_size() == step_cache_size, "unexpected step recompilation"
    env.close()
