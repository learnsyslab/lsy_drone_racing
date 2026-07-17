from pathlib import Path

import gymnasium
import jax
import jax.numpy as jp
import pytest
from drone_models import available_models

import lsy_drone_racing  # noqa: F401, environment registrations
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import load_config
from tests.conftest import skip_if_headless

CONFIG_FILES = {
    "DroneRacing-v0": ["level0.toml", "level1.toml", "level2.toml"],
    "MultiDroneRacing-v0": ["multi_level0.toml", "multi_level2.toml"],
}
DEVICES = ["cpu", "gpu"]


def available_backends() -> list[str]:
    """Return list of available JAX backends."""
    backends = []
    for backend in ["tpu", "gpu", "cpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


def skip_unavailable_device(device: str):
    if device not in available_backends():
        pytest.skip(f"{device} device not available")


@pytest.mark.parametrize("physics", available_models.keys())
@pytest.mark.parametrize("config_file", CONFIG_FILES["DroneRacing-v0"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.integration
def test_single_drone_envs(config_file: str, physics: str, device: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics  # override physics mode
    assert hasattr(config.env, "id"), "Environment ID is not set"
    skip_unavailable_device(device)

    kwargs = {
        "freq": config.env.freq,
        "sim_config": config.sim,
        "sensor_range": config.env.sensor_range,
        "track": config.env.track,
        "disturbances": config.env.get("disturbances"),
        "randomizations": config.env.get("randomizations"),
        "seed": config.env.seed,
        "device": device,
    }
    if "n_drones" in config.env:
        kwargs["n_drones"] = config.env.n_drones

    device = jax.devices(device)[0]
    env = gymnasium.make("DroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()

    kwargs["num_envs"] = 3
    env = gymnasium.make_vec("DroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()


@pytest.mark.parametrize("physics", available_models.keys())
@pytest.mark.parametrize("config_file", CONFIG_FILES["MultiDroneRacing-v0"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.integration
def test_multi_drone_envs(config_file: str, physics: str, device: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics
    assert hasattr(config.env, "id"), "Environment ID is not set"
    skip_unavailable_device(device)

    kwargs = {
        "freq": config.env.kwargs[0]["freq"],
        "sim_config": config.sim,
        "sensor_range": config.env.kwargs[0]["sensor_range"],
        "track": config.env.track,
        "disturbances": config.env.get("disturbances"),
        "randomizations": config.env.get("randomizations"),
        "seed": config.env.seed,
        "device": device,
    }
    if "n_drones" in config.env:
        kwargs["n_drones"] = config.env.n_drones

    device = jax.devices(device)[0]
    env = gymnasium.make("MultiDroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()
    kwargs["num_envs"] = 3
    env = gymnasium.make_vec("MultiDroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()


@pytest.mark.parametrize("config_file", ["level2.toml", "level3.toml"])
def test_vector_envs_randomization(config_file: str):
    """Test track randomization works correctly with vectorized environments."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)

    env: DroneRaceEnv = gymnasium.make_vec(
        "DroneRacing-v0",
        num_envs=2,
        control_mode="attitude",
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        randomizations=config.env.get("randomizations"),
    )

    def get_obj_data(env: DroneRaceEnv) -> jax.Array:
        return env.data.gates_pos, env.data.gates_quat, env.data.obstacles_pos

    # Backup nominal track data (before reset, gates_pos is initialized to nominal values).
    nominal_gates_pos, nominal_gates_quat, nominal_obstacles_pos = get_obj_data(env)

    env.reset()

    # Check if track is randomized
    gates_pos, gates_quat, obstacles_pos = get_obj_data(env)
    drone_pos = env.data.sim_data.states.pos[:, 0, :]
    nominal_drone_pos = env.sim.default_data.states.pos[:, 0, :]
    assert not jp.allclose(gates_pos, nominal_gates_pos), "gates_pos not randomized"
    assert not jp.allclose(gates_quat, nominal_gates_quat), "gates_quat not randomized"
    assert not jp.allclose(obstacles_pos, nominal_obstacles_pos), "obstacles_pos not randomized"
    assert not jp.allclose(drone_pos, nominal_drone_pos), "drone_pos not randomized"

    action = jp.tile(jp.array([0.0, 0.0, 0.0, 0.6]), (2, 1))  # ensure drone doesn't crash
    env.step(action)
    gates_pos_1, gates_quat_1, obstacles_pos_1 = get_obj_data(env)

    # Force reset 2nd world
    env.data = env.data.replace(marked_for_reset=env.data.marked_for_reset.at[1].set(True))

    # Step once to trigger autoreset
    env.step(action)
    gates_pos_2, gates_quat_2, obstacles_pos_2 = get_obj_data(env)
    drone_pos = env.data.sim_data.states.pos[:, 0, :]
    nominal_drone_pos = env.sim.default_data.states.pos[:, 0, :]

    # Check if track is re-randomized for the 2nd world but not for the 1st world
    assert jp.allclose(gates_pos_2[0], gates_pos_1[0]), "unexpected gates_pos[0] change"
    assert jp.allclose(gates_quat_2[0], gates_quat_1[0]), "unexpected gates_quat[0] change"
    assert jp.allclose(obstacles_pos_2[0], obstacles_pos_1[0]), "unexpected obstacles_pos[0] change"

    assert not jp.allclose(gates_pos_2[1], nominal_gates_pos[1]), "gates_pos[1] not randomized"
    assert not jp.allclose(gates_quat_2[1], nominal_gates_quat[1]), "gates_quat[1] not randomized"
    assert not jp.allclose(obstacles_pos_2[1], nominal_obstacles_pos[1]), (
        "obstacles_pos[1] not randomized"
    )
    assert not jp.allclose(drone_pos[1], nominal_drone_pos[1]), "drone_pos[1] not randomized"

    assert not jp.allclose(gates_pos_2[1], gates_pos_1[1]), "gates_pos[1] not re-randomized"
    assert not jp.allclose(gates_quat_2[1], gates_quat_1[1]), "gates_quat[1] not re-randomized"
    assert not jp.allclose(obstacles_pos_2[1], obstacles_pos_1[1]), (
        "obstacles_pos[1] not re-randomized"
    )

    env.close()


@skip_if_headless
@pytest.mark.integration
def test_render():
    """Smoke test: env.render() runs at every lifecycle stage for DroneRacing-v0."""
    config = load_config(Path(__file__).parents[2] / "config" / "level0.toml")
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    # Right after reset
    env.reset()
    env.render()
    # Mid-episode: render after several steps
    for _ in range(5):
        env.step(env.action_space.sample())
        env.render()

    # Force terminal state
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(
        n_gates_passed=data.n_gates_passed.at[...].set(data.gate_sequence.shape[0])
    )
    env.step(env.action_space.sample())
    assert jp.all(env.unwrapped.data.disabled_drones), "drone not actually disabled"
    env.render()
    env.close()


@skip_if_headless
@pytest.mark.integration
def test_render_multi_drone():
    """Smoke test: env.render() runs at every lifecycle stage for MultiDroneRacing-v0."""
    config = load_config(Path(__file__).parents[2] / "config" / "multi_level0.toml")
    env = gymnasium.make(
        "MultiDroneRacing-v0",
        freq=config.env.kwargs[0]["freq"],
        sim_config=config.sim,
        sensor_range=config.env.kwargs[0]["sensor_range"],
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    # Right after reset
    env.reset()
    env.render()
    # Mid-episode: render after several steps
    for _ in range(5):
        env.step(env.action_space.sample())
        env.render()

    # Force only drone 0 (in world 0) into a terminal state by setting its gate-pass count to the
    # sequence length. The other drones stay active, so autoreset will not fire (it
    # only triggers when all drones in a world are disabled). The next step will
    # mark drone 0 as disabled and warp it below the ground via
    # `_warp_disabled_drones`, exercising the renderer with a mix of active and
    # warped drones.
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(
        n_gates_passed=data.n_gates_passed.at[0, 0].set(data.gate_sequence.shape[0])
    )
    env.step(env.action_space.sample())

    # Verify drone 0 is disabled and the others are still active.
    disabled = env.unwrapped.data.disabled_drones
    assert bool(disabled[0, 0]), "drone 0 should be disabled after finishing the gate sequence"
    assert not bool(jp.all(disabled)), "only drone 0 should be disabled, not all drones"

    env.render()

    # Force ALL drones into a terminal state by setting every gate-pass count to the sequence
    # length. On the next step, every drone will be disabled and (because the env
    # has autoreset enabled) the world will be auto-reset before render is
    # called again.
    data = env.unwrapped.data
    env.unwrapped.data = data.replace(
        n_gates_passed=data.n_gates_passed.at[...].set(data.gate_sequence.shape[0])
    )
    env.step(env.action_space.sample())
    env.render()
    env.close()
