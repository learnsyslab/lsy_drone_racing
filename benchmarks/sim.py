from __future__ import annotations

import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

load_config_code = f"""
from pathlib import Path

from lsy_drone_racing.utils import load_config

config = load_config(Path('{Path(__file__).parents[1] / "config/level2.toml"}'))
"""

env_setup_code = """
import gymnasium
import jax

import lsy_drone_racing

env = gymnasium.make_vec(
    config.env.id,
    num_envs={num_envs},
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    seed=config.env.seed,
    device='{device}',
)

# JIT compile the reset and step functions
env.reset()
env.step(env.action_space.sample())
jax.block_until_ready(env.unwrapped.data)
# JIT masked reset (used in autoreset)
mask = env.unwrapped.data.marked_for_reset
mask = mask.at[0].set(True)
env.unwrapped._reset(mask=mask)  # enforce masked reset compile
jax.block_until_ready(env.unwrapped.data)
env.action_space.seed(2)
"""

attitude_env_setup_code = """
import gymnasium
import jax

import lsy_drone_racing

env = gymnasium.make_vec('DroneRacingAttitude-v0',
    num_envs={num_envs},
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    seed=config.env.seed,
    device='{device}',
)

# JIT compile the reset and step functions
env.reset()
env.step(env.action_space.sample())
jax.block_until_ready(env.unwrapped.data)
# JIT masked reset (used in autoreset)
mask = env.unwrapped.data.marked_for_reset
mask = mask.at[0].set(True)
env.unwrapped._reset(mask=mask)  # enforce masked reset compile
jax.block_until_ready(env.unwrapped.data)
env.action_space.seed(2)
"""

load_multi_drone_config_code = f"""
from pathlib import Path

from lsy_drone_racing.utils import load_config

config = load_config(Path('{Path(__file__).parents[1] / "config/multi_level3.toml"}'))
"""

multi_drone_env_setup_code = """
import gymnasium
import jax

import lsy_drone_racing
from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv


env = gymnasium.make_vec('MultiDroneRacing-v0',
    num_envs={num_envs},
    freq=config.env.kwargs[0]["freq"],
    sim_config=config.sim,
    sensor_range=config.env.kwargs[0]["sensor_range"],
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    seed=config.env.seed,
    device='{device}',
)

# JIT compile the reset and step functions
env.reset()
env.step(env.action_space.sample())
jax.block_until_ready(env.unwrapped.data)
# JIT masked reset (used in autoreset)
mask = env.unwrapped.data.marked_for_reset
mask = mask.at[0].set(True)
env.unwrapped._reset(mask=mask)  # enforce masked reset compile
jax.block_until_ready(env.unwrapped.data)
env.action_space.seed(2)
"""


def time_sim_reset(
    n_tests: int = 10, number: int = 1, n_envs: int = 1, device: str = "cpu"
) -> NDArray[np.floating]:
    setup = load_config_code + env_setup_code.format(num_envs=n_envs, device=device)
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_sim_step(
    n_tests: int = 10,
    number: int = 1,
    dynamics: str = "first_principles",
    n_envs: int = 1,
    device: str = "cpu",
) -> NDArray[np.floating]:
    modify_config_code = f"""config.sim.dynamics = '{dynamics}'\n"""
    _env_setup_code = env_setup_code.format(num_envs=n_envs, device=device)
    setup = load_config_code + modify_config_code + _env_setup_code + "\nenv.reset()"
    stmt = """env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_sim_attitude_step(
    n_tests: int = 10, number: int = 1, n_envs: int = 1, device: str = "cpu"
) -> NDArray[np.floating]:
    env_setup_code = attitude_env_setup_code.format(num_envs=n_envs, device=device)
    setup = load_config_code + env_setup_code + "\nenv.reset()"
    stmt = """env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_multi_drone_reset(
    n_tests: int = 10, number: int = 1, n_envs: int = 1, device: str = "cpu"
) -> NDArray[np.floating]:
    env_setup_code = multi_drone_env_setup_code.format(num_envs=n_envs, device=device)
    setup = load_multi_drone_config_code + env_setup_code + "\nenv.reset()"
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_multi_drone_step(
    n_tests: int = 10,
    number: int = 100,
    dynamics: str = "first_principles",
    n_envs: int = 1,
    device: str = "cpu",
) -> NDArray[np.floating]:
    modify_config_code = f"""config.sim.dynamics = '{dynamics}'\n"""
    env_setup_code = multi_drone_env_setup_code.format(num_envs=n_envs, device=device)

    setup = load_multi_drone_config_code + modify_config_code + env_setup_code + "\nenv.reset()"
    stmt = """env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))
