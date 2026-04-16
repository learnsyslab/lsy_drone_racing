"""Simulate a multi-drone race.

Run as:

    $ python scripts/multi_sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.multi_drone_race import MultiDroneRacingEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "multi_level0.toml",
    controllers: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controllers: Comma-separated controller filenames in `lsy_drone_racing/control/` or None.
            If None, the controllers specified in the config file are used.
        n_runs: The number of episodes.
        render: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    logger.warning(
        "The simulation currently only supports running with one controller type and one set of "
        "environment parameters (i.e. frequencies, control mode etc.). Only using the settings for "
        "the first drone."
    )
    # Load the controller modules
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    if controllers is None:
        controller_names = [controller["file"] for controller in config.controller]
    else:
        controller_names = [controller.strip() for controller in controllers.split(",")]

    controller_classes = [
        load_controller(control_path / controller) for controller in controller_names
    ]  # This returns a list of classes, not a list of instances

    # Create the racing environment
    env: MultiDroneRacingEnv = gymnasium.make(
        "MultiDroneRacing-v0",
        freq=config.env.kwargs[0]["freq"],
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.kwargs[0]["sensor_range"],
        control_mode=config.env.kwargs[0]["control_mode"],
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    # We use the same example controllers for this script as for the single-drone case. These expect
    # the config to have env.freq set, so we copy it here. Actual multi-drone controllers should not
    # rely on this.
    config.env.freq = config.env.kwargs[0]["freq"]
    env = JaxToNumpy(env)
    n_drones = env.unwrapped.sim.n_drones

    for _ in range(n_runs):  # Run n_runs episodes with the controllers
        obs, info = env.reset()
        controller_instances: list[Controller] = [
            cls(obs, _info_with_rank(info, rank), config)
            for rank, cls in enumerate(controller_classes)
        ]
        finish_times = np.full(n_drones, np.nan, dtype=np.float32)
        controller_finished = np.full(n_drones, False, dtype=bool)

        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq
            ranked_infos = [_info_with_rank(info, rank) for rank in range(n_drones)]

            actions = np.stack(
                [
                    ctrl.compute_control(obs, ctrl_info)
                    for ctrl, ctrl_info in zip(controller_instances, ranked_infos)
                ],
                dtype=np.float32,
            )

            obs, reward, terminated, truncated, info = env.step(actions)

            newly_finished = (obs["target_gate"] == -1) & np.isnan(finish_times)
            finish_times[newly_finished] = curr_time
            # Update the controllers' internal state and models.
            for rank, (ctrl, ctrl_info) in enumerate(zip(controller_instances, ranked_infos)):
                controller_finished[rank] = ctrl.step_callback(
                    actions[rank], obs, reward, terminated, truncated, ctrl_info
                )

            # Synchronize the GUI.
            if config.sim.render:
                if ((i * fps) % config.env.freq) < fps:
                    try:
                        env.render()
                    # TODO: JaxToNumpy not working with None (returned by env.render()). Open issue
                    # in gymnasium and fix this.
                    except Exception as e:
                        if not e.args[0].startswith("No known conversion for Jax type"):
                            raise e
            i += 1
            if terminated | truncated | controller_finished.all():
                break

        for ctrl in controller_instances:
            ctrl.episode_callback()  # Update the controller internal state and models.
            ctrl.episode_reset()
        log_episode_stats(obs, info, config, finish_times, controller_names)

    # Close the environment
    env.close()


def _info_with_rank(info: dict, rank: int) -> dict:
    """Return a controller-specific info dict with the drone rank attached."""
    return {**info, "rank": rank}


def log_episode_stats(
    obs: dict, info: dict, config: ConfigDict, finish_times: np.ndarray, controller_names: list[str]
):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    n_gates = len(config.env.track.gates)
    gates_passed = np.where(gates_passed == -1, n_gates, gates_passed)
    finished = gates_passed == n_gates

    time_strings = [
        "DNF" if np.isnan(finish_time) else f"{finish_time:.2f}" for finish_time in finish_times
    ]
    name_width = max(len("controller"), max(len(name) for name in controller_names))
    time_width = max(len("time [s]"), max(len(time_str) for time_str in time_strings))
    finished_width = len("finished")
    gates_width = len("gates")

    lines = [
        f"{'controller':<{name_width}} | {'time [s]':>{time_width}} | "
        f"{'finished':>{finished_width}} | {'gates':>{gates_width}}"
    ]
    for i, controller_name in enumerate(controller_names):
        lines.append(
            f"{controller_name:<{name_width}} | {time_strings[i]:>{time_width}} | "
            f"{str(finished[i]):>{finished_width}} | {gates_passed[i]:>{gates_width}}"
        )

    table = "\n".join(lines)
    logger.info(f"Episode stats:\n{table}\n")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
