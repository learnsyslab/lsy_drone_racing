"""Sweep a scalar controller attribute over several seeds and report the success rate.

Run as:

    $ python scripts/sweep.py --controller=attitude_mppi.py --attr=_lam --values=0.01,0.05

Every value is evaluated on the seeds 1 to `n_seeds` of the given config. The attribute is set on
the controller instance after construction, so it overrides whatever the constructor assigned.
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

logger = logging.getLogger(__name__)


def run_episode(
    config: ConfigDict, controller_cls: type[Controller], seed: int, attr: str, value: float
) -> tuple[int, float, float]:
    """Run a single episode headless with a controller attribute overridden.

    Args:
        config: The race configuration.
        controller_cls: The controller class to instantiate.
        seed: The seed of the environment.
        attr: Name of the controller attribute to override after construction.
        value: The value of the attribute.

    Returns:
        The number of gates passed, the episode time in seconds, and the median of the
        controller's `_ess` attribute over the episode (NaN if it has none).
    """
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=seed,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset()
    controller = controller_cls(obs, info, config)
    setattr(controller, attr, value)  # override the swept attribute after construction

    steps, ess = 0, []
    while True:
        action = controller.compute_control(obs, info)
        ess.append(getattr(controller, "_ess", np.nan))  # controllers without ESS give NaN
        obs, reward, terminated, truncated, info = env.step(action)
        finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
        if finished or terminated or truncated:
            break
        steps += 1
    env.close()
    return int(obs["n_gates_passed"]), steps / config.env.freq, float(np.nanmedian(ess))


def sweep(
    controller: str | None = None,
    attr: str = "_lam",
    values: float | tuple[float, ...] = (0.05,),
    n_seeds: int = 8,
    config: str = "level0.toml",
    control_mode: str | None = None,
):
    """Evaluate a controller for several values of one attribute.

    Args:
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If
            None, the controller specified in the config file is used.
        attr: Name of the scalar controller attribute to sweep, e.g. `_lam`.
        values: The values to test, given as comma-separated numbers on the command line.
        n_seeds: Number of seeds, which are 1 to `n_seeds`.
        config: The path to the configuration file. Assumes the file is in `config/`.
        control_mode: Overrides the control mode of the config (`state` or `attitude`) if given.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    config.sim.render = False
    if control_mode is not None:
        config.env.control_mode = control_mode  # lets a config run with a different interface
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_cls = load_controller(control_path / (controller or config.controller.file))

    for value in np.atleast_1d(values):
        # Same seeds for every value, so the values are compared on identical tracks
        results = [
            run_episode(config, controller_cls, seed, attr, float(value))
            for seed in range(1, n_seeds + 1)
        ]
        gates = [r[0] for r in results]
        n_total = len(config.env.track.gate_order)
        n_success = sum(g == n_total for g in gates)  # success means all gates were passed
        times = [r[1] for r in results if r[0] == n_total]  # time only counts for finished runs
        mean_time = f"{np.mean(times):.2f}s" if times else "n/a"
        logger.info(
            f"{attr}={value}: success {n_success}/{n_seeds}, gates per seed {gates}, "
            f"mean time of finished runs {mean_time}, "
            f"median ESS {np.nanmedian([r[2] for r in results]):.0f}"
        )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(sweep, serialize=lambda _: None)
