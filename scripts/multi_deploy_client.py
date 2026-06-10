"""Deployment script for multi-drone racing clients (host-client architecture).

Each client runs on a drone's computing unit and communicates with the central host
via Zenoh. Run one instance per drone, specifying its rank.

Usage:

    python deploy_client.py --config multi_level2.toml --drone_rank 0
    python deploy_client.py --config multi_level2.toml --drone_rank 1

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rclpy

from lsy_drone_racing.utils import extract_config_for_rank, load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.envs.real_race_env_client import RealMultiDroneRaceEnvClient

logger = logging.getLogger(__name__)


def main(
    config: str = "multi_level2.toml", controller: str | None = None, drone_rank: int | None = None
):
    """Deploy and run a client for multi-drone racing.

    Args:
        config: Configuration file name, assumed to be in ``config/``.
        controller: Controller file name in ``lsy_drone_racing/control/``, or ``None``
            to use the value from the config file.
        drone_rank: Rank of this drone in the multi-drone setup. Required.
    """
    assert drone_rank is not None, "drone_rank must be specified"

    rclpy.init()
    config_obj = load_config(Path(__file__).parents[1] / "config" / config)
    if controller is not None:
        config_obj.controller[drone_rank]["file"] = controller
    env: RealMultiDroneRaceEnvClient = gymnasium.make(
        "RealMultiDroneRaceEnvClient-v0",
        drones=config_obj.deploy.drones,
        rank=drone_rank,
        freq=config_obj.env.kwargs[drone_rank]["freq"],
        track=config_obj.env.track,
        randomizations=config_obj.env.randomizations,
        sensor_range=config_obj.env.kwargs[drone_rank]["sensor_range"],
        control_mode=config_obj.env.kwargs[drone_rank]["control_mode"],
    )
    try:
        obs, info = env.reset(options=config_obj.deploy)

        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        controller_cls = load_controller(control_path / config_obj.controller[drone_rank]["file"])
        controller = controller_cls(obs, info, extract_config_for_rank(config_obj, drone_rank))

        env.unwrapped.lock_until_race_start(timeout=120.0)
        logger.info(
            f"Client {drone_rank}: Starting control loop at "
            f"{config_obj.env.kwargs[drone_rank]['freq']} Hz"
        )
        start_time = time.time()

        while rclpy.ok():
            t_loop = time.time()
            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                logger.debug(
                    f"Client {drone_rank}: Episode finished "
                    f"({terminated=}, "
                    f"{truncated=}, "
                    f"{controller_finished=})"
                )
                break

            dt = time.time() - t_loop
            sleep_time = 1.0 / config_obj.env.kwargs[drone_rank]["freq"] - dt
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(
                    f"Client {drone_rank}: Control loop overrun by {-sleep_time * 1000:.1f}ms"
                )

        ep_time = time.time() - start_time
        finished = obs["target_gate"][drone_rank] == -1
        logger.info(
            f"Client {drone_rank}: Episode completed in {ep_time:.3f}s (finished={finished})"
        )
    except KeyboardInterrupt:
        logger.info(f"Client {drone_rank}: Interrupted by user")
    finally:
        env.close()
        rclpy.shutdown()
        logger.debug(f"Client {drone_rank}: Shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
