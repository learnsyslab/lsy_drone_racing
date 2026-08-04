#!/usr/bin/env python
"""Launch script for the real race with multiple drones.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rclpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from multiprocessing.synchronize import Barrier

    from ml_collections import ConfigDict

    from lsy_drone_racing.envs.real_race_env import RealMultiDroneRaceEnv

logger = logging.getLogger(__name__)


def control_loop(rank: int, config: ConfigDict, start_barrier: Barrier):
    """Control loop for the drone."""
    rclpy.init()  # Start the ROS library
    node = rclpy.create_node(f"drone{rank}")
    # Override the env config with the kwargs for this particular drone
    config.env.freq = config.env.kwargs[rank]["freq"]
    config.env.sensor_range = config.env.kwargs[rank]["sensor_range"]
    config.env.control_mode = config.env.kwargs[rank]["control_mode"]

    env: RealMultiDroneRaceEnv = gymnasium.make(
        "RealMultiDroneRacing-v0",
        drones=config.deploy.drones,
        rank=rank,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
    )
    try:
        options = {
            "check_drone_start_pos": config.deploy.check_drone_start_pos,
            "check_race_track": config.deploy.check_race_track,
            "real_track_objects": config.deploy.real_track_objects,
        }
        obs, info = env.reset(options=options)
        next_obs = obs  # Set next_obs to avoid errors when the loop never enters

        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        # Will take the absolute path if provided in config.controller.file
        controller_path = control_path / config.controller[rank]["file"]
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)

        start_barrier.wait(timeout=10.0)  # Wait for all drones to be ready at the same time
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            # Enable this if you want to test with single drone controllers. TODO: Remove
            obs = {k: v[rank] for k, v in obs.items()}
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller.step_callback(action, next_obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                node.get_logger().warning(
                    f"Controller {rank} exceeded loop frequency by {exc:.3f}s",
                    throttle_duration_sec=2,
                )
        ep_time = time.perf_counter() - start_time
        finished_track = (next_obs["n_gates_passed"] == next_obs["gate_sequence"].shape[-1])[rank]
        print(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
    finally:
        node.destroy_node()
        env.close()


def main(config: str = "multi_level3.toml"):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    n_drones = len(config.deploy.drones)
    assert len(config.controller) == n_drones, "Number of drones and controllers must match."
    assert len(config.env.kwargs) == n_drones, "Number of drones and env kwargs must match."
    assert len(config.env.track.drones) == n_drones, "Number of drones and track drones must match."
    n_drones = len(config.controller)
    ctx = mp.get_context("spawn")
    start_barrier = ctx.Barrier(n_drones)
    drone_processes = [
        ctx.Process(target=control_loop, args=(i, config, start_barrier)) for i in range(n_drones)
    ]
    for p in drone_processes:
        p.start()

    while any(p.is_alive() for p in drone_processes):
        time.sleep(0.2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
