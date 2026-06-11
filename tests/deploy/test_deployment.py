"""Manual deployment smoke test for the cflib2 Crazyflie wrapper.

This is intentionally a script, not a pytest test. Run it from a deploy shell with ROS running:

    python tests/deploy/test_deployment.py --config config/level0.toml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/level0.toml")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--radio-id", type=int, default=None)
    parser.add_argument("--height", type=float, default=0.6)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--freq", type=float, default=50.0)
    return parser.parse_args()


def current_obs(ros_connector: Any, drone_name: str) -> dict[str, np.ndarray]:
    """Return the single-drone observation fields needed for return_to_start."""
    return {
        "pos": ros_connector.pos[drone_name],
        "quat": ros_connector.quat[drone_name],
        "vel": ros_connector.vel[drone_name],
        "ang_vel": ros_connector.ang_vel[drone_name],
    }


def sleep_step(t_start: float, step: int, freq: float) -> None:
    """Sleep until the next control tick."""
    dt = time.perf_counter() - t_start
    wait = (step + 1) / freq - dt
    if wait > 0:
        time.sleep(wait)


def state_circle(
    drone: Any,
    drone_params: dict[str, float],
    center: np.ndarray,
    height: float,
    radius: float,
    freq: float,
) -> None:
    """Take off and fly one small circle with position/yaw state commands."""
    takeoff_duration = 2.0
    circle_duration = 4.0
    steps = int(takeoff_duration * freq)
    start = center.copy()
    target = center.copy()
    target[2] = height

    t_start = time.perf_counter()
    for step in range(steps):
        alpha = (step + 1) / steps
        action = np.zeros(13, dtype=np.float32)
        action[:3] = (1 - alpha) * start + alpha * target
        drone.send_action(action, control_mode="state", drone_parameters=drone_params)
        drone.send_external_pose()
        sleep_step(t_start, step, freq)

    steps = int(circle_duration * freq)
    t_start = time.perf_counter()
    for step in range(steps):
        theta = 2 * np.pi * step / steps
        action = np.zeros(13, dtype=np.float32)
        action[:3] = target + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        action[9] = theta + np.pi / 2
        drone.send_action(action, control_mode="state", drone_parameters=drone_params)
        drone.send_external_pose()
        sleep_step(t_start, step, freq)


def attitude_circle(drone: Any, drone_params: dict[str, float], freq: float) -> None:
    """Take off and fly one small open-loop attitude-command circle."""
    takeoff_duration = 2.0
    circle_duration = 4.0
    hover_thrust = drone_params["mass"] * 9.81
    steps = int(takeoff_duration * freq)

    t_start = time.perf_counter()
    for step in range(steps):
        alpha = (step + 1) / steps
        thrust = hover_thrust * (1.15 - 0.1 * alpha)
        action = np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
        drone.send_action(action, control_mode="attitude", drone_parameters=drone_params)
        drone.send_external_pose()
        sleep_step(t_start, step, freq)

    steps = int(circle_duration * freq)
    t_start = time.perf_counter()
    for step in range(steps):
        theta = 2 * np.pi * step / steps
        action = np.array(
            [0.08 * np.sin(theta), 0.08 * np.cos(theta), 0.0, hover_thrust], dtype=np.float32
        )
        drone.send_action(action, control_mode="attitude", drone_parameters=drone_params)
        drone.send_external_pose()
        sleep_step(t_start, step, freq)


def stream_external_pose(drone: Any, duration: float, freq: float) -> None:
    """Stream external pose while high-level commands are running."""
    steps = int(duration * freq)
    t_start = time.perf_counter()
    for step in range(steps):
        drone.send_external_pose()
        sleep_step(t_start, step, freq)


def main() -> None:
    """Run the manual deployment smoke test."""
    import rclpy
    from drone_estimators.ros_nodes.ros2_connector import ROSConnector
    from drone_models.core import load_params

    from lsy_drone_racing.utils import load_config
    from lsy_drone_racing.utils.crazyflie import RacingCrazyflie

    args = parse_args()
    config = load_config(Path(args.config))
    drone_config = config.deploy.drones[args.rank]
    drone_name = f"cf{drone_config['id']}"
    radio_id = args.rank if args.radio_id is None else args.radio_id
    home_pos = np.array(config.env.track.drones[args.rank]["pos"], dtype=np.float32)
    drone_params = load_params("first_principles", drone_config["drone_model"])

    rclpy.init()
    obs_connector = ROSConnector(tf_names=[drone_name], timeout=10.0)
    drone = RacingCrazyflie.from_radio(
        radio_id=radio_id,
        radio_channel=drone_config["channel"],
        drone_id=drone_config["id"],
        drone_name=drone_name,
    )
    try:
        drone.connect_and_reset()
        state_circle(
            drone,
            drone_params,
            obs_connector.pos[drone_name].copy(),
            args.height,
            args.radius,
            args.freq,
        )
        drone.return_to_start(home_pos, current_obs(obs_connector, drone_name), check_ok=rclpy.ok)

        drone.connect_and_reset(unlock_thrust=True)
        attitude_circle(drone, drone_params, args.freq)
        drone.return_to_start(home_pos, current_obs(obs_connector, drone_name), check_ok=rclpy.ok)

        drone.connect_and_reset()
        target = obs_connector.pos[drone_name].copy()
        target[2] = args.height
        drone.go_to(target, duration=3.0)
        stream_external_pose(drone, 1.5, args.freq)
        drone.emergency_stop()
        time.sleep(0.2)
    finally:
        drone.close(emergency_stop=False)
        obs_connector.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
