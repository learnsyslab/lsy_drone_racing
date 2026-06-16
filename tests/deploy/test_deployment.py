"""Manual deployment smoke test for the cflib2 Crazyflie wrapper.

This is intentionally a script, not a pytest test. Run it from a deploy shell with ROS running:

    python tests/deploy/test_deployment.py --config config/level0.toml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from drone_models.core import load_params

from lsy_drone_racing.control.attitude_controller import AttitudeController
from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)

try:
    import rclpy
    from drone_estimators.ros_nodes.ros2_connector import ROSConnector

    from lsy_drone_racing.utils.crazyflie import Crazyflie
except ImportError as e:
    logger.error("Failed to import modules: %s", e)


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


def state_circle(drone: Any, center: np.ndarray, height: float, radius: float, freq: float) -> None:
    """Take off and fly one small circle with full-state commands."""
    takeoff_duration = 2.0
    circle_duration = 4.0
    steps = int(takeoff_duration * freq)
    start = center.copy()
    target = center.copy()
    target[2] = height

    logger.info("Taking off with state commands to %.2f m.", height)
    t_start = time.perf_counter()
    for step in range(steps):
        alpha = (step + 1) / steps
        action = np.zeros(13, dtype=np.float32)
        action[:3] = (1 - alpha) * start + alpha * target
        action[3:6] = (target - start) / takeoff_duration
        drone.send_action_state(action[:3], action[3:6], action[6:9], action[9], action[10:])
        drone.send_external_pose()
        sleep_step(t_start, step, freq)

    steps = int(circle_duration * freq)
    omega = 2 * np.pi / circle_duration
    logger.info("Flying state-command circle with %.2f m radius.", radius)
    t_start = time.perf_counter()
    for step in range(steps):
        theta = omega * step / freq
        action = np.zeros(13, dtype=np.float32)
        action[:3] = target + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        action[3:6] = [-radius * omega * np.sin(theta), radius * omega * np.cos(theta), 0.0]
        action[6:9] = [-radius * omega**2 * np.cos(theta), -radius * omega**2 * np.sin(theta), 0.0]
        action[9] = theta + np.pi / 2
        action[12] = omega
        drone.send_action_state(action[:3], action[3:6], action[6:9], action[9], action[10:])
        drone.send_external_pose()
        sleep_step(t_start, step, freq)
    logger.info("Finished state-command circle.")


def override_attitude_reference(
    controller: Any,
    start: np.ndarray,
    height: float,
    radius: float,
    takeoff_duration: float,
    circle_duration: float,
) -> None:
    """Overwrite the attitude controller reference with the smoke-test circle."""
    target = start.copy()
    target[2] = height
    omega = 2 * np.pi / circle_duration

    def des_pos(t: float) -> np.ndarray:
        if t < takeoff_duration:
            alpha = np.clip(t / takeoff_duration, 0.0, 1.0)
            return (1 - alpha) * start + alpha * target

        theta = omega * (t - takeoff_duration)
        return target + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])

    def des_vel(t: float) -> np.ndarray:
        if t < takeoff_duration:
            return (target - start) / takeoff_duration

        theta = omega * (t - takeoff_duration)
        return np.array([-radius * omega * np.sin(theta), radius * omega * np.cos(theta), 0.0])

    controller._des_pos_spline = des_pos
    controller._des_vel_spline = des_vel
    controller._t_total = takeoff_duration + circle_duration
    controller._tick = 0
    controller._finished = False


def attitude_circle(
    drone: Any,
    obs_connector: Any,
    drone_name: str,
    drone_params: dict[str, float],
    controller: Any,
    height: float,
    radius: float,
    freq: float,
) -> None:
    """Take off and fly one small circle with the attitude controller."""
    takeoff_duration = 2.0
    circle_duration = 4.0
    start = obs_connector.pos[drone_name].copy()
    override_attitude_reference(
        controller, start, height, radius, takeoff_duration, circle_duration
    )
    controller._freq = freq

    logger.info("Taking off with attitude controller to %.2f m.", height)
    steps = int(takeoff_duration * freq)
    t_start = time.perf_counter()
    for step in range(steps):
        obs = current_obs(obs_connector, drone_name)
        action = controller.compute_control(obs).astype(np.float32)
        drone.send_action_attitude(action[:3], action[3], drone_params)
        drone.send_external_pose()
        controller.step_callback(action, obs, 0.0, False, False, {})
        sleep_step(t_start, step, freq)

    logger.info("Flying attitude-controller circle with %.2f m radius.", radius)
    steps = int(circle_duration * freq)
    t_start = time.perf_counter()
    for step in range(steps):
        obs = current_obs(obs_connector, drone_name)
        action = controller.compute_control(obs).astype(np.float32)
        drone.send_action_attitude(action[:3], action[3], drone_params)
        drone.send_external_pose()
        controller.step_callback(action, obs, 0.0, False, False, {})
        sleep_step(t_start, step, freq)
    logger.info("Finished attitude-controller circle.")


def stream_external_pose(drone: Any, duration: float, freq: float) -> None:
    """Stream external pose while high-level commands are running."""
    steps = int(duration * freq)
    logger.info("Streaming external pose for %.1f s.", duration)
    t_start = time.perf_counter()
    for step in range(steps):
        drone.send_external_pose()
        sleep_step(t_start, step, freq)
    logger.info("Finished streaming external pose.")


def main() -> None:
    """Run the manual deployment smoke test."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info("Loading deployment config from %s.", args.config)
    config = load_config(Path(args.config))
    drone_config = config.deploy.drones[args.rank]
    drone_name = f"cf{drone_config['id']}"
    radio_id = args.rank if args.radio_id is None else args.radio_id
    home_pos = np.array(config.env.track.drones[args.rank]["pos"], dtype=np.float32)
    drone_params = load_params("first_principles", drone_config["drone_model"])

    logger.info("Initializing ROS for %s.", drone_name)
    rclpy.init()
    obs_connector = ROSConnector(estimator_names=[drone_name], timeout=10.0)
    logger.info(
        "Creating Crazyflie wrapper for %s on radio %d, channel %d.",
        drone_name,
        radio_id,
        drone_config["channel"],
    )
    drone = Crazyflie.from_radio(
        radio_id=radio_id,
        radio_channel=drone_config["channel"],
        drone_id=drone_config["id"],
        drone_name=drone_name,
    )
    try:
        logger.info("Connecting and resetting for state-command test.")
        drone.connect()
        drone.reset(arm=True)
        state_circle(
            drone, obs_connector.pos[drone_name].copy(), args.height, args.radius, args.freq
        )
        logger.info("Returning to start after state-command test.")
        drone.return_to_start(home_pos, current_obs(obs_connector, drone_name), check_ok=rclpy.ok)
        logger.info("Finished state-command test.")

        logger.info("Connecting and resetting for attitude-command test.")
        drone.reset(arm=True)
        attitude_controller = AttitudeController(current_obs(obs_connector, drone_name), {}, config)
        attitude_circle(
            drone,
            obs_connector,
            drone_name,
            drone_params,
            attitude_controller,
            args.height,
            args.radius,
            args.freq,
        )
        logger.info("Returning to start after attitude-command test.")
        drone.return_to_start(home_pos, current_obs(obs_connector, drone_name), check_ok=rclpy.ok)
        logger.info("Finished attitude-command test.")

        logger.info("Connecting and resetting for high-level goto test.")
        drone.reset(arm=True)
        target = obs_connector.pos[drone_name].copy()
        target[2] = args.height
        logger.info("Sending high-level goto to %s.", np.array2string(target, precision=3))
        drone.go_to(target, duration=3.0)
        stream_external_pose(drone, 1.5, args.freq)
        logger.info("Sending emergency stop.")
        drone.emergency_stop()
        time.sleep(0.2)
        logger.info("Deployment smoke test finished.")
    finally:
        logger.info("Cleaning up.")
        drone.close(emergency_stop=True)
        obs_connector.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
