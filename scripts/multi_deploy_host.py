"""Deployment script for the multi-drone racing host.

The host connects to all Crazyflie drones, coordinates client synchronization via
Zenoh, and supervises the race until all clients finish.

Usage:

    python deploy_host.py --config multi_level2.toml

"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import rclpy

from lsy_drone_racing.envs.real_race_host import CrazyflieRealRaceHost
from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main(config: str = "multi_level2.toml"):
    """Deploy and run the race host.

    Args:
        config: Configuration file name, assumed to be in ``config/``.
    """
    rclpy.init()
    config_obj = load_config(Path(__file__).parents[1] / "config" / config)
    host = CrazyflieRealRaceHost(
        track=config_obj.env.track,
        deploy_args=config_obj.deploy,
        control_args=config_obj.env.kwargs,
    )
    try:
        host.update_poses(
            track_obj=config_obj.deploy.real_track_objects,
            drones=config_obj.deploy.check_drone_start_pos,
        )
        host.check_track(
            rng_config=config_obj.env.randomizations,
            check_objects=config_obj.deploy.real_track_objects
            and config_obj.deploy.check_race_track,
            check_drones=config_obj.deploy.check_drone_start_pos,
        )
        host.connect_drones()
        host.host_main_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Host encountered error: {e}", exc_info=True)
    finally:
        host.close()
        rclpy.shutdown()
        logger.info("Host shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("cflib").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.DEBUG)
    fire.Fire(main)
