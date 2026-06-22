"""Lighthouse bench check (read-only, no flight).

Connects to a single Crazyflie in lighthouse mode and continuously prints the onboard state
estimate so you can validate the Lighthouse setup before ever arming the drone:

- confirms the Lighthouse deck is detected (``deck.bcLighthouse4``),
- starts the same background state-log reader used by the real environment,
- prints ``pos`` / ``rpy`` / ``vel`` / ``ang_vel`` from :meth:`Crazyflie.get_obs`.

Move and rotate the drone **by hand** and check that the values are sane and in the right frame:
position should match a tape measure in the Lighthouse frame (origin = track origin), and the
velocity / angular velocity signs should follow your motion. This is the make-or-break check for
the frames/units before any closed-loop flight.

The drone is never armed and no setpoints are sent, so the motors stay off.

Usage:

    python scripts/lighthouse_bench.py --drone_id 10 --channel 100
"""

from __future__ import annotations

import logging
import time

import fire
import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils.crazyflie import Crazyflie

logger = logging.getLogger(__name__)


def main(drone_id: int = 10, channel: int = 100, radio_id: int = 0, rate: float = 10.0) -> None:
    """Print the onboard Lighthouse state estimate of a single drone (read-only, no flight).

    Args:
        drone_id: Crazyflie id (the last byte of the radio address, e.g. 10 for cf10).
        channel: Radio channel of the drone.
        radio_id: Crazyradio device index.
        rate: Print frequency in Hz.
    """
    drone = Crazyflie.from_radio(
        radio_id=radio_id, radio_channel=channel, drone_id=drone_id, lighthouse=True
    )
    logger.info("Read-only bench check: the drone is never armed and no setpoints are sent.")
    logger.info("Connecting to cf%d (channel %d)...", drone_id, channel)
    drone.connect(timeout=10.0)  # also verifies the Lighthouse deck
    drone.reset(arm=False)  # applies settings, resets the estimator, starts the state-log reader
    logger.info("Connected. Move the drone by hand and watch the values (Ctrl-C to stop).")

    period = 1.0 / rate
    try:
        while True:
            obs = drone.get_obs()
            pos, vel, ang_vel = obs["pos"], obs["vel"], obs["ang_vel"]
            rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=True)
            speed = float(np.linalg.norm(vel))
            print(
                f"pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}] m | "
                f"rpy=[{rpy[0]:+6.1f} {rpy[1]:+6.1f} {rpy[2]:+6.1f}] deg | "
                f"vel=[{vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f}] |v|={speed:.2f} m/s | "
                f"ang_vel=[{ang_vel[0]:+.2f} {ang_vel[1]:+.2f} {ang_vel[2]:+.2f}] rad/s"
            )
            time.sleep(period)
    except KeyboardInterrupt:
        logger.info("Stopping bench check.")
    finally:
        drone.close(emergency_stop=False)  # never armed, so no emergency stop is needed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
