"""ROS2 node for the communication between host and clients in multi-drone races."""

from __future__ import annotations

import logging
import threading

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.parameter import Parameter

logger = logging.getLogger(__name__)


class RaceCommNode:
    """ROS2 node for race coordination, spinning in a background daemon thread.

    Access the underlying rclpy node via :attr:`node` to create publishers and
    subscriptions directly. All cleanup is handled by :meth:`close`.

    Args:
        name: ROS2 node name (must be unique within the process).
        use_sim_time: Whether the node reads its time from the ``/clock`` topic instead of the
            system clock.
    """

    def __init__(self, name: str, *, use_sim_time: bool = False):
        """Initialize and spin the ROS2 node in a background thread."""
        use_sim_time_parameter = Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)
        self.node = rclpy.create_node(name, parameter_overrides=[use_sim_time_parameter])
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        def _spin():
            try:
                self._executor.spin()
            except ExternalShutdownException:
                logger.debug(f"RaceCommNode '{name}' spin thread stopped")
            except Exception as e:
                if type(e).__name__ == "RCLError":
                    # Raised when the context is shut down while the thread is still spinning
                    logger.debug(f"RaceCommNode '{name}' spin thread stopped (context invalid)")
                else:
                    raise

        self._thread = threading.Thread(target=_spin, daemon=True, name=f"spin-{name}")
        self._thread.start()
        logger.debug(f"RaceCommNode '{name}' started")

    def close(self):
        """Shut down the executor and destroy the node."""
        self._executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        logger.debug("RaceCommNode closed")
