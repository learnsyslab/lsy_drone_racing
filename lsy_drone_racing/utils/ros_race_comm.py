"""ROS2-based communication for multi-drone racing coordination.

Uses custom ``lsy_race_msgs`` message types for type-safe pub/sub and the
``CalibrateClock`` service for clock offset estimation. Each participant
creates a :class:`RaceCommNode` which spins a
:class:`~rclpy.executors.SingleThreadedExecutor` in a background daemon thread.

Clock offset estimation uses the midpoint method over N round-trips::

    offset = host_timestamp - (t_send + t_recv) / 2

Clients apply this offset when timestamping every ``ClientState`` message so
the host observes accurate one-way latency without clock skew.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import rclpy
from drone_racing_msgs.srv import RealCalibrateClock  # type: ignore[import-untyped]
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

if TYPE_CHECKING:
    from rclpy.client import Client

logger = logging.getLogger(__name__)


def _suppress_shutdown_thread_errors():
    """Install a threading.excepthook that silences expected ROS2 shutdown exceptions.

    Replaces the noisy default traceback for :class:`~rclpy.executors.ExternalShutdownException`
    and :class:`KeyboardInterrupt` in background spin threads with a single DEBUG log line.
    Any other uncaught thread exception still goes through the default handler.
    """
    _original = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type in (ExternalShutdownException, KeyboardInterrupt) or (
            args.exc_type.__name__ == "RCLError"
        ):
            logger.debug(f"Thread '{args.thread.name}' stopped (shutdown)")
        else:
            _original(args)

    threading.excepthook = _hook


def compute_latency_ms(timestamp: float, clock_offset: float = 0.0) -> float:
    """Compute one-way latency in milliseconds from a sent timestamp.

    Args:
        timestamp: Time the message was sent.
        clock_offset: Calibrated offset (host_time - client_time) in seconds.
            Zero when called on the host side (timestamps are already in host time).

    Returns:
        Estimated one-way latency in milliseconds.
    """
    return (time.time() - timestamp - clock_offset) * 1000


def calibrate_clock(client: Client, n: int = 5, timeout: float = 60.0) -> float:
    """Estimate clock offset (host_time - client_time) in seconds via N round-trips.

    Blocks until the :class:`~lsy_race_msgs.srv.CalibrateClock` service becomes
    available or ``timeout`` is reached. Each call records the send and receive
    times; the offset is estimated as::

        offset = host_timestamp - (t_send + t_recv) / 2

    and averaged over all ``n`` calls.

    Args:
        client: rclpy service client for the ``CalibrateClock`` service.
        n: Number of round-trips to average.
        timeout: Maximum time in seconds to wait for the service to become available.

    Returns:
        Estimated clock offset in seconds. Add this to ``time.time()`` on the client
        to get the equivalent host-clock time.

    Raises:
        TimeoutError: If the service is not available within ``timeout`` seconds.
    """
    if not client.wait_for_service(timeout_sec=timeout):
        raise TimeoutError(f"Calibration service not available after {timeout}s")
    offsets = []
    for _ in range(n):
        t_send = time.time()
        future = client.call_async(RealCalibrateClock.Request())
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        if not ready.wait(timeout=timeout):
            raise TimeoutError("Clock calibration call timed out")
        t_recv = time.time()
        offsets.append(future.result().host_timestamp - (t_send + t_recv) / 2)
    return sum(offsets) / len(offsets)


class RaceCommNode:
    """ROS2 node for race coordination, spinning in a background daemon thread.

    Access the underlying rclpy node via :attr:`node` to create publishers,
    subscriptions, and services directly. All cleanup is handled by :meth:`close`.

    Args:
        name: ROS2 node name (must be unique within the process).
    """

    def __init__(self, name: str):
        """Initialize and spin the ROS2 node in a background thread."""
        _suppress_shutdown_thread_errors()
        self.node = rclpy.create_node(name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        def _spin():
            try:
                self._executor.spin()
            except (ExternalShutdownException, KeyboardInterrupt):
                logger.debug(f"RaceCommNode '{name}' spin thread stopped")
            except Exception as e:
                if type(e).__name__ == "RCLError":
                    # TODO: RCLError can not be imported
                    # but is raised when the context is already shutdown
                    # while the thread is still spinning
                    # this is stupid workaround
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
