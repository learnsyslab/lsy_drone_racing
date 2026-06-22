"""Crazyflie cflib2 wrapper for drone racing."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cflib2 import Crazyflie as CflibCrazyflie
from cflib2 import LinkContext
from cflib2.error import CrazyflieError
from cflib2.toc_cache import FileTocCache
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.transform import force2pwm
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils.lighthouse import (
    LIGHTHOUSE_DECK_PARAM,
    ORI_RATE_VARS,
    POS_VEL_VARS,
    decode_state,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from concurrent.futures import Future

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["Crazyflie"]

_POWER_CYCLE_BOOT_WAIT = 3.0  # 3 seconds is sufficient for a reboot
# Argument passed to cflib2's log block ``start`` (matches swarmGPT usage). The resulting onboard
# log rate should be verified on hardware; see the Lighthouse implementation plan.
_STATE_LOG_START_ARG = 10
_STATE_LOG_FIRST_SAMPLE_TIMEOUT = 5.0  # seconds to wait for the first lighthouse sample


class Crazyflie:
    """Synchronous single-drone wrapper around the asynchronous cflib2 API.

    The environment owns ROS and observation assembly. This class owns only the Crazyflie radio
    link, firmware parameters, command streaming, external-pose injection, and shutdown.

    In lighthouse mode the drone localizes itself from the Lighthouse base stations. We then read
    the onboard state estimate back from the drone (see :meth:`get_obs`) instead of using ROS, and
    do not inject external poses.
    """

    def __init__(
        self,
        uri: str,
        drone_name: str,
        cache_dir: str | Path | None = None,
        power_cycle_on_connect: bool = True,
        lighthouse: bool = False,
    ):
        """Create a Crazyflie wrapper.

        Args:
            uri: Crazyflie radio URI.
            drone_name: Name of the drone in ROS, e.g. cf10.
            cache_dir: Directory used for cflib2 TOC caching.
            power_cycle_on_connect: Whether to power-cycle the STM32 domain before connecting.
            lighthouse: If True, use the onboard Lighthouse state estimate (read back from the
                drone) instead of an external motion capture system. No ROS connection is created
                and external poses are not pushed to the drone.
        """
        self.uri = uri
        self.drone_name = drone_name
        self.power_cycle_on_connect = power_cycle_on_connect
        self.lighthouse = lighthouse
        self.context = LinkContext()
        cache_dir = Path(__file__).parent / ".cache" if cache_dir is None else Path(cache_dir)
        self.toc_cache = FileTocCache(str(cache_dir))
        self._ros_connector = (
            None
            if lighthouse
            else ROSConnector(
                tf_names=[self.drone_name],
                cmd_topic=f"/drones/{self.drone_name}/command",
                timeout=10.0,
            )
        )

        self._cf: CflibCrazyflie | None = None
        self._commander_level: Literal["low", "high"] | None = None
        self._state_setpoint_fallback_warned = False
        self._loop = asyncio.new_event_loop()
        # Lighthouse state-log reader (started in reset() when lighthouse is enabled).
        self._loop_thread: threading.Thread | None = None
        self._log_stop: threading.Event | None = None
        self._log_future: Future[None] | None = None
        self._latest_state: dict[str, NDArray[np.floating]] | None = None

    @classmethod
    def from_radio(
        cls,
        radio_id: int,
        radio_channel: int,
        drone_id: int,
        drone_name: str | None = None,
        cache_dir: str | Path | None = None,
        power_cycle_on_connect: bool = True,
        lighthouse: bool = False,
    ) -> Crazyflie:
        """Create a Crazyflie wrapper from deployment radio settings."""
        return cls(
            f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7{drone_id:02X}",
            f"cf{drone_id}" if drone_name is None else drone_name,
            cache_dir=cache_dir,
            power_cycle_on_connect=power_cycle_on_connect,
            lighthouse=lighthouse,
        )

    @property
    def cf(self) -> CflibCrazyflie | None:
        """Return the underlying cflib2 Crazyflie instance."""
        return self._cf

    @property
    def is_connected(self) -> bool:
        """Return whether the Crazyflie is currently connected."""
        return self._cf is not None

    def connect(self, timeout: float = 10.0) -> None:
        """Connect to the Crazyflie."""
        self._run(self._connect, timeout)

    def reset(self, arm: bool = False) -> None:
        """Apply race settings, reset the estimator, and optionally arm the drone."""
        self._run(self._apply_settings)
        self._run(self._reset_estimator)
        if self.lighthouse:
            self._start_state_log()
        if arm:
            self.arm()

    def arm(self) -> None:
        """Arm the drone and unlock thrust for low-level setpoints."""
        self._run(self._arm)
        self._run(self._unlock_thrust)

    def send_external_pose(self) -> None:
        """Send an external mocap pose to the Crazyflie estimator (no-op in lighthouse mode)."""
        if self.lighthouse:
            return
        self._run(self._send_external_pose)

    def get_obs(self) -> dict[str, NDArray[np.floating]]:
        """Return the latest onboard state estimate (lighthouse mode only).

        Returns:
            A dictionary with ``float32`` ``pos``, ``quat`` (xyzw), ``vel`` and ``ang_vel``, read
            from the drone's onboard estimator by the background state-log reader. The fields match
            the per-drone simulation observation.
        """
        if not self.lighthouse:
            raise RuntimeError("get_obs() is only available in lighthouse mode.")
        state = self._latest_state
        if state is None:
            raise RuntimeError("No lighthouse state available yet. Call reset() first.")
        return state

    def send_action_attitude(
        self,
        attitude: NDArray[np.floating],
        thrust: float,
        drone_parameters: dict[str, float],
        publish_to_ros: bool = True,
    ) -> None:
        """Send a roll, pitch, yaw-rate, and collective-thrust command."""
        pwm = force2pwm(thrust, drone_parameters["thrust_max"] * 4, drone_parameters["pwm_max"])
        pwm = np.clip(pwm, drone_parameters["pwm_min"], drone_parameters["pwm_max"])
        command = (*np.rad2deg(attitude), int(pwm))
        self._run(self._send_attitude_setpoint, *command)
        if publish_to_ros and self._ros_connector is not None:
            self._ros_connector.publish_cmd(command)

    def send_action_state(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating] | None = None,
        acc: NDArray[np.floating] | None = None,
        yaw: float | None = None,
        body_rates: NDArray[np.floating] | None = None,
    ) -> None:
        """Send a state command with yaw-only orientation."""
        if vel is None:
            vel = np.zeros(3)
        if acc is None:
            acc = np.zeros(3)
        if yaw is None:
            yaw = 0.0
        if body_rates is None:
            body_rates = np.zeros(3)
        quat = R.from_euler("z", yaw).as_quat()
        # TODO have quat as argument and just forward it -> need to change action interface
        self._run(self._send_full_state_setpoint, pos, vel, acc, quat, body_rates)

    def return_to_start(
        self,
        return_pos: NDArray[np.floating],
        initial_obs: dict[str, NDArray[np.floating]],
        check_ok: Callable[[], bool] | None = None,
        return_height: float = 1.75,
        breaking_distance: float = 1.0,
        breaking_duration: float = 3.0,
        return_duration: float = 5.0,
        land_duration: float = 3.0,
    ) -> None:
        """Return to a start position using the high-level commander."""
        self._run(self._prepare_high_level)

        def wait_for_action(duration: float) -> None:
            end_time = self._loop.time() + duration
            while self._loop.time() < end_time:
                if check_ok is not None and not check_ok():
                    raise RuntimeError("Return-to-start was interrupted")
                if not self.is_connected:
                    raise RuntimeError("Drone connection lost")
                self.send_external_pose()
                self._run(asyncio.sleep, 0.05)

        vel_norm = np.linalg.norm(initial_obs["vel"])
        break_pos = initial_obs["pos"].copy()
        if vel_norm > 1e-6:
            break_pos += initial_obs["vel"] / vel_norm * breaking_distance
        break_pos[2] = return_height
        self._run(self._go_to, break_pos, 0.0, breaking_duration)
        wait_for_action(breaking_duration)

        return_pos = return_pos.copy()
        return_pos[2] = return_height
        self._run(self._go_to, return_pos, 0.0, return_duration)
        wait_for_action(return_duration)

        return_pos[2] = 0.05
        self._run(self._go_to, return_pos, 0.0, land_duration)
        wait_for_action(land_duration)

    def go_to(
        self,
        pos: NDArray[np.floating],
        yaw: float = 0.0,
        duration: float = 3.0,
        linear: bool = False,
    ) -> None:
        """Send a high-level goto command."""
        self._run(self._go_to, pos, yaw, duration, linear=linear)

    def emergency_stop(self) -> None:
        """Send the Crazyflie emergency stop command."""
        self._run(self._emergency_stop)

    def close(self, emergency_stop: bool = True) -> None:
        """Emergency-stop, disconnect, and close the cflib2 event loop."""
        if self._loop.is_closed():
            return
        try:
            self._stop_state_log()
            if emergency_stop and self.is_connected:
                self._run(self._emergency_stop)
                self._run(asyncio.sleep, 0.1)

            if self._cf is not None:
                self._run(self._disconnect)
        finally:
            try:
                if self._ros_connector is not None:
                    self._ros_connector.close()
            finally:
                self._stop_loop_thread()
                self._loop.close()

    def _run(self, operation: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Run an asynchronous operation on this drone's event loop.

        In lighthouse mode the event loop is driven by a background thread (for the state-log
        reader), so coroutines are scheduled onto it thread-safely. Otherwise the loop is driven
        directly via ``run_until_complete``.
        """
        if self._loop.is_closed():
            raise RuntimeError("Crazyflie wrapper is already closed.")
        coro = operation(*args, **kwargs)
        if self._loop_thread is not None:
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return self._loop.run_until_complete(coro)

    # region Lighthouse state log

    def _start_state_log(self) -> None:
        """Start the background reader that streams the onboard state estimate."""
        if self._loop_thread is not None:
            return
        self._latest_state = None
        self._log_stop = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name=f"cf-{self.drone_name}-loop", daemon=True
        )
        self._loop_thread.start()
        self._log_future = asyncio.run_coroutine_threadsafe(
            self._state_log_loop(self._log_stop), self._loop
        )
        deadline = time.time() + _STATE_LOG_FIRST_SAMPLE_TIMEOUT
        while self._latest_state is None and time.time() < deadline:
            if self._log_future.done():  # surface reader errors instead of waiting for the timeout
                self._log_future.result()
            time.sleep(0.01)
        if self._latest_state is None:
            raise RuntimeError(
                f"No lighthouse state sample within {_STATE_LOG_FIRST_SAMPLE_TIMEOUT}s. Is the "
                "Lighthouse system powered and the drone within the tracked volume?"
            )

    def _stop_state_log(self) -> None:
        """Signal the background state-log reader to stop and wait for it to finish."""
        if self._log_stop is not None:
            self._log_stop.set()
        if self._log_future is not None:
            try:
                self._log_future.result(timeout=2.0)
            except Exception as exc:
                logger.warning(f"Stopping the lighthouse state log failed: {exc}")
        self._log_future = None
        self._log_stop = None

    def _stop_loop_thread(self) -> None:
        """Stop the background event loop thread, if one is running."""
        if self._loop_thread is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop_thread = None

    async def _state_log_loop(self, stop: threading.Event) -> None:
        """Continuously decode the onboard state estimate into ``self._latest_state``.

        Two log blocks are used because a single CRTP log packet cannot hold all twelve values.
        """
        log = self.cf.log()
        pos_vel_block = await log.create_block()
        for variable in POS_VEL_VARS:
            await pos_vel_block.add_variable(variable)
        ori_rate_block = await log.create_block()
        for variable in ORI_RATE_VARS:
            await ori_rate_block.add_variable(variable)

        pos_vel_stream = await pos_vel_block.start(_STATE_LOG_START_ARG)
        ori_rate_stream = await ori_rate_block.start(_STATE_LOG_START_ARG)
        try:
            while not stop.is_set():
                pos_vel = (await pos_vel_stream.next()).data
                ori_rate = (await ori_rate_stream.next()).data
                self._latest_state = decode_state(pos_vel, ori_rate)
        finally:
            await pos_vel_stream.stop()
            await ori_rate_stream.stop()

    async def _check_lighthouse_deck(self) -> None:
        """Verify a Lighthouse deck is attached and detected."""
        value = await self.cf.param().get(LIGHTHOUSE_DECK_PARAM)
        if value != 1:
            raise RuntimeError(
                f"Lighthouse deck not detected ({LIGHTHOUSE_DECK_PARAM}={value!r}). Is the deck "
                "attached and flashed, and are the base stations powered?"
            )

    # region Async cflib2 operations

    async def _connect(self, timeout: float) -> None:
        if self.is_connected:
            return

        async def _power_cycle(uri: str) -> None:
            try:
                await CflibCrazyflie.power_off_stm32_domain(self.context, uri)
                await asyncio.sleep(0.1)
                await CflibCrazyflie.power_on_stm32_domain(self.context, uri)
            except CrazyflieError as exc:
                logger.warning(f"Power cycling {uri} failed: {exc}")

        if self.power_cycle_on_connect:
            await asyncio.gather(_power_cycle(self.uri))
            await asyncio.sleep(_POWER_CYCLE_BOOT_WAIT)

        logger.info(f"Connecting to Crazyflie at {self.uri}...")
        results = await asyncio.gather(
            asyncio.wait_for(
                CflibCrazyflie.connect_from_uri(self.context, self.uri, self.toc_cache),
                timeout=timeout,
            ),
            return_exceptions=True,
        )
        result = results[0]
        if isinstance(result, BaseException):
            self._cf = None
            self._commander_level = None
            raise RuntimeError(f"Connecting to Crazyflie failed: {self.uri}: {result}") from result

        self._cf = result
        logger.info(f"Crazyflie connected to {self.uri}")
        if self.lighthouse:
            await self._check_lighthouse_deck()

    async def _disconnect(self) -> None:
        if self._cf is None:
            return
        try:
            await self._cf.disconnect()
        except CrazyflieError as exc:
            logger.error(f"Disconnecting {self.uri} failed: {exc}")
        finally:
            self._cf = None
            self._commander_level = None

    async def _reset_estimator(self) -> None:
        param = self.cf.param()
        # In lighthouse mode the drone derives its absolute pose from the base stations, so we do
        # not seed the estimator with an external pose and let it converge on its own.
        if not self.lighthouse:
            pos = self._ros_connector.pos[self.drone_name]
            quat = self._ros_connector.quat[self.drone_name]
            await param.set("kalman.initialX", pos[0])
            await param.set("kalman.initialY", pos[1])
            await param.set("kalman.initialZ", pos[2])
            yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]
            await param.set("kalman.initialYaw", yaw)
        await param.set("kalman.resetEstimation", 1)
        await asyncio.sleep(0.1)
        await param.set("kalman.resetEstimation", 0)

    async def _apply_settings(self) -> None:
        param = self.cf.param()
        # Estimators: 1: complementary, 2: Kalman. We recommend Kalman from real-world tests.
        await param.set("stabilizer.estimator", 2)
        await asyncio.sleep(0.1)
        # Enable/disable tumble control. Required 0 for aggressive maneuvers.
        await param.set("supervisor.tmblChckEn", 1)
        # Choose controller: 1: PID; 2: Mellinger.
        await param.set("stabilizer.controller", 2)
        # Rate: 0, angle: 1.
        await param.set("flightmode.stabModeRoll", 1)
        await param.set("flightmode.stabModePitch", 1)
        await param.set("flightmode.stabModeYaw", 1)
        await asyncio.sleep(0.1)

    async def _unlock_thrust(self) -> None:
        await self._change_commander_level("low")
        await self.cf.commander().send_setpoint_rpyt(0.0, 0.0, 0.0, 0)

    async def _send_external_pose(self) -> None:
        pos = self._ros_connector.pos[self.drone_name]
        quat = self._ros_connector.quat[self.drone_name]
        await self.cf.localization().external_pose().send_external_pose(pos=pos, quat=quat)

    async def _send_attitude_setpoint(
        self, roll: float, pitch: float, yaw_rate: float, thrust: int
    ) -> None:
        await self._change_commander_level("low")
        await self.cf.commander().send_setpoint_rpyt(roll, pitch, yaw_rate, thrust)

    async def _send_full_state_setpoint(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        acc: NDArray[np.floating],
        quat: NDArray[np.floating],
        body_rates: NDArray[np.floating],
    ) -> None:
        await self._change_commander_level("low")
        await self.cf.commander().send_setpoint_full_state(
            pos, vel, acc, quat, body_rates[0], body_rates[1], body_rates[2]
        )

    async def _stop_setpoint(self) -> None:
        await self.cf.commander().send_stop_setpoint()

    async def _prepare_high_level(self) -> None:
        await self._stop_setpoint()
        await self._change_commander_level("high")

    async def _go_to(
        self, pos: NDArray[np.floating], yaw: float, duration: float, linear: bool = False
    ) -> None:
        await self._change_commander_level("high")
        await self.cf.high_level_commander().go_to(
            pos[0], pos[1], pos[2], yaw, duration, False, linear, None
        )

    async def _arm(self) -> None:
        await self.cf.platform().send_arming_request(do_arm=True)
        await asyncio.sleep(0.8)

    async def _emergency_stop(self) -> None:
        await self.cf.localization().emergency().send_emergency_stop()

    async def _change_commander_level(self, level: Literal["low", "high"]) -> None:
        if self._commander_level == level:
            return

        cf = self.cf
        if level == "high":
            await cf.commander().send_notify_setpoint_stop(0)
        await cf.param().set("commander.enHighLevel", int(level == "high"))
        self._commander_level = level
