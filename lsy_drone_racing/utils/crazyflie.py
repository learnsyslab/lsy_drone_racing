"""Crazyflie cflib2 wrapper for drone racing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cflib2 import Crazyflie as CflibCrazyflie
from cflib2 import LinkContext
from cflib2.error import CrazyflieError, DisconnectedError, LinkError, TimeoutError
from cflib2.toc_cache import FileTocCache
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.transform import force2pwm
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["Crazyflie"]

_DISCONNECT_ERRORS = (DisconnectedError, LinkError, TimeoutError)
_POWER_CYCLE_BOOT_WAIT = 3.0


class Crazyflie:
    """Synchronous single-drone wrapper around the asynchronous cflib2 API.

    The environment owns ROS and observation assembly. This class owns only the Crazyflie radio
    link, firmware parameters, command streaming, external-pose injection, and shutdown.
    """

    def __init__(
        self,
        uri: str,
        drone_name: str,
        cache_dir: str | Path | None = None,
        power_cycle_on_connect: bool = True,
    ):
        """Create a Crazyflie wrapper.

        Args:
            uri: Crazyflie radio URI.
            drone_name: Name of the drone in ROS, e.g. cf10.
            cache_dir: Directory used for cflib2 TOC caching.
            power_cycle_on_connect: Whether to power-cycle the STM32 domain before connecting.
        """
        self.uri = uri
        self.drone_name = drone_name
        self.power_cycle_on_connect = power_cycle_on_connect
        self.context = LinkContext()
        cache_dir = Path(__file__).parent / ".cache" if cache_dir is None else Path(cache_dir)
        self.toc_cache = FileTocCache(str(cache_dir))
        self._ros_connector = ROSConnector(
            tf_names=[self.drone_name], cmd_topic=f"/drones/{self.drone_name}/command", timeout=10.0
        )

        self.cf: CflibCrazyflie | None = None
        self.active = False
        self._commander_level: Literal["low", "high"] | None = None
        self._loop = asyncio.new_event_loop()

    @classmethod
    def from_radio(
        cls,
        radio_id: int,
        radio_channel: int,
        drone_id: int,
        drone_name: str | None = None,
        cache_dir: str | Path | None = None,
        power_cycle_on_connect: bool = True,
    ) -> Crazyflie:
        """Create a Crazyflie wrapper from deployment radio settings."""
        return cls(
            f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7{drone_id:02X}",
            f"cf{drone_id}" if drone_name is None else drone_name,
            cache_dir=cache_dir,
            power_cycle_on_connect=power_cycle_on_connect,
        )

    @property
    def is_connected(self) -> bool:
        """Return whether the Crazyflie is currently connected."""
        return self.active and self.cf is not None

    def connect(self, timeout: float = 10.0) -> None:
        """Connect to the Crazyflie."""
        if self._loop.is_closed():
            raise RuntimeError("Crazyflie wrapper is already closed.")
        self._run(self._connect(timeout))

    def reset(self, arm: bool = False) -> None:
        """Apply race settings, reset the estimator, and optionally arm the drone."""
        self._run(self._apply_settings())
        self._run(self._reset_estimator())
        if arm:
            self._run(self._arm())
            self._run_command("Unlocking thrust", self._unlock_thrust())

    def send_external_pose(self) -> None:
        """Send an external mocap pose to the Crazyflie estimator."""
        self._run_command("External pose update", self._send_external_pose())

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
        self._run_command("Attitude setpoint", self._send_attitude_setpoint(*command))
        if publish_to_ros:
            self._ros_connector.publish_cmd(command)

    def send_action_state(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating] | None = None,
        acc: NDArray[np.floating] | None = None,
        yaw: float | None = None,
        body_rates: NDArray[np.floating] | None = None,
    ) -> None:
        """Send a full-state command with yaw-only orientation."""
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
        self._run_command(
            "Full-state setpoint", self._send_full_state_setpoint(pos, vel, acc, quat, body_rates)
        )

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
        self._run_command("Preparing high-level commander", self._prepare_high_level())

        def wait_for_action(duration: float) -> None:
            end_time = self._loop.time() + duration
            while self._loop.time() < end_time:
                if check_ok is not None and not check_ok():
                    raise RuntimeError("Return-to-start was interrupted")
                if not self.is_connected:
                    raise RuntimeError("Drone connection lost")
                self.send_external_pose()
                self._run(asyncio.sleep(0.05))

        vel_norm = np.linalg.norm(initial_obs["vel"])
        break_pos = initial_obs["pos"].copy()
        if vel_norm > 1e-6:
            break_pos += initial_obs["vel"] / vel_norm * breaking_distance
        break_pos[2] = return_height
        self._run_command("High-level goto", self._go_to(break_pos, 0.0, breaking_duration))
        wait_for_action(breaking_duration)

        return_pos = return_pos.copy()
        return_pos[2] = return_height
        self._run_command("High-level goto", self._go_to(return_pos, 0.0, return_duration))
        wait_for_action(return_duration)

        return_pos[2] = 0.05
        self._run_command("High-level goto", self._go_to(return_pos, 0.0, land_duration))
        wait_for_action(land_duration)

    def go_to(
        self,
        pos: NDArray[np.floating],
        yaw: float = 0.0,
        duration: float = 3.0,
        linear: bool = False,
    ) -> None:
        """Send a high-level goto command."""
        self._run_command("High-level goto", self._go_to(pos, yaw, duration, linear=linear))

    def emergency_stop(self) -> None:
        """Send the Crazyflie emergency stop command."""
        self._run_command("Emergency stop", self._emergency_stop())

    def close(self, emergency_stop: bool = True) -> None:
        """Emergency-stop, disconnect, and close the cflib2 event loop."""
        if self._loop.is_closed():
            return
        try:
            if emergency_stop and self.is_connected:
                try:
                    self._run(self._emergency_stop())
                    self._run(asyncio.sleep(0.1))
                except RuntimeError as exc:
                    logger.warning(f"Emergency stop failed during shutdown: {exc}")
                except _DISCONNECT_ERRORS as exc:
                    self._mark_disconnected("Emergency stop", exc)
                except CrazyflieError as exc:
                    logger.warning(f"Emergency stop failed during shutdown: {exc}")

            if self.cf is not None:
                self._run(self._disconnect())
        finally:
            try:
                self._ros_connector.close()
            finally:
                self._loop.close()

    def _run(self, coroutine: Awaitable[Any]) -> Any:
        """Run a cflib2 coroutine on this drone's event loop."""
        return self._loop.run_until_complete(coroutine)

    def _run_command(self, action_name: str, coroutine: Awaitable[None]) -> None:
        if not self.is_connected:
            close = getattr(coroutine, "close", None)
            if close is not None:
                close()
            return
        try:
            self._run(coroutine)
        except _DISCONNECT_ERRORS as exc:
            # Mark disconnected
            self.active = False
            self._commander_level = None
            logger.error(f"{self.uri} disconnected or unreachable. {action_name} failed: {exc}")

    def _cf(self) -> CflibCrazyflie:
        if not self.is_connected:
            raise RuntimeError(f"Drone {self.uri} is not active.")
        assert self.cf is not None
        return self.cf

        

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
            self.cf = None
            self.active = False
            self._commander_level = None
            raise RuntimeError(f"Connecting to Crazyflie failed: {self.uri}: {result}") from result

        self.cf = result
        self.active = True
        logger.info(f"Crazyflie connected to {self.uri}")

    async def _disconnect(self) -> None:
        if self.cf is None:
            self.active = False
            return
        try:
            await self.cf.disconnect()
        except CrazyflieError as exc:
            logger.error(f"Disconnecting {self.uri} failed: {exc}")
        finally:
            self.cf = None
            self.active = False
            self._commander_level = None

    async def _reset_estimator(self) -> None:
        pos = self._ros_connector.pos[self.drone_name]
        quat = self._ros_connector.quat[self.drone_name]
        param = self._cf().param()
        await param.set("kalman.initialX", pos[0])
        await param.set("kalman.initialY", pos[1])
        await param.set("kalman.initialZ", pos[2])
        yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]
        await param.set("kalman.initialYaw", yaw)
        await param.set("kalman.resetEstimation", 1)
        await asyncio.sleep(0.1)
        await param.set("kalman.resetEstimation", 0)

    async def _apply_settings(self) -> None:
        param = self._cf().param()
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
        await self._cf().commander().send_setpoint_rpyt(0.0, 0.0, 0.0, 0)

    async def _send_external_pose(self) -> None:
        pos = self._ros_connector.pos[self.drone_name]
        quat = self._ros_connector.quat[self.drone_name]
        await self._cf().localization().external_pose().send_external_pose(pos=pos, quat=quat)

    async def _send_attitude_setpoint(
        self, roll: float, pitch: float, yaw_rate: float, thrust: int
    ) -> None:
        await self._change_commander_level("low")
        await self._cf().commander().send_setpoint_rpyt(roll, pitch, yaw_rate, thrust)

    async def _send_full_state_setpoint(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        acc: NDArray[np.floating],
        quat: NDArray[np.floating],
        body_rates: NDArray[np.floating],
    ) -> None:
        await self._change_commander_level("low")
        await (
            self._cf()
            .commander()
            .send_setpoint_full_state(
                pos, vel, acc, quat, body_rates[0], body_rates[1], body_rates[2]
            )
        )

    async def _stop_setpoint(self) -> None:
        await self._cf().commander().send_stop_setpoint()

    async def _prepare_high_level(self) -> None:
        await self._stop_setpoint()
        await self._change_commander_level("high")

    async def _go_to(
        self, pos: NDArray[np.floating], yaw: float, duration: float, linear: bool = False
    ) -> None:
        await self._change_commander_level("high")
        await (
            self._cf()
            .high_level_commander()
            .go_to(pos[0], pos[1], pos[2], yaw, duration, False, linear, None)
        )

    async def _arm(self) -> None:
        await self._cf().platform().send_arming_request(do_arm=True)
        await asyncio.sleep(0.8)

    async def _emergency_stop(self) -> None:
        await self._cf().localization().emergency().send_emergency_stop()

    async def _change_commander_level(self, level: Literal["low", "high"]) -> None:
        if self._commander_level == level:
            return

        cf = self._cf()
        if level == "high":
            await cf.commander().send_notify_setpoint_stop(0)
        await cf.param().set("commander.enHighLevel", int(level == "high"))
        self._commander_level = level
