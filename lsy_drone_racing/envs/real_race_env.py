"""Real-world drone racing environments.

This module contains the environments for controlling a single or multiple drones in a real-world
race track. It mirrors the :mod:`~lsy_drone_racing.envs.drone_race` module as closely as possible,
but uses data from real-world observations from motion capture systems and sends actions to the
real drones.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cflib
import jax
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm
from gymnasium import Env
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment."""

    target_gate: NDArray
    gates_visited: NDArray
    obstacles_visited: NDArray
    last_drone_pos: NDArray[np.float32]
    taken_off: bool = False
    drone_connected: bool = False

    @classmethod
    def create(cls, n_drones: int, n_gates: int, n_obstacles: int) -> EnvData:
        """Create an instance of the EnvData class."""
        return EnvData(
            target_gate=np.zeros(n_drones, dtype=int),
            gates_visited=np.zeros((n_drones, n_gates), dtype=bool),
            obstacles_visited=np.zeros((n_drones, n_obstacles), dtype=bool),
            last_drone_pos=np.zeros((n_drones, 3), dtype=np.float32),
        )

    def reset(self, last_drone_pos: NDArray[np.float32]):
        """Reset the environment data."""
        self.target_gate[...] = 0
        self.gates_visited[...] = False
        self.obstacles_visited[...] = False
        self.last_drone_pos[...] = last_drone_pos
        self.taken_off = False
        self.drone_connected = False


# region CoreEnv
class RealRaceCoreEnv:
    """Deployable version of the (multi-agent) drone racing environments.

    This class acts as a generic core implementation of the environment logic that can be reused for
    both single-agent and multi-agent deployments.
    """

    POS_UPDATE_FREQ = 30  # Frequency of position updates to the drone estimator in Hz

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Create a deployable version of the drone racing environment.

        Args:
            drones: List of all drones in the race, including their channel and id.
            rank: Rank of the drone that is controlled by this environment.
            freq: Environment step frequency.
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            randomizations: Randomization configuration.
            sensor_range: Sensor range. Determines at which distance the exact position of the
                gates and obstacles is reveiled.
            control_mode: Control mode of the drone.
        """
        assert rclpy.ok(), "ROS2 is not running. Please start ROS2 before creating a deploy env."
        # Static env data
        self.n_drones = len(drones)
        self.gates, self.obstacles, self.drones = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(track.safety_limits["pos_limit_high"])
        self.sensor_range = sensor_range
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]
        self.drone_channel = drones[rank]["channel"]
        self.drone_id = drones[rank]["id"]
        self.rank = rank
        self.freq = freq
        self.device = jax.devices("cpu")[0]
        assert control_mode in ["state", "attitude"], f"Invalid control mode {control_mode}"
        self.control_mode = control_mode
        self.randomizations = randomizations
        self.drone_parameters = load_params("first_principles", drones[rank]["drone_model"])
        self.drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
        self._drone_healthy = mp.Event()
        self._drone_healthy.set()
        self._ros_connector = ROSConnector(
            estimator_names=self.drone_names,
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )
        # Dynamic data
        self.data = EnvData.create(
            n_drones=self.n_drones, n_gates=self.n_gates, n_obstacles=self.n_obstacles
        )
        self._jit()

    def _reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        options = {} if options is None else options
        # Update the position of gates and obstacles with the real positions measured from Mocap. If
        # disabled, they are equal to the nominal positions defined in the track config.
        if options.get("real_track_objects", True):
            self._update_track_poses()
        if options.get("check_race_track", True):
            check_race_track(
                gates_pos=self.gates.pos,
                nominal_gates_pos=self.gates.nominal_pos,
                gates_quat=self.gates.quat,
                nominal_gates_quat=self.gates.nominal_quat,
                obstacles_pos=self.obstacles.pos,
                nominal_obstacles_pos=self.obstacles.nominal_pos,
                rng_config=self.randomizations,
            )
        if options.get("check_drone_start_pos", True):
            check_drone_start_pos(
                nominal_pos=self.drones.pos[self.rank],
                real_pos=self._ros_connector.pos[self.drone_name],
                rng_config=self.randomizations,
                drone_name=self.drone_name,
            )
        self.data.reset(np.stack([self._ros_connector.pos[n] for n in self.drone_names]))

        self._connect_radio(
            radio_id=self.rank, radio_channel=self.drone_channel, drone_id=self.drone_id
        )
        self._last_drone_pos_update = 0  # Last time a position was sent to the drone estimator
        self._reset_drone()

        if self.control_mode == "attitude":
            # Unlock thrust mode protection by sending a zero thrust command
            self.drone.commander.send_setpoint(0, 0, 0, 0)

        return self.obs(), self.info()

    def _step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment."""
        # Note: We do not send the action to the drone here.
        self.send_action(action)

        drone_pos = np.stack([self._ros_connector.pos[drone] for drone in self.drone_names])
        assert drone_pos.dtype == np.float32, "Drone position must be of type float32"
        drone_quat = np.stack([self._ros_connector.quat[drone] for drone in self.drone_names])
        assert drone_quat.dtype == np.float32, "Drone quaternion must be of type float32"
        # Check if the drone is in the sensor range of the gates and obstacles
        dpos = drone_pos[:, None, :2] - self.gates.pos[None, :, :2]
        self.data.gates_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range
        dpos = drone_pos[:, None, :2] - self.obstacles.pos[None, :, :2]
        self.data.obstacles_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range

        gate_pos = self.gates.pos[self.data.target_gate]
        gate_quat = self.gates.quat[self.data.target_gate]

        with jax.default_device(self.device):  # Ensure gate_passed runs on the CPU
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45)
            )
        self.data.target_gate += np.asarray(passed)
        self.data.target_gate[self.data.target_gate >= self.n_gates] = -1
        self.data.last_drone_pos[...] = drone_pos
        self.data.taken_off |= drone_pos[self.rank, 2] > 0.1
        # Send vicon position updates to the drone at a fixed frequency irrespective of the env freq
        # Sending too many updates may deteriorate the performance of the drone, hence the limiter
        if (t := time.perf_counter()) - self._last_drone_pos_update > 1 / self.POS_UPDATE_FREQ:
            self.drone.extpos.send_extpose(*drone_pos[self.rank], *drone_quat[self.rank])
            self._last_drone_pos_update = t
        return self.obs(), self.reward(), self.terminated(), self.truncated(), self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the observation of the environment."""
        # If gates/obstacles are in sensor range use the actual pose, otherwise use the nominal pose
        # The actual pose is measured at the beginning of the episode and is not updated during the
        # episode. If we want to use dynamic gates/obstacles, we need to update the poses here.
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(
            np.float32
        )
        drone_pos = np.stack([self._ros_connector.pos[drone] for drone in self.drone_names])
        drone_quat = np.stack([self._ros_connector.quat[drone] for drone in self.drone_names])
        drone_vel = np.stack([self._ros_connector.vel[drone] for drone in self.drone_names])
        drone_ang_vel = np.stack([self._ros_connector.ang_vel[drone] for drone in self.drone_names])
        obs = {
            "pos": drone_pos,
            "quat": drone_quat,
            "vel": drone_vel,
            "ang_vel": drone_ang_vel,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
        return obs

    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 * (self.data.target_gate == -1)  # Implicit float conversion

    def terminated(self) -> NDArray:
        """Check if the episode is terminated."""
        terminated = self.data.target_gate == -1
        terminated[self.rank] |= not self._drone_healthy.is_set()
        terminated |= np.any(
            (self.pos_limit_low > self.data.last_drone_pos)
            | (self.data.last_drone_pos > self.pos_limit_high)
        )

        return terminated

    def truncated(self) -> NDArray:
        """Check if the episode is truncated."""
        return np.zeros(self.n_drones, dtype=bool)

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    def send_action(self, action: NDArray):
        """Send the action to the drone."""
        if self.control_mode == "attitude":
            pwm = force2pwm(
                action[3], self.drone_parameters["thrust_max"] * 4, self.drone_parameters["pwm_max"]
            )
            pwm = np.clip(pwm, self.drone_parameters["pwm_min"], self.drone_parameters["pwm_max"])
            action = (*np.rad2deg(action[:3]), int(pwm))
            self.drone.commander.send_setpoint(*action)
            self._ros_connector.publish_cmd(action)
        else:
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            # TODO: We currently limit ourselves to yaw rotation only because the simulation is
            # based on the old crazyswarm full_state command definition. Once the simulation does
            # support the real full_state command, we can remove this limitation and use full
            # quaternions as inputs
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )
            # TODO: The estimators can't handle state commands, so we simply don't send anything
            # Make sure to use the legacy estimator with the state interface

    def _update_track_poses(self):
        """Update the track poses from the motion capture system."""
        tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
        tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
        ros_connector: ROSConnector | None = None
        try:
            ros_connector = ROSConnector(tf_names=tf_names, timeout=10.0)
            pos, quat = ros_connector.pos, ros_connector.quat
        finally:
            if ros_connector is not None:
                ros_connector.close()
        try:
            for i in range(self.n_gates):
                self.gates.pos[i, ...] = pos[f"gate{i + 1}"]
                self.gates.quat[i, ...] = quat[f"gate{i + 1}"]
            for i in range(self.n_obstacles):
                self.obstacles.pos[i, ...] = pos[f"obstacle{i + 1}"]
        except KeyError as e:
            raise KeyError(
                f"Could not find all track objects in the ROS TF tree: {e}. Have you enabled the "
                "track objects in Vicon and started the motion capture tracking node?"
            ) from e

    def _connect_radio(self, radio_id: int, radio_channel: int, drone_id: int):
        """Connect to the drone via radio.

        Raises:
            TimeoutError: If the drone is not reachable within 10 seconds.
        """
        cflib.crtp.init_drivers()
        uri = f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7" + f"{drone_id:02x}".upper()

        power_switch = PowerSwitch(uri)
        power_switch.stm_power_cycle()
        time.sleep(2)

        event = mp.Event()

        def connect_callback(_: str):
            self._drone_healthy.set()
            event.set()

        self.drone.fully_connected.add_callback(connect_callback)
        self.drone.disconnected.add_callback(lambda _: self._drone_healthy.clear())
        self.drone.connection_failed.add_callback(
            lambda _, msg: logger.warning(f"Connection failed: {msg}")
        )
        self.drone.connection_lost.add_callback(
            lambda _, msg: logger.warning(f"Connection lost: {msg}")
        )
        self.drone.open_link(uri)

        logger.info(f"Waiting for drone {drone_id} to connect...")
        connected = event.wait(10.0)
        if not connected:
            raise TimeoutError("Timed out while waiting for the drone.")
        self.data.drone_connected = True
        logger.info(f"Drone {drone_id} connected to {uri}")

    def _reset_drone(self):
        """Arm the drone, reset estimation."""
        # Arm the drone
        self.drone.platform.send_arming_request(True)
        self._apply_drone_settings()
        pos = self._ros_connector.pos[self.drone_name]
        # Reset Kalman filter values
        self.drone.param.set_value("kalman.initialX", pos[0])
        self.drone.param.set_value("kalman.initialY", pos[1])
        self.drone.param.set_value("kalman.initialZ", pos[2])
        quat = self._ros_connector.quat[self.drone_name]
        yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]
        self.drone.param.set_value("kalman.initialYaw", yaw)
        self.drone.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.drone.param.set_value("kalman.resetEstimation", "0")

    def _apply_drone_settings(self):
        """Apply firmware settings to the drone.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        # Estimators: 1: complementary, 2: kalman. We recommend kalman based on real-world tests
        self.drone.param.set_value("stabilizer.estimator", 2)
        time.sleep(0.1)  # TODO: Maybe remove
        # enable/disable tumble control. Required 0 for agressive maneuvers
        self.drone.param.set_value("supervisor.tmblChckEn", 1)
        # Choose controller: 1: PID; 2:Mellinger
        self.drone.param.set_value("stabilizer.controller", 2)
        # rate: 0, angle: 1
        self.drone.param.set_value("flightmode.stabModeRoll", 1)
        self.drone.param.set_value("flightmode.stabModePitch", 1)
        self.drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)  # Wait for settings to be applied

    def _return_to_start(self):
        """Returning controller to fly the drone back to the starting position."""
        # Enable high-level functions of the drone and disable low-level control access
        self.drone.commander.send_stop_setpoint()
        self.drone.commander.send_notify_setpoint_stop()
        self.drone.param.set_value("commander.enHighLevel", "1")
        self.drone.platform.send_arming_request(True)
        # Fly back to the start position
        RETURN_HEIGHT = 1.75  # m
        BREAKING_DISTANCE = 1.0  # m
        BREAKING_DURATION = 3.0  # s
        RETURN_DURATION = 7.0  # s
        LAND_DURATION = 3.0  # s

        def wait_for_action(dt: float):
            tstart = time.perf_counter()
            # Wait for the action to be completed and send the current position to the drone
            while time.perf_counter() - tstart < dt:
                if not rclpy.ok():
                    raise RuntimeError("ROS has already stopped")
                if not self._drone_healthy.is_set():
                    raise RuntimeError("Drone connection lost")
                obs = self.obs()
                self.drone.extpos.send_extpose(*obs["pos"][self.rank], *obs["quat"][self.rank])
                time.sleep(0.05)

        pos = self._ros_connector.pos[self.drone_name]
        vel = self._ros_connector.vel[self.drone_name]
        break_pos = pos + vel / np.linalg.norm(vel) * BREAKING_DISTANCE
        break_pos[2] = RETURN_HEIGHT
        self.drone.high_level_commander.go_to(*break_pos, 0, BREAKING_DURATION)
        wait_for_action(BREAKING_DURATION)
        return_pos = self.drones.pos[self.rank]  # Starting position from the config file
        return_pos[2] = RETURN_HEIGHT
        self.drone.high_level_commander.go_to(*return_pos, 0, RETURN_DURATION)
        wait_for_action(RETURN_DURATION)
        return_pos[2] = 0.05
        self.drone.high_level_commander.go_to(*return_pos, 0, LAND_DURATION)
        wait_for_action(LAND_DURATION)

    def _jit(self):
        """JIT compile jax functions.

        We compile all jit-compiled functions at startup to avoid the overhead of compiling them
        at the first call when the drone is already in the air.
        """
        drone_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        gate_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        gate_quat = np.zeros((self.n_drones, 4), dtype=np.float32)
        with jax.default_device(self.device):
            jax.block_until_ready(
                gate_passed(drone_pos, drone_pos, gate_pos, gate_quat, (0.45, 0.45))
            )

    def close(self):
        """Close the environment.

        If the drone has finished the track, it will try to return to the start position.
        Irrespective of succeeding or not, the drone will be stopped immediately afterwards or in
        case of errors, and close the connections to the ROSConnector.
        """
        try:
            if self.data.taken_off:
                self._return_to_start()
        finally:
            try:
                # Kill the drone
                pk = CRTPPacket()
                pk.port = CRTPPort.LOCALIZATION
                pk.channel = Localization.GENERIC_CH
                pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
                self.drone.send_packet(pk)
                self.drone.close_link()
            finally:
                # Close all ROS connections
                self._ros_connector.close()


# region Single Drone Env
class RealDroneRaceEnv(RealRaceCoreEnv, Env):
    """A Gymnasium environment for controlling a real Crazyflie drone in a physical race track.

    This environment provides a standardized interface for deploying drone racing algorithms on
    physical hardware. It handles communication with the drone through the cflib library and tracks
    the drone's position using a motion capture system via ROS2.

    The environment maintains the same observation and action space as its simulation counterpart,
    allowing for seamless transition from simulation to real-world deployment. It processes sensor
    data, handles gate passing detection, and manages the drone's state throughout the race.

    Features:
    - Interfaces with physical Crazyflie drones through radio communication
    - Tracks drone position and orientation using motion capture data via ROS2
    - Supports both state-based and attitude-based control modes
    - Provides sensor range simulation for gates and obstacles
    - Handles automatic return-to-home behavior when the race is completed

    Note:
        This environment is designed for single-drone racing. For multi-drone racing, use the
        :class:`~lsy_drone_racing.envs.real_race_env.RealMultiDroneRaceEnv` class instead.
    """

    def __init__(
        self,
        drones: list[dict[str, int]],
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the multi-drone environment.

        Action space:
            The action space is a single action vector for the drone with the environment rank.
            See :class:`~.RealRaceCoreEnv` for more information. Depending on the control mode, it
            is either a 13D desired drone state setpoint, or a 4D desired attitude and collective
            thrust setpoint.

        Observation space:
            The observation space is a dictionary containing the state of all drones in the race.
            It mimics exactly the observation space of
            :class:`lsy_drone_racing.envs.drone_race.DroneRaceEnv`.

        Note:
            rclpy must be initialized before creating this environment.

        Args:
            drones: List of all drones in the race, including their channel and id.
            freq: Environment step frequency.
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            randomizations: Randomization configuration.
            sensor_range: Sensor range. Determines at which distance the exact position of the
                gates and obstacles is reveiled.
            control_mode: Control mode of the drone.
        """
        super().__init__(
            drones=drones,
            rank=0,
            freq=freq,
            track=track,
            randomizations=randomizations,
            sensor_range=sensor_range,
            control_mode=control_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        obs, info = self._reset(seed=seed, options=options)
        return {k: v[0, ...] for k, v in obs.items()}, info

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment.

        Args:
            action: Action to be taken by the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        return {k: v[0, ...] for k, v in obs.items()}, reward[0], terminated[0], truncated[0], info


# region Multi Drone Env
class RealMultiDroneRaceEnv(RealRaceCoreEnv, Env):
    """A Gymnasium environment for controlling a specific drone in a multi-drone physical race.

    This environment extends the functionality of `RealRaceCoreEnv` to support multi-drone racing
    scenarios. Each instance of this environment controls a single drone identified by its rank, but
    maintains awareness of all drones in the race. This allows for coordinated multi-drone
    deployments where each drone runs in a separate process with its own controller.

    The environment handles communication with the specific drone through cflib and tracks all
    drones' positions using a motion capture system via ROS2. It provides observations that include
    the state of all drones, allowing controllers to implement collision avoidance or cooperative
    strategies.

    Features:
    - Controls a specific drone in a multi-drone race based on its rank
    - Tracks all drones' positions and states via ROS2
    - Supports both state-based and attitude-based control modes
    - Provides sensor range simulation for gates and obstacles
    - Handles automatic return-to-home behavior when the race is completed

    Action space:
        The action space is a **single** action vector for the drone with the environment rank.
        See :class:`~.RealRaceCoreEnv` for more information.

    Warning:
        The action space differs from the action space of the simulated counterpart. This deviation
        is necessary to run different controller types at different frequencies that asynchronously
        publish ther commands to the drone.

    Observation space:
        The observation space is a dictionary containing the state of all drones in the race.
        It mimics exactly the observation space of
        :class:`lsy_drone_racing.envs.multi_drone_race.MultiDroneRaceEnv`.

    Note:
        Each instance of this environment controls only one drone (specified by rank), but provides
        observations for all drones in the race. This allows us to run controllers at different
        frequencies for different drones. Consequently the step method applies actions only to the
    """

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the multi-drone environment.

        Args:
            drones: List of all drones in the race, including their channel and id.
            rank: Rank of the drone that is controlled by this environment.
            freq: Environment step frequency.
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            randomizations: Randomization configuration.
            sensor_range: Sensor range. Determines at which distance the exact position of the
                gates and obstacles is reveiled.
            control_mode: Control mode of the drone.
        """
        super().__init__(
            drones=drones,
            rank=rank,
            freq=freq,
            track=track,
            randomizations=randomizations,
            sensor_range=sensor_range,
            control_mode=control_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        return self._reset(seed=seed, options=options)

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment.

        Note:
            The action is applied only to the drone with the environment rank!

        Args:
            action: Action to be taken by the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        return obs, reward[self.rank], terminated[self.rank], truncated[self.rank], info
