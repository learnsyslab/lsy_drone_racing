"""Client-side environment for multi-drone racing with host-client architecture.

The RealMultiDroneRaceEnvClient operates as a client in a host-client system:
- Receives coordination messages from the host via ROS2
- Manages a single drone's state and control
- Sends control actions and state updates to the host for supervision
- Handles local observation and gate tracking
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm
from drone_racing_msgs.msg import RealClientAction, RealHostState  # type: ignore[import-untyped]
from drone_racing_msgs.srv import RealCalibrateClock  # type: ignore[import-untyped]
from gymnasium import Env

from lsy_drone_racing.envs.real_race_env import EnvData
from lsy_drone_racing.envs.utils import gate_passed, load_gate_order, load_track
from lsy_drone_racing.utils.ros import track_poses
from lsy_drone_racing.utils.ros_race_comm import RaceCommNode, calibrate_clock

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RealMultiDroneRaceEnvClient(Env):
    """Client-side Gymnasium environment for multi-drone racing.

    Runs on each drone's computing unit. Receives host coordination messages via ROS2,
    computes observations and gate tracking locally, and forwards actions to the host
    which relays them to the physical drone.

    Observation space:
        A dictionary containing the state of all drones in the race, mirroring
        :class:`lsy_drone_racing.envs.multi_drone_race.MultiDroneRaceEnv`.

    Action space:
        A single action vector for the drone identified by ``rank``. See
        :class:`~lsy_drone_racing.envs.real_race_host.CrazyflieWorker` for format details.

    Note:
        rclpy must be initialized before creating this environment.
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
        """Initialize the client-side multi-drone environment.

        Args:
            drones: List of all drones in the race, each with ``id``, ``channel``, and
                ``drone_model`` keys.
            rank: Index of this drone among all drones in the race.
            freq: Control frequency in Hz.
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            randomizations: Randomization configuration (unused on the client side).
            sensor_range: Distance in metres at which gate/obstacle true poses are revealed.
            control_mode: Either ``"state"`` or ``"attitude"``.
        """
        self.n_drones = len(drones)
        self.rank = rank
        self.freq = freq
        self.sensor_range = sensor_range
        self.control_mode = control_mode
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]
        self.drone_parameters: dict = load_params(
            physics="first_principles", drone_model=drones[rank]["drone_model"]
        )

        self.gates, self.obstacles, self.drones_track = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.gate_sequence, self.gate_sequence_direction = load_gate_order(track, self.n_gates)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(track.safety_limits["pos_limit_high"])

        self.device = jax.devices("cpu")[0]
        self._ros_connector: ROSConnector | None = None
        self.data = EnvData.create(self.n_drones, self.n_gates, self.n_obstacles)

        self._comm: RaceCommNode | None = None
        self._client_action_pub: Any = None
        self._clock_calib_client: Any = None

        self._host_ready_event = threading.Event()
        self._race_started = False
        self._race_start_time = 0.0
        self._clock_offset = 0.0
        self._host_finished = False
        self._client_ready = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and wait for the host to signal readiness.

        Args:
            seed: Unused in real environments.
            options: Deploy options to determine whether to load real track object poses

        Returns:
            Initial observation and info dictionaries.

        Raises:
            TimeoutError: If the host does not respond within 120 seconds.
        """
        if options and options.get("real_track_objects", False):
            self.gates.pos, self.gates.quat, self.obstacles.pos = track_poses(
                self.n_gates, self.n_obstacles
            )

        if self._ros_connector is None:
            self._init_ros_connectors()
        if self._comm is None:
            self._init_comm()

        current_pos, _, _, _ = self._all_drone_states()
        self.data.reset(current_pos)

        logger.debug("Environment reset complete")
        return self.obs(), self.info()

    def lock_until_race_start(self, timeout: float = 60.0):
        """Sends dummy messages at the control frequency (``self.freq`` Hz) until the race starts.

        After receiving host ready message, the client will calibrate the clock offset and
            until the host broadcasts :class:`RaceStartMessage`.

        Args:
            timeout: Maximum time in seconds to wait for calibration and race start.

        Raises:
            TimeoutError: If calibration or race start exceeds ``timeout`` seconds.
        """
        logger.info("Waiting for host ready message...")
        stop_sending = threading.Event()

        def send_action_messages():
            while not stop_sending.is_set():
                if self.control_mode == "attitude":
                    dummy_action = np.zeros(4, dtype=np.float32)
                else:
                    dummy_action = np.zeros(13, dtype=np.float32)
                    dummy_action[:3] = self._ros_connector.pos[self.drone_name]
                self._send_action_update(dummy_action, stopped=False)
                time.sleep(1 / self.freq)

        threading.Thread(target=send_action_messages, daemon=True).start()

        if not self._host_ready_event.wait(timeout=timeout):
            stop_sending.set()
            raise TimeoutError(
                "Timeout waiting for host ready. "
                "Host may not be running or network connection failed."
            )

        logger.info("Received host ready message.")
        self._clock_offset = calibrate_clock(self._clock_calib_client, n=5, timeout=timeout)
        logger.info(f"Clock offset = {self._clock_offset * 1000:.2f}ms")
        logger.info("Waiting for race start")

        t_start = time.time()
        while not self._race_started:
            if time.time() - t_start > timeout:
                raise TimeoutError(f"Timeout waiting for race start after {timeout}s.")
            time.sleep(0.001)
        stop_sending.set()
        logger.info("Race starts!")

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a control step: update gate tracking, check bounds, and send the action.

        Args:
            action: Control action for this drone.

        Returns:
            Observation, reward (always 0.0), terminated, truncated (always False), info.
        """
        drone_pos, _, _, _ = self._all_drone_states()

        dpos = drone_pos[:, None, :2] - self.gates.pos[None, :, :2]
        self.data.gates_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range
        dpos = drone_pos[:, None, :2] - self.obstacles.pos[None, :, :2]
        self.data.obstacles_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range

        # Allow for different gate ordering
        gate_id = self.gate_sequence[self.data.n_gates_passed[self.rank]]
        gate_reverse = self.gate_sequence_direction[self.data.n_gates_passed[self.rank]] < 0
        gate_pos = self.gates.pos[gate_id]
        gate_quat = self.gates.quat[gate_id]

        with jax.default_device(self.device):
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, gate_reverse, (0.45, 0.45)
            )
        self.data.n_gates_passed = self.data.n_gates_passed + np.asarray(passed)
        self.data.last_drone_pos[...] = drone_pos
        self.data.taken_off |= drone_pos[self.rank, 2] > 0.1

        terminated = bool(self.data.n_gates_passed[self.rank] >= len(self.gate_sequence))

        within_bound = np.all(
            (drone_pos[self.rank] >= self.pos_limit_low)
            & (drone_pos[self.rank] <= self.pos_limit_high)
        )
        if not within_bound:
            logger.warning("Drone exceeded safety bounds")
            terminated = True

        if self.control_mode == "attitude" and self._ros_connector:
            pwm = force2pwm(
                action[3], self.drone_parameters["thrust_max"] * 4, self.drone_parameters["pwm_max"]
            )
            pwm = np.clip(pwm, self.drone_parameters["pwm_min"], self.drone_parameters["pwm_max"])
            command = (*np.rad2deg(action[:3]), int(pwm))
            self._ros_connector.publish_cmd(command)

        self._send_action_update(action, terminated)

        return self.obs(), 0.0, terminated, False, self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the current observation dictionary."""
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(
            np.float32
        )
        drone_pos, drone_quat, drone_vel, drone_ang_vel = self._all_drone_states()
        return {
            "pos": drone_pos,
            "quat": drone_quat,
            "vel": drone_vel,
            "ang_vel": drone_ang_vel,
            "n_gates_passed": self.data.n_gates_passed,
            "gate_sequence": np.broadcast_to(
                self.gate_sequence, (self.n_drones, len(self.gate_sequence))
            ),
            "gate_sequence_direction": np.broadcast_to(
                self.gate_sequence_direction, (self.n_drones, len(self.gate_sequence))
            ),
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }

    def info(self) -> dict:
        """Return the info dictionary."""
        n_gates_passed = int(self.data.n_gates_passed[self.rank])
        return {
            "rank": self.rank,
            "n_gates_passed": n_gates_passed,
            "finished_track": n_gates_passed >= len(self.gate_sequence),
        }

    def close(self):
        """Send a final stop message and close all ROS connections."""
        logger.info("Closing environment...")
        if self._client_action_pub:
            stop_action = np.zeros(4 if self.control_mode == "attitude" else 13)
            for _ in range(5):
                self._send_action_update(stop_action, stopped=True)
                time.sleep(0.05)
        if self._comm:
            self._comm.close()
        if self._ros_connector:
            self._ros_connector.close()
        logger.debug("Environment closed")

    def set_client_ready(self):
        """Mark the client as fully initialized and ready for race start."""
        self._client_ready = True

    def _send_action_update(self, action: NDArray, stopped: bool):
        """Publish a :class:`ClientActionMessage` to the host.

        The timestamp is adjusted by the calibrated clock offset so the host can
        measure accurate latency without clock skew.

        Args:
            action: Current control action.
            stopped: Whether this client has finished or crashed.
        """
        elapsed_time = time.time() - self._race_start_time if self._race_started else 0.0
        msg = RealClientAction()
        msg.drone_rank = self.rank
        msg.action = action.tolist() if isinstance(action, np.ndarray) else list(action)
        msg.elapsed_time = elapsed_time
        msg.timestamp = time.time() + self._clock_offset
        msg.client_ready = self._client_ready
        msg.controller_stopped = stopped
        self._client_action_pub.publish(msg)

    def _init_ros_connectors(self):
        """Open ROS connector for own drone (estimator) and others (TF)."""
        self._ros_connector = ROSConnector(
            estimator_names=self.drone_names,
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )

    def _init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode(f"lsy_race_client_{self.rank}")
        node = self._comm.node

        def on_host_state(msg: RealHostState):
            latency_ms = (time.time() - msg.timestamp) * 1000
            if msg.host_ready and not self._host_ready_event.is_set():
                self._host_ready_event.set()
                logger.debug(f"Host ready (latency: {latency_ms:.2f}ms)")
            if msg.race_started and not self._race_started:
                self._race_started = True
                self._race_start_time = time.time() - msg.elapsed_time
                logger.debug(f"Race started (latency: {latency_ms:.2f}ms)")
            self._host_finished = bool(msg.race_finished)

        # TODO Why do I need to save this if unused?
        self._sub = node.create_subscription(
            RealHostState, "lsy_drone_racing/host_state", on_host_state, 10
        )
        self._client_action_pub = node.create_publisher(
            RealClientAction, f"lsy_drone_racing/client/drone_{self.rank}/action", 10
        )
        self._clock_calib_client = node.create_client(
            RealCalibrateClock, "lsy_drone_racing/calibrate_clock"
        )
        logger.debug("ROS2 communication initialized")

    def _all_drone_states(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read positions, quaternions, velocities, and angular velocities for all drones.

        Own drone state comes from the high-precision estimator; other drones from TF.
        Fields for unreachable drones are filled with NaN.

        Returns:
            Tuple of ``(pos, quat, vel, ang_vel)``, each of shape ``(n_drones, ...)``.
        """
        pos = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        quat = np.full((self.n_drones, 4), np.nan, dtype=np.float32)
        vel = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        ang_vel = np.full((self.n_drones, 3), np.nan, dtype=np.float32)
        for i, name in enumerate(self.drone_names):
            if i == self.rank:
                pos[i] = self._ros_connector.pos[name]
                quat[i] = self._ros_connector.quat[name]
                vel[i] = self._ros_connector.vel[name]
                ang_vel[i] = self._ros_connector.ang_vel[name]
            else:
                pos[i] = self._ros_connector.pos.get(name, np.nan)
                quat[i] = self._ros_connector.quat.get(name, np.nan)
                vel[i] = self._ros_connector.vel.get(name, np.nan)
                ang_vel[i] = self._ros_connector.ang_vel.get(name, np.nan)
        return pos, quat, vel, ang_vel
