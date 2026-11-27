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
from drone_estimators.ros import ROSConnector
from gymnasium import Env
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Float64MultiArray, Int32

from lsy_drone_racing.envs.real_race_env import EnvData
from lsy_drone_racing.envs.utils import gate_passed, load_gate_order, load_track
from lsy_drone_racing.utils.ros import track_poses
from lsy_drone_racing.utils.ros_race_comm import RaceCommNode

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
        A single action vector for the drone identified by ``rank``.

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
                ``drone`` keys.
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
        self.drone_id = drones[rank]["id"]

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

        self._race_start_event = threading.Event()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and open the ROS connections.

        Args:
            seed: Unused in real environments.
            options: Deploy options to determine whether to load real track object poses

        Returns:
            Initial observation and info dictionaries.

        Raises:
            TimeoutError: If the state of a drone is not received within 10 seconds.
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
        """Register this drone with the host and block until the race starts.

        Args:
            timeout: Maximum time in seconds to wait for the race start.

        Raises:
            TimeoutError: If the race does not start within ``timeout`` seconds.
        """
        logger.info("Registering with host, waiting for race start...")
        self._register_pub.publish(Int32(data=self.drone_id))

        if not self._race_start_event.wait(timeout):
            raise TimeoutError(f"Timeout waiting for race start after {timeout}s.")

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

        # Each drone is checked against its own next gate. Finished drones index the last gate
        n_track = len(self.gate_sequence)
        idx = np.minimum(self.data.n_gates_passed, n_track - 1)
        gate_ids = self.gate_sequence[idx]
        gate_reverse = self.gate_sequence_direction[idx] < 0
        gate_pos = self.gates.pos[gate_ids]
        gate_quat = self.gates.quat[gate_ids]

        with jax.default_device(self.device):
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, gate_reverse, (0.45, 0.45)
            )
        passed = np.asarray(passed) & (self.data.n_gates_passed < n_track)  # Finished drones stop
        self.data.n_gates_passed = self.data.n_gates_passed + passed
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

        self._send_action_update(action)

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

    def close(self, deregister: bool = True):
        """Deregister from the host and close all ROS connections.

        Args:
            deregister: Whether to deregister from the host, which makes the drone return to its
                start position. If False, the drone worker detects the missing actions and
                emergency stops the drone.
        """
        logger.info("Closing environment...")
        if deregister and self._comm:
            self._deregister_pub.publish(Int32(data=self.drone_id))
            time.sleep(0.2)  # Give the message time to go out before the node is destroyed
        if self._comm:
            self._comm.close()
        if self._ros_connector:
            self._ros_connector.close()
        logger.debug("Environment closed")

    def _send_action_update(self, action: NDArray):
        """Publish the action to the drone worker on the host.

        Args:
            action: Current control action.
        """
        msg = Float64MultiArray(data=np.asarray(action, dtype=np.float64).tolist())
        self._client_action_pub.publish(msg)

    def _init_ros_connectors(self):
        """Open the ROS connector for the estimators of all drones."""
        self._ros_connector = ROSConnector(estimator_names=self.drone_names, timeout=10.0)

    def _init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode(f"lsy_race_client_{self.rank}")
        node = self._comm.node

        # TODO make this more generic. This code duplicates in race host
        latching_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._register_pub = node.create_publisher(Int32, "lsy_drone_racing/register", latching_qos)

        def race_started(msg: Empty):
            self._race_start_event.set()

        self._race_start_sub = node.create_subscription(
            Empty, "lsy_drone_racing/race_start", race_started, latching_qos
        )
        self._client_action_pub = node.create_publisher(
            Float64MultiArray, f"lsy_drone_racing/{self.drone_name}/action", 10
        )
        self._deregister_pub = node.create_publisher(
            Int32, "lsy_drone_racing/deregister", latching_qos
        )
        logger.debug("ROS2 communication initialized")

    def _all_drone_states(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read positions, quaternions, velocities, and angular velocities for all drones.

        Returns:
            Tuple of ``(pos, quat, vel, ang_vel)``, each of shape ``(n_drones, ...)``.
        """
        c = self._ros_connector
        pos, quat, vel, ang_vel = (
            np.stack([states[name] for name in self.drone_names])
            for states in (c.est_pos, c.est_quat, c.est_vel, c.est_ang_vel)
        )
        return pos, quat, vel, ang_vel
