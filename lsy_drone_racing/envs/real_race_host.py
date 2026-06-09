"""Real-world drone racing host for multi-drone coordination.

The RealRaceHost manages the central coordination of multi-drone racing:
track validation, drone connection, client synchronization, race operation,
and graceful shutdown. Communication with clients is handled via ROS2.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import struct
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cflib
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm
from drone_racing_msgs.msg import RealClientState, RealHostReady, RealRaceStart
from drone_racing_msgs.srv import RealCalibrateClock
from scipy.spatial.transform import RigidTransform as Tr
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.utils import load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.ros_race_comm import RaceCommNode

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CrazyflieWorker:
    """Manages a single Crazyflie drone in a dedicated subprocess.

    Connects to the drone via radio, resets it to its initial state, and runs
    a control loop that forwards actions received from the client over ROS2.
    External position updates from the ROS estimator are forwarded to the drone's
    Kalman filter at a fixed rate.
    """

    POS_UPDATE_FREQ = 30  # Hz at which MoCap poses are forwarded to the drone's Kalman filter

    def __init__(
        self,
        rank: int,
        drone_id: int,
        drone_channel: int,
        drone_model: str,
        stop_event: mp.synchronize.Event,
        init_barrier: mp.synchronize.Barrier,
        init_pose: Tr,
        control_mode: str,
        control_freq: float = 50.0,
    ):
        """Initialize the Crazyflie worker.

        Args:
            rank: Index of this drone among all drones in the race.
            drone_id: Crazyflie hardware ID (used to build the radio URI).
            drone_channel: Radio channel to connect on.
            drone_model: Drone model name for loading thrust/PWM parameters.
            stop_event: Set by the host to request a shutdown.
            init_barrier: Shared barrier; all workers and the host call :meth:`wait` once
                initialized so that all drones start simultaneously. Any worker that fails
                calls :meth:`abort` to propagate the failure to everyone.
            init_pose: Initial pose used to seed the drone's Kalman filter.
            control_mode: Either ``"attitude"`` or ``"state"``.
            control_freq: Frequency in Hz at which actions are forwarded to the drone.
        """
        self.rank = rank
        self.drone_id = drone_id
        self.drone_channel = drone_channel
        self.drone_model = drone_model
        self.stop_event = stop_event
        self.init_barrier = init_barrier
        self.connected = False  # Once connected, set to True.
        self.connection_failed = (
            False  # Set to True if connection fails during the initial connection phase
        )
        self.connection_lost = False  # Set to True if connection is lost after being established

        self.init_pose = init_pose
        self.control_mode = control_mode.lower()
        self.control_freq = control_freq

        logging.basicConfig(level=logging.INFO, format=f"[Drone {rank}] %(levelname)s: %(message)s")
        logging.getLogger("cflib").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.drone_name = f"cf{drone_id}"
        self.drone: Crazyflie | None = None
        self.params: dict | None = None
        self.last_msg: RealClientState | None = None
        self.action_lock = threading.Lock()
        self._comm: RaceCommNode | None = None
        self._ros_connector: ROSConnector | None = None
        self._last_drone_pos_update: float = 0.0

    def _apply_drone_settings(self):
        """Apply firmware settings required for racing.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        self.drone.param.set_value("stabilizer.estimator", 2)  # 1: complementary, 2: kalman
        time.sleep(0.1)
        self.drone.param.set_value("supervisor.tmblChckEn", 1)
        self.drone.param.set_value("stabilizer.controller", 2)  # 1: PID, 2: Mellinger
        self.drone.param.set_value("flightmode.stabModeRoll", 1)  # 0: rate, 1: angle
        self.drone.param.set_value("flightmode.stabModePitch", 1)
        self.drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)

    def _crazyflie_reset(self):
        """Arm the drone and reset the Kalman filter to the initial pose."""
        self.drone.platform.send_arming_request(True)
        self._apply_drone_settings()
        pos = self.init_pose.translation
        self.drone.param.set_value("kalman.initialX", pos[0])
        self.drone.param.set_value("kalman.initialY", pos[1])
        self.drone.param.set_value("kalman.initialZ", pos[2])
        yaw = self.init_pose.rotation.as_euler("xyz", degrees=False)[2]
        self.drone.param.set_value("kalman.initialYaw", yaw)
        self.drone.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.drone.param.set_value("kalman.resetEstimation", "0")
        if self.control_mode == "attitude":
            # Required to unlock the firmware's thrust protection before the first setpoint
            self.drone.commander.send_setpoint(0, 0, 0, 0)

    def _send_action(self, action: NDArray[np.float32]):
        """Forward an action to the drone.

        Args:
            action: For attitude mode, a 4-element array ``[roll, pitch, yaw, thrust]`` in
                radians and Newtons. For state mode, a 13-element array
                ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rollrate, pitchrate, yawrate]``.
        """
        if self.control_mode == "attitude":
            if action.shape[0] != 4:
                raise ValueError(f"Attitude action must have shape (4,), got {action.shape}")
            pwm = force2pwm(action[3], self.params["thrust_max"] * 4, self.params["pwm_max"])
            pwm = np.clip(pwm, self.params["pwm_min"], self.params["pwm_max"])
            self.drone.commander.send_setpoint(*np.rad2deg(action[:3]), int(pwm))
        else:
            if action.shape[0] != 13:
                raise ValueError(f"State action must have shape (13,), got {action.shape}")
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )

    def _on_client_state(self, msg: RealClientState):
        """Store the latest action from the client state message."""
        with self.action_lock:
            self.last_msg = msg

        latency_ms = (time.time() - msg.timestamp) * 1000

        self.logger.debug(f"Action received (gate={msg.next_gate_idx}, latency={latency_ms:.2f}ms)")

    def _connect_drone(self):
        """Connect to the Crazyflie drone via radio.

        Power-cycles the drone first, then opens the radio link. Raises on connection
        failure, link loss (e.g. "Too many packets lost"), or timeout.

        Raises:
            RuntimeError: If the connection fails or the link is lost before full connection.
            TimeoutError: If the drone does not connect within 10 seconds.
        """
        self.logger.info(f"Connecting to drone {self.drone_id} on channel {self.drone_channel}...")
        self.drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))

        cflib.crtp.init_drivers()
        uri = f"radio://{self.rank}/{self.drone_channel}/2M/E7E7E7E7{self.drone_id:02X}"
        PowerSwitch(uri).stm_power_cycle()
        time.sleep(2)

        def on_connected(_: str):
            self.connected = True

        def on_connection_failed(uri_failed: str, msg: str):
            self.logger.error(f"Connection failed to {uri_failed}: {msg}")
            self.connection_failed = True

        def on_connection_lost(uri_lost: str, msg: str):
            if self.connected:
                self.logger.warning(f"Connection lost to {uri_lost}: {msg}")
                self.connection_lost = True
                self.init_barrier.abort()

        self.drone.fully_connected.add_callback(on_connected)
        self.drone.connection_failed.add_callback(on_connection_failed)
        self.drone.connection_lost.add_callback(on_connection_lost)
        self.drone.open_link(uri)

        start_time = time.time()
        while time.time() - start_time < 10.0:
            if self.connection_failed or self.connection_lost:
                raise RuntimeError(f"Connection failed to drone {self.drone_id}")
            if self.connected:
                break
            time.sleep(0.05)

        if not self.connected:
            raise TimeoutError(
                f"Timed out waiting for drone {self.drone_id} on channel {self.drone_channel}."
            )

        self.logger.info(f"Connected to {uri}")

    def _init_ros_comm(self):
        """Subscribe to client state messages for this drone via ROS2."""
        self._comm = RaceCommNode(f"lsy_race_worker_{self.rank}")
        self._sub = self._comm.node.create_subscription(
            RealClientState,
            f"lsy_drone_racing/client/drone_{self.rank}/state",
            self._on_client_state,
            10,
        )

    def _init_ros_connector(self):
        """Open a ROS connector for reading this drone's pose from the estimator."""
        self.logger.info(f"Initializing ROS connector for {self.drone_name}...")
        self._ros_connector = ROSConnector(
            estimator_names=[self.drone_name],
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )

    def _control_loop(self):
        """Send actions to the drone at the configured control frequency."""
        with self.action_lock:
            self.last_msg = None  # Clear any stale message received during initialization

        dt = 1.0 / self.control_freq

        self._last_drone_pos_update = time.perf_counter()
        while not self.stop_event.is_set() and not self.connection_lost:
            t_start = time.time()

            with self.action_lock:
                if self.last_msg and (t_start - self.last_msg.timestamp) > 10 * dt:
                    self.logger.error(
                        f"No command received for 10 * {dt:.2f}s, handover control to host..."
                    )
                    break
                if self.last_msg and self.last_msg.controller_finished:
                    self.logger.info(
                        "Received stop signal from client, handover control to host..."
                    )
                    break
                action = list(self.last_msg.action) if self.last_msg else None

            if action is not None:
                action_array = (
                    np.array(action) if isinstance(action, (list, tuple)) else np.array([action])
                )
                self._send_action(action_array)

            if (t := time.perf_counter()) - self._last_drone_pos_update > 1 / self.POS_UPDATE_FREQ:
                pos = self._ros_connector.pos[self.drone_name]
                quat = self._ros_connector.quat[self.drone_name]
                self.drone.extpos.send_extpose(*pos, *quat)
                self._last_drone_pos_update = t
            elapsed = time.time() - t_start

            time.sleep(max(0.0, dt - elapsed))

    def _cleanup(self):
        """Send an emergency stop, close all connections, and shut down ROS."""
        if self._ros_connector:
            self._ros_connector.close()
        if self._comm:
            self._comm.close()
        if self.drone:
            try:
                if self.connected and not self.connection_lost:
                    pk = CRTPPacket()
                    pk.port = CRTPPort.LOCALIZATION
                    pk.channel = Localization.GENERIC_CH
                    pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
                    self.drone.send_packet(pk)
            finally:
                self.drone.close_link()
        rclpy.shutdown()
        self.logger.info("Drone process finished")

    def run(self):
        """Run the worker: connect to the drone, initialize, and enter the control loop.

        Exceptions during initialization abort the shared barrier to notify the host and
        all other workers. A BrokenBarrierError means another worker already failed.
        """
        rclpy.init()
        try:
            assert self.control_mode in ["attitude", "state"]
            self.params = load_params(physics="first_principles", drone_model=self.drone_model)
            tasks = [
                self._connect_drone,
                self._crazyflie_reset,
                self._init_ros_connector,
                self._init_ros_comm,
            ]
            for task in tasks:
                if self.stop_event.is_set():
                    return
                task()
            self.logger.info("Waiting for start signal...")
            self.init_barrier.wait(timeout=None)
            self._control_loop()
        except threading.BrokenBarrierError:
            # This will ONLY trigger during initilization phase,
            # since no further wait() will be called here
            pass
        except TimeoutError:
            self.logger.error("Initialization timed out, aborting...")
            self.init_barrier.abort()
        except Exception:
            self.init_barrier.abort()
            raise
        finally:
            self._cleanup()

    @staticmethod
    def crazyflie_process_worker(
        rank: int,
        drone_id: int,
        drone_channel: int,
        drone_model: str,
        stop_event: mp.synchronize.Event,
        init_pose: Tr,
        control_mode: str,
        init_barrier: mp.synchronize.Barrier,
        control_freq: float = 50.0,
    ):
        """Multiprocessing entry point that creates and runs a :class:`CrazyflieWorker`.

        SIGINT is ignored so that only the host process handles keyboard interrupts.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        CrazyflieWorker(
            rank=rank,
            drone_id=drone_id,
            drone_channel=drone_channel,
            drone_model=drone_model,
            stop_event=stop_event,
            init_pose=init_pose,
            control_mode=control_mode,
            control_freq=control_freq,
            init_barrier=init_barrier,
        ).run()


class CrazyflieRealRaceHost:
    """Race host implementation for multi-drone racing with Crazyflie drones.

    Each drone runs in its own subprocess (:class:`CrazyflieWorker`) that handles
    radio communication independently. The host coordinates the race lifecycle via
    ROS2 messages to the client processes.
    """

    gates: ConfigDict
    obstacles: ConfigDict
    drones_pose: ConfigDict
    n_gates: int
    pos_limit_high: NDArray[np.float32]
    pos_limit_low: NDArray[np.float32]

    _num_drones: int
    _drone_names: list[str]
    _drone_ids: list[int]
    _drone_channels: list[int]
    _drone_models: list[str]
    _processes: list[mp.Process]
    _drone_control_freq: list[float]
    _drone_control_mode: list[str]
    _stop_event: mp.synchronize.Event | None
    _init_barrier: mp.synchronize.Barrier | None
    _mp_ctx: mp.context.BaseContext

    def __init__(self, track: ConfigDict, deploy_args: list[dict], control_args: list[dict]):
        """Initialize the host.

        Args:
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            deploy_args: List of drone configs, each with ``id``, ``channel``, and ``drone_model``.
            control_args: Per-drone kwargs, each with ``freq`` and ``control_mode``.
        """
        self.gates, self.obstacles, self.drones_pose = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(track.safety_limits["pos_limit_high"])
        self._drone_names = [f"cf{drone['id']}" for drone in deploy_args.drones]
        self._drone_ids = [drone["id"] for drone in deploy_args.drones]
        self._drone_channels = [drone["channel"] for drone in deploy_args.drones]
        self._drone_models = [drone["drone_model"] for drone in deploy_args.drones]
        self._drone_control_freq = [kwargs["freq"] for kwargs in control_args]
        self._drone_control_mode = [kwargs["control_mode"] for kwargs in control_args]
        self._num_drones = len(deploy_args.drones)
        self._mp_ctx = mp.get_context("spawn")
        self._processes = []
        self._stop_event = None
        self._init_barrier = None
        self._shutdown_event = threading.Event()
        self._clients_ready: dict[int, bool] = {rank: False for rank in range(self._num_drones)}
        self._clients_stopped: dict[int, bool] = {rank: False for rank in range(self._num_drones)}
        self._start_time = time.time()
        self._comm = None
        self._host_ready_pub = None
        self._race_start_pub = None
        self.init_comm()

    def init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode("lsy_race_host")
        node = self._comm.node
        self._host_ready_pub = node.create_publisher(
            RealHostReady, "lsy_drone_racing/host/ready", 10
        )
        self._race_start_pub = node.create_publisher(
            RealRaceStart, "lsy_drone_racing/host/race_start", 10
        )
        self._subs = []
        for rank in range(self._num_drones):

            def on_client_state(msg: RealClientState, rank: int = rank):
                if not self._clients_ready[rank]:
                    logger.debug(f"Client {rank} ready")
                    self._clients_ready[rank] = True
                if msg.controller_finished:
                    logger.info(f"Client {rank} stopped (gate={msg.next_gate_idx})")
                    self._clients_stopped[rank] = True

            self._subs.append(
                node.create_subscription(
                    RealClientState,
                    f"lsy_drone_racing/client/drone_{rank}/state",
                    on_client_state,
                    10,
                )
            )
        logger.debug("ROS2 communication initialized")

    def check_track(
        self, rng_config: ConfigDict, check_objects: bool = True, check_drones: bool = True
    ) -> None:
        """Verify that gates, obstacles, and drones are within their allowed tolerances.

        Args:
            rng_config: Randomization config used to determine position tolerances.
            check_objects: Whether to validate gate and obstacle positions.
            check_drones: Whether to validate drone start positions.
        """
        if not check_objects and not check_drones:
            return
        logger.debug("Checking track configuration...")
        if check_objects:
            check_race_track(
                gates_pos=self.gates.pos,
                nominal_gates_pos=self.gates.nominal_pos,
                gates_quat=self.gates.quat,
                nominal_gates_quat=self.gates.nominal_quat,
                obstacles_pos=self.obstacles.pos,
                nominal_obstacles_pos=self.obstacles.nominal_pos,
                rng_config=rng_config,
            )
            logger.debug("Track object check passed")
        if check_drones:
            for rank, drone_name in enumerate(self._drone_names):
                check_drone_start_pos(
                    nominal_pos=self.drones_pose.nominal_pos[rank],
                    real_pos=self.drones_pose.pos[rank],
                    rng_config=rng_config,
                    drone_name=drone_name,
                )
            logger.debug("Drone start position check passed")

    def connect_drones(self):
        """Spawn one subprocess per drone and wait until all workers finish initialization.

        Returns once all workers are waiting at the init barrier, or immediately if the
        barrier is broken by a worker failure. The race start is triggered in
        :meth:`host_main_loop` when the host calls :meth:`~mp.Barrier.wait` on the barrier.
        """
        logger.debug(f"Spawning processes for {self._num_drones} Crazyflie drones...")
        self._processes = []
        self._stop_event = self._mp_ctx.Event()
        self._init_barrier = self._mp_ctx.Barrier(self._num_drones + 1)

        for rank in range(self._num_drones):
            init_pose = Tr.from_components(
                translation=self.drones_pose.pos[rank],
                rotation=R.from_quat(self.drones_pose.quat[rank]),
            )
            process = self._mp_ctx.Process(
                target=CrazyflieWorker.crazyflie_process_worker,
                args=(
                    rank,
                    self._drone_ids[rank],
                    self._drone_channels[rank],
                    self._drone_models[rank],
                    self._stop_event,
                    init_pose,
                    self._drone_control_mode[rank],
                    self._init_barrier,
                    self._drone_control_freq[rank],
                ),
                name=f"CrazyflieProcess-{rank}",
            )
            process.start()
            self._processes.append(process)
            logger.debug(f"Spawned process for drone {rank} (PID: {process.pid})")

        logger.debug("Waiting for drones to connect...")
        while not self._init_barrier.broken:
            if self._init_barrier.n_waiting == self._num_drones:
                break
            time.sleep(0.05)

    def update_poses(self, track_obj: bool = False, drones: bool = False) -> None:
        """Update gate, obstacle, and/or drone poses from the motion capture system.

        Initializes temporary ROS connectors, reads the current TF poses, writes them
        into ``self.gates``, ``self.obstacles``, and ``self.drones_pose``, then closes
        the connectors.

        Args:
            track_obj: Whether to update gate and obstacle poses.
            drones: Whether to update drone start poses.
        """
        if not track_obj and not drones:
            return

        ros_connector: ROSConnector | None = None
        if track_obj:
            try:
                tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
                tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
                ros_connector = ROSConnector(estimator_names=tf_names, timeout=5.0)
                for i in range(self.n_gates):
                    self.gates.pos[i] = ros_connector.pos[f"gate{i + 1}"]
                    self.gates.quat[i] = ros_connector.quat[f"gate{i + 1}"]
                for i in range(self.n_obstacles):
                    self.obstacles.pos[i] = ros_connector.pos[f"obstacle{i + 1}"]
                    self.obstacles.quat[i] = ros_connector.quat[f"obstacle{i + 1}"]
            finally:
                if ros_connector:
                    ros_connector.close()

        if drones:
            ros_connector = None
            try:
                ros_connector = ROSConnector(estimator_names=self._drone_names, timeout=5.0)
                for rank, drone_name in enumerate(self._drone_names):
                    self.drones_pose.pos[rank] = ros_connector.pos[drone_name]
                    self.drones_pose.quat[rank] = ros_connector.quat[drone_name]
            finally:
                if ros_connector:
                    ros_connector.close()

    def close(self):
        """Stop all drone subprocesses and close ROS communication."""
        self._race_start_pub.publish(
            RealRaceStart(elapsed_time=-1.0, timestamp=time.time(), race_finished=True)
        )
        if self._init_barrier is not None:
            self._init_barrier.abort()
        if self._stop_event is not None:
            self._stop_event.set()
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()

        if self._comm:
            self._comm.close()
        logger.info("Host shutdown complete")

    def _calibrate_client_clocks(self):
        """Expose the clock calibration service and wait for all clients to calibrate.

        Creates a ``lsy_drone_racing/calibrate_clock`` service. Clients discover it via
        ``wait_for_service`` and call it N times to estimate the clock offset using the
        midpoint method.
        """
        logger.info("Starting clock calibration service...")

        def _handler(
            _: RealCalibrateClock.Request, response: RealCalibrateClock.Response
        ) -> RealCalibrateClock.Response:
            response.host_timestamp = time.time()
            return response

        self._calib_srv = self._comm.node.create_service(
            RealCalibrateClock, "lsy_drone_racing/calibrate_clock", _handler
        )
        time.sleep(1.0)
        logger.info("Clock calibration complete")

    def host_main_loop(self, race_update_freq: float = 50.0):
        """Run the host coordination loop.

        Broadcasts :class:`HostReady` until all clients signal readiness, then performs
        clock calibration and releases the drone workers via the init barrier. Enters the
        race loop broadcasting :class:`RaceStart` until all clients report stopping.
        Returns early without error if the init barrier was already broken by a worker failure.

        Args:
            race_update_freq: Frequency in Hz at which :class:`RaceStart` is broadcast.

        Raises:
            TimeoutError: If clients do not become ready within 300 seconds.
        """
        if self._init_barrier.broken:
            return
        logger.info("Drones connected, waiting for clients...")
        t_start = time.time()
        while time.time() - t_start < 300.0:
            self._host_ready_pub.publish(RealHostReady(elapsed_time=0.0, timestamp=time.time()))
            if all(self._clients_ready.values()):
                logger.info("All clients ready")
                break
            time.sleep(0.1)

        if not all(self._clients_ready.values()):
            raise TimeoutError("Timeout waiting for all clients to become ready")

        self._calibrate_client_clocks()

        try:
            self._init_barrier.wait(timeout=None)
        except threading.BrokenBarrierError:
            return

        logger.info("Race started")
        self._start_time = time.time()

        while True:
            elapsed_time = time.time() - self._start_time
            finished = all(self._clients_stopped.values())
            self._race_start_pub.publish(
                RealRaceStart(elapsed_time=elapsed_time, timestamp=time.time(), race_finished=False)
            )
            if finished:
                logger.info("All clients stopped")
                break
            time.sleep(1.0 / race_update_freq)
