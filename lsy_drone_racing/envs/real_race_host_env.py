"""Real-world drone racing host for multi-drone coordination.

The RealRaceHost manages the central coordination of multi-drone racing:
track validation, drone connection, client synchronization, race operation,
and graceful shutdown. Communication with clients is handled via ROS2.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rclpy
from drone_estimators.ros import ROSConnector
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Float64MultiArray, Int32

from lsy_drone_racing.envs.utils import gate_passed, load_drone_params, load_gate_order, load_track
from lsy_drone_racing.utils.checks import check_drone_start_pos, check_race_track
from lsy_drone_racing.utils.crazyflie import Crazyflie
from lsy_drone_racing.utils.ros import drone_poses, track_poses
from lsy_drone_racing.utils.ros_race_comm import RaceCommNode

if TYPE_CHECKING:
    from crazyflow.dynamics import Dynamics
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# region Worker


class CrazyflieWorker:
    """Manages a single Crazyflie drone in a dedicated subprocess.

    Connects to the drone via radio, resets it to its initial state, and runs
    a control loop that forwards actions received from the client over ROS2.
    The :class:`Crazyflie` wrapper streams the motion capture pose to the drone's Kalman filter.
    """

    # TODO this should not be needed since the client registers after it has jit compiled everything
    FIRST_ACTION_TIMEOUT = 5.0  # Seconds after the race start until the first action must arrive

    def __init__(
        self,
        rank: int,
        drone: dict,
        dynamics: Dynamics,
        start_pos: NDArray[np.floating],
        return_to_start_event: mp.synchronize.Event,
        return_height_ready_event: mp.synchronize.Event,
        assigned_return_heights: mp.Array,
        stop_event: mp.synchronize.Event,
        init_barrier: mp.synchronize.Barrier,
        control_mode: Literal["attitude", "state"],
        control_freq: float = 50.0,
    ):
        """Initialize the Crazyflie worker.

        Args:
            rank: Index of this drone among all drones in the race.
            drone: This drone's entry of ``deploy.drones`` with ``id``, ``channel``, ``drone`` and
                optionally ``radio``.
            dynamics: Dynamics model used to load the drone's physical parameters.
            start_pos: Start position the drone should return to after the race.
            return_to_start_event: Set while the worker is executing the return maneuver.
            return_height_ready_event: Set by the host once this drone's return height is ready.
            assigned_return_heights: Shared host-assigned return heights for all drones.
            stop_event: Set by the host to request a shutdown.
            init_barrier: Shared barrier; all workers and the host call :meth:`wait` once
                initialized so that all drones start simultaneously. Any worker that fails
                calls :meth:`abort` to propagate the failure to everyone.
            control_mode: Either ``"attitude"`` or ``"state"``.
            control_freq: Frequency in Hz at which actions are forwarded to the drone.
        """
        self.rank = rank
        self.drone_cfg = drone
        self.drone_id = drone["id"]
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.return_to_start_event = return_to_start_event
        self.return_height_ready_event = return_height_ready_event
        self.assigned_return_heights = assigned_return_heights
        self.stop_event = stop_event
        self.init_barrier = init_barrier

        assert control_mode in ["attitude", "state"], "control_mode must be 'attitude' or 'state'"
        self._control_mode: Literal["attitude", "state"] = control_mode
        self._control_freq = control_freq

        logging.basicConfig(level=logging.INFO, format=f"[Drone {rank}] %(levelname)s: %(message)s")
        logging.getLogger("cflib").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.drone_name = f"cf{self.drone_id}"
        self.drone: Crazyflie | None = None
        self.drone_params = load_drone_params(dynamics, drone["drone"])
        self.last_msg: Float64MultiArray | None = None
        self.action_lock = threading.Lock()
        self._comm: RaceCommNode | None = None
        self._ros_connector: ROSConnector | None = None
        self._last_msg_time: float = 0.0  # Local receive time of last_msg

    def _on_client_action(self, msg: Float64MultiArray):
        """Store the latest action received from the client."""
        with self.action_lock:
            self.last_msg = msg
            self._last_msg_time = time.perf_counter()

    def _init_cf(self):
        """Connect to the Crazyflie, reset it, and initialize its Kalman filter."""
        self.logger.info(f"Connecting to Crazyflie with ID {self.drone_id}...")
        cfg = self.drone_cfg
        self.drone = Crazyflie.from_radio(cfg.get("radio"), cfg["channel"], self.drone_id)
        self.drone.connect()
        self.drone.reset(arm=True)
        self.logger.info("Connected to Crazyflie")

    def _init_ros_comm(self):
        """Subscribe to client action messages for this drone via ROS2."""
        self._comm = RaceCommNode(f"lsy_race_worker_{self.rank}")
        self._sub = self._comm.node.create_subscription(
            Float64MultiArray,
            f"lsy_drone_racing/{self.drone_name}/action",
            self._on_client_action,
            10,
        )

    def _init_ros_connector(self):
        """Open a ROS connector for reading this drone's pose from the estimator."""
        self.logger.info(f"Initializing ROS connector for {self.drone_name}...")
        self._ros_connector = ROSConnector(estimator_names=[self.drone_name], timeout=10.0)

    def _control_loop(self):
        """Send actions to the drone at the configured control frequency."""
        with self.action_lock:
            self.last_msg = None  # Clear any stale message received during initialization

        dt = 1.0 / self._control_freq

        t_race_start = time.perf_counter()
        while not self.stop_event.is_set():
            t_start = time.time()

            if self.return_height_ready_event.is_set():
                self._return_to_start()  # Check before action staleness to not confuse with crash
                break

            with self.action_lock:
                t = time.perf_counter()
                if self.last_msg and (t - self._last_msg_time) > 10 * dt:
                    self.logger.error(
                        f"No command received for 10 * {dt:.2f}s, handover control to host..."
                    )
                    break
                # The first step includes the first controller solve and JIT compilation
                if self.last_msg is None and t - t_race_start > self.FIRST_ACTION_TIMEOUT:
                    self.logger.error(
                        f"No command received within {self.FIRST_ACTION_TIMEOUT}s of the race "
                        "start, stopping the drone..."
                    )
                    break
                action = list(self.last_msg.data) if self.last_msg else None

            if action is not None:
                action_array = (
                    np.array(action) if isinstance(action, (list, tuple)) else np.array([action])
                )
                if self._control_mode == "attitude":
                    self.drone.send_action_attitude(
                        action_array[:3], action_array[3], self.drone_params
                    )
                else:
                    self.drone.send_action_state(
                        action_array[0:3],
                        action_array[3:6],
                        action_array[6:9],
                        action_array[9:13],
                        action_array[13:16],
                    )

            elapsed = time.time() - t_start

            time.sleep(max(0.0, dt - elapsed))

    def _cleanup(self):
        """Send an emergency stop, close all connections, and shut down ROS."""
        try:
            if self.drone:
                self.drone.close(emergency_stop=True)
        finally:
            if self._ros_connector:
                self._ros_connector.close()
            if self._comm:
                self._comm.close()
            rclpy.shutdown()
        self.logger.info("Drone process finished")

    def _return_to_start(self):
        """Return the drone to its configured start position if it has taken off."""
        pos = self._ros_connector.est_pos[self.drone_name]
        if pos[2] <= 0.1:
            return
        obs = {
            "pos": pos,
            "quat": self._ros_connector.est_quat[self.drone_name],
            "vel": self._ros_connector.est_vel[self.drone_name],
            "ang_vel": self._ros_connector.est_ang_vel[self.drone_name],
        }
        self.return_to_start_event.set()
        try:
            return_height = self._wait_for_return_height()
            self.logger.info(f"Returning to start at height {return_height:.2f}m")
            self.drone.return_to_start(
                self.start_pos, obs, check_ok=rclpy.ok, return_height=return_height
            )
        finally:
            self.return_to_start_event.clear()

    def _wait_for_return_height(self) -> float:
        """Wait until the host assigns a return height for this drone."""
        while rclpy.ok() and not self.stop_event.is_set():
            if self.return_height_ready_event.wait(timeout=0.1):
                with self.assigned_return_heights.get_lock():
                    return float(self.assigned_return_heights[self.rank])
        raise RuntimeError("Return height assignment was interrupted")

    def run(self):
        """Run the worker: connect to the drone, initialize, and enter the control loop.

        Exceptions during initialization abort the shared barrier to notify the host and
        all other workers. A BrokenBarrierError means another worker already failed.
        """
        rclpy.init()
        try:
            tasks = [self._init_ros_comm, self._init_cf, self._init_ros_connector]
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


def _worker_main(**kwargs: Any) -> None:
    """Process entry point that creates and runs a :class:`CrazyflieWorker`.

    SIGINT is ignored so that only the host process handles keyboard interrupts.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    CrazyflieWorker(**kwargs).run()


# region Host


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

    _drones: list[dict]
    _rank_of_id: dict[int, int]
    _control_args: list[dict]
    _num_drones: int
    _drone_names: list[str]
    _dynamics: Dynamics
    _processes: list[mp.Process]
    _stop_event: mp.synchronize.Event
    _init_barrier: mp.synchronize.Barrier
    _mp_ctx: mp.context.BaseContext

    def __init__(
        self,
        track: ConfigDict,
        deploy_args: ConfigDict,
        control_args: list[dict],
        dynamics: Dynamics,
    ):
        """Initialize the host.

        Args:
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            deploy_args: Deploy configuration with the return heights and the list of ``drones``,
                each with ``id``, ``channel``, ``radio``, and ``drone``.
            control_args: Per-drone kwargs, each with ``freq`` and ``control_mode``.
            dynamics: Dynamics model used to load the drones' physical parameters.
        """
        self.gates, self.obstacles, self.drones_pose = load_track(track)
        self.drones_pose.nominal_pos = self.drones_pose.pos.copy()  # update_poses overwrites pos
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)

        self._drones = [dict(drone) for drone in deploy_args.drones]
        self._drone_names = [f"cf{drone['id']}" for drone in self._drones]
        self._rank_of_id = {drone["id"]: rank for rank, drone in enumerate(self._drones)}
        self._control_args = control_args
        self._dynamics = dynamics
        self._num_drones = len(self._drones)

        self._mp_ctx = mp.get_context("spawn")
        self._processes = []
        # Created here so that the deregister callback can use them before connect_drones
        n = self._num_drones
        self._return_to_start_events = [self._mp_ctx.Event() for _ in range(n)]
        self._return_height_ready_events = [self._mp_ctx.Event() for _ in range(n)]
        self._assigned_return_heights = self._mp_ctx.Array("d", [float("nan")] * n, lock=True)
        self._stop_event = self._mp_ctx.Event()
        self._init_barrier = self._mp_ctx.Barrier(n + 1)
        self._clients_registered = np.zeros(self._num_drones, dtype=bool)
        self._clients_stopped = np.zeros(self._num_drones, dtype=bool)

        # For tracking the finish line
        self.gate_sequence, self.gate_sequence_direction = load_gate_order(track, self.n_gates)
        self._n_gates_passed = np.zeros(self._num_drones, dtype=np.int32)
        self._finish_times = np.full(self._num_drones, np.nan)
        self._ros_connector: ROSConnector | None = None

        # For returning home
        self._finish_order: list[int] = []
        self._return_height_min = float(deploy_args.return_height_min)
        self._return_height_max = float(deploy_args.return_height_max)
        if self._return_height_min > self._return_height_max:
            raise ValueError("return_height_min must be less than or equal to return_height_max")
        self._comm = None
        self._race_start_pub = None
        self.init_comm()

    def init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode("lsy_race_host")
        node = self._comm.node
        # Late joiners still receive the last message on these topics
        register_qos = QoSProfile(
            depth=self._num_drones,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        race_start_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._race_start_pub = node.create_publisher(
            Empty, "lsy_drone_racing/race_start", race_start_qos
        )

        def on_register(msg: Int32):
            if (rank := self._rank_of_id.get(msg.data)) is None:
                logger.warning(f"Unknown drone ID {msg.data} tried to register")
            elif not self._clients_registered[rank]:
                logger.info(f"Client cf{msg.data} registered")
                self._clients_registered[rank] = True

        self._register_sub = node.create_subscription(
            Int32, "lsy_drone_racing/register", on_register, register_qos
        )

        def on_deregister(msg: Int32):
            if (rank := self._rank_of_id.get(msg.data)) is None:
                logger.warning(f"Unknown drone ID {msg.data} tried to deregister")
                return
            if not self._clients_stopped[rank]:
                finish_order = len(self._finish_order)
                return_height = self._return_height_for_finish_order(finish_order)
                logger.info(
                    f"Client cf{msg.data} stopped, assigning return height {return_height:.2f}m"
                )
                with self._assigned_return_heights.get_lock():
                    self._assigned_return_heights[rank] = return_height
                self._return_height_ready_events[rank].set()
                self._finish_order.append(rank)
                self._clients_stopped[rank] = True

        self._deregister_sub = node.create_subscription(
            Int32, "lsy_drone_racing/deregister", on_deregister, register_qos
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

    def connect_drones(self, timeout: float = 60.0):
        """Spawn one subprocess per drone and wait until all workers finish initialization.

        Returns once all workers are waiting at the init barrier. If a worker fails or exits, or
        the workers do not finish within ``timeout`` seconds, the barrier is aborted so that
        :meth:`host_main_loop` raises. The race start is triggered in :meth:`host_main_loop` when
        the host calls :meth:`~mp.Barrier.wait` on the barrier.

        Args:
            timeout: Maximum time in seconds for all workers to finish initialization.
        """
        logger.debug(f"Spawning processes for {self._num_drones} Crazyflie drones...")
        self._processes = []
        for rank, drone in enumerate(self._drones):
            process = self._mp_ctx.Process(
                target=_worker_main,
                kwargs=dict(
                    rank=rank,
                    drone=drone,
                    dynamics=self._dynamics,
                    start_pos=self.drones_pose.pos[rank],
                    return_to_start_event=self._return_to_start_events[rank],
                    return_height_ready_event=self._return_height_ready_events[rank],
                    assigned_return_heights=self._assigned_return_heights,
                    stop_event=self._stop_event,
                    init_barrier=self._init_barrier,
                    control_mode=self._control_args[rank]["control_mode"],
                    control_freq=self._control_args[rank]["freq"],
                ),
                name=f"CrazyflieProcess-{rank}",
            )
            process.start()
            self._processes.append(process)
            logger.debug(f"Spawned process for drone {rank} (PID: {process.pid})")

        logger.debug("Waiting for drones to connect...")
        deadline = time.monotonic() + timeout
        while not self._init_barrier.broken:
            if self._init_barrier.n_waiting == self._num_drones:
                return
            if dead := [p.name for p in self._processes if not p.is_alive()]:
                logger.error(f"Drone worker(s) {dead} exited during initialization")
                self._init_barrier.abort()
            elif time.monotonic() > deadline:
                logger.error(f"Drone workers did not finish initialization within {timeout}s")
                self._init_barrier.abort()
            time.sleep(0.05)

    def update_poses(self, track_obj: bool = False, drones: bool = False) -> None:
        """Update the gate and obstacle poses and/or the drone start poses from motion capture.

        Args:
            track_obj: Whether to update gate and obstacle poses.
            drones: Whether to update drone start poses.
        """
        if track_obj:
            self.gates.pos, self.gates.quat, self.obstacles.pos = track_poses(
                self.n_gates, self.n_obstacles
            )

        if drones:
            self.drones_pose.pos, self.drones_pose.quat = drone_poses(self._drone_names)

    def close(self):
        """Stop all drone subprocesses and close ROS communication."""
        self._init_barrier.abort()
        if self._clients_stopped.all():
            for process in self._processes:
                if process.is_alive():
                    process.join(timeout=15)
        else:
            for process, return_event in zip(
                self._processes, self._return_to_start_events, strict=False
            ):
                if process.is_alive() and return_event.is_set():
                    process.join(timeout=15)

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

        if self._ros_connector:
            self._ros_connector.close()

        if self._comm:
            self._comm.close()
        logger.info("Host shutdown complete")

    def _return_height_for_finish_order(self, finish_order: int) -> float:
        """Map finish order to a return height between configured max and min."""
        if self._num_drones <= 1:
            return self._return_height_max
        alpha = finish_order / (self._num_drones - 1)
        return self._return_height_max + alpha * (self._return_height_min - self._return_height_max)

    def _update_gates_passed(self, last_pos: NDArray, t_race: float) -> NDArray:
        """Count gate passes of unfinished drones and return the current positions."""
        pos = np.stack([self._ros_connector.est_pos[name] for name in self._drone_names])
        n_track = len(self.gate_sequence)
        idx = np.minimum(self._n_gates_passed, n_track - 1)  # Finished drones index the last gate
        gate_ids = self.gate_sequence[idx]
        passed = np.array(
            gate_passed(
                pos,
                last_pos,
                self.gates.pos[gate_ids],
                self.gates.quat[gate_ids],
                self.gate_sequence_direction[idx] < 0,
                (0.45, 0.45),
            )
        )
        passed &= self._n_gates_passed < n_track  # Finished drones stop counting
        self._n_gates_passed += passed
        self._finish_times[passed & (self._n_gates_passed == n_track)] = t_race
        return pos

    def _print_leaderboard(self):
        """Log finished drones by time, then unfinished drones by gates passed."""
        n_track = len(self.gate_sequence)
        finished = self._n_gates_passed == n_track
        ranking = sorted(
            range(self._num_drones),
            key=lambda r: (
                not finished[r],
                self._finish_times[r] if finished[r] else -self._n_gates_passed[r],
            ),
        )
        lines = ["Leaderboard:"]
        for place, r in enumerate(ranking, 1):
            time_str = f"{self._finish_times[r]:.3f}s" if finished[r] else "DNF"
            gates = f"{self._n_gates_passed[r]}/{n_track}"
            lines.append(f"  {place}. {self._drone_names[r]}  {time_str}  gates {gates}")
        logger.info("\n".join(lines))

    def host_main_loop(self, race_update_freq: float = 50.0):
        """Run the host coordination loop.

        Waits until all clients have registered, then releases the drone workers via the init
        barrier and publishes the race start. Afterwards tracks the gates passed by each drone until
        all drone workers have exited, then logs the leaderboard.

        Args:
            race_update_freq: Frequency in Hz at which the host tracks gate passes and checks
                whether the workers exited.

        Raises:
            RuntimeError: If a drone worker failed during initialization.
            TimeoutError: If clients do not register within 300 seconds.
        """
        if self._init_barrier.broken:
            raise RuntimeError("A drone worker failed during initialization")
        self._ros_connector = ROSConnector(estimator_names=self._drone_names, timeout=10.0)
        # Drones are on the ground, so this only JIT-compiles gate_passed and sets last_pos
        last_pos = self._update_gates_passed(
            np.stack([self._ros_connector.est_pos[n] for n in self._drone_names]), 0.0
        )
        logger.info("Drones connected, waiting for clients to register...")
        t_start = time.time()
        while time.time() - t_start < 300.0:
            if self._clients_registered.all():
                logger.info("All clients registered")
                break
            if self._init_barrier.broken or not all(p.is_alive() for p in self._processes):
                raise RuntimeError("A drone worker failed while waiting for the clients")
            time.sleep(0.1)

        if not self._clients_registered.all():
            raise TimeoutError("Timeout waiting for all clients to register")

        try:
            self._init_barrier.wait(timeout=None)
        except threading.BrokenBarrierError as error:
            raise RuntimeError("A drone worker failed during initialization") from error

        logger.info("Race started")
        self._race_start_pub.publish(Empty())
        t_start = time.perf_counter()

        while any(p.is_alive() for p in self._processes):
            last_pos = self._update_gates_passed(last_pos, time.perf_counter() - t_start)
            time.sleep(1.0 / race_update_freq)
        logger.info("All drone workers finished")
        self._print_leaderboard()
