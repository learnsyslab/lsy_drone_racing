"""The harness automates the task of spawning a process in a new terminal and monitoring it."""

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage

from lsy_drone_racing.utils import load_config

REPO = Path(__file__).parents[2]
CONFIG = REPO / "config/level2.toml"
CONTROL_DIR = REPO / "lsy_drone_racing/control"
FAULT_CONTROLLER = Path(__file__).parent / "fault_controller.py"


def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.05) -> bool:
    """Poll a predicate until it holds true or times out and return whether it held."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def write_config(path: Path, controller: Path, control_mode: str) -> Path:
    """Write a copy of the base config that deploys the controller to the simulator."""
    text = CONFIG.read_text()
    for key, value in [
        ("crazyflow_itl", "true"),
        ("file", f'"{controller}"'),
        ("control_mode", f'"{control_mode}"'),
    ]:
        text, n = re.subn(rf"(?m)^{key}\s*=.*$", f"{key} = {value}", text)
        assert n == 1, f"Expected one {key} entry in {CONFIG}, found {n}"
    path.write_text(text)
    return path


# region Process
class Proc:
    """A subprocess in its own process group, with its output in a log file."""

    def __init__(self, name: str, cmd: list[str], log_dir: Path, env: dict, cwd: Path = REPO):
        """Start the process."""
        self.name = name
        self.log_path = log_dir / f"{name}.log"
        self._log = open(self.log_path, "w")  # noqa: SIM115, closed in kill
        self.popen = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def log(self) -> str:
        """Return the log so far."""
        return self.log_path.read_text(errors="replace")

    def wait_for(self, pattern: str, timeout: float) -> bool:
        """Wait until the log matches a regex."""
        return wait_until(lambda: re.search(pattern, self.log()) is not None, timeout, 0.1)

    def wait_exit(self, timeout: float) -> int | None:
        """Wait until the process exits and return its exit code, or None on timeout."""
        try:
            return self.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self):
        """Kill the whole process group and close the log."""
        try:
            os.killpg(self.popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already exited
        self.popen.wait()
        self._log.close()


# region TFWatcher
class TFWatcher:
    """Track the latest positions of frames on /tf."""

    def __init__(self, domain_id: int):
        """Subscribe to /tf in its own ROS context and spin in a background thread."""
        self._context = rclpy.Context()
        rclpy.init(context=self._context, domain_id=domain_id)
        self.node = rclpy.create_node("deploy_itl_test", context=self._context)
        self._lock = threading.Lock()
        self._pos: dict[str, np.ndarray] = {}
        self.node.create_subscription(TFMessage, "/tf", self._callback, 100)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _callback(self, msg: TFMessage):
        with self._lock:
            for tf in msg.transforms:
                t = tf.transform.translation
                self._pos[tf.child_frame_id] = np.array([t.x, t.y, t.z])

    def pos(self, name: str) -> np.ndarray | None:
        """Return the latest position of a frame, or None if it was not seen yet."""
        with self._lock:
            return self._pos.get(name)

    def close(self):
        """Stop spinning, then destroy the node and shut down the context."""
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=5.0)
        self.node.destroy_node()
        rclpy.shutdown(context=self._context)


# region DeployStack
class DeployStack:
    """The simulator, estimator and deploy script of one single-drone deployment."""

    def __init__(self, log_dir: Path, domain_id: int):
        """Prepare the environment and /tf watcher without starting any process."""
        self.log_dir = log_dir
        self.watcher = TFWatcher(domain_id)
        self.env = os.environ | {
            "JAX_PLATFORMS": "cpu",
            "PYTHONUNBUFFERED": "1",
            "ROS_DOMAIN_ID": str(domain_id),
        }
        base = load_config(CONFIG)
        self.drone_name = f"cf{base.deploy.drones[0]['id']}"
        self.start_pos = np.array(base.env.track.drones[0]["pos"])
        self.procs: list[Proc] = []

    def _start(self, name: str, cmd: list[str], env: dict | None = None, cwd: Path = REPO) -> Proc:
        proc = Proc(name, cmd, self.log_dir, env or self.env, cwd)
        self.procs.append(proc)
        return proc

    def bring_up(self, timeout: float = 240.0):
        """Start the simulator and estimator and wait until both are ready."""
        sim = self._start(
            "sim", [sys.executable, "scripts/crazyflow_itl_sim.py", "--config", str(CONFIG)]
        )
        ready = wait_until(lambda: self.watcher.pos(self.drone_name) is not None, timeout, 0.2)
        assert ready, f"Simulator did not publish {self.drone_name}\n{sim.log()[-3000:]}"
        cmd = [
            sys.executable,
            "-m",
            "drone_estimators.ros.node",
            "--drone_name",
            self.drone_name,
            "--estimator_type",
            "ukf",
        ]
        est = self._start("estimator", cmd, cwd=self.log_dir)
        assert est.wait_for(r"Started estimator", 180), f"Estimator failed\n{est.log()[-3000:]}"

    def start_deploy(
        self,
        controller: Path = FAULT_CONTROLLER,
        control_mode: Literal["state", "attitude"] = "state",
        fault: str = "none",
    ) -> Proc:
        """Start deploy.py with a controller and an injected fault."""
        config = write_config(self.log_dir / "config.toml", controller, control_mode)
        cmd = [sys.executable, "scripts/deploy.py", "--config", str(config)]
        env = self.env | {"LSY_FAULT": fault, "LSY_CONTROL_MODE": control_mode}
        return self._start("deploy", cmd, env=env)

    def pos(self) -> np.ndarray:
        """Return the latest true position of the drone."""
        return self.watcher.pos(self.drone_name)

    def height(self) -> float:
        """Return the latest true height of the drone."""
        return float(self.pos()[2])

    def dist_to_start(self) -> float:
        """Return the horizontal distance of the drone to its start position."""
        return float(np.linalg.norm(self.pos()[:2] - self.start_pos[:2]))

    def wait_height(self, predicate: Callable[[float], bool], timeout: float) -> bool:
        """Wait until the height of the drone satisfies a predicate."""
        return wait_until(lambda: predicate(self.height()), timeout)

    def close(self) -> str:
        """Kill all processes and return the ends of their logs."""
        for proc in reversed(self.procs):
            proc.kill()
        report = "\n\n".join(f"===== {p.name} =====\n{p.log()[-3000:]}" for p in self.procs)
        self.watcher.close()
        return report
