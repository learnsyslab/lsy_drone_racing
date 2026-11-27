"""Deployment tests against drones simulated by crazyflow_itl.

Each test starts the simulator, the estimator and ``scripts/deploy.py``, and checks the true drone
pose that the simulator publishes on /tf. Run them with ``pixi run -e crazyflow-itl itl-tests``.
"""

from __future__ import annotations

import re
import signal
import time
from typing import TYPE_CHECKING

import pytest

from tests.deploy_itl.harness import CONTROL_DIR, wait_until

if TYPE_CHECKING:
    from tests.deploy_itl.harness import DeployStack, Proc

pytestmark = pytest.mark.deploy_itl


def _takeoff(stack: DeployStack, deploy: Proc):
    """Wait until the drone has taken off, which includes the startup of deploy.py."""
    assert stack.wait_height(lambda z: z > 0.2, 60), f"No takeoff\n{deploy.log()[-2000:]}"


def _landed_at_start(stack: DeployStack) -> bool:
    return stack.height() < 0.15 and stack.dist_to_start() < 0.2


def _where(stack: DeployStack) -> str:
    return f"height={stack.height():.2f} m, dist_to_start={stack.dist_to_start():.2f} m"


@pytest.mark.parametrize(
    ("controller", "mode"),
    [("state_controller.py", "state"), ("attitude_controller.py", "attitude")],
)
def test_full_track(stack: DeployStack, controller: str, mode: str):
    """The drone finishes the track and returns to its start."""
    stack.bring_up()
    deploy = stack.start_deploy(CONTROL_DIR / controller, mode)
    assert deploy.wait_exit(90) == 0, f"deploy.py failed or hung\n{deploy.log()[-2000:]}"
    match = re.search(r"Track time: ([0-9.]+)s", deploy.log())
    assert match, "Track not completed"
    assert float(match[1]) < 40, f"Track took {match[1]} s"
    assert _landed_at_start(stack), _where(stack)


def test_crash_returns_home(stack: DeployStack):
    """A controller exception flies the drone home at the return height."""
    stack.bring_up()
    deploy = stack.start_deploy(control_mode="attitude", fault="raise@3")
    _takeoff(stack, deploy)
    assert stack.wait_height(lambda z: z > 1.7, 10), f"No climb to return height, {_where(stack)}"
    assert wait_until(lambda: _landed_at_start(stack), 20), f"Not landed at start, {_where(stack)}"
    assert deploy.wait_exit(30) == 1
    assert "RuntimeError: Injected controller fault" in deploy.log()


def test_out_of_bounds_returns_home(stack: DeployStack):
    """Leaving the safety limits ends the run and flies the drone home."""
    stack.bring_up()
    deploy = stack.start_deploy(control_mode="attitude", fault="oob@3")
    _takeoff(stack, deploy)
    assert stack.wait_height(lambda z: z > 2.0, 10), f"Never left the bounds, {_where(stack)}"
    assert wait_until(lambda: _landed_at_start(stack), 25), f"Not landed at start, {_where(stack)}"
    assert deploy.wait_exit(30) == 0
    assert "Task not completed" in deploy.log()


def test_sigint_emergency_stops(stack: DeployStack):
    """Ctrl-C stops the motors at once instead of flying home."""
    stack.bring_up()
    deploy = stack.start_deploy(control_mode="attitude")
    _takeoff(stack, deploy)
    time.sleep(3.0)
    deploy.popen.send_signal(signal.SIGINT)
    assert stack.wait_height(lambda z: z < 0.1, 2.0), f"Drone did not drop, {_where(stack)}"
    assert not stack.wait_height(lambda z: z > 0.1, 3.0), "Drone climbed again after the stop"
    assert deploy.wait_exit(10) == -signal.SIGINT
    assert "KeyboardInterrupt" in deploy.log()
