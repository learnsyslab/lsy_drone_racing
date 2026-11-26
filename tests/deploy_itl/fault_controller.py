"""Controller for crazyflow_itl deployment tests that can inject faults.

``LSY_CONTROL_MODE`` selects the base controller (``state`` or ``attitude``), and ``LSY_FAULT``
the fault, timed from the first control step:

* ``none``: Follow the trajectory of the base controller.
* ``raise@T``: Raise an exception after ``T`` seconds.
* ``oob@T``: After ``T`` seconds, report a position 1 m too low to the base controller, so that it
  climbs out of the safety limits.
"""

from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.attitude_controller import AttitudeController
from lsy_drone_racing.control.state_controller import StateController

if TYPE_CHECKING:
    from numpy.typing import NDArray

# deploy.py builds the controller itself, so the harness selects the base through the environment
Base = AttitudeController if os.environ.get("LSY_CONTROL_MODE") == "attitude" else StateController


class FaultController(Base):
    """Base controller with an injected fault."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Parse LSY_FAULT and initialize the base controller."""
        super().__init__(obs, info, config)
        fault = os.environ.get("LSY_FAULT", "none")
        match = re.fullmatch(r"(none|raise|oob)(?:@([0-9.]+))?", fault)
        assert match, f"Invalid LSY_FAULT {fault!r}"
        self._fault, self._after = match[1], float(match[2] or 0.0)
        self._t0: float | None = None

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Inject the fault once its time has passed and compute the base controller's action."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        if now - self._t0 >= self._after:
            if self._fault == "raise":
                raise RuntimeError("Injected controller fault")
            if self._fault == "oob":
                obs = obs | {"pos": obs["pos"] - np.array([0.0, 0.0, 1.0])}
        return super().compute_control(obs, info)
