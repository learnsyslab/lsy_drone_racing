"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

To give you an idea of what you need to do, we also include some example implementations:

* [Controller][lsy_drone_racing.control.controller.Controller]: The abstract base class defining the
  interface for all controllers.
* [StateController][lsy_drone_racing.control.state_controller.StateController]:
  A controller that follows a pre-defined trajectory using cubic spline interpolation.
* [AttitudeController][lsy_drone_racing.control.attitude_controller.AttitudeController]: A
  controller that follows a pre-defined attitude using cubic spline interpolation.
* [PathPlanner][lsy_drone_racing.control.path_planning.PathPlanner]: Generates an
  obstacle-avoiding, arc-length parameterized path through the gates of the track.
* [MPCCController][lsy_drone_racing.control.my_controller.MPCCController]: A Model Predictive
  Contouring Control controller that tracks the planned path at the maximum feasible speed.
"""

from lsy_drone_racing.control.controller import Controller

__all__ = ["Controller"]
