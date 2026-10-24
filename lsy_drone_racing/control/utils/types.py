import numpy as np
from typing import NewType
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from dataclasses import dataclass


ObservationDict = NewType("ObservationDict", dict[str, NDArray[np.floating]])
State = NewType("State", NDArray)

StateCommand = NewType("StateCommand", NDArray[np.floating]) # (16, ) [x, y, z, vx, ..., wz]
AttitudeCommand = NewType("AttitudeCommand", NDArray[np.floating]) # (4, ) [thrust, roll, pitch, yaw]

@dataclass(frozen=True) # immutable
class Reference:
    path: CubicSpline
    sparse_waypoints: NDArray[np.floating] # (K, 3) knots of the editable sparse curve
    sparse_tangents: NDArray[np.floating] # (K, 3) first derivative at the knots (w.r.t. chord parameter)
    sparse_accs: NDArray[np.floating] # (K, 3) second derivative at the knots (w.r.t. chord parameter)
    dense_waypoints: NDArray[np.floating] # dense waypoints used to generate the spline
    sparse_s: NDArray[np.floating] # progress values of sparse waypoints
    length: float
    version: int