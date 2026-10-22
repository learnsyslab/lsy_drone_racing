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
    pos_spline: CubicSpline
    length: float
    version: int