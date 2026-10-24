import crazyflow
import numpy as np

import xml.etree.ElementTree as ET
from pathlib import Path

from lsy_drone_racing.control.utils.types import (
    ObservationDict,
    State
)

from scipy.spatial.transform import Rotation as R
import numpy.linalg as la

def obs_to_state(obs: ObservationDict) -> State:
    pos = obs["pos"]
    vel = obs["vel"]
    rpy = R.from_quat(obs["quat"]).as_euler("xyz")
    
    #ang_vel = #TODO

    
    state = np.concat([pos, rpy, vel], axis=-1).reshape(-1)
    
    return state

def obs_to_gate_state(obs: ObservationDict):
    
    gates_pos = obs["gates_pos"]
    gates_rpy = R.from_quat(obs["gates_quat"]).as_euler("xyz")
    
    state = np.concat([gates_pos, gates_rpy], axis=-1).reshape(-1)
    
    return state

def drone_collision_radius(drone: str) -> float:
    """Radius of the drone's collision sphere from crazyflow's MuJoCo model."""
    xml_path = Path(crazyflow.__file__).parent / "drones" / f"{drone}.xml"
    geom = ET.parse(xml_path).getroot().find(".//geom[@name='col_sphere']")
    return float(geom.get("size").split()[0])

def obstacle_radius() -> float:
    """Radius of the obstacle poles from the environment's MuJoCo model."""
    xml_path = Path(__file__).parents[2] / "envs" / "assets" / "obstacle.xml"
    geom = ET.parse(xml_path).getroot().find(".//geom[@name='obstacle']")
    return float(geom.get("size").split()[0])

def dist_to_segment(pts, a, b):
    """
    Pointwise distance to the segment described by a->b
    
    pts: (N, 3)
    a: (M, 3)
    b: (M, 3)

    returns: (N, M, 1) distance of every point to every segment
    """

    ab = b - a # (M, 3)
    ap = pts[:, None, :] - a[None, :, :] # (N, M, 3)
    t = np.sum(ap * ab, axis=-1, keepdims=True) / np.sum(ab * ab, axis=-1, keepdims=True) # (N, M, 1)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t * ab[None, :, :] # (N, M, 3)
    return la.norm(pts[:, None, :] - closest, axis=-1, keepdims=True) # (N, M, 1)
