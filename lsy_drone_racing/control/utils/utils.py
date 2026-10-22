import numpy as np

from lsy_drone_racing.control.utils.types import (
    ObservationDict,
    State
)

from scipy.spatial.transform import Rotation as R

def obs_to_state(obs: ObservationDict) -> State:
    pos = obs["pos"]
    vel = obs["vel"]
    rpy = R.from_quat(obs["quat"]).as_euler("xyz")
    
    #ang_vel = #TODO

    
    state = np.concat([pos, rpy, vel], axis=-1).reshape(-1)
    
    return state

def obs_to_gate_states(obs: ObservationDict):
    
    gates_pos = obs["gates_pos"]
    gates_rpy = R.from_quat(obs["gates_quat"]).as_euler("xyz")
    
    state = np.concat([gates_pos, gates_rpy], axis=-1).reshape(-1)
    
    return state