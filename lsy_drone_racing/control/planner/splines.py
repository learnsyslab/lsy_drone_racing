"""
Planner to which generates paths through control points
"""
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import crazyflow
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline, BSpline

from dataclasses import dataclass

from lsy_drone_racing.control.utils.utils import (
    obs_to_state,
    obs_to_gate_states
)

from lsy_drone_racing.control.utils.types import (
    Reference,
    ObservationDict
)

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

class SplinePlanner:

    """
    Spline Planner through the gates which pushes the path away from obstacles
    """

    def __init__(
        self,
        obs: ObservationDict,
        drone: str = "cf21B_500",
        obstacle_margin: float = 0.1,
        insert_margin: float = 0.05,
        sample_ds: float = 0.01,
        min_waypoint_spacing: float = 0.05,
        max_collision_iters: int = 10,
    ):
        """
        Args:
            obs: Initial observation.
            drone: Drone model name (config.sim.drone), used to read the drone's collision radius.
            obstacle_margin: Extra clearance [m] on top of drone radius + obstacle radius.
            insert_margin: Inserted safe points are placed this much [m] beyond the clearance so
                the spline between neighbouring waypoints also tends to stay clear.
            sample_ds: Spacing [m] of the dense samples used for the collision check.
            min_waypoint_spacing: A violation closer than this [m] to an existing waypoint moves
                that waypoint instead of inserting a new one.
            max_collision_iters: Maximum number of check / insert / refit rounds.
        """

        # required distance between path and obstacle center (xy)
        self.clearance = drone_collision_radius(drone) + obstacle_radius() + obstacle_margin
        self.insert_margin = insert_margin
        self.sample_ds = sample_ds
        self.min_waypoint_spacing = min_waypoint_spacing
        self.max_collision_iters = max_collision_iters

        self._current_ego_state = obs_to_state(obs)
        self.gate_state = obs_to_gate_states(obs)
        
        self.gate_sequence = obs["gate_sequence"]
        self.gate_sequence_dir = obs["gate_sequence_direction"]
    
        self._current_reference = None
        
        self._reference_version = 1
        
        
    def plan(self, obs: ObservationDict) -> Reference:
        
        # replan if position changes
        current_ego_state = obs_to_state(obs)
        gate_state = obs_to_gate_states(obs)
        
        if self._current_reference is None:
            # initial path
            ego_pos = current_ego_state[:3]

            waypoints = np.vstack([
                ego_pos,
                self.gate_waypoints(
                    obs["gates_pos"],
                    obs["gates_quat"]
                )]
            )

            reference = self.build_reference(waypoints)
            
            self._current_reference = self.ensure_collision_free_reference(obs, reference, waypoints)
            
            return self._current_reference

        if not np.array_equal(self.gate_state, gate_state):
            # TODO replanning
            
            # gates positions changed
            
            # replan path from current position
            
            # do we need to use new center? maybe use minimal change to fly through gate
            
            # update gate state
            pass
        
        # return path segment to track
        return self._current_reference
    
    def gate_waypoints(self, gates_pos, gates_quat, dist=0.3):
        normals = R.from_quat(gates_quat).apply([1.0, 0.0, 0.0])
        normals_seq = normals[self.gate_sequence]
        
        normals_seq *= self.gate_sequence_dir[:, None]
        gates_pos_seq = gates_pos[self.gate_sequence]
        
        pre = gates_pos_seq - dist * normals_seq
        post = gates_pos_seq +  dist * normals_seq
        
        return np.stack([pre, gates_pos_seq, post], axis=1).reshape(-1, 3)
        

    def build_reference(self, waypoints) -> CubicSpline:
        
        s_chord = np.append([0], np.cumsum(la.norm(np.diff(waypoints, axis=0), axis=1))) # (n_wp)

        spline = CubicSpline(s_chord, waypoints, axis=0)
    
        return spline
    
    def ensure_collision_free_reference(self, obs, reference, waypoints) -> Reference:
        """
        Push the reference away from obstacles by inserting safe waypoints and refitting.

        Obstacles are vertical poles, so distances are measured in xy. Points above the top of a
        pole (plus clearance) do not count as violations. The given waypoints (ego + gates) are
        never moved, only inserted waypoints are.
        """
        fixed = np.ones(len(waypoints), dtype=bool)
        obstacles = obs["obstacles_pos"]

        for _ in range(self.max_collision_iters):

            # 1. densely sample the reference spline
            knots = reference.x
            n_samples = max(2, int(np.ceil((knots[-1] - knots[0]) / self.sample_ds)) + 1)
            s = np.linspace(knots[0], knots[-1], n_samples)
            pts = reference(s) # (S, 3)

            # arc length approximated by the polyline through the dense samples
            length = la.norm(np.diff(pts, axis=0), axis=1).sum()
            
            if len(obstacles) == 0:
                return Reference(
                    pos_spline=reference,
                    length=length,
                    version=self._reference_version
                )
            
            # 2. xy distance of every sample to every obstacle
            diff = pts[:, None, :2] - obstacles[None, :, :2] # (S, O, 2)
            dist = la.norm(diff, axis=-1) # (S, O)
            below_top = pts[:, None, 2] < obstacles[None, :, 2] + self.clearance

            # 3. check distance constraint
            violated = (dist < self.clearance) & below_top
            if not violated.any():
                return Reference(
                    pos_spline=reference,
                    length=length,
                    version=self._reference_version
                )

            # deepest violation per violated obstacle
            obs_ids = np.flatnonzero(violated.any(axis=0))
            sample_ids = np.argmin(np.where(violated, dist, np.inf)[:, obs_ids], axis=0)
            s_viol = s[sample_ids]
            p_viol = pts[sample_ids]

            # 4. safe point: shift the curve point orthogonally to the spline (in xy) until it is
            # clearance + insert_margin away from the obstacle along that direction
            tangents = reference(s_viol, 1)
            normals = np.stack(
                [-tangents[:, 1], tangents[:, 0], np.zeros(len(tangents))], axis=1
            )
            normal_len = la.norm(normals, axis=1)
            # vertical tangent (e.g. take-off) has no xy normal -> push straight away from obstacle
            away = np.zeros_like(normals)
            away[:, :2] = diff[sample_ids, obs_ids]
            normals = np.where(normal_len[:, None] > 1e-6, normals, away)
            normals /= np.maximum(la.norm(normals, axis=1, keepdims=True), 1e-9)

            # orient normals away from the obstacle
            offset = np.einsum("ij,ij->i", diff[sample_ids, obs_ids], normals[:, :2])
            normals *= np.where(offset >= 0, 1.0, -1.0)[:, None]
            shift = self.clearance + self.insert_margin - np.abs(offset)
            safe = p_viol + normals * shift[:, None]

            # insert in descending order of s so earlier insertion indices stay valid
            changed = False
            for k in np.argsort(-s_viol):
                i = np.searchsorted(knots, s_viol[k])
                prev_closer = i == len(knots) or (i > 0 and s_viol[k] - knots[i - 1] < knots[i] - s_viol[k])
                near = i - 1 if prev_closer else i

                if abs(knots[near] - s_viol[k]) < self.min_waypoint_spacing:
                    if fixed[near]:
                        warnings.warn(
                            f"Obstacle {obs_ids[k]} violates clearance at a fixed waypoint "
                            f"{waypoints[near]}, cannot move it."
                        )
                        continue
                    waypoints[near] = safe[k]
                else:
                    waypoints = np.insert(waypoints, i, safe[k], axis=0)
                    fixed = np.insert(fixed, i, False)
                changed = True

            if not changed:
                break
            reference = self.build_reference(waypoints)

        warnings.warn("Could not make the reference collision free.")
        
        return Reference(
            pos_spline=reference,
            length=length,
            version=self._reference_version
        )