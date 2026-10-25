"""
Planner to which generates paths through control points
"""

from abc import abstractmethod

import crazyflow
import time
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import PPoly


from lsy_drone_racing.control.utils.utils import (
    drone_collision_radius,
    obstacle_radius,
    dist_to_segment
)

from lsy_drone_racing.control.utils.types import (
    ObservationDict
)

from lsy_drone_racing.control.planner.reference import Reference

class SplinePlanner:

    """
    Spline Planner through the gates which pushes the path away from obstacles
    """

    def __init__(
        self,
        obs: ObservationDict,
        config: dict
    ):
        """
        Args:
            obs: Initial observation.
            config: Environment config.
        """

        self.config = config
        
        self.gate_sequence = obs["gate_sequence"]
        self.gate_sequence_dir = obs["gate_sequence_direction"]
        
        self.obstacles_visited = obs["obstacles_visited"]
        self.gates_visited = obs["gates_visited"]

        self.is_track_randomized = self.config.env.track.randomize
        self._current_reference = None
        
        self.ds = 0.01 # reference is sampled every ds [m]
        margin = 0.04 # safety margin for collisions
        
        d_r = drone_collision_radius(config.sim.drone)
        o_r = obstacle_radius()
        
        self.obstacle_clearance = (d_r + o_r + margin)
        self.gate_clearance_inside = 0.2 - d_r - margin
        self.gate_clearance_outside = 0.36 + d_r + margin
        
    def plan(self, obs: ObservationDict) -> Reference:
        
        if self._current_reference is None:
            # initial path
            if not self.is_track_randomized:
                self._current_reference = Reference.default_reference(obs)
            else:
                self._current_reference = Reference(*self.find_waypoints(obs)) # compute waypoints from gate positions

            return self._current_reference
        
        updated_obstacles = np.flatnonzero(obs["obstacles_visited"] & ~self.obstacles_visited)
        updated_gates = np.flatnonzero(obs["gates_visited"] & ~self.gates_visited)
        
        if updated_gates.size > 0:
            # update gates visited
            self.gates_visited = obs["gates_visited"].copy()
            self._handle_gates_update(obs, updated_gates)

        if updated_obstacles.size > 0:
            # update obstacles visited
            self.obstacles_visited = obs["obstacles_visited"].copy()
            self._handle_obstacle_update(obs, updated_obstacles)
        
        #TODO handle points close to plane as well (+ joint handling of obstacles and gate frames)
        
        # return path segment to track
        return self._current_reference
    
    def _handle_gates_update(self, obs, updated_gates):
        """
        Returns the updated sparse curve (knots, tangents, accs) or None if the current reference
        is still valid.
        """
        assert updated_gates.size > 0
        forward, other = self.gate_plane_crossings(obs, updated_gates)
        
        # project to closest point inside
        safe_points = self.closest_safe_point_inside(forward, obs)
        self._current_reference.update(safe_points, forward["s"])
        
        inside_mask = np.max(np.abs(other["offset"]), axis=-1) <= 0.2 + 0.08
    
        if inside_mask.any():
            safe_points = self.closest_safe_point_inside(other, obs)
            self._current_reference.update(safe_points[inside_mask], other["s"][inside_mask])
        
        safe_points, _ = self.closest_safe_point_outside(other, obs)
        self._current_reference.update(safe_points[~inside_mask], other["s"][~inside_mask])
        
    def _handle_obstacle_update(self, obs, new_obstacles, eps=0.01):
        # TODO: only handles collision of the path in one place
        obstacle_pos = obs["obstacles_pos"][new_obstacles]
        ref = self._current_reference
             
        dist = self.min_distance_to_obstacles(obstacle_pos, ref._dense_knots)[..., 0]
        path_collides = (dist < self.obstacle_clearance).any()
        
        if not path_collides:
            return
        
        min_i = np.argmin(dist, axis=0)
        
        closest = ref._dense_knots[min_i]
        dir_vec = (closest - obstacle_pos) * np.array([1.0, 1.0, 0.0])
        dir_vec = dir_vec / la.norm(dir_vec, axis=-1)[..., None]
        
        clearance = self.obstacle_clearance + eps
        
        safe_points = closest + clearance * dir_vec
        s = ref.path.x[min_i]
        
        self._current_reference.update(safe_points, s)

    def find_waypoints(self, obs, tangent_scale=1.0, overshoot=1.0) -> tuple:
        ego_pos = obs["pos"]
        ego_vel = obs["vel"]
        
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        
        gates = gates_pos[self.gate_sequence]
        normals = R.from_quat(gates_quat[self.gate_sequence]).apply([1.0, 0.0, 0.0])
        normals *= self.gate_sequence_dir[:, None]
        
        # stack ego position, gate positions and an extension along the normal of the final gate for control points
        pts = np.vstack([ego_pos, gates, gates[-1] + overshoot * normals[-1]])
        
        start_dir = ego_vel if la.norm(ego_vel) > 0.1 else pts[1] - pts[0]
        tangents = tangent_scale * np.vstack([start_dir / la.norm(start_dir), normals, normals[-1]])
        accs = np.zeros_like(tangents) # straight through the gates

        return pts, tangents, accs
    
    def gate_plane_crossings(self, obs, gates_to_check, search_radius=0.4):
        """
        Find where the current reference crosses the planes of the given gates.

        Crossings are split into
            forward: crossings in a direction the gate has to be passed in according to the gate
                sequence and close to the gate (within search_radius in the gate plane). These
                have to lie inside the safe opening.
            other: all remaining crossings (wrong direction or far away from the gate). These only
                must not hit the frame.

        Args:
            obs: Current observation.
            gates_to_check: (G,) gate ids.
            search_radius: Max. offset [m] (per axis, in the gate plane) for a crossing in the
                required direction to count as a pass of that gate.

        Returns:
            forward, other: dicts with
                s: (C,) progress of the crossing on the reference
                gate: (C,) gate id
                offset: (C, 2) (y, z) of the crossing point in the gate frame
        """
        gates_pos = obs["gates_pos"][gates_to_check] # (G, 3)
        gates_quat = obs["gates_quat"][gates_to_check] # (G, 4)
        spline = self._current_reference.path

        rot = R.from_quat(gates_quat)
        normals = rot.apply([1.0, 0.0, 0.0]) # (G, 3)

        # signed distance to every gate plane as one piecewise polynomial with G columns
        c = spline.c @ normals.T # (4, m, 3) @ (3, G) -> (4, m, G)
        c[-1] -= np.sum(normals * gates_pos, axis=1) # subtract n_g · g_g in the constant term
        f = PPoly(c, spline.x)

        roots = f.roots(extrapolate=False) # (G,) object array, one root array per gate

        # flatten into one list of crossings
        s_cross = np.concatenate(roots) # (C,)
        gate_idx = np.repeat(np.arange(len(roots)), [len(r) for r in roots]) # (C,) index into gates_to_check

        # (y, z) of the crossing point relative to the gate center in the gate frame
        local = rot[gate_idx].apply(spline(s_cross) - gates_pos[gate_idx], inverse=True)
        offset = local[:, 1:] # (C, 2)

        # +1 if the path crosses along the gate normal (-x -> +x), -1 for the reverse
        slope = f.derivative()(s_cross)[np.arange(len(s_cross)), gate_idx]
        crossing_dir = np.sign(slope)

        # a crossing is forward if any pass of its gate in the sequence requires this direction
        passes = self.gate_sequence[None, :] == gates_to_check[:, None] # (G, P)
        dir_matches = self.gate_sequence_dir[None, :] == crossing_dir[:, None] # (C, P)
        required_dir = (passes[gate_idx] & dir_matches).any(axis=1) # (C,)

        # and it is within a given search radius
        near = np.all(np.abs(offset) < search_radius, axis=1)
        forward = required_dir & near

        def select(mask):
            return dict(
                s=s_cross[mask],
                gate=gates_to_check[gate_idx[mask]],
                offset=offset[mask],
                dir=crossing_dir[mask],
            )

        return select(forward), select(~forward)

    def closest_safe_point_inside(self, d, obs, eps=0.04):
        
        gate_indices = d["gate"]
        
        gates_pos = obs["gates_pos"][gate_indices]
        rot = R.from_quat(obs["gates_quat"][gate_indices])
        
        clearance = self.gate_clearance_inside - eps
        
        safe_yz = np.clip(d["offset"], -clearance, clearance)
        safe_local = np.column_stack([np.zeros(len(safe_yz)), safe_yz]) # (x, y, z) with x=0
        safe_world = gates_pos + rot.apply(safe_local)
        
        moved = (safe_yz != d["offset"]).any(axis=-1)
        
        return safe_world
    
    def closest_safe_point_outside(self, d, obs, eps=0.02):
        
        gate_indices = d["gate"]
        gates_pos = obs["gates_pos"][gate_indices]
        rot = R.from_quat(obs["gates_quat"][gate_indices]) 
        
        clearance = self.gate_clearance_outside + eps
        offset = d["offset"] # (C, 2)
        y, z = offset[:, 0], offset[:, 1]

        # shift needed to leave the square over each edge: right (+y), left (-y), top (+z), bottom (-z)
        shifts = np.stack([clearance - y, clearance + y, clearance - z, clearance + z], axis=1) # (C, 4)

        # leaving over the bottom edge would put the point onto the stand (|y| < 0.05 below the frame)
        on_stand = np.abs(y) < 0.05 + self.gate_clearance_outside - 0.36
        shifts[on_stand, 3] = np.inf

        # push out over the nearest allowed edge, points that are already outside are not moved
        edge = np.argmin(shifts, axis=1)
        moved = np.max(np.abs(offset), axis=1) < clearance

        safe_yz = offset.copy()
        rows = np.flatnonzero(moved)
        axis = edge[rows] // 2 # 0: y, 1: z
        sign = np.where(edge[rows] % 2 == 0, 1.0, -1.0)
        safe_yz[rows, axis] = sign * clearance

        safe_local = np.column_stack([np.zeros(len(safe_yz)), safe_yz]) # (x, y, z) with x=0
        safe_world = gates_pos + rot.apply(safe_local)

        return safe_world, moved
    
    def min_distance_to_obstacles(self, obstacle_pos, waypoints):
    
        return dist_to_segment(
            waypoints,
            obstacle_pos * np.array([1.0, 1.0, 0.0]), # set z component to zero,
            obstacle_pos
        )