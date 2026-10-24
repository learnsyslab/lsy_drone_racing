"""
Planner to which generates paths through control points
"""
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import crazyflow
import time
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import BPoly, PPoly, CubicSpline

from dataclasses import dataclass

from lsy_drone_racing.control.utils.utils import (
    obs_to_gate_state,
    drone_collision_radius,
    obstacle_radius,
    dist_to_segment
)

from lsy_drone_racing.control.utils.types import (
    Reference,
    ObservationDict
)
class SplinePlanner():

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
        # self.clearance = drone_collision_radius(drone) + obstacle_radius() + obstacle_margin
        self.config = config
        
        self.gate_sequence = obs["gate_sequence"]
        self.gate_sequence_dir = obs["gate_sequence_direction"]
        
        self.obstacles_visited = obs["obstacles_visited"]
        self.gates_visited = obs["gates_visited"]

        self.is_track_randomized = self.config.env.track.randomize
        self._current_reference = None
        self._reference_version = 0
        
        self._default_waypoints = np.array(
            [
                obs["pos"],
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [1.2, -0.15, 1.2],
                [1.05, 0.75, 1.2],
                [0.25, 1.25, 1.2],
            ]
        )
        
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
                knots, tangents, accs = self.default_curve() # use provided waypoints
            else:
                knots, tangents, accs = self.find_waypoints(obs) # compute waypoints from gate positions

            self._current_reference = self.build_reference(knots, tangents, accs)
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
        
        # return path segment to track
        return self._current_reference
    
    def _handle_gates_update(self, obs, updated_gates):
        """
        Returns the updated sparse curve (knots, tangents, accs) or None if the current reference
        is still valid.
        """
        assert updated_gates.size > 0
        forward, other = self.gate_plane_crossings(obs, updated_gates)

        updated = None
        
        # project to closest point inside
        safe_points = self.closest_safe_point_inside(forward, obs)
        tangents = self._current_reference.path(forward["s"], 1)
        accs = self._current_reference.path(forward["s"], 2)

        # insert / replace with safe points
        updated = self.insert_waypoints(safe_points, tangents, accs, forward["s"])

        if updated is not None:
            self._current_reference = self.build_reference(*updated)
            updated = None
        
        """ TODO handle points close to plane as well
        safe_points, _ = self.closest_safe_point_outside(other, obs)
        tangents = self._current_reference.path(other["s"], 1)
        accs = self._current_reference.path(other["s"], 2)
           
        # insert / replace with safe points
        updated = self.insert_waypoints(safe_points, tangents, accs, other["s"])

        if updated is not None:
            self._current_reference = self.build_reference(*updated)
        """
        
    def _handle_obstacle_update(self, obs, new_obstacles):
        obstacle_pos = obs["obstacles_pos"][new_obstacles]
        ref = self._current_reference
             
        dist = self.min_distance_to_obstacles(obstacle_pos, ref.dense_waypoints)
        path_collides = (dist < self.obstacle_clearance).any()
        
        if not path_collides:
            return
        
        min_i = np.argmin(dist, axis=0)
        
        closest = ref.dense_waypoints[min_i]
        dir_vec = (closest - obstacle_pos) * np.array([1.0, 1.0, 0.0])
        dir_vec = dir_vec / la.norm(dir_vec)
        
        safe_points = closest + self.obstacle_clearance * dir_vec
        
        s = ref.path.x[min_i]
        tangents = ref.path(s, 1)
        accs = ref.path(s, 2)
        
        updated = self.insert_waypoints(safe_points, tangents, accs, s)
        
        if updated is not None:
            self._current_reference = self.build_reference(*updated)
            updated = None
        
    def default_curve(self):
        """
        Sparse curve through the default waypoints.

        The derivatives are taken from a cubic spline through the waypoints, so the quintic
        Hermite curve built from them reproduces exactly that cubic spline.
        """
        knots = self._default_waypoints
        u = np.append(0, np.cumsum(la.norm(np.diff(knots, axis=0), axis=1)))
        spline = CubicSpline(u, knots, axis=0)
        return knots, spline(u, 1), spline(u, 2)

    def find_waypoints(self, obs, tangent_scale=1.0, overshoot=1.0):
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

    def build_reference(self, knots, tangents, accs) -> Reference:
        """
        Build the reference from the sparse curve.

        The sparse quintic Hermite curve is densely sampled and refit by a cubic spline that is
        parameterized by arc length.
        """
        curve, u_knots = self.build_sparse_curve(knots, tangents, accs)

        # dense samples, uniform in the chord parameter u
        u, dense_waypoints = self.resample_spline(curve, u_knots[-1])

        # reparameterize by arc length
        s = np.append(0, np.cumsum(la.norm(np.diff(dense_waypoints, axis=0), axis=1)))
        spline = CubicSpline(s, dense_waypoints, axis=0)

        self._reference_version = self._reference_version + 1

        return Reference(
            path=spline,
            sparse_waypoints=knots,
            sparse_tangents=tangents,
            sparse_accs=accs,
            dense_waypoints=dense_waypoints,
            sparse_s=np.interp(u_knots, u, s), # map knot parameters to arc length
            length=s[-1],
            version=self._reference_version
        )

    @staticmethod
    def build_sparse_curve(knots, tangents, accs):
        """
        Quintic Hermite curve through the knots with the given first and second derivatives.

        The curve is parameterized by the chord length between the knots. It is C2 and local:
        changing a knot only changes the two adjacent segments.

        Returns:
            curve: BPoly, evaluated as curve(u) -> (..., 3)
            u: (K,) chord parameter of the knots
        """
        u = np.append(0, np.cumsum(la.norm(np.diff(knots, axis=0), axis=1)))
        if np.any(np.diff(u) <= 0):
            raise ValueError("Consecutive knots must not coincide.")

        curve = BPoly.from_derivatives(u, np.stack([knots, tangents, accs], axis=1)) # (K, 3 derivs, 3 dims)
        return curve, u

    def resample_spline(self, spline, length):
        u = np.linspace(0, length, int(length / self.ds) + 1)
        return u, spline(u)
    
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

    def insert_waypoints(self, new_pts, new_tangents, new_accs, new_s, min_spacing=0.3):
        """
        Insert knots into the sparse curve of the current reference, ordered by progress.

        A new knot closer than min_spacing [m] (in s) to an existing knot replaces that knot
        instead (except for the start knot).

        Args:
            new_pts, new_tangents, new_accs: (M, 3) knots and their derivatives.
            new_s: (M,) progress of the new knots on the current reference.

        Returns:
            knots, tangents, accs of the updated sparse curve.
        """
        ref = self._current_reference
        sparse_s = ref.sparse_s

        # sort by progress so knots with the same insertion index stay ordered
        order = np.argsort(new_s)
        new_s = new_s[order]
        new = [new_pts[order], new_tangents[order], new_accs[order]]

        idx = np.searchsorted(sparse_s, new_s) # insert before knot idx

        left = np.clip(idx - 1, 0, len(sparse_s) - 1)
        right = np.clip(idx, 0, len(sparse_s) - 1)
        nearest = np.where(np.abs(sparse_s[left] - new_s) < np.abs(sparse_s[right] - new_s), left, right)
        replace = (np.abs(sparse_s[nearest] - new_s) < min_spacing) & (nearest != 0)

        updated = []
        for old, values in zip([ref.sparse_waypoints, ref.sparse_tangents, ref.sparse_accs], new):
            old = old.copy()
            old[nearest[replace]] = values[replace]
            updated.append(np.insert(old, idx[~replace], values[~replace], axis=0))

        return tuple(updated)
    
    def min_distance_to_obstacles(self, obstacle_pos, waypoints):
    
        return dist_to_segment(
            waypoints,
            obstacle_pos * np.array([1.0, 1.0, 0.0]), # set z component to zero,
            obstacle_pos
        )