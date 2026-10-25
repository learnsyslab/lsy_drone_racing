"""Path planning for the drone racing track.

The planner turns the race track (gates in a signed passing order plus cylindrical obstacles) into a
continuously differentiable, arc-length parameterized 3D path that the
[MPCC controller][lsy_drone_racing.control.my_controller.MPCCController] tracks.

The implementation follows the reference solution ``path_planning.py`` of the successful
`lsy_drone_racing_ws26`_ entry:

* Each gate contributes ``num_intermediate_points`` waypoints along the signed gate normal, so the
  drone enters the gate on one side and leaves on the other.
* A detour waypoint is inserted whenever the chord between two consecutive gate passes points more
  than ``angle_threshold_deg`` away from the gate normal, which would otherwise require a sharp
  turn around a gate post.
* Obstacle crossings are detected on a dense spline resampling and replaced by a single waypoint
  that pushes the path out of the obstacle's safety radius.
* The resulting polyline is fitted with a cubic spline and reparameterized by arc length.

Deviations from the reference implementation (all required for this environment):

* The gate order is taken from ``obs["gate_sequence"]`` and ``obs["gate_sequence_direction"]``
  instead of the raw gate array order, so repeated gates and reverse passes are handled.
* The blocked interval's entry sample is kept and a degenerate (dead-centre) obstacle crossing
  falls back to the chord perpendicular instead of placing the waypoint inside the obstacle.
* The path is extended past the last gate and defines a linear continuation outside its parameter
  range, so the controller may evaluate it slightly ahead or behind without extrapolating a cubic.

.. _lsy_drone_racing_ws26: https://github.com/lihengrongfei/lsy_drone_racing_ws26
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Weight of the proximity bump around a gate centre. Mirrors the reference `qc_dyn` allocation and
#: the Gaussian contouring-weight allocation of the T-RO22 paper (Sec. IV-C).
GATE_PROXIMITY_WEIGHT = 0.4
#: Weight of the (slightly weaker) proximity bump around an obstacle.
OBSTACLE_PROXIMITY_WEIGHT = 0.2
#: Width of the proximity bumps, `exp(-_PROXIMITY_DECAY * d^2)`.
_PROXIMITY_DECAY = 8.0


@dataclass(frozen=True)
class PathConfig:
    """Configuration of the path planner.

    Attributes:
        approach_distance: Half width [m] of the waypoint spread along each gate normal.
        num_intermediate_points: Number of waypoints generated per gate pass.
        angle_threshold_deg: Chord angle [deg] above which a detour waypoint is inserted.
        detour_distance: Offset [m] of a detour waypoint from the gate centre.
        min_offset: Smallest approach offset [m] the obstacle-aware shortening may produce.
        safety_distance: Required XY clearance [m] between the path and an obstacle centre.
        push_scale: Factor applied to ``safety_distance`` when placing a detour waypoint. The
            spline between the old and the new waypoint cuts inside the detour point, so the
            waypoint is placed farther out than the required clearance.
        obstacle_iter_max: Maximum number of obstacle avoidance passes. The spline is refitted
            after each pass and the loop stops as soon as the required clearance is achieved.
        length_growth_max: Maximum factor by which obstacle avoidance may lengthen the path before
            it is rejected. A rejected pass keeps the previous, shorter path.
        smooth_iterations: Number of Laplacian smoothing sweeps applied to the polyline after
            obstacle avoidance. The push-out inserts waypoints that create near-cusps, which would
            otherwise cap the flyable path speed.
        anchor_window: Arc length [m] around every gate centre that is kept fixed during smoothing,
            so that the path cannot drift sideways inside a gate plane.
        gate_straight_length: Half length [m] of the straight passage forced through each gate.
        gate_straight_blend: Length [m] over which the straight passage blends into the rest of the
            path.
        gate_frame_clearance: In-plane distance [m] from a gate centre that the path must keep when
            it crosses the gate plane but does not pass through the gate. The solid frame ends at
            0.36 m and the drone is a 0.09 m sphere, so anything closer would hit the frame.
        gate_frame_window: Arc length [m] over which an unintended frame crossing is pushed away.
        climb_height: Height [m] added above the start position before heading to the first gate.
        runout: Extra length [m] the path continues behind the last gate.
        duration: Duration [s] of the auxiliary time-parameterized spline.
        sampling_freq: Sampling frequency [Hz] of the obstacle collision scan.
        arc_step: Target spacing [m] of the arc-length reparameterization.
        arc_tol: Relative segment-length deviation at which the arc-length loop stops.
        arc_iter_max: Maximum number of arc-length reparameterization iterations.
    """

    approach_distance: float = 0.6
    num_intermediate_points: int = 5
    angle_threshold_deg: float = 120.0
    detour_distance: float = 0.65
    min_offset: float = 0.15
    safety_distance: float = 0.26
    push_scale: float = 1.4
    obstacle_iter_max: int = 2
    length_growth_max: float = 1.25
    smooth_iterations: int = 600
    anchor_window: float = 0.25
    gate_straight_length: float = 0.55
    gate_straight_blend: float = 0.25
    gate_frame_clearance: float = 0.55
    gate_frame_window: float = 0.3
    climb_height: float = 0.25
    runout: float = 0.4
    duration: float = 30.0
    sampling_freq: float = 50.0
    arc_step: float = 0.05
    arc_tol: float = 1e-3
    arc_iter_max: int = 20


@dataclass(frozen=True)
class GateFrames:
    """Body frames of all gates in the track.

    Attributes:
        normals: Passing direction of each gate, shape ``(n_gates, 3)``.
        y_axes: In-plane lateral axis of each gate, shape ``(n_gates, 3)``.
        z_axes: In-plane vertical axis of each gate, shape ``(n_gates, 3)``.
    """

    normals: NDArray[np.floating]
    y_axes: NDArray[np.floating]
    z_axes: NDArray[np.floating]

    @staticmethod
    def from_quaternions(gates_quat: NDArray[np.floating]) -> GateFrames:
        """Extract the gate body frames from their xyzw quaternions.

        Gates have to be crossed along their local x-axis, see
        [gate_passed][lsy_drone_racing.envs.utils.gate_passed].

        Args:
            gates_quat: Gate orientations as (x, y, z, w) quaternions, shape ``(n_gates, 4)``.

        Returns:
            The gate frames.
        """
        rot = R.from_quat(np.atleast_2d(np.asarray(gates_quat, dtype=float))).as_matrix()
        return GateFrames(normals=rot[:, :, 0], y_axes=rot[:, :, 1], z_axes=rot[:, :, 2])


@dataclass(frozen=True)
class PathInfo:
    """Metadata of a planned path.

    Attributes:
        waypoints: The fitted waypoints, shape ``(n_waypoints, 3)``.
        gate_centers: Gate centre of every gate pass in passing order, shape ``(n_passes, 3)``.
        obstacles: Obstacle positions used for collision avoidance, shape ``(n_obstacles, 3)``.
    """

    waypoints: NDArray[np.floating]
    gate_centers: NDArray[np.floating]
    obstacles: NDArray[np.floating]


class ArcPath:
    """Arc-length parameterized, continuously differentiable 3D path.

    Inside its parameter range ``[0, length]`` the path is a natural cubic spline through the
    waypoints with knots equal to the cumulative chord length. Outside the range it continues
    linearly along the terminal tangents, so the controller can evaluate it slightly ahead of the
    end of the track without extrapolating a cubic.
    """

    def __init__(self, points: NDArray[np.floating]):
        """Fit the path.

        Args:
            points: Path waypoints in world coordinates, shape ``(n_points, 3)``.

        Raises:
            ValueError: If fewer than two distinct waypoints are provided.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate(([True], seg > 1e-9))
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("A path requires at least two distinct waypoints.")
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        self._spline = CubicSpline(s, pts, bc_type="natural", extrapolate=False)
        self.points = pts
        self.length = float(s[-1])
        self._pos_start = pts[0]
        self._pos_end = pts[-1]
        self._tan_start = self._unit_tangent(0.0)
        self._tan_end = self._unit_tangent(self.length)

    def _unit_tangent(self, tau: float) -> NDArray[np.floating]:
        """Evaluate the unit tangent at a single parameter value."""
        tangent = np.asarray(self._spline(min(max(tau, 0.0), self.length), 1), dtype=float)
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _linear_tangent(self, tau: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the linear-extension tangent for parameters outside the path range."""
        out = np.empty((len(tau), 3))
        high = tau > self.length
        out[high] = self._tan_end
        out[~high] = self._tan_start
        return out

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the path position.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The path position, shape ``tau.shape + (3,)``.
        """
        tau_arr = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(tau_arr).ravel()
        clipped = np.clip(flat, 0.0, self.length)
        out = np.asarray(self._spline(clipped), dtype=float).reshape(-1, 3)
        high = flat > self.length
        if np.any(high):
            out[high] = self._pos_end + np.outer(flat[high] - self.length, self._tan_end)
        low = flat < 0.0
        if np.any(low):
            out[low] = self._pos_start + np.outer(flat[low], self._tan_start)
        return out.reshape(*tau_arr.shape, 3)

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent of the path.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        tau_arr = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(tau_arr).ravel()
        clipped = np.clip(flat, 0.0, self.length)
        out = np.asarray(self._spline(clipped, 1), dtype=float).reshape(-1, 3)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, np.maximum(norm, 1e-9))
        outside = (flat < 0.0) | (flat > self.length)
        if np.any(outside):
            out[outside] = self._linear_tangent(flat[outside])
        return out.reshape(*tau_arr.shape, 3)

    def curvature(self, tau: NDArray[np.floating] | float, step: float = 0.05) -> NDArray:
        """Evaluate the curvature of the path.

        The curvature is measured over a finite arc length rather than from the analytic second
        derivative, because the analytic value of the fitted spline is very sensitive to the dense
        waypoint sample of the obstacle avoidance.

        Args:
            tau: Arc length parameter, any shape.
            step: Half width [m] over which the tangent change is measured.

        Returns:
            The curvature in 1/m, shape ``tau.shape``.
        """
        tau_arr = np.asarray(tau, dtype=float)
        forward = self.tangent(np.clip(tau_arr + step, 0.0, self.length))
        backward = self.tangent(np.clip(tau_arr - step, 0.0, self.length))
        return np.linalg.norm(forward - backward, axis=-1) / (2.0 * step)

    def closest(
        self,
        point: NDArray[np.floating],
        tau_prev: float,
        window: tuple[float, float] = (-0.15, 0.6),
        step: float = 0.02,
    ) -> float:
        """Project a position onto the path, searching around the previous progress.

        The search is limited to a window around ``tau_prev`` and refined locally, which keeps the
        projection robust for paths that revisit the same region (a loop through a gate twice).

        Args:
            point: Position in world coordinates, shape ``(3,)``.
            tau_prev: Previous arc length parameter, used to center the search window.
            window: Search range relative to ``tau_prev``, in meters.
            step: Sampling distance of the coarse search, in meters.

        Returns:
            The arc length parameter closest to ``point``.
        """
        point = np.asarray(point, dtype=float).reshape(3)
        low = max(0.0, tau_prev + window[0])
        high = min(self.length, tau_prev + window[1])
        if high <= low:
            high = min(self.length, low + step)
        grid = np.arange(low, high + 1e-9, step)
        if len(grid) < 2:
            grid = np.array([low, high])
        dists = np.linalg.norm(self.pos(grid) - point, axis=1)
        best = int(np.argmin(dists))
        # Parabolic refinement around the best sample for sub-step accuracy.
        if 0 < best < len(grid) - 1:
            y0, y1, y2 = dists[best - 1], dists[best], dists[best + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                offset = 0.5 * (y0 - y2) / denom
                return float(np.clip(grid[best] + offset * step, low, high))
        return float(grid[best])


class PathPlanner:
    """Generate an obstacle-avoiding, arc-length parameterized path through the track."""

    def __init__(self, config: PathConfig | None = None):
        """Initialize the planner.

        Args:
            config: Planner configuration. Defaults to [PathConfig][lsy_drone_racing.control.
                path_planning.PathConfig].
        """
        self.config = config or PathConfig()

    def plan(
        self, obs: dict[str, NDArray[np.floating]], start_pos: NDArray[np.floating]
    ) -> tuple[ArcPath, PathInfo]:
        """Plan the path from the start position through all gates in the configured order.

        Args:
            obs: Environment observation. Requires ``gates_pos``, ``gates_quat``,
                ``gate_sequence``, ``gate_sequence_direction`` and ``obstacles_pos``.
            start_pos: Position the path starts at, shape ``(3,)``.

        Returns:
            The arc-length parameterized path and its metadata.
        """
        gate_pos = np.asarray(obs["gates_pos"], dtype=float).reshape(-1, 3)
        frames = GateFrames.from_quaternions(np.asarray(obs["gates_quat"], dtype=float))
        obstacles = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)
        sequence = np.asarray(obs["gate_sequence"], dtype=int).ravel()
        direction = np.sign(np.asarray(obs["gate_sequence_direction"], dtype=float)).ravel()
        direction = np.where(direction == 0, 1.0, direction)

        pass_pos = gate_pos[sequence]
        pass_normals = frames.normals[sequence] * direction[:, None]
        pass_y_axes = frames.y_axes[sequence]
        pass_z_axes = frames.z_axes[sequence]

        start = np.asarray(start_pos, dtype=float).reshape(3)
        blocks = self._gate_blocks(start, pass_pos, pass_normals, obstacles)
        blocks = self._insert_detours(blocks, pass_pos, pass_normals, pass_y_axes, pass_z_axes)
        waypoints = np.concatenate(
            [np.concatenate(blocks, axis=0), self._runout(blocks, pass_pos, pass_normals)]
        )
        waypoints = self._avoid_obstacles(waypoints, obstacles)
        waypoints = self._smooth_path(waypoints, obstacles)
        waypoints = self._straighten_gate_passages(waypoints, pass_pos, pass_normals)
        waypoints = self._avoid_gate_frames(
            waypoints, pass_pos, pass_normals, pass_y_axes, pass_z_axes
        )
        path = self._to_arc_path(waypoints)
        info = PathInfo(
            waypoints=waypoints,
            gate_centers=pass_pos,
            obstacles=obstacles if len(obstacles) else np.zeros((0, 3)),
        )
        return path, info

    def _gate_blocks(
        self,
        start: NDArray[np.floating],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
        obstacles: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Build one waypoint block per gate pass, preceded by start and climb blocks."""
        cfg = self.config
        climb = start + np.array([0.0, 0.0, cfg.climb_height])
        blocks = [start.reshape(1, 3), climb.reshape(1, 3)]
        for pos, normal in zip(pass_pos, pass_normals, strict=True):
            d_in = self._side_offset(pos, normal, -1.0, obstacles)
            d_out = self._side_offset(pos, normal, +1.0, obstacles)
            offsets = np.linspace(-d_in, d_out, cfg.num_intermediate_points)
            blocks.append(pos.reshape(1, 3) + offsets[:, None] * normal.reshape(1, 3))
        return blocks

    def _side_offset(
        self,
        gate_pos: NDArray[np.floating],
        normal: NDArray[np.floating],
        sign: float,
        obstacles: NDArray[np.floating],
    ) -> float:
        """Return the largest approach offset on one side of a gate that clears all obstacles.

        On this track the nominal approach points of three gate passes are within a few
        centimeters of an obstacle pole, so a fixed offset would make the path graze a pole. The
        offset is shortened (down to ``min_offset``) until the waypoint clears every obstacle by the
        same distance a detour waypoint is placed at.

        Args:
            gate_pos: Gate centre, shape ``(3,)``.
            normal: Signed traversal normal of the gate, shape ``(3,)``.
            sign: ``-1`` for the approach side, ``+1`` for the exit side.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The approach offset in meters.
        """
        cfg = self.config
        if obstacles is None or len(obstacles) == 0:
            return cfg.approach_distance
        required = cfg.safety_distance * cfg.push_scale
        offset = cfg.approach_distance
        while offset > cfg.min_offset:
            point = gate_pos + sign * offset * normal
            clearance = float(np.min(np.linalg.norm(obstacles[:, :2] - point[:2], axis=1)))
            if clearance >= required:
                break
            offset -= 0.05
        return offset

    def _runout(
        self,
        blocks: list[NDArray[np.floating]],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return the final waypoint behind the last gate."""
        distance = self.config.approach_distance + self.config.runout
        return (pass_pos[-1] + distance * pass_normals[-1]).reshape(1, 3)

    def _insert_detours(
        self,
        blocks: list[NDArray[np.floating]],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
        pass_y_axes: NDArray[np.floating],
        pass_z_axes: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Insert a lateral detour waypoint where the path would double back on itself."""
        cfg = self.config
        threshold = np.deg2rad(cfg.angle_threshold_deg)
        out: list[NDArray[np.floating]] = list(blocks[:2])
        for i, block in enumerate(blocks[2:]):
            out.append(block)
            if i + 1 >= len(blocks) - 2:
                continue
            p1 = block[-1]
            p2 = blocks[2 + i + 1][0]
            chord = p2 - p1
            chord_norm = np.linalg.norm(chord)
            if chord_norm < 1e-6:
                continue
            normal = pass_normals[i]
            cos_angle = float(np.clip(np.dot(chord, normal) / chord_norm, -1.0, 1.0))
            if np.arccos(cos_angle) <= threshold:
                continue
            in_plane = chord - np.dot(chord, normal) * normal
            if np.linalg.norm(in_plane) < 1e-6:
                detour_dir = pass_y_axes[i]
            else:
                proj_angle = np.degrees(
                    np.arctan2(np.dot(in_plane, pass_z_axes[i]), np.dot(in_plane, pass_y_axes[i]))
                )
                if -90 <= proj_angle < 45:
                    detour_dir = pass_y_axes[i]
                elif 45 <= proj_angle < 135:
                    detour_dir = pass_z_axes[i]
                else:
                    detour_dir = -pass_y_axes[i]
            out.append((pass_pos[i] + cfg.detour_distance * detour_dir).reshape(1, 3))
        return out

    def _avoid_obstacles(
        self, waypoints: NDArray[np.floating], obstacles: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Push the path out of every obstacle's safety radius.

        A single push-out pass is not enough in general, because the spline that is refitted through
        the modified waypoints cuts inside the inserted detour point. The pass is therefore repeated
        with the refitted spline until the required clearance is met or the iteration limit is hit.

        Args:
            waypoints: Path waypoints, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The obstacle-free waypoints, shape ``(n_points, 3)``.
        """
        if len(obstacles) == 0:
            return waypoints
        cfg = self.config
        points = self._sample(self._create_spline(waypoints, cfg.duration), cfg)
        length_0 = self._path_length(points)
        for _ in range(cfg.obstacle_iter_max):
            for obstacle in obstacles:
                points = self._push_out(points, obstacle)
            if len(points) < 2:
                raise RuntimeError("Obstacle avoidance removed all path points.")
            samples = self._sample(self._create_spline(points, cfg.duration), cfg)
            if self._path_length(samples) > cfg.length_growth_max * length_0:
                # The spline zig-zags around the obstacles instead of going around them. Reject
                # the pass and keep the shorter path.
                break
            points = samples
            if self.clearance(samples, obstacles) >= cfg.safety_distance:
                return samples
        return points

    @staticmethod
    def _path_length(points: NDArray[np.floating]) -> float:
        """Return the polyline length of a sampled path."""
        deltas = np.diff(np.asarray(points, dtype=float), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _smooth_path(
        self, points: NDArray[np.floating], obstacles: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Round the corners of the polyline while keeping the obstacle clearance.

        The waypoint insertion of the obstacle avoidance creates near-cusps: the polyline jumps from
        the entry sample to the detour waypoint and back. Those cusps have a curvature of tens per
        meter and would limit the flyable speed to a fraction of a meter per second. Laplacian
        smoothing rounds them, and every sweep re-projects points that came closer to an obstacle
        than the required clearance onto the clearance circle.

        Smoothing also rounds the corners at the gates, which pushes the path sideways inside the
        gate plane; the drone then follows it into the gate frame. The gate passages are restored
        afterwards, see [PathPlanner._straighten_gate_passages][].

        Args:
            points: Densely sampled path points, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The smoothed path points, shape ``(n_points, 3)``.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        if len(pts) < 3:
            return pts
        radius = self.config.safety_distance * 0.92
        for _ in range(self.config.smooth_iterations):
            pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
            if obstacles is None or len(obstacles) == 0:
                continue
            delta = pts[:, None, :2] - np.asarray(obstacles, dtype=float)[None, :, :2]
            dist = np.linalg.norm(delta, axis=-1)
            nearest = np.argmin(dist, axis=1)
            closest = dist[np.arange(len(pts)), nearest]
            proximity = closest < radius
            if not proximity.any():
                continue
            direction = delta[np.arange(len(pts)), nearest]
            span = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
            offset = np.asarray(obstacles, dtype=float)[nearest, :2]
            pts[proximity, :2] = offset[proximity] + radius * direction[proximity] / span[proximity]
        return pts

    def _avoid_gate_frames(
        self,
        points: NDArray[np.floating],
        gate_pos: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Keep the path out of the solid ring of every gate it is not passing through.

        A gate is a solid frame: its collision geometry is a ring covering 0.2 m to 0.36 m from the
        gate centre in the gate plane, while the drone is a sphere of about 0.09 m radius. A path
        that crosses a gate plane between 0.2 m and 0.45 m from the centre therefore flies straight
        through the frame. Since the legs between gates can pass close to a gate that is not the
        current target, every unintended plane crossing inside that band is pushed outwards, with a
        cosine-tapered displacement so that the correction stays local.

        Args:
            points: Path points, shape ``(n_points, 3)``.
            gate_pos: Gate centres in passing order, shape ``(n_passes, 3)``.
            gate_normals: Signed traversal normal of every pass, shape ``(n_passes, 3)``.
            gate_y_axes: In-plane lateral axis of every pass, shape ``(n_passes, 3)``.
            gate_z_axes: In-plane vertical axis of every pass, shape ``(n_passes, 3)``.

        Returns:
            The corrected path points, shape ``(n_points, 3)``.
        """
        clearance = self.config.gate_frame_clearance
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        window = max(2, int(round(self.config.gate_frame_window / max(spacing, 1e-3))))
        # Group the passes by gate plane: a gate can appear several times in the gate order, and all
        # of its crossings through the opening are legitimate while every other crossing is not.
        planes: dict[tuple, dict] = {}
        for centre, normal, y_axis, z_axis in zip(
            gate_pos, gate_normals, gate_y_axes, gate_z_axes, strict=True
        ):
            centre = np.asarray(centre, dtype=float)
            normal = np.asarray(normal, dtype=float)
            key = tuple(np.round(np.concatenate((centre, normal)), 6))
            plane = planes.setdefault(
                key,
                {
                    "centre": centre,
                    "normal": normal,
                    "y": np.asarray(y_axis, dtype=float),
                    "z": np.asarray(z_axis, dtype=float),
                    "passes": 0,
                },
            )
            plane["passes"] += 1
        for plane in planes.values():
            centre, normal = plane["centre"], plane["normal"]
            offset = pts - centre
            along = offset @ normal
            crossing = np.where(np.diff(np.sign(along)) != 0)[0]
            if len(crossing) == 0:
                continue
            # The legitimate passages are the crossings through the opening, i.e. the ones closest
            # to the gate centre; the number of passes of this plane tells how many there are.
            crossings = []
            for index in crossing:
                index = int(index)
                fraction = -along[index] / (along[index + 1] - along[index])
                local = offset[index] + fraction * (offset[index + 1] - offset[index])
                lateral = float(local @ plane["y"])
                vertical = float(local @ plane["z"])
                # The frame is a square ring, so the relevant measure is the Chebyshev distance in
                # the gate plane, not the radial one: a point at (0.31, 0.40) is 0.51 m from the
                # centre diagonally, but only 0.04 m below the bottom bar.
                crossings.append((index, max(abs(lateral), abs(vertical)), lateral, vertical))
            crossings.sort(key=lambda item: item[1])
            intended = {item[0] for item in crossings[: plane["passes"]]}
            for index, extent, lateral, vertical in crossings:
                if index in intended or extent >= clearance:
                    continue
                # Push along the dominant in-plane axis, which grows the square clearance fastest.
                if abs(lateral) >= abs(vertical) and abs(lateral) > 1e-6:
                    direction = plane["y"] if lateral > 0 else -plane["y"]
                elif abs(vertical) > 1e-6:
                    direction = plane["z"] if vertical > 0 else -plane["z"]
                else:
                    direction = plane["y"]
                shift = (clearance - extent) * direction
                indices = np.arange(max(0, index - window), min(len(pts), index + window + 1))
                weight = 0.5 * (1.0 + np.cos(np.pi * (indices - index) / window))
                pts[indices] += weight[:, None] * shift
        return pts

    def _straighten_gate_passages(
        self,
        points: NDArray[np.floating],
        gate_pos: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Make every gate passage straight and perpendicular to the gate plane.

        Passing through the gate centre is not sufficient: if the path approaches the gate at an
        angle, the drone crosses the plane in the middle but its body sweeps through the frame's
        side bar, and after the gate it can clip the frame while turning away. The gate frame is a
        solid ring with a 0.4 m opening, so the path has to run along the gate normal through the
        opening.

        The samples within ``gate_straight_length`` of the gate plane (and within the gate's local
        neighbourhood) are moved onto the normal line through the gate centre, with a smooth blend
        at the ends so that the transition does not become a corner.

        Args:
            points: Smoothed path points, shape ``(n_points, 3)``.
            gate_pos: Gate centres in passing order, shape ``(n_passes, 3)``.
            gate_normals: Signed traversal normal of every pass, shape ``(n_passes, 3)``.

        Returns:
            The corrected path points, shape ``(n_points, 3)``.
        """
        length = self.config.gate_straight_length
        blend = max(self.config.gate_straight_blend, 1e-3)
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        half_window = max(2, int(round(length / max(spacing, 1e-3))))
        for centre, normal in zip(gate_pos, gate_normals, strict=True):
            centre = np.asarray(centre, dtype=float)
            normal = np.asarray(normal, dtype=float)
            # Straighten only the crossing window of this gate, so a path that legitimately passes
            # by a gate at some distance is not pulled through it.
            crossing = int(np.argmin(np.linalg.norm(pts - centre, axis=1)))
            low = max(1, crossing - half_window)
            high = min(len(pts) - 1, crossing + half_window + 1)
            indices = np.arange(low, high)
            along = (pts[indices] - centre) @ normal
            target = centre + along[:, None] * normal
            distance = np.abs(indices - crossing) * spacing
            ramp = np.clip((half_window * spacing - distance) / blend, 0.0, 1.0)
            weight = 0.5 * (1.0 - np.cos(np.pi * ramp))
            pts[indices] = (1.0 - weight)[:, None] * pts[indices] + weight[:, None] * target
        return pts

    @staticmethod
    def clearance(points: NDArray[np.floating], obstacles: NDArray[np.floating]) -> float:
        """Return the minimum XY distance between a sampled path and the obstacle centres.

        Args:
            points: Sampled path points, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The minimum XY clearance in meters, or ``inf`` if there are no obstacles.
        """
        if len(obstacles) == 0:
            return float("inf")
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        delta = points[:, None, :2] - np.asarray(obstacles, dtype=float)[None, :, :2]
        return float(np.min(np.linalg.norm(delta, axis=-1)))

    def _sample(self, spline: CubicSpline, cfg: PathConfig) -> NDArray[np.floating]:
        """Densely sample a time-parameterized spline."""
        n_samples = max(2, int(cfg.sampling_freq * cfg.duration))
        return np.asarray(spline(np.linspace(0.0, cfg.duration, n_samples)), dtype=float)

    def _push_out(
        self, points: NDArray[np.floating], obstacle: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Replace one blocked interval of the path by a waypoint next to the obstacle."""
        safety = self.config.safety_distance
        dist = np.linalg.norm(points[:, :2] - obstacle[:2], axis=1)
        keep: list[NDArray[np.floating]] = []
        i = 0
        n = len(points)
        while i < n:
            if dist[i] >= safety:
                keep.append(points[i])
                i += 1
                continue
            # Find the first free sample after the blocked interval. The entry sample is kept so
            # the path still approaches the obstacle before turning away.
            j = i
            while j < n and dist[j] < safety:
                j += 1
            keep.append(points[i])
            if j < n:
                keep.append(self._detour_point(points[i], points[j], obstacle))
            i = j
        return np.asarray(keep, dtype=float).reshape(-1, 3)

    def _detour_point(
        self,
        entry: NDArray[np.floating],
        exit_: NDArray[np.floating],
        obstacle: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute the waypoint that leads the path around an obstacle."""
        distance = self.config.safety_distance * self.config.push_scale
        dir_in = entry[:2] - obstacle[:2]
        dir_out = exit_[:2] - obstacle[:2]
        dir_in = dir_in / (np.linalg.norm(dir_in) + 1e-9)
        dir_out = dir_out / (np.linalg.norm(dir_out) + 1e-9)
        direction = dir_in + dir_out
        if np.linalg.norm(direction) < 1e-6:
            # Dead-centre crossing: use the chord perpendicular instead of the (zero) bisector.
            chord = exit_[:2] - entry[:2]
            direction = np.array([-chord[1], chord[0]])
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([0.0, 1.0])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        xy = obstacle[:2] + distance * direction
        return np.array([xy[0], xy[1], 0.5 * (entry[2] + exit_[2])])

    def _create_spline(self, waypoints: NDArray[np.floating], duration: float) -> CubicSpline:
        """Fit a cubic spline through the waypoints, parameterized by chord length."""
        pts = np.asarray(waypoints, dtype=float).reshape(-1, 3)
        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        keep = np.concatenate(([True], segment_lengths > 1e-9))
        pts, cumulative = pts[keep], cumulative[keep]
        knots = cumulative / (cumulative[-1] + 1e-9) * duration
        return CubicSpline(knots, pts, bc_type="natural")

    def _reparametrize(self, spline: CubicSpline) -> CubicSpline:
        """Reparameterize a spline so that its parameter is the arc length."""
        cfg = self.config
        current = spline
        for _ in range(cfg.arc_iter_max):
            span = float(current.x[-1] - current.x[0])
            n_segments = max(2, int(span / cfg.arc_step))
            params = np.linspace(float(current.x[0]), float(current.x[-1]), n_segments)
            points = np.asarray(current(params), dtype=float)
            segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            current = CubicSpline(cumulative, points, bc_type="natural")
            mean = float(np.mean(segment_lengths))
            if mean > 0.0 and float(np.std(segment_lengths)) / mean < cfg.arc_tol:
                break
        return current

    def _to_arc_path(self, waypoints: NDArray[np.floating]) -> ArcPath:
        """Fit and reparameterize the path, then hand it to an [ArcPath][...ArcPath]."""
        cfg = self.config
        spline = self._reparametrize(self._create_spline(waypoints, cfg.duration))
        span = float(spline.x[-1] - spline.x[0])
        n_points = max(2, int(span / 0.02))
        return ArcPath(
            np.asarray(
                spline(np.linspace(float(spline.x[0]), float(spline.x[-1]), n_points)), dtype=float
            ).reshape(-1, 3)
        )


def track_reference(
    path: ArcPath,
    thetas: NDArray[np.floating],
    gate_centers: NDArray[np.floating],
    obstacles: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate the path reference used by the MPCC stage cost.

    The proximity weight ``qc`` follows the Gaussian contouring-weight allocation of the T-RO22
    paper: it peaks at the gates and obstacles and decays to zero elsewhere. The controller scales
    its contouring and lag weights with it, so the drone hugs the path exactly where it has to pass
    through a gate and is free to cut corners in between.

    Args:
        path: The planned path.
        thetas: Arc length parameters at which to evaluate, shape ``(n, )``.
        gate_centers: Gate centres in passing order, shape ``(n_passes, 3)``.
        obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

    Returns:
        The reference positions ``(n, 3)``, unit tangents ``(n, 3)`` and proximity weights
        ``(n, )`` in ``[0, 1]``.
    """
    thetas = np.asarray(thetas, dtype=float).ravel()
    positions = path.pos(thetas)
    tangents = path.tangent(thetas)
    weights = np.zeros(len(thetas))
    if gate_centers is not None and len(gate_centers):
        dist_sq = np.sum((positions[:, None, :] - np.asarray(gate_centers)[None, :, :]) ** 2, -1)
        weights = np.maximum(
            weights, (GATE_PROXIMITY_WEIGHT * np.exp(-_PROXIMITY_DECAY * dist_sq)).max(axis=1)
        )
    if obstacles is not None and len(obstacles):
        delta = positions[:, None, :2] - np.asarray(obstacles)[None, :, :2]
        dist_sq = np.sum(delta**2, -1)
        weights = np.maximum(
            weights, (OBSTACLE_PROXIMITY_WEIGHT * np.exp(-_PROXIMITY_DECAY * dist_sq)).max(axis=1)
        )
    return positions, tangents, weights
