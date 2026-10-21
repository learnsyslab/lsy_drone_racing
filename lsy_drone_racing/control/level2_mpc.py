"""Four-dimensional attitude MPC with observed-map collision-aware routing.

The finite-horizon QP predicts position, velocity and acceleration with a first-order
attitude response. Only its first acceleration command is applied as roll, pitch,
yaw and collective thrust (N). No simulator state is inspected by this controller.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from ml_collections import ConfigDict
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


class Level2MPC(Controller):
    """Replan on observed map updates and track with constrained linear MPC."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Build a 1.2-second, 20-stage prediction model and initialize navigation."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("Level2MPC requires the 4D attitude interface.")
        self.dt = 1.0 / config.env.freq
        params = load_params(config.sim.dynamics, config.sim.drone)
        self.mass0 = float(params["mass"])
        self.mass = self.mass0
        self.thrust_bounds = np.array([params["thrust_min"], params["thrust_max"]]) * 4
        self.phase = "takeoff"
        self.passed = 0
        self.target = np.asarray(obs["pos"], float).copy()
        self.target[2] = 0.65
        self.route = []
        self.map_key = None
        self.n = 20
        self.h = 0.06
        # Exact ZOH discretization of p'=v, v'=a, a'=(u-a)/tau.
        continuous = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, -1 / 0.10, 1 / 0.10], [0, 0, 0, 0.0]]
        )
        discrete = expm(continuous * self.h)
        A, B = discrete[:3, :3], discrete[:3, 3:]
        F = np.vstack([np.linalg.matrix_power(A, k + 1) for k in range(self.n)])
        G = np.zeros((3 * self.n, self.n))
        for k in range(self.n):
            for j in range(k + 1):
                G[3 * k : 3 * k + 3, j] = (np.linalg.matrix_power(A, k - j) @ B).ravel()
        weights = np.tile([60.0, 8.0, 0.08], self.n)
        weights[-3:] *= 3
        D = np.eye(self.n) - np.eye(self.n, k=-1)
        self.F, self.G = F, G
        self.GQ = G.T * weights
        self.H = self.GQ @ G + 0.35 * np.eye(self.n) + 0.4 * D.T @ D
        self.invH = np.linalg.inv(self.H)
        self.L = np.linalg.eigvalsh(self.H)[-1]
        self.u = np.zeros((self.n, 3))
        self.last_vel = np.asarray(obs["vel"], float).copy()
        self.last_action = np.array([0.0, 0.0, 0.0, self.mass * 9.81])
        self.accel = np.zeros(3)
        self.tick = 0
        self.failures = 0
        self.lower = np.asarray(config.env.track.safety_limits.pos_limit_low, float) + 0.14
        self.upper = np.asarray(config.env.track.safety_limits.pos_limit_high, float) - 0.14
        self.lower[2] = 0.15

    def geometry(self, obs: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the active signed gate entry, including repeated physical gates."""
        gid = int(obs["gate_sequence"][index])
        rot = Rotation.from_quat(obs["gates_quat"][gid]).as_matrix()
        return np.asarray(obs["gates_pos"][gid], float), rot[:, 0] * int(
            obs["gate_sequence_direction"][index]
        )

    def update_map(self, obs: dict) -> bool:
        """Use only public observations; inflate objects not yet observed accurately."""
        self.centers = np.asarray(obs["gates_pos"], float)
        self.rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        self.poles = np.asarray(obs["obstacles_pos"], float)
        self.pole_r = np.where(obs["obstacles_visited"], 0.22, 0.35)
        self.margin = np.where(obs["gates_visited"], 0.105, 0.24)
        key = (
            self.centers.tobytes(),
            self.rot.tobytes(),
            self.poles.tobytes(),
            np.asarray(obs["gates_visited"]).tobytes(),
            np.asarray(obs["obstacles_visited"]).tobytes(),
        )
        changed = key != self.map_key
        self.map_key = key
        return changed

    def clear_segments(self, a: np.ndarray, ends: np.ndarray, escape: bool = False) -> np.ndarray:
        """Check segments against inflated poles and the gate.xml collision boxes.

        Escape edges may leave a pole's planning margin, but cannot approach it
        closer than the starting distance (minus 5 mm), or within 12 cm. Their
        endpoints must clear the full margin. This is not a collision guarantee
        for arbitrary tracking errors or a different drone geometry.
        """
        ends = np.atleast_2d(ends)
        good = np.ones(len(ends), bool)
        delta = ends - a
        for pole, radius in zip(self.poles, self.pole_r):
            full_radius = radius
            if escape:
                radius = min(radius, max(0.12, np.linalg.norm(a[:2] - pole[:2]) - 0.005))
            alpha = np.clip(
                ((pole[:2] - a[:2]) * delta[:, :2]).sum(axis=1)
                / np.maximum((delta[:, :2] ** 2).sum(axis=1), 1e-12),
                0,
                1,
            )
            nearest = a + alpha[:, None] * delta
            good &= ~(
                (np.linalg.norm(nearest[:, :2] - pole[:2], axis=1) < radius)
                & (np.minimum(a[2], ends[:, 2]) < pole[2] + 0.12)
            )
            good &= (np.linalg.norm(ends[:, :2] - pole[:2], axis=1) >= full_radius) | (
                ends[:, 2] >= pole[2] + 0.12
            )
        for center, rot, margin in zip(self.centers, self.rot, self.margin):
            local = (a - center) @ rot
            local_delta = delta @ rot
            parallel = np.abs(local_delta) < 1e-10
            safe_delta = np.where(parallel, 1.0, local_delta)
            for box, half in [
                ([0, 0, 0.28], [0.01, 0.36, 0.08]),
                ([0, 0, -0.28], [0.01, 0.36, 0.08]),
                ([0, -0.28, 0], [0.01, 0.08, 0.36]),
                ([0, 0.28, 0], [0.01, 0.08, 0.36]),
                ([0, 0, -0.86], [0.05, 0.05, 0.5]),
            ]:
                low = np.array(box) - np.array(half) - margin
                high = np.array(box) + np.array(half) + margin
                t1 = (low - local) / safe_delta
                t2 = (high - local) / safe_delta
                near = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=1)
                far = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=1)
                outside = (parallel & ((local < low) | (local > high))).any(axis=1)
                good &= ~((np.maximum(near, 0.0) <= np.minimum(far, 1.0)) & ~outside)
        good &= ((ends >= self.lower) & (ends <= self.upper)).all(axis=1)
        return good

    def plan(self, pos: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Search a visibility graph with lazy A* and exact segment-box intersections."""
        if self.clear_segments(pos, goal, escape=True)[0]:
            return [goal.copy()]
        nodes = [pos, goal]
        heights = [goal[2], pos[2]]
        for z in heights:
            for pole, radius in zip(self.poles, self.pole_r):
                for angle in np.arange(12) * np.pi / 6:
                    nodes.append(
                        np.array(
                            [
                                pole[0] + (radius + 0.10) * np.cos(angle),
                                pole[1] + (radius + 0.10) * np.sin(angle),
                                z,
                            ]
                        )
                    )
            for center, rot, margin in zip(self.centers, self.rot, self.margin):
                for x in [-margin - 0.12, margin + 0.12]:
                    for y in [-0.36 - margin - 0.12, 0.36 + margin + 0.12]:
                        point = center + rot @ np.array([x, y, 0])
                        point[2] = z
                        nodes.append(point)
        nodes = np.asarray(nodes)
        distance = np.full(len(nodes), np.inf)
        distance[0] = 0
        prev = np.full(len(nodes), -1, int)
        queue = [(np.linalg.norm(pos - goal), 0)]
        visited = set()
        while queue:
            _, i = heapq.heappop(queue)
            if i in visited:
                continue
            if i == 1:
                break
            visited.add(i)
            ok = self.clear_segments(nodes[i], nodes, escape=(i == 0))
            costs = distance[i] + np.linalg.norm(nodes - nodes[i], axis=1)
            for j in np.flatnonzero(ok & (costs < distance)):
                distance[j] = costs[j]
                prev[j] = i
                heapq.heappush(queue, (costs[j] + np.linalg.norm(nodes[j] - goal), int(j)))
        if not np.isfinite(distance[1]):
            self.failures += 1
            return [pos.copy()]
        route, idx = [], 1
        while idx != 0:
            route.append(nodes[idx])
            idx = int(prev[idx])
        return route[::-1]

    def navigation(self, obs: dict, changed: bool) -> tuple[np.ndarray, float]:
        """Take off, observe and align, cross the center, then clear the gate plane."""
        pos = np.asarray(obs["pos"], float)
        passed = int(obs["n_gates_passed"])
        if passed >= len(obs["gate_sequence"]):
            return pos, 0.0
        reset_route = changed
        if passed != self.passed:
            center, normal = self.geometry(obs, self.passed)
            self.target = center + 0.20 * normal
            self.passed = passed
            self.phase = "exit"
            self.route = [self.target.copy()]
        if self.phase == "takeoff":
            if np.linalg.norm(pos - self.target) < 0.10:
                self.phase = "approach"
                reset_route = True
            else:
                return self.target, 0.65
        if self.phase == "exit":
            if np.linalg.norm(pos - self.target) > 0.04:
                return self.target, 0.65
            self.phase = "approach"
            reset_route = True
        center, normal = self.geometry(obs, passed)
        if self.phase == "approach":
            gid = int(obs["gate_sequence"][passed])
            ready = False
            if obs["gates_visited"][gid]:
                # Choose a staging distance that also clears poles near the opening.
                entries = [center - d * normal for d in [0.23, 0.20, 0.17, 0.14, 0.28, 0.34]]
                feasible = [p for p in entries if self.clear_segments(p, center)[0]]
                self.target = feasible[0] if feasible else entries[0]
                ready = bool(feasible)
            else:
                self.target = center - 0.28 * normal
            if not self.clear_segments(self.target, self.target)[0]:
                lateral = self.rot[gid][:, 1]
                candidates = [center - 0.30 * normal + s * 0.27 * lateral for s in [-1, 1]]
                candidates += [center + 0.30 * normal + s * 0.27 * lateral for s in [-1, 1]]
                feasible = [p for p in candidates if self.clear_segments(p, p)[0]]
                if feasible:
                    self.target = min(feasible, key=lambda p: np.linalg.norm(pos - p))
                    ready = False
            if reset_route or not self.route:
                self.route = self.plan(pos, self.target)
            while len(self.route) > 1 and np.linalg.norm(pos - self.route[0]) < 0.06:
                self.route.pop(0)
            if (
                ready
                and np.linalg.norm(pos - self.target) < 0.05
                and np.linalg.norm(obs["vel"]) < 0.25
            ):
                self.phase = "cross"
                self.route = []
            else:
                return self.route[0], 0.95
        self.target = center + 0.28 * normal
        return self.target, 0.65

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return [roll, pitch, yaw, collective thrust], in radians and newtons."""
        changed = self.update_map(obs)
        goal, speed = self.navigation(obs, changed)
        pos, vel = np.asarray(obs["pos"], float), np.asarray(obs["vel"], float)
        measured = (vel - self.last_vel) / self.dt
        self.last_vel = vel.copy()
        self.accel = 0.7 * self.accel + 0.3 * measured
        zaxis = Rotation.from_quat(obs["quat"]).as_matrix()[:, 2]
        # Estimate mass using observed vertical impulse and the previous applied thrust.
        if self.tick > 20 and pos[2] > 0.25 and abs(measured[2]) < 5 and zaxis[2] > 0.85:
            estimate = self.last_action[3] * zaxis[2] / max(4.0, 9.81 + measured[2])
            self.mass = 0.98 * self.mass + 0.02 * np.clip(
                estimate, 0.75 * self.mass0, 1.25 * self.mass0
            )
        delta = goal - pos
        distance = np.linalg.norm(delta)
        direction = delta / max(distance, 1e-8)
        # Build position/velocity references, braking as the waypoint approaches.
        ref = np.zeros((self.n, 3, 3))
        state = np.stack([pos, vel, self.accel])
        limits = np.array([2.5, 2.5, 3.0])
        # Limit requested speed by a reachable position reference over the horizon.
        vref = direction * min(speed, 2.0 * distance)
        tt = np.arange(1, self.n + 1) * self.h
        travel = np.minimum(np.linalg.norm(vref) * tt, distance)
        ref[:, 0, :] = pos + travel[:, None] * direction
        ref[:, 1, :] = vref
        ref[travel >= distance, 1, :] = 0
        acceleration = self.solve_mpc(state, ref, limits)
        force = acceleration + np.array([0.0, 0.0, 9.81])
        pitch = np.arctan2(force[0], force[2])
        roll = np.arctan2(-force[1], np.hypot(force[0], force[2]))
        thrust = self.mass * np.dot(force, zaxis)
        action = np.array(
            [
                np.clip(roll, -0.35, 0.35),
                np.clip(pitch, -0.35, 0.35),
                0.0,
                np.clip(thrust, *self.thrust_bounds),
            ],
            dtype=np.float32,
        )
        self.last_action = action.copy()
        self.tick += 1
        return action

    def solve_mpc(self, state: np.ndarray, reference: np.ndarray, limits: np.ndarray) -> np.ndarray:
        """Approximately minimize the box-constrained condensed quadratic objective.

        Cost: position/velocity/acceleration tracking plus command magnitude and
        within-horizon command differences. Projected-gradient refinement starts
        at the clipped unconstrained optimum and has a fixed iteration budget.
        Only the first planned command is used; the problem is rebuilt next tick.
        """
        gradient = self.GQ @ (self.F @ state - reference.reshape(3 * self.n, 3))
        self.u = np.clip(-self.invH @ gradient, -limits, limits)
        for _ in range(12):
            self.u = np.clip(self.u - (self.H @ self.u + gradient) / self.L, -limits, limits)
        return self.u[0].copy()

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Finish only when every configured gate-order entry has been passed."""
        return int(obs["n_gates_passed"]) >= len(obs["gate_sequence"])

    def render_callback(self, sim: Sim):
        """Display the current route and gate target in the simulation viewer."""
        from crazyflow.sim.visualize import draw_line, draw_points

        draw_points(sim, self.target.reshape(1, 3), rgba=(1.0, 0.2, 0.1, 1.0), size=0.025)
        if len(self.route) > 1:
            draw_line(sim, np.asarray(self.route), rgba=(0.1, 0.8, 0.2, 1.0))
