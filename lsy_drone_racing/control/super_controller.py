import heapq
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cho_factor, cho_solve
from lsy_drone_racing.control.controller import Controller


class SuperController(Controller):
    def __init__(self, obs, info, config):
        self._init_path(obs, info, config)
        self.horizon = 18
        self.h = 0.06
        coefficients = np.array([[1.1799780814151852, -0.2502303831084363, 0.010193999406397404, 0.07011809633883681, -0.038025938086136676, 0.0037011446618741917], [1.168196460147705, -0.23488897900774877, 0.008862868392270126, 0.06978913351435123, -0.056606233939362226, -0.0031405316455949852], [1.0982376478528746, -0.2250331327146906, 0.10706156414715436, 0.04102644089286392, -0.07397828887438959, 0.05161382721586473]])
        rho = np.exp(-self.dt / np.array([.20, .20, .07]))
        original = np.column_stack((rho, np.zeros(3), 1.2 * (1. - rho), np.zeros((3, 3))))
        coefficients = 1.0 * coefficients + 0.0 * original
        step_A, step_B = np.zeros((16, 16)), np.zeros((16, 3))
        for axis, (a, previous_a, b, previous_b, drag, offset) in enumerate(coefficients):
            row = np.zeros(16)
            row[6 + axis], row[9 + axis], row[12 + axis] = a, previous_a, previous_b
            row[3 + axis], row[15] = drag, offset
            raw = 2. * row
            raw[6 + axis] -= 1.
            step_A[axis] = .5 * self.dt ** 2 * raw
            step_A[axis, axis] += 1.
            step_A[axis, 3 + axis] += self.dt
            step_A[3 + axis] = self.dt * raw
            step_A[3 + axis, 3 + axis] += 1.
            step_A[6 + axis] = row
            step_A[9 + axis, 6 + axis] = 1.
            step_B[axis, axis] = self.dt ** 2 * b
            step_B[3 + axis, axis] = 2. * self.dt * b
            step_B[6 + axis, axis] = b
            step_B[12 + axis, axis] = 1.
        step_A[15, 15] = 1.
        A, B = np.eye(16), np.zeros((16, 3))
        for _ in range(round(self.h / self.dt)):
            A, B = step_A @ A, step_A @ B + step_B
        self.model_old_acc = np.zeros(3)
        self.model_old_u = np.zeros(3)
        self.model_tick = -100
        state_map = np.eye(16)
        input_map = np.zeros((16, 3 * self.horizon))
        states, inputs = [], []
        for j in range(self.horizon):
            state_map = A @ state_map
            input_map = A @ input_map
            input_map[:, 3 * j:3 * j + 3] += B
            states.append(state_map.copy())
            inputs.append(input_map.copy())
        self.state_map = np.stack(states)
        self.G = np.stack(inputs)
        self.P = self.G[:, :3].reshape(-1, 3 * self.horizon)
        self.V = self.G[:, 3:6].reshape(-1, 3 * self.horizon)
        self.Q = np.tile([90.0, 90.0, 400.0], self.horizon)
        difference = np.eye(3 * self.horizon) - np.eye(3 * self.horizon, k=-3)
        self.H = self.P.T @ (self.Q[:, None] * self.P) + 4.0 * self.V.T @ self.V
        self.H += 0.03 * np.eye(3 * self.horizon) + 0.03 * difference.T @ difference
        self.cost_scale = float(np.mean(np.diag(self.H)))
        self.solution = np.zeros(3 * self.horizon)
        self.commit_gate = -1
        self.speed_gate = -1
        self.speed_since = 0
        self.stall_ticks = 0
        self.last_recovery = -100
        self.recovery_count = 0
        self.recovery_goal = None

    def _init_path(self, obs, info, config):
        super().__init__(obs, info, config)
        self.dt = 1.0 / float(config.env.freq)
        self.speed = 7.465
        self.acc = 1.3
        self.clearance = 0.20
        self.tick = 0
        self.update_environment(obs)
        self.sequence = np.asarray(obs['gate_sequence'], dtype=int)
        self.directions = np.asarray(obs['gate_sequence_direction'], dtype=int)
        self.ref = self.pos.copy()
        self.ref_vel = np.zeros(3)
        direction = self.gates[self.sequence[0]] - self.pos
        direction[2] = 0.0
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        self.takeoff_direction = direction
        self.takeoff = self.pos + 0.35 * direction + [0.0, 0.0, 0.26]
        self.route = []
        self.index = 0
        self.planned_gate = -1
        self.last_plan = -100
        self.snapshot = None
        self.phase = 'takeoff'
        self.integral = np.zeros(3)
        self.path = None
        self.arc = None
        self.curve = None
        self.progress = 0.0
        self.force_bias = np.array([0.0, 0.0, 2.0])
        self.last_state = None
        self.replan_check = 0
        self.replan_position = self.pos.copy()
        self.measured_acc = np.zeros(3)
        self.tight = False

    def predict_command(self, command, velocity, fallback):
        scales = np.linspace(1.0, 0.0, 11)
        neutral = self.pos - self.force_bias / np.array([11.4, 11.4, 35.7])
        commands = neutral + scales[:, None] * (command - neutral)
        velocities = scales[:, None] * velocity
        commands[-1] = neutral - 0.15 * self.vel
        velocities[-1] = 0.0
        commands = np.vstack((commands, fallback))
        velocities = np.vstack((velocities, velocity))
        scales = np.r_[scales, 0.5]
        position = np.broadcast_to(self.pos, commands.shape).copy()
        speed = np.broadcast_to(self.vel, commands.shape).copy()
        acceleration = np.broadcast_to(self.measured_acc, commands.shape).copy()
        points = [position.copy()]
        for _ in range(12):
            desired = np.array([11.4, 11.4, 35.7]) * (commands - position)
            desired += np.array([5.7, 5.7, 14.3]) * (velocities - speed) + self.force_bias
            limit = 7.0 if self.planned_gate <= 0 else 10.0
            desired = np.clip(desired, [-limit, -limit, -3.0], [limit, limit, 6.0])
            acceleration += (desired - acceleration) * (0.05 / 0.12)
            position += 0.05 * speed + 0.00125 * acceleration
            speed += 0.05 * acceleration
            points.append(position.copy())
        points = np.stack(points, axis=1)
        clear = self.clear_segments(points[:, :-1], points[:, 1:])
        safe = np.all(clear, axis=1)
        if safe[0]:
            return command, velocity
        progress = np.sum(np.cumprod(clear, axis=1), axis=1)
        score = 100.0 * safe + progress + 0.01 * scales
        chosen = int(np.argmax(score))
        return commands[chosen], velocities[chosen]

    def filter_command(self, command, velocity):
        passed = self.passed > self.planned_gate
        kp = np.array([11.4, 11.4, 35.7])
        kd = np.array([5.7, 5.7, 14.3])
        if self.last_state is not None:
            old_pos, old_vel, old_command, old_velocity = self.last_state
            measured = (self.vel - old_vel) / self.dt
            self.measured_acc = 0.5 * self.measured_acc + 0.5 * np.clip(measured, -12.0, 12.0)
            nominal = kp * (old_command - old_pos) + kd * (old_velocity - old_vel)
            residual = np.clip(measured - nominal, [-1.0, -1.0, -4.0], [1.0, 1.0, 4.0])
            rate = np.array([0.005, 0.005, 0.01])
            self.force_bias = (1.0 - rate) * self.force_bias + rate * residual
        command = command - np.clip(self.force_bias / kp, -0.16, 0.16)
        self.free_command = command.copy()
        nominal = kp * (command - self.pos) + kd * (velocity - self.vel) + self.force_bias
        constraints = []
        for centre, known in zip(self.obstacles, self.obstacle_known):
            offset = self.pos[:2] - centre[:2]
            length = max(float(np.linalg.norm(offset)), 1e-6)
            normal = np.r_[offset / length, 0.0]
            distance = length - self.obstacle_radius(centre, known)
            if distance < 0.45:
                constraints.append((normal, -16.0 * distance - 8.0 * float(normal @ self.vel)))
        boxes = [([0.0, 0.0, 0.28], [0.01, 0.36, 0.08]),
                 ([0.0, 0.0, -0.28], [0.01, 0.36, 0.08]),
                 ([0.0, -0.28, 0.0], [0.01, 0.08, 0.36]),
                 ([0.0, 0.28, 0.0], [0.01, 0.08, 0.36]),
                 ([0.0, 0.0, -0.86], [0.05, 0.05, 0.50])]
        for gate_id, (centre, rotation) in enumerate(zip(self.gates, self.rot)):
            local = (self.pos - centre) @ rotation
            padding = np.abs(rotation.T @ self.body_rotation) @ np.array([0.07, 0.07, 0.02]) + 0.02
            opening = 0.20 - padding[1:]
            active = 0 <= self.planned_gate < len(self.sequence) and gate_id == self.sequence[self.planned_gate]
            limit = padding[0] + 0.03 if self.passed > self.planned_gate else 0.50
            corridor = active and self.phase == 'route' and -0.5 < local[0] * self.directions[self.planned_gate] < limit and np.all(np.abs(local[1:]) < opening)
            if corridor:
                for axis in (1, 2):
                    for sign in (-1.0, 1.0):
                        normal = -sign * rotation[:, axis]
                        predicted = local[axis] + 0.12 * float(self.vel @ rotation[:, axis])
                        distance = opening[axis - 1] - sign * predicted
                        constraints.append((normal, -16.0 * distance - 8.0 * float(normal @ self.vel)))
            previous = self.closed_gate(gate_id)
            gate_boxes = boxes + [([0.0, 0.0, 0.0], [0.01, 0.36, 0.36])] if previous else boxes
            for box_id, (origin, size) in enumerate(gate_boxes):
                if corridor and box_id < 4:
                    continue
                point = local - origin
                half = np.asarray(size) + padding
                closest = np.clip(point, -half, half)
                delta = point - closest
                distance = float(np.linalg.norm(delta))
                if distance > 0.35:
                    continue
                if distance > 1e-6:
                    normal = delta / distance
                else:
                    gap = np.abs(point) - half
                    axis = int(np.argmax(gap))
                    normal = np.zeros(3)
                    normal[axis] = 1.0 if point[axis] >= 0 else -1.0
                    distance = float(gap[axis])
                normal = rotation @ normal
                constraints.append((normal, -16.0 * distance - 8.0 * float(normal @ self.vel)))
        safe = nominal.copy()
        for _ in range(4):
            for normal, bound in constraints:
                safe += max(0.0, bound - float(normal @ safe)) * normal
        correction = np.clip((safe - nominal) / kp, -0.30, 0.30)
        command = command + correction
        self.last_state = (self.pos.copy(), self.vel.copy(), command.copy(), velocity.copy())
        return command

    def update_environment(self, obs):
        self.pos = np.asarray(obs['pos'], dtype=float).copy()
        self.vel = np.asarray(obs['vel'], dtype=float).copy()
        self.gates = np.asarray(obs['gates_pos'], dtype=float)
        self.quat = np.asarray(obs['gates_quat'], dtype=float)
        self.rot = R.from_quat(self.quat).as_matrix()
        self.body_rotation = R.from_quat(np.asarray(obs['quat'], dtype=float)).as_matrix()
        self.obstacles = np.asarray(obs['obstacles_pos'], dtype=float).reshape(-1, 3)
        self.obstacle_known = np.asarray(obs.get('obstacles_visited', np.ones(len(self.obstacles))), dtype=bool).reshape(-1)
        self.gate_known = np.asarray(obs.get('gates_visited', np.ones(len(self.gates))), dtype=bool).reshape(-1)
        self.passed = int(obs['n_gates_passed'])

    def obstacle_radius(self, centre, known):
        if not known:
            return 0.28
        if hasattr(self, 'sequence'):
            idx = self.sequence[min(max(self.planned_gate, 0), len(self.sequence) - 1)]
            if known and self.gate_known[idx] and np.linalg.norm(centre[:2] - self.gates[idx, :2]) < 0.38:
                return 0.14 if self.tight else 0.16
        return self.clearance

    def closed_gate(self, gate_id):
        if getattr(self, 'future_gate', None) == gate_id:
            return True
        if self.passed == 0 or self.planned_gate != self.passed or self.passed >= len(self.sequence):
            return False
        previous = self.sequence[self.passed - 1]
        target = self.sequence[self.passed]
        close = np.linalg.norm(self.gates[target, :2] - self.gates[previous, :2]) < 0.95
        return gate_id == previous and gate_id != target

    def clear_segments(self, a, b, margin=0.0):
        a, b = np.broadcast_arrays(a, b)
        d = b - a
        clear = np.ones(a.shape[:-1], dtype=bool)
        for centre, known in zip(self.obstacles, self.obstacle_known):
            radius = self.obstacle_radius(centre, known)
            offset = a[..., :2] - centre[:2]
            den = np.maximum(np.sum(d[..., :2] ** 2, axis=-1), 1e-10)
            dot = np.sum(offset * d[..., :2], axis=-1)
            t = np.clip(-dot / den, 0.0, 1.0)
            distance = np.linalg.norm(offset + t[..., None] * d[..., :2], axis=-1)
            escaping = (np.linalg.norm(offset, axis=-1) < radius) & (dot >= 0)
            clear &= (distance >= radius) | escaping
        for gate_id, (centre, rotation) in enumerate(zip(self.gates, self.rot)):
            local = (a - centre) @ rotation
            delta = d @ rotation
            boxes = [([0, 0, 0.28], [0.01, 0.36, 0.08]),
                     ([0, 0, -0.28], [0.01, 0.36, 0.08]),
                     ([0, -0.28, 0], [0.01, 0.08, 0.36]),
                     ([0, 0.28, 0], [0.01, 0.08, 0.36]),
                     ([0, 0, -0.86], [0.05, 0.05, 0.50])]
            if self.closed_gate(gate_id):
                boxes.append(([0.0, 0.0, 0.0], [0.01, 0.36, 0.36]))
            padding = np.abs(rotation.T @ self.body_rotation) @ np.array([0.07, 0.07, 0.02]) + 0.02
            padding = np.maximum(padding + 0.005, [0.11, 0.11, 0.08])
            if 0 <= self.planned_gate < len(self.sequence) and gate_id != self.sequence[self.planned_gate]:
                padding = padding + margin
            origins, sizes = np.asarray(boxes).transpose(1, 0, 2)
            half = sizes + padding
            point = local[..., None, :] - origins
            delta = delta[..., None, :]
            moving = np.abs(delta) > 1e-10
            den = np.where(moving, delta, 1.0)
            t1, t2 = (-half - point) / den, (half - point) / den
            low = np.where(moving, np.minimum(t1, t2), -np.inf)
            high = np.where(moving, np.maximum(t1, t2), np.inf)
            parallel_clear = np.any(~moving & (np.abs(point) > half), axis=-1)
            enter = np.maximum(np.max(low, axis=-1), 0.0)
            leave = np.minimum(np.min(high, axis=-1), 1.0)
            clear &= np.all(parallel_clear | (enter > leave), axis=-1)
        clear &= np.all((b >= [-2.35, -1.35, 0.12]) & (b <= [2.35, 1.35, 1.85]), axis=-1)
        return clear

    def route_search(self, start, goal):
        if self.clear_segments(start, goal) and self.clear_segments(start, goal, margin=0.10):
            return [goal.copy()]
        nodes = [start.copy(), goal.copy()]
        for centre in self.obstacles:
            for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
                p = np.array([centre[0] + 0.43 * np.cos(angle),
                              centre[1] + 0.43 * np.sin(angle), goal[2]])
                nodes.append(p)
        for centre, rotation in zip(self.gates, self.rot):
            for x in (-0.3, 0.3):
                for y in (-0.60, 0.60):
                    p = centre + rotation @ [x, y, 0.0]
                    nodes.append(p)
                for z in (-0.55, 0.55):
                    p = centre + rotation @ [x, 0.0, z]
                    nodes.append(p)
        nodes = np.asarray(nodes)
        valid = self.clear_segments(nodes, nodes)
        valid[:2] = True
        nodes = nodes[valid]
        clear = self.clear_segments(nodes[:, None, :], nodes[None, :, :])
        costs = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
        roomy = self.clear_segments(nodes[:, None, :], nodes[None, :, :], margin=0.10)
        costs += 2.0 * (~roomy)
        dist = np.full(len(nodes), np.inf)
        prev = np.full(len(nodes), -1, dtype=int)
        dist[0] = 0.0
        queue = [(0.0, 0)]
        while queue:
            cost, u = heapq.heappop(queue)
            if cost > dist[u]:
                continue
            if u == 1:
                path = []
                while u != 0:
                    path.append(nodes[u].copy())
                    u = prev[u]
                return path[::-1]
            for v in np.flatnonzero(clear[u]):
                candidate = cost + costs[u, v]
                if candidate < dist[v]:
                    dist[v], prev[v] = candidate, u
                    heapq.heappush(queue, (candidate, v))
        return []

    def plan_route(self, start, goal):
        forward = goal - start
        speed = float(np.linalg.norm(self.vel))
        near_start = np.linalg.norm(start - self.pos) < 1e-5
        alignment = float(forward @ self.vel) / max(float(np.linalg.norm(forward)) * speed, 1e-6)
        if self.planned_gate == 1 and near_start and speed > 0.6 and np.linalg.norm(forward) > 0.5 and alignment < 0.70:
            lead = start + 0.25 * self.vel
            if self.clear_segments(start, lead):
                route = self.route_search(lead, goal)
                if route:
                    return [lead] + route
        return self.route_search(start, goal)

    def _plan_gate_base(self):
        if self.planned_gate != self.passed:
            self.integral[:2] *= 0.25
        if self.planned_gate != self.passed:
            self.tight = False
        self.last_plan = self.tick
        self.snapshot = np.concatenate((self.gates.ravel(), self.quat.ravel(), self.obstacles.ravel(), self.gate_known, self.obstacle_known))
        self.planned_gate = self.passed
        self.path = None
        if self.passed >= len(self.sequence):
            self.phase = 'finished'
            return
        idx = self.sequence[self.passed]
        centre = self.gates[idx]
        normal = self.rot[idx, :, 0] * self.directions[self.passed]
        if self.passed + 1 < len(self.sequence):
            next_idx = self.sequence[self.passed + 1]
            next_normal = self.rot[next_idx, :, 0] * self.directions[self.passed + 1]
            incoming = centre - self.pos
            outgoing = self.gates[next_idx] - 0.30 * next_normal - centre
            incoming /= max(float(np.linalg.norm(incoming)), 1e-8)
            outgoing /= max(float(np.linalg.norm(outgoing)), 1e-8)
            direction = incoming + outgoing
            direction -= (direction @ normal) * normal
            length = float(np.linalg.norm(direction))
        choices = []
        joint = self.passed == 3
        for offset in (0.0, -0.06, 0.06):
            crossing = centre + offset * self.rot[idx, :, 1]
            exits = [crossing + distance * normal for distance in (0.24, 0.18, 0.32, 0.12, 0.10, 0.08)]
            exits = [p for p in exits if self.clear_segments(crossing, p)]
            if not exits:
                continue
            after = exits[0]
            incoming_alignment = float((centre - self.pos) @ normal) / max(float(np.linalg.norm(centre - self.pos)), 1e-6)
            long_entry = self.passed == len(self.sequence) - 1 or incoming_alignment < 0.65
            approaches = (0.55, 0.45, 0.35, 0.30, 0.22, 0.16, 0.10) if long_entry else (0.30, 0.22, 0.16, 0.10, 0.40, 0.06, 0.04)
            for approach in approaches:
                before = crossing - approach * normal
                if not self.clear_segments(before, crossing):
                    continue
                route = self.plan_route(self.pos, before)
                if not route:
                    continue
                points = np.vstack((self.pos, route, crossing, after))
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
                points = points[np.r_[True, lengths > 1e-6]]
                self.path = points
                self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                self.entry_arc = float(self.arc[-3])
                self.cross_arc = float(self.arc[-2])
                if self.passed + 1 < len(self.sequence):
                    next_idx = self.sequence[self.passed + 1]
                    next_normal = self.rot[next_idx, :, 0] * self.directions[self.passed + 1]
                    self.future_gate = idx
                    approach_alignment = float((self.gates[next_idx] - after) @ next_normal) / max(float(np.linalg.norm(self.gates[next_idx] - after)), 1e-6)
                    distances = (0.55, 0.45, 0.35, 0.30, 0.22, 0.16) if approach_alignment < 0.65 or self.passed + 1 == len(self.sequence) - 1 else (0.30, 0.22, 0.16, 0.10)
                    for distance in distances:
                        next_before = self.gates[next_idx] - distance * next_normal
                        extension = self.plan_route(after, next_before)
                        if extension:
                            self.path = np.vstack((points, extension))
                            lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
                            self.path = self.path[np.r_[True, lengths > 1e-6]]
                            self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
                            break
                    self.future_gate = None
                self.curve = None
                self.progress = 0.0
                self.ref = self.pos.copy()
                self.exit_centre = centre.copy()
                self.exit_normal = normal.copy()
                self.exit_point = after.copy()
                self.phase = 'route'
                self.crossing_started = False
                if not joint:
                    return
                delta = np.diff(self.path, axis=0)
                sizes = np.linalg.norm(delta, axis=1)
                headings = delta / np.maximum(sizes[:, None], 1e-8)
                turns = np.arccos(np.clip(np.sum(headings[:-1] * headings[1:], axis=1), -1., 1.))
                score = float(np.sum(sizes) + .50 * np.sum(turns))
                names = ('path', 'arc', 'entry_arc', 'cross_arc', 'exit_centre', 'exit_normal', 'exit_point')
                choices.append((score, {name: getattr(self, name).copy() if isinstance(getattr(self, name), np.ndarray) else getattr(self, name) for name in names}))
        if choices:
            _, selected = min(choices, key=lambda item: item[0])
            for name, value in selected.items():
                setattr(self, name, value)
            self.curve, self.progress = None, 0.
            self.ref = self.pos.copy()
            self.phase, self.crossing_started = 'route', False
            return
        if not self.tight and self.gate_known[idx]:
            self.tight = True
            self._plan_gate_base()
            return
        self.phase = 'blocked'
        if not self.gate_known[idx] or np.any(~self.obstacle_known):
            direction = self.pos - centre
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            for angle in (0.0, -0.5, 0.5):
                distance = 0.30 if self.gate_known[idx] else 0.55
                point = centre + distance * R.from_euler('z', angle).apply(direction)
                route = self.plan_route(self.pos, point)
                if route:
                    self.path = np.vstack((self.pos, route))
                    self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
                    self.progress = 0.0
                    self.ref = self.pos.copy()
                    self.curve = None
                    self.exit_centre = centre.copy()
                    self.exit_normal = normal.copy()
                    self.exit_point = centre + 0.24 * normal
                    self.entry_arc = float(self.arc[-1])
                    self.cross_arc = float(self.arc[-1])
                    self.phase = 'survey'
                    break

    def plan_gate(self):
        self._plan_gate_base()
        if self.phase != 'route' or self.planned_gate != len(self.sequence) - 1:
            return
        gate = self.sequence[self.planned_gate]
        if not self.gate_known[gate]:
            return
        centre = self.gates[gate]
        normal = self.rot[gate, :, 0] * self.directions[self.planned_gate]
        lateral = centre - self.pos
        lateral -= float(lateral @ normal) * normal
        length = float(np.linalg.norm(lateral))
        if length < .05:
            return
        lateral /= length
        def shape(points):
            delta = np.diff(points, axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            directions = delta / np.maximum(lengths[:, None], 1e-6)
            angles = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1., 1.))
            return float(np.sum(lengths)), float(np.sum(angles))
        best_length, best_turn = shape(self.path)
        selected = None
        for y, height in ((0., 0.), (-.03, 0.), (.03, 0.), (0., -.03), (0., .03), (-.03, -.03), (-.03, .03), (.03, -.03), (.03, .03)):
            crossing = centre + self.rot[gate] @ [0., y, height]
            for slope in (.20, .35, .45):
                direction = normal + slope * lateral
                before = crossing - .45 * direction
                after = crossing + .24 * direction
                if not np.all(self.clear_segments(np.array([before, crossing]), np.array([crossing, after]), margin=.02)):
                    continue
                route = self.plan_route(self.pos, before)
                if not route:
                    continue
                points = np.vstack((self.pos, route, crossing, after))
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
                points = points[np.r_[True, lengths > 1e-6]]
                if not np.all(self.clear_segments(points[:-1], points[1:])):
                    continue
                distance, turning = shape(points)
                if distance < best_length - .02 and turning <= best_turn + .001:
                    best_length, best_turn = distance, turning
                    selected = points, before, after, direction, crossing
        if selected is None:
            return
        self.path, before, self.exit_point, direction, crossing = selected
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.entry_arc = float(self.arc[np.argmin(np.linalg.norm(self.path-before, axis=1))])
        self.cross_arc = float(self.arc[np.argmin(np.linalg.norm(self.path-crossing, axis=1))])
        self.exit_normal = direction / np.linalg.norm(direction)
        self.curve, self.progress = None, 0.
        self.ref = self.pos.copy()

    def point_at(self, progress):
        if self.curve is not None:
            return self.curve(progress)
        return np.array([np.interp(progress, self.arc, self.path[:, axis]) for axis in range(3)])

    def _path_control(self, obs, info=None):
        self.update_environment(obs)
        path_velocity = np.zeros(3)
        if self.phase == 'takeoff':
            delta = self.takeoff - self.ref
            self.ref += delta * min(1.0, 1.2 * self.dt / max(float(np.linalg.norm(delta)), 1e-8))
            path_velocity = 0.8 * self.takeoff_direction * float(np.clip((self.pos[2] - 0.05) / 0.10, 0.0, 1.0))
            path_velocity[2] = 1.0
            if self.pos[2] > self.takeoff[2] - 0.10:
                self.plan_gate()
        else:
            state = np.concatenate((self.gates.ravel(), self.quat.ravel(), self.obstacles.ravel(), self.gate_known, self.obstacle_known))
            changed = self.snapshot is None or np.max(np.abs(state - self.snapshot), initial=0.0) > 0.025
            passed = self.passed > self.planned_gate
            stalled = False
            if self.tick - self.replan_check >= 50:
                deviation = 0.0
                if self.path is not None:
                    segments = np.diff(self.path, axis=0)
                    offset = self.pos - self.path[:-1]
                    fraction = np.clip(np.sum(offset * segments, axis=1) /
                                       np.maximum(np.sum(segments ** 2, axis=1), 1e-8), 0.0, 1.0)
                    deviation = float(np.min(np.linalg.norm(offset - fraction[:, None] * segments, axis=1)))
                stalled = self.phase == 'route' and (np.linalg.norm(self.pos - self.replan_position) < 0.06
                           or deviation > 0.15)
                self.replan_check = self.tick
                self.replan_position = self.pos.copy()
            exiting = passed
            exit_distance = float((self.exit_point - self.exit_centre) @ self.exit_normal) if exiting else 0.24
            clear_exit = exiting and float((self.pos - self.exit_centre) @ self.exit_normal) > max(0.08, min(0.20, exit_distance - 0.02))
            if exiting and not clear_exit:
                if self.path is None:
                    self.path = np.vstack((self.pos, 0.5 * (self.pos + self.exit_point), self.exit_point))
                    self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
                    self.phase = 'route'
                self.progress = max(self.progress, float(self.cross_arc))
                self.ref = self.point_at(self.progress)
            if clear_exit or (not exiting and (changed or stalled or self.passed != self.planned_gate
                               or (self.phase == 'blocked' and self.tick - self.last_plan > 10))):
                self.plan_gate()
            if self.path is not None:
                error_vector = self.pos - self.ref
                near_gate = self.phase == 'route' and len(self.arc) >= 3 and self.progress > self.entry_arc
                if near_gate:
                    along = float(error_vector @ self.exit_normal)
                    lateral = float(np.linalg.norm(error_vector - along * self.exit_normal))
                    lateral_limit = 0.10 if self.planned_gate <= 0 else 0.191
                    along_limit = 0.70 if self.planned_gate <= 0 else 1.0
                    scale = min(float(np.clip(1.0 - lateral / lateral_limit, 0.0, 1.0)),
                                float(np.clip(1.0 - abs(along) / along_limit, 0.0, 1.0)))
                else:
                    remaining = self.entry_arc - self.progress if len(self.arc) >= 3 else self.arc[-1] - self.progress
                    lead = 0.25 + 0.20 * float(np.clip((remaining - 0.35) / 0.65, 0.0, 1.0))
                    gate_distance = float(np.min(np.linalg.norm(self.gates - self.pos, axis=1)))
                    obstacle_distance = float(np.min(np.linalg.norm(self.obstacles[:, :2] - self.pos[:2], axis=1), initial=np.inf))
                    room = min(gate_distance - 0.45, obstacle_distance - 0.35)
                    lead = min(lead, 0.25 + 0.20 * float(np.clip(room / 0.35, 0.0, 1.0)))
                    factor = 1.20 if self.planned_gate <= 0 else 2.05
                    scale = float(np.clip(1.0 - np.linalg.norm(error_vector) / (factor * lead), 0.0, 1.0))
                speed = (2.4 if near_gate else 4.2) if self.planned_gate <= 0 else (3.946 if near_gate else self.speed)
                if not self.gate_known[self.sequence[self.planned_gate]]:
                    speed = min(speed, 2.8 if self.planned_gate <= 0 else 4.511)
                if near_gate and self.passed == len(self.sequence) - 1:
                    speed = 3.946
                progress = min(self.arc[-1], self.progress + speed * scale * self.dt)
                if self.phase == 'route' and not passed:
                    idx = self.sequence[self.planned_gate]
                    side = float((self.pos - self.gates[idx]) @ self.rot[idx, :, 0]) * self.directions[self.planned_gate]
                    if self.progress >= self.entry_arc - 0.02 and side < -0.025:
                        self.crossing_started = True
                    if not self.crossing_started:
                        progress = min(progress, self.entry_arc)
                candidate = self.point_at(progress)
                if self.clear_segments(self.ref, candidate):
                    path_velocity = (candidate - self.ref) / self.dt
                    self.progress = progress
                    self.ref = candidate
                else:
                    self.path = None
                    self.phase = 'blocked'
        error = self.ref - self.pos
        self.integral[:2] *= 0.99
        self.integral = np.clip(self.integral + [0.5, 0.5, 1.5] * error * self.dt, [-0.03, -0.03, -0.18], [0.03, 0.03, 0.12])
        command = self.ref + self.integral
        if self.phase != 'takeoff':
            for scale in (1.0, 0.5, 0.0):
                command = self.ref + scale * self.integral
                if self.clear_segments(self.pos, command):
                    break
            else:
                brake = self.pos - 0.20 * self.vel
                hold = brake if self.clear_segments(self.pos, brake) else self.pos.copy()
                self.ref = hold.copy()
                command = hold + self.integral
                self.path = None
                self.phase = 'blocked'
        desired_velocity = path_velocity
        cap = 1.8 if self.planned_gate <= 0 else 3.507
        slew = 3.0 if self.planned_gate <= 0 else 8.847
        desired_velocity *= min(1.0, cap / max(float(np.linalg.norm(desired_velocity)), 1e-8))
        self.ref_vel += np.clip(desired_velocity - self.ref_vel, -slew * self.dt, slew * self.dt)
        if self.phase in ('blocked', 'finished'):
            self.ref_vel[:] = 0.0
        gain = 0.55
        if self.path is not None and self.phase == 'route':
            segment = min(int(np.searchsorted(self.arc, self.progress, side='right')) - 1, len(self.path) - 2)
            remaining = float(self.arc[segment + 1] - self.progress)
            tangent = self.path[segment + 1] - self.path[segment]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
            speed = float(np.linalg.norm(self.vel))
            alignment = 1.0 if speed < 0.15 else float(np.clip((self.vel @ tangent / speed - 0.50) / 0.40, 0.0, 1.0))
            extra = 0.60 if self.planned_gate <= 0 else 1.284
            gain += extra * alignment * float(np.clip((remaining - 0.25) / 0.50, 0.0, 1.0))
        velocity = gain * self.ref_vel
        if self.phase == 'route' and self.path is not None and self.progress > self.entry_arc and self.passed == self.planned_gate:
            offset = self.pos - self.exit_centre
            idx = self.sequence[self.planned_gate]
            normal = self.rot[idx, :, 0] * self.directions[self.planned_gate]
            local = (self.pos - self.gates[idx]) @ self.rot[idx]
            lateral = np.linalg.norm(local[1:] + 0.12 * (self.vel @ self.rot[idx])[1:])
            limit = 0.85 if self.planned_gate <= 0 else 1.225
            tolerance = 0.08 if self.planned_gate <= 0 else 0.12
            gain = limit * float(np.clip(1.0 - lateral / tolerance, 0.0, 1.0))
            velocity += gain * (self.ref_vel @ normal) * normal
        command = self.filter_command(command, velocity)
        if self.phase != 'takeoff':
            command, velocity = self.predict_command(self.free_command, velocity, command)
        vertical = 35.7 * (command[2] - self.pos[2]) + 14.3 * (velocity[2] - self.vel[2]) + self.force_bias[2]
        horizontal = 11.4 * (command[:2] - self.pos[:2]) + 5.7 * (velocity[:2] - self.vel[:2]) + self.force_bias[:2]
        limit = 7.0 if self.planned_gate <= 0 else 10.0
        command[:2] += (np.clip(horizontal, -limit, limit) - horizontal) / 11.4
        command[2] += (np.clip(vertical, -3.0, 6.0) - vertical) / 35.7
        self.last_state = (self.pos.copy(), self.vel.copy(), command.copy(), velocity.copy())
        return np.concatenate((command, velocity, np.zeros(3), [0.0, 0.0, 0.0, 1.0], np.zeros(3))).astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self.tick += 1
        return False

    def episode_callback(self):
        pass

    def _takeoff_control(self, obs, info):
        action = self._path_control(obs, info)
        nominal = np.array([11.4, 11.4, 35.7]) * (action[:3] - self.pos)
        nominal += np.array([5.7, 5.7, 14.3]) * (action[3:6] - self.vel) + self.force_bias
        self.solution = np.tile(np.clip(nominal, [-7.0, -7.0, -3.0], [7.0, 7.0, 6.0]), self.horizon)
        return action

    def passage_speed(self):
        gate = self.sequence[self.planned_gate]
        normal = self.rot[gate, :, 0] * self.directions[self.planned_gate]
        before = self.point_at(max(0., self.cross_arc - .45))
        after = self.point_at(min(self.arc[-1], self.cross_arc + .55))
        incoming = self.exit_centre - before
        outgoing = after - self.exit_centre
        if np.linalg.norm(outgoing) < .12 and self.planned_gate + 1 < len(self.sequence):
            outgoing = self.gates[self.sequence[self.planned_gate + 1]] - self.exit_centre
        incoming /= max(float(np.linalg.norm(incoming)), 1e-6)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-6)
        alignment = min(float(incoming @ normal), float(outgoing @ normal))
        smooth = float(np.clip((alignment - .15) / .75, 0., 1.))
        speed = 2.0 + (2.9 - 2.0) * smooth
        if self.planned_gate == len(self.sequence) - 1:
            speed = 2.9
        if self.planned_gate + 1 < len(self.sequence):
            following = self.gates[self.sequence[self.planned_gate + 1]] - self.exit_centre
            if float(normal @ following) < -.1:
                speed = 1.45
        if not self.gate_known[gate]:
            speed = min(speed, 1.8)
        return speed

    def prepare_speed_path(self):
        original = self.path.copy()
        entry = self.point_at(self.entry_arc)
        crossing = self.point_at(self.cross_arc)
        protected = np.array([crossing, self.exit_point])
        points = [original[0]]
        for j in range(1, len(original) - 1):
            centre = original[j]
            left, right = centre - original[j - 1], original[j + 1] - centre
            a, b = np.linalg.norm(left), np.linalg.norm(right)
            cut = min(0.45, .40 * a, .40 * b)
            if np.min(np.linalg.norm(protected - centre, axis=1)) < 1e-5 or self.planned_gate == 0 or cut < .025:
                points.append(centre)
                continue
            before, after = centre - cut * left / a, centre + cut * right / b
            t = np.linspace(0., 1., 9)[:, None]
            curve = (1. - t) ** 2 * before + 2. * t * (1. - t) * centre + t ** 2 * after
            if np.all(self.clear_segments(curve[:-1], curve[1:], margin=.04)):
                points.extend(curve)
            else:
                points.append(centre)
        points.append(original[-1])
        path = np.asarray(points)
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.path = path[np.r_[True, lengths > 1e-7]]
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.entry_arc = float(self.arc[np.argmin(np.linalg.norm(self.path - entry, axis=1))])
        self.cross_arc = float(self.arc[np.argmin(np.linalg.norm(self.path - crossing, axis=1))])
        self.curve = None
        sample = np.unique(np.r_[np.arange(0., self.arc[-1], .04), self.arc])
        xyz = np.column_stack([np.interp(sample, self.arc, self.path[:, axis]) for axis in range(3)])
        u, v = np.diff(xyz, axis=0)[:-1], np.diff(xyz, axis=0)[1:]
        denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * np.linalg.norm(u + v, axis=1)
        curvature = np.r_[0., 2. * np.linalg.norm(np.cross(u, v), axis=1) / np.maximum(denominator, 1e-8), 0.]
        limit = np.minimum(4.3, np.sqrt(24.0 / np.maximum(curvature, .001)))
        limit = np.maximum(limit, 1.3)
        near = np.abs(sample - self.cross_arc) < .12
        limit[near] = np.minimum(limit[near], self.passage_speed())
        if self.planned_gate + 1 < len(self.sequence):
            normal = self.rot[self.sequence[self.planned_gate], :, 0] * self.directions[self.planned_gate]
            outgoing = self.gates[self.sequence[self.planned_gate + 1]] - self.exit_centre
            if float(normal @ outgoing) < -.10:
                limit[near] = np.minimum(limit[near], 1.45)
        for j in range(len(sample) - 2, -1, -1):
            limit[j] = min(limit[j], np.sqrt(limit[j + 1] ** 2 + 2. * 5.0 * (sample[j + 1] - sample[j])))
        for j in range(1, len(sample)):
            limit[j] = min(limit[j], np.sqrt(limit[j - 1] ** 2 + 2. * 5.0 * (sample[j] - sample[j - 1])))
        self.speed_arc, self.speed_limit = sample, limit
        self.speed_path = self.path

    def start_recovery(self, gate, normal):
        goal = self.gates[gate] - .45 * normal
        route = self.plan_route(self.pos, goal)
        if not route:
            return False
        points = np.vstack((self.pos, route))
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.path = points[np.r_[True, lengths > 1e-6]]
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.entry_arc = self.cross_arc = float(self.arc[-1])
        self.exit_point = goal.copy()
        self.recovery_goal = goal.copy()
        self.curve, self.progress = None, 0.
        self.ref = self.pos.copy()
        self.phase = 'route'
        return True

    def compute_control(self, obs, info=None):
        self.update_environment(obs)
        if self.phase == 'takeoff':
            return self._takeoff_control(obs, info)
        if self.recovery_goal is not None:
            arrived = np.linalg.norm(self.pos - self.recovery_goal) < .10 and np.linalg.norm(self.vel) < .6
            if arrived or self.passed != self.planned_gate:
                self.recovery_goal = None
                self.plan_gate()
        waiting_gate = self.passed == self.planned_gate and self.passed < len(self.sequence)
        slow = float(np.linalg.norm(self.vel)) < .25
        self.stall_ticks = self.stall_ticks + 1 if waiting_gate and slow else 0
        if self.recovery_goal is None and self.stall_ticks >= 40 and self.tick - self.last_recovery >= 100:
            gate = self.sequence[self.passed]
            normal = self.rot[gate, :, 0] * self.directions[self.passed]
            side = float((self.pos - self.gates[gate]) @ normal)
            if self.gate_known[gate] and side > .06 and self.start_recovery(gate, normal):
                self.last_recovery, self.stall_ticks = self.tick, 0
                self.recovery_count += 1
        state = np.concatenate((self.gates.ravel(), self.quat.ravel(), self.obstacles.ravel(), self.gate_known, self.obstacle_known))
        changed = self.snapshot is None or np.max(np.abs(state - self.snapshot), initial=0.0) > 0.025
        passed = self.passed > self.planned_gate
        plane_normal = self.exit_normal
        if 0 <= self.planned_gate < len(self.sequence):
            plane_normal = self.rot[self.sequence[self.planned_gate], :, 0] * self.directions[self.planned_gate]
        exit_distance = float((self.exit_point - self.exit_centre) @ plane_normal) if passed else 0.24
        clearance = max(0.08, min(0.14, exit_distance - 0.02))
        departed = passed and (float((self.pos - self.exit_centre) @ plane_normal) > clearance or float(self.vel @ plane_normal) < -0.05)
        if self.recovery_goal is None and (departed or (not passed and (changed or self.planned_gate != self.passed
                                        or (self.path is None and self.tick - self.last_plan >= 10)))):
            self.plan_gate()
        if self.path is None or self.phase != 'route':
            return self._path_control(obs, info)
        if getattr(self, 'speed_path', None) is not self.path:
            self.prepare_speed_path()
        passed = self.passed > self.planned_gate
        kp = np.array([11.4, 11.4, 35.7])
        kd = np.array([5.7, 5.7, 14.3])
        if self.last_state is not None:
            old_pos, old_vel, old_command, old_velocity = self.last_state
            measured = (self.vel - old_vel) / self.dt
            self.measured_acc = 0.5 * self.measured_acc + 0.5 * np.clip(measured, -12.0, 12.0)
            nominal = kp * (old_command - old_pos) + kd * (old_velocity - old_vel)
            residual = np.clip(measured - nominal, [-1.0, -1.0, -4.0], [1.0, 1.0, 4.0])
            rate = np.array([0.005, 0.005, 0.01])
            self.force_bias = (1.0 - rate) * self.force_bias + rate * residual
        segments = np.diff(self.path, axis=0)
        lengths = np.diff(self.arc)
        fraction = np.clip(np.sum((self.pos - self.path[:-1]) * segments, axis=1) /
                           np.maximum(lengths ** 2, 1e-8), 0.0, 1.0)
        distance = np.linalg.norm(self.pos - self.path[:-1] - fraction[:, None] * segments, axis=1)
        projected = self.arc[:-1] + fraction * lengths
        if not passed:
            distance = np.where(projected <= self.cross_arc + 0.01, distance, np.inf)
        nearest = int(np.argmin(distance))
        self.progress = float(projected[nearest])
        self.ref = self.point_at(self.progress)
        if self.speed_gate != self.planned_gate:
            self.speed_gate = self.planned_gate
            self.speed_since = self.tick
        known = self.gate_known[self.sequence[self.planned_gate]]
        recovering = self.recovery_goal is not None
        cruise = 1.0 if recovering else 4.8
        if not known and np.linalg.norm(self.pos - self.gates[self.sequence[self.planned_gate]]) < 1.1:
            cruise = min(cruise, 2.3)
        if self.planned_gate == 0:
            cruise = min(cruise, 2.1)
        starting_speed = max(1.1, float(np.linalg.norm(self.vel)))
        gate = self.sequence[self.planned_gate]
        normal = self.rot[gate, :, 0] * self.directions[self.planned_gate]
        gate_speed = self.passage_speed()
        if self.planned_gate + 1 < len(self.sequence):
            outgoing = self.gates[self.sequence[self.planned_gate + 1]] - self.gates[gate]
            if float(normal @ outgoing) < -0.10:
                gate_speed = 1.45
        else:
            gate_speed = self.passage_speed()
        samples = []
        progress = self.progress
        for forecast_step in range(self.horizon):
            distance = abs(self.cross_arc - progress)
            available = max(0.0, distance - 0.20 * float(np.linalg.norm(self.vel)))
            if passed:
                available = max(available, .5)
            forecast_cap = starting_speed + .5 + (1.5 if self.planned_gate == len(self.sequence) - 1 else 0.) * forecast_step * self.h
            speed = min(cruise, forecast_cap, float(np.interp(progress + 0.05 * np.linalg.norm(self.vel), self.speed_arc, self.speed_limit)), np.sqrt(gate_speed ** 2 + 8.0 * available))
            if recovering:
                speed = min(speed, np.sqrt(6. * max(0., self.arc[-1] - progress)))
            progress += self.h * speed
            if recovering:
                progress = min(progress, self.arc[-1])
            samples.append(progress)
        samples = np.asarray(samples)
        points = np.column_stack([np.interp(samples, self.arc, self.path[:, j]) for j in range(3)])
        if self.planned_gate == len(self.sequence) - 1 and not recovering:
            points += np.maximum(samples - self.arc[-1], 0.0)[:, None] * self.exit_normal
        target_velocity = np.diff(np.vstack((self.ref, points)), axis=0) / self.h
        if self.tick - self.model_tick != 1:
            self.model_old_acc = self.measured_acc.copy()
            self.model_old_u = self.solution[:3].copy()
        free = self.state_map @ np.r_[self.pos, self.vel, self.measured_acc, self.model_old_acc, self.model_old_u, 1.]
        self.model_old_acc = self.measured_acc.copy()
        self.model_tick = self.tick
        warm_start = (1. - self.dt / self.h) * self.solution + self.dt / self.h * np.r_[self.solution[3:], self.solution[-3:]]
        motion_points = free[:, :3] + np.einsum('nij,j->ni', self.G[:, :3], warm_start)
        distance = np.linalg.norm(points - self.gates[self.sequence[self.planned_gate]], axis=1)
        altitude_weight = np.full(self.horizon, 400.0) if self.planned_gate == len(self.sequence) - 1 else 60.0 + 340.0 * np.exp(-(distance / 0.5) ** 2)
        self.Q = np.column_stack((np.full(self.horizon, 140.0), np.full(self.horizon, 140.0), altitude_weight)).ravel()
        difference = np.eye(3 * self.horizon) - np.eye(3 * self.horizon, k=-3)
        self.H = self.P.T @ (self.Q[:, None] * self.P) + 4.0 * self.V.T @ self.V
        self.H += 0.03 * np.eye(3 * self.horizon) + 0.03 * difference.T @ difference
        self.cost_scale = float(np.mean(np.diag(self.H)))
        gradient = self.P.T @ (self.Q * (free[:, :3] - points).ravel())
        gradient += 4.0 * self.V.T @ (free[:, 3:6] - target_velocity).ravel()
        gradient[:3] -= 0.03 * self.solution[:3]
        rows, bounds = [], []
        for centre, known in zip(self.obstacles, self.obstacle_known):
            offset = points[:, :2] - centre[:2]
            length = np.linalg.norm(offset, axis=1)
            selected = np.flatnonzero(length < 1.0)
            normal = np.column_stack((offset[selected] / np.maximum(length[selected, None], 1e-6), np.zeros(len(selected))))
            rows.extend(np.einsum('ni,nij->nj', normal, self.G[selected, :3]))
            bounds.extend(self.obstacle_radius(centre, known) + np.sum(normal * (centre - free[selected, :3]), axis=1))
        boxes = np.array([[[0.0, 0.0, 0.28], [0.01, 0.36, 0.08]],
                          [[0.0, 0.0, -0.28], [0.01, 0.36, 0.08]],
                          [[0.0, -0.28, 0.0], [0.01, 0.08, 0.36]],
                          [[0.0, 0.28, 0.0], [0.01, 0.08, 0.36]],
                          [[0.0, 0.0, -0.86], [0.05, 0.05, 0.50]]])
        for gate, (centre, rotation) in enumerate(zip(self.gates, self.rot)):
            gate_boxes = np.concatenate((boxes, [[[0.0, 0.0, 0.0], [0.01, 0.36, 0.36]]])) if self.closed_gate(gate) else boxes
            origins, sizes = gate_boxes.transpose(1, 0, 2)
            padding = np.abs(rotation.T @ self.body_rotation) @ np.array([0.07, 0.07, 0.02]) + 0.025
            padding = np.maximum(padding, [0.11, 0.11, 0.08])
            half = sizes + padding
            local = ((points - centre) @ rotation)[:, None, :] - origins
            nearest_point = np.clip(local, -half, half)
            selected, box = np.nonzero(np.linalg.norm(local - nearest_point, axis=-1) < 0.65)
            if not len(selected):
                continue
            axis = np.argmax(np.abs(local[selected, box]) - half[box], axis=1)
            sign = np.where(local[selected, box, axis] >= 0.0, 1.0, -1.0)
            face = np.eye(3)[axis] * sign[:, None]
            surface = nearest_point[selected, box].copy()
            surface[np.arange(len(selected)), axis] = half[box, axis] * sign
            surface = centre + (surface + origins[box]) @ rotation.T
            normal = face @ rotation.T
            rows.extend(np.einsum('ni,nij->nj', normal, self.G[selected, :3]))
            bounds.extend(np.sum(normal * (surface - free[selected, :3]), axis=1))
        matrix = np.asarray(rows).reshape(-1, 3 * self.horizon)
        bound = np.asarray(bounds)
        norm = np.maximum(np.linalg.norm(matrix, axis=1), 1e-8)
        matrix = matrix / norm[:, None]
        bound = bound / norm
        weight = 50000.0 * norm ** 2 / self.cost_scale
        obstacle_count = len(bound)
        matrix = np.vstack((matrix, np.eye(3 * self.horizon)))
        low = np.r_[bound, np.tile([-7.0, -7.0, -3.0], self.horizon)]
        high = np.r_[np.full(obstacle_count, np.inf), np.tile([7.0, 7.0, 6.0], self.horizon)]
        rho = 1.0
        hessian = 2.0 * self.H / self.cost_scale
        factor = cho_factor(hessian + rho * np.einsum('ij,ik->jk', matrix, matrix) + 1e-6 * np.eye(len(self.solution)))
        linear = 2.0 * gradient / self.cost_scale
        solution = (1. - self.dt / self.h) * self.solution + self.dt / self.h * np.r_[self.solution[3:], self.solution[-3:]]
        projected = np.einsum('ij,j->i', matrix, solution)
        z = np.clip(projected, low, high)
        dual = np.zeros(len(low))
        identity = np.eye(len(self.solution))
        inverse = np.column_stack([cho_solve(factor, identity[:, j], check_finite=False) for j in range(len(self.solution))])
        offset = -np.einsum('ij,j->i', inverse, linear)
        response = np.einsum('ij,jk->ik', inverse, matrix.T, order='C')
        for _ in range(320):
            solution = offset + np.einsum('ij,j->i', response, rho * z - dual)
            projected = np.einsum('ij,j->i', matrix, solution)
            value = projected + dual / rho
            z = np.clip(value, low, high)
            z[:obstacle_count] = np.where(value[:obstacle_count] >= bound, value[:obstacle_count],
                                         (rho * value[:obstacle_count] + 2.0 * weight * bound) / (rho + 2.0 * weight))
            dual += rho * (projected - z)
        self.solution = np.clip(solution, np.tile([-7.0, -7.0, -3.0], self.horizon), np.tile([7.0, 7.0, 6.0], self.horizon))
        control = self.solution[:3].copy()
        gate = self.sequence[self.planned_gate]
        rotation = self.rot[gate]
        local = (self.pos - self.gates[gate]) @ rotation
        direction = self.directions[self.planned_gate]
        side = float(local[0]) * direction
        use_servo = self.planned_gate == 0
        if use_servo and self.commit_gate != self.planned_gate and self.gate_known[gate] and -0.70 < side < 0.10:
            entry = self.point_at(self.entry_arc)
            if abs(float(local[1])) < 0.45 and self.clear_segments(self.pos, entry):
                self.commit_gate = self.planned_gate
        if not recovering and self.commit_gate == self.planned_gate and side < 0.22:
            crossing_local = (self.point_at(self.cross_arc) - self.gates[gate]) @ rotation
            tracking = local - crossing_local
            velocity_local = self.vel @ rotation
            acceleration_local = self.measured_acc @ rotation
            y_error = float(local[1] + 0.15 * velocity_local[1])
            z_error = float(local[2] + 0.10 * velocity_local[2])
            room = max(0.0, -side - 0.18)
            y_window = 0.10 + 0.25 * room
            z_window = 0.13 + 0.25 * room
            centred = min(float(np.clip((y_window - abs(y_error)) / 0.06, 0.0, 1.0)),
                          float(np.clip((z_window - abs(z_error)) / 0.07, 0.0, 1.0)))
            crossing_speed = 2.3
            reverse = False
            if self.planned_gate + 1 < len(self.sequence):
                next_gate = self.sequence[self.planned_gate + 1]
                next_normal = self.rot[next_gate, :, 0] * self.directions[self.planned_gate + 1]
                outgoing = self.gates[next_gate] - 0.45 * next_normal - self.gates[gate]
                reverse = float(outgoing @ rotation[:, 0]) * direction < -0.10
            if reverse:
                crossing_speed = 0.70
                if side > 0.0:
                    exit_distance = float((self.exit_point - self.exit_centre) @ rotation[:, 0]) * direction
                    crossing_speed = min(crossing_speed, max(0.25, 4.0 * (exit_distance - side)))
            target_velocity = crossing_speed * (max(0.8, centred) if passed else centred)
            normal_velocity = float(velocity_local[0]) * direction
            normal_acceleration = float(acceleration_local[0]) * direction
            local_control = np.array([direction * (9.375 * (target_velocity - normal_velocity) - 1.667 * normal_acceleration),
                                      -85.333 * tracking[1] - 32.0 * velocity_local[1] - 3.167 * acceleration_local[1],
                                      -28.0 * tracking[2] - 10.3 * velocity_local[2] - 0.52 * acceleration_local[2]])
            control = np.clip(rotation @ local_control, [-7.0, -7.0, -3.0], [7.0, 7.0, 6.0])
        self.solution[:3] = control
        self.model_old_u = control.copy()
        command = self.pos + (self.solution[:3] - self.force_bias) / kp
        self.last_state = (self.pos.copy(), self.vel.copy(), command.copy(), self.vel.copy())
        return np.r_[command, self.vel, np.zeros(3), [0.0, 0.0, 0.0, 1.0], np.zeros(3)].astype(np.float32)

