"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and a minimum-snap reference trajectory through hand-picked waypoints.

Improvements over the original version:
    * Segment times are allocated by distance (constant average speed), not uniformly.
    * Minimum-snap with polynomial degree 7 (well conditioned) instead of degree len(waypoints).
    * Attitude / thrust references come from the planned acceleration (differential flatness)
      instead of "upright + hover".
    * Replanning starts from the current state and only contains the not-yet-passed waypoints.
    * One single planning function (no duplicated waypoint code), gate approach/exit points for
      every gate, obstacle changes also trigger a replan.
    * Coarser MPC step with longer horizon, higher xy weights, solver status check with fallback.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import minsnap_trajectories as ms
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------------------------
V_AVG = float(
    os.environ.get("V_AVG", 1.4)
)  # [m/s] average speed used to allocate time to the trajectory segments
V_GATE = float(
    os.environ.get("V_GATE", 1.0)
)  # [m/s] speed used on the segments around gates (poses are revealed late)
MIN_SEG_TIME = 0.25  # [s] lower bound for a segment duration (avoids huge accelerations)
MIN_WP_DIST = 0.10  # [m] waypoints closer than this to the previous one are dropped
PASS_RADIUS = 0.20  # [m] a waypoint counts as "passed" once the drone is this close
GATE_APPROACH = 0.35  # [m] distance of the pre-gate point along the gate normal
GATE_EXIT = 0.25  # [m] distance of the post-gate point (longer: drone must clear the frame)
OBS_OFFSET = 0.20  # [m] lateral offset used to pass obstacles
MPC_DT = 0.04  # [s] MPC discretisation step (coarser than the control period)
MPC_N = 30  # MPC horizon steps -> horizon = MPC_N * MPC_DT = 1.2 s
MAX_TILT = float(
    os.environ.get("MAX_TILT", 0.45)
)  # [rad] clip for reference roll/pitch (state/input limit is 0.5)
# Speeds on segments whose objects are already seen. The track is either exactly known up front
# (no randomization) or only becomes known while flying (randomized) -> two speed sets.
V_AVG_EXACT = float(os.environ.get("V_AVG_EXACT", 3.5))
V_GATE_EXACT = float(os.environ.get("V_GATE_EXACT", 2.6))
V_AVG_SEEN = float(os.environ.get("V_AVG_SEEN", 2.0))
V_GATE_SEEN = float(os.environ.get("V_GATE_SEEN", 1.5))
KNOWN_DIST = 1.0  # [m] a segment counts as "known" if no unseen object is this close to its end
LAG_TOL = float(
    os.environ.get("LAG_TOL", 0.15)
)  # [m] reference time is frozen while the drone lags further than this behind it
ENV_CHANGE_TOL = 1e-3  # [m] / quat tolerance to detect updated gate/obstacle poses


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic drone dynamics."""
    # For more info on the models, check out https://github.com/learnsyslab/crazyflow
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "mpcplus_v2"
    model.f_expl_expr = X_dot  # Can be changed even after compilation
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            400.0,  # pos x
            400.0,  # pos y
            400.0,  # pos z
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights. The reference is now the attitude/thrust required by the planned
    # acceleration, so the rpy weights are small: the attitude may move freely.
    R_w = np.diag(
        [
            0.5,  # rpy
            0.5,  # rpy
            0.5,  # rpy
            50.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R_w)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    lim = MAX_TILT + 0.05
    ocp.constraints.lbx = np.array([-lim, -lim, -lim])
    ocp.constraints.ubx = np.array([lim, lim, lim])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    total_thrust_min = parameters["thrust_min"] * 4
    total_thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbu = np.array([-lim, -lim, -lim, total_thrust_min])
    ocp.constraints.ubu = np.array([lim, lim, lim, total_thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI (faster, see notes)
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    cmake_builder = None
    # On Windows, use CMake with MinGW to build the generated MPC solver DLL.
    if os.name == "nt":
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpcplus_v2.json",
        verbose=verbose,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )

    return acados_ocp_solver, ocp


def rpy2NormVec(rpy: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the gate normal (horizontal) from the gate's yaw angle."""
    return np.array([np.cos(rpy[2]), np.sin(rpy[2]), 0])


class AttitudeMPCPlusV2(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        freq = config.env.freq
        self._freq = freq
        # MPC runs on a coarser grid than the control loop: reference is sampled with a stride.
        self._stride = max(1, int(round(MPC_DT * freq)))
        self._N = MPC_N
        self._dt = self._stride / freq
        self._T_HORIZON = self._N * self._dt

        self.drone_params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._mass = self.drone_params["mass"]
        self._g = -self.drone_params["gravity_vec"][-1]

        # Planning
        self._start_pos = np.array(obs["pos"], dtype=float)
        self._wp_idx = 0  # index of the next not-yet-passed nominal waypoint
        self._seen_env = None
        rnd = config.env.get("randomizations") or {}
        # Without gate/obstacle randomization the nominal track is the true track
        self._track_known = not any(k in rnd for k in ("gate_pos", "gate_rpy", "obstacle_pos"))
        self._v_avg_known = V_AVG_EXACT if self._track_known else V_AVG_SEEN
        self._v_gate_known = V_GATE_EXACT if self._track_known else V_GATE_SEEN
        self._plan(obs, self._start_pos, np.zeros(3))
        self._initial_plan = self._snapshot()

        # MPC
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._last_u = np.array([0.0, 0.0, 0.0, self._mass * self._g])

        self._tick = 0
        self._config = config
        self._finished = False

    # ------------------------------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _gate_normal(gate_quat: NDArray[np.floating]) -> NDArray[np.floating]:
        """Horizontal normal of a gate from its quaternion."""
        rpy = R.from_quat(gate_quat).as_euler("xyz")
        return rpy2NormVec(rpy)

    def _gate_pass(
        self,
        obs: dict,
        gate: int,
        prev: NDArray[np.floating],
        approach: float = GATE_APPROACH,
        exit_: float = GATE_EXIT,
    ) -> list[NDArray[np.floating]]:
        """Pre-gate, gate and post-gate point. Direction follows the direction of travel."""
        pos = obs["gates_pos"][gate]
        n = self._gate_normal(obs["gates_quat"][gate])
        if np.dot(n, pos - prev) < 0:  # flip so we cross the gate in the direction of travel
            n = -n
        return [
            (pos - n * approach, "gate", None),
            (np.array(pos), "gate", n * V_GATE),  # cross the gate along its normal
            (pos + n * exit_, "gate", None),
        ]

    def _build_waypoints(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Nominal waypoints (excluding the start) as a function of the current gate/obstacle poses.

        The structure (number and order of waypoints) must not depend on the observation, since
        ``self._wp_idx`` indexes into this list across replans.
        """
        gp, op = obs["gates_pos"], obs["obstacles_pos"]
        wps: list[NDArray[np.floating]] = []

        meta: list[tuple[str, NDArray | None]] = []

        def add(p: ArrayLike, tag: str = "free", vel: NDArray | None = None):
            wps.append(np.asarray(p, dtype=float))
            meta.append((tag, vel))

        add([-1.0, 0.75, 0.4])
        add([op[0][0] - 0.1, op[0][1] - 0.4, gp[0][2]])
        for p, tag, vel in self._gate_pass(obs, 0, wps[-1]):  # gate 1
            add(p, tag, vel)
        add([op[1][0] + 0.05, op[1][1] - 0.35, (gp[0][2] + gp[1][2]) / 2])
        add([op[1][0] + 0.3, op[1][1] + 0.05, (gp[0][2] + gp[1][2]) / 2])  # wrap around, no cutting
        for p, tag, vel in self._gate_pass(obs, 1, wps[-1], approach=0.45):  # gate 2
            add(p, tag, vel)
        add([op[0][0] + 0.15, op[0][1] - 0.4, 0.95])  # pass obstacle 1 on its -y side
        add([-0.5, -0.05, 0.7])
        for p, tag, vel in self._gate_pass(obs, 2, wps[-1]):  # gate 3
            add(p, tag, vel)
        # U-turn after gate 3 away from its frame posts (obstacle 3 sits right behind it)
        add([gp[2][0] - 0.25, gp[2][1] - 0.55, gp[2][2] + 0.1])
        add([op[3][0] - 0.3, op[3][1] - 0.4, 0.85])
        add([op[3][0] + 0.1, op[3][1] - 0.4, 1.0])  # pass obstacle 4 on its -y side
        for p, tag, vel in self._gate_pass(obs, 3, wps[-1], approach=0.2):  # gate 4
            add(p, tag, vel)
        # Return leg (kept from the original waypoint list)
        add([op[1][0] + 0.05, op[1][1] - 0.35, gp[1][2]])
        add([op[1][0] + 0.3, op[1][1] + 0.05, gp[1][2]])
        for p, tag, vel in self._gate_pass(obs, 1, wps[-1], approach=0.45):  # gate 2 (second time)
            add(p, tag, vel)
        add([0.25, 1.25, 1.2])
        self._nominal_meta = meta
        return np.array(wps)

    def _plan(
        self,
        obs: dict[str, NDArray[np.floating]],
        start_pos: NDArray[np.floating],
        start_vel: NDArray[np.floating],
    ):
        """(Re)plan a minimum-snap trajectory from the given state through the remaining waypoints.

        Used for the initial plan and for every replan, so there is only one code path.
        """
        self._nominal_wps = self._build_waypoints(obs)
        remaining = self._nominal_wps[self._wp_idx :]

        # Drop waypoints that are (almost) on top of the previous one. The final one is kept.
        remaining_meta = self._nominal_meta[self._wp_idx :]
        pts = [np.asarray(start_pos, dtype=float)]
        meta = [("free", None)]
        for k, p in enumerate(remaining):
            is_last = k == len(remaining) - 1
            if np.linalg.norm(p - pts[-1]) < MIN_WP_DIST:
                if is_last and len(pts) > 1:
                    pts[-1] = p  # replace instead of creating a degenerate segment
                    meta[-1] = remaining_meta[k]
                continue
            pts.append(p)
            meta.append(remaining_meta[k])
        pts = np.array(pts)
        self._waypoints = pts

        self._seen_vis = len(self._unseen_objects(obs))
        self._seen_env = (
            np.array(obs["gates_pos"]),
            np.array(obs["gates_quat"]),
            np.array(obs["obstacles_pos"]),
        )

        pad = self._stride * self._N + 1  # so that the horizon never runs off the end

        if len(pts) < 2:  # nothing left to fly: hold position
            self._t_total = 0.0
            self._n_real = 1
            self._ref_pos = np.repeat(pts[:1], 1 + pad, axis=0)
            self._ref_vel = np.zeros_like(self._ref_pos)
            self._ref_acc = np.zeros_like(self._ref_pos)
            return

        # Time allocation proportional to the segment length
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        unseen = self._unseen_objects(obs)
        seg_v = []
        for p, m in zip(pts[1:], meta[1:]):
            known = len(unseen) == 0 or np.min(np.linalg.norm(unseen - p[:2], axis=1)) > KNOWN_DIST
            if m[0] == "gate":
                seg_v.append(self._v_gate_known if known else V_GATE)
            else:
                seg_v.append(self._v_avg_known if known else V_AVG)
        seg_v = np.array(seg_v)
        seg_time = np.maximum(seg_len / seg_v, MIN_SEG_TIME)
        t_wp = np.concatenate([[0.0], np.cumsum(seg_time)])
        self._t_total = float(t_wp[-1])

        ms_waypoints = []
        for i, pos in enumerate(pts):
            if i == 0:
                # Start from the current state (zero at the very first plan)
                w = ms.Waypoint(
                    time=t_wp[i],
                    position=pos,
                    velocity=np.asarray(start_vel, dtype=float),
                    acceleration=np.zeros(3),
                )
            elif i == len(pts) - 1:
                w = ms.Waypoint(
                    time=t_wp[i], position=pos, velocity=np.zeros(3), acceleration=np.zeros(3)
                )
            elif meta[i][1] is not None:  # gate centre: fix the velocity along the gate normal
                w = ms.Waypoint(time=t_wp[i], position=pos, velocity=meta[i][1])
            else:  # only the position is fixed, the solver chooses velocity/acceleration
                w = ms.Waypoint(time=t_wp[i], position=pos)
            ms_waypoints.append(w)

        # Degree 7 + minimising the 4th derivative (snap) is the classic well-conditioned choice.
        trajectory = ms.generate_trajectory(
            ms_waypoints, degree=7, idx_minimized_orders=4, num_continuous_orders=3
        )

        # Sample exactly at the control rate
        n_real = int(np.floor(self._t_total * self._freq)) + 1
        time_grid = np.minimum(np.arange(n_real) / self._freq, self._t_total)
        ref_pos = np.zeros((n_real, 3))
        ref_vel = np.zeros((n_real, 3))
        ref_acc = np.zeros((n_real, 3))
        for idx, t in enumerate(time_grid):
            d = ms.compute_trajectory_derivatives(trajectory, t, 3)  # pos, vel, acc
            ref_pos[idx], ref_vel[idx], ref_acc[idx] = d[0], d[1], d[2]

        # Hold the final point after the end of the trajectory
        self._n_real = n_real
        self._ref_pos = np.vstack([ref_pos, np.repeat(ref_pos[-1:], pad, axis=0)])
        self._ref_vel = np.vstack([ref_vel, np.zeros((pad, 3))])
        self._ref_acc = np.vstack([ref_acc, np.zeros((pad, 3))])

    def _unseen_objects(self, obs: dict) -> NDArray[np.floating]:
        """Xy positions of the gates/obstacles whose true pose has not been observed yet."""
        if self._track_known:
            return np.zeros((0, 2))
        g = ~np.asarray(obs["gates_visited"], dtype=bool).reshape(-1)
        o = ~np.asarray(obs["obstacles_visited"], dtype=bool).reshape(-1)
        return np.vstack(
            [np.asarray(obs["gates_pos"])[g][:, :2], np.asarray(obs["obstacles_pos"])[o][:, :2]]
        )

    def _snapshot(self) -> dict:
        return {
            k: np.copy(getattr(self, k))
            if isinstance(getattr(self, k), np.ndarray)
            else getattr(self, k)
            for k in (
                "_waypoints",
                "_ref_pos",
                "_ref_vel",
                "_ref_acc",
                "_n_real",
                "_t_total",
                "_nominal_wps",
                "_nominal_meta",
                "_seen_vis",
            )
        } | {"_seen_env": tuple(np.copy(a) for a in self._seen_env)}

    def _env_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """True if any gate or obstacle pose differs from the one the current plan is based on."""
        g_pos, g_quat, o_pos = self._seen_env
        if len(self._unseen_objects(obs)) != self._seen_vis:
            return True
        return not (
            np.allclose(obs["gates_pos"], g_pos, atol=ENV_CHANGE_TOL)
            and np.allclose(obs["gates_quat"], g_quat, atol=ENV_CHANGE_TOL)
            and np.allclose(obs["obstacles_pos"], o_pos, atol=ENV_CHANGE_TOL)
        )

    def _advance_wp_idx(self, pos: NDArray[np.floating]):
        """Mark nominal waypoints as passed once the drone has come close to them."""
        last = len(self._nominal_wps) - 1
        while self._wp_idx < last and (
            np.linalg.norm(pos - self._nominal_wps[self._wp_idx]) < PASS_RADIUS
        ):
            self._wp_idx += 1

    # ------------------------------------------------------------------------------------------
    # Feedforward from differential flatness
    # ------------------------------------------------------------------------------------------
    def _acc_to_ref(
        self, acc: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Roll, pitch and collective thrust that produce the planned acceleration (yaw ref = 0)."""
        f = self._mass * (acc + np.array([0.0, 0.0, self._g]))  # (N, 3) required force
        thrust = np.linalg.norm(f, axis=1)
        zb = f / thrust[:, None]
        roll = np.arcsin(np.clip(-zb[:, 1], -1.0, 1.0))
        pitch = np.arctan2(zb[:, 0], zb[:, 2])
        return np.clip(roll, -MAX_TILT, MAX_TILT), np.clip(pitch, -MAX_TILT, MAX_TILT), thrust

    # ------------------------------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # Replan from the current state if gates/obstacles moved (only the remaining waypoints)
        self._advance_wp_idx(obs["pos"])
        if self._env_changed(obs):
            self._plan(
                obs, np.asarray(obs["pos"], dtype=float), np.asarray(obs["vel"], dtype=float)
            )
            self._tick = 0

        # Project the drone onto the upcoming part of the reference: if it is ahead of the
        # (frozen) reference time, jump forward to the closest reference sample.
        hi = min(self._tick + 75, self._n_real)
        if hi > self._tick:
            d = np.linalg.norm(self._ref_pos[self._tick : hi] - np.asarray(obs["pos"]), axis=1)
            self._tick += int(np.argmin(d))
        i = min(self._tick, self._n_real - 1)
        self._lag = float(np.linalg.norm(np.asarray(obs["pos"]) - self._ref_pos[i]))
        if self._tick >= self._n_real - 1:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Reference samples for stages 0..N (stage k is k*MPC_DT in the future)
        idx = i + self._stride * np.arange(self._N + 1)
        roll, pitch, thrust = self._acc_to_ref(self._ref_acc[idx])

        # Stage reference: [x(12), u(4)]
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._ref_pos[idx[:-1]]  # position
        yref[:, 3] = roll[:-1]  # roll, pitch from planned acceleration (yaw ref = 0)
        yref[:, 4] = pitch[:-1]
        yref[:, 6:9] = self._ref_vel[idx[:-1]]  # velocity
        # zero drpy
        yref[:, 12] = roll[:-1]  # input roll / pitch
        yref[:, 13] = pitch[:-1]
        yref[:, 15] = thrust[:-1]  # input thrust
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Terminal reference (state part only)
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._ref_pos[idx[-1]]
        yref_e[3] = roll[-1]
        yref_e[4] = pitch[-1]
        yref_e[6:9] = self._ref_vel[idx[-1]]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)  # was "y_ref" (typo)

        # Solve; on failure keep the previous command instead of using a garbage solution
        status = self._acados_ocp_solver.solve()
        if status in (0, 2):  # 0: success, 2: max iterations reached (still usable)
            self._last_u = self._acados_ocp_solver.get(0, "u")
        return self._last_u

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        if getattr(self, "_lag", 0.0) < LAG_TOL:  # progress-based: wait for the drone if it lags
            self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the tick counter and restore the initial plan."""
        self._tick = 0
        self._finished = False
        self._wp_idx = 0
        for k, v in self._initial_plan.items():
            if k == "_seen_env":
                v = tuple(np.copy(a) for a in v)
            elif isinstance(v, np.ndarray):
                v = np.copy(v)
            setattr(self, k, v)
        self._last_u = np.array([0.0, 0.0, 0.0, self._mass * self._g])

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        draw_points(sim, self._waypoints, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        draw_line(sim, self._ref_pos[: self._n_real : 10], rgba=(0.0, 1.0, 0.0, 1.0))
