"""Model Predictive Contouring Control (MPCC) for time-optimal drone racing.

The controller follows the formulation of *Model Predictive Contouring Control for Time-Optimal
Quadrotor Flight* (Romero et al., IEEE T-RO 2022) and the reference implementation of the successful
`lsy_drone_racing_ws26`_ entry. Instead of tracking a time-parameterized reference, the drone tracks
a 3D path while the optimizer chooses how fast to traverse it:

* The path [p_d(s)][lsy_drone_racing.control.path_planning.ArcPath] is generated once per episode by
  the [PathPlanner][lsy_drone_racing.control.path_planning.PathPlanner].
* The stage cost penalizes the *lag* error along the path tangent and the *contour* error normal to
  it, and rewards progress along the path. Both weights are scaled by a proximity weight that peaks
  at the gates and obstacles (the Gaussian contouring-weight allocation of the paper), so the drone
  hugs the path exactly where it has to pass through a gate.
* The optimal control problem is solved with acados at the environment frequency (50 Hz) using an
  RTI SQP iteration, which is fast enough to run in the control loop.

The predicted arc length of every stage is obtained from the current projection of the drone onto
the path plus a reference speed that is the closed-form minimizer of the progress cost. This mirrors
Algorithm 1 of the paper ("predict theta_k from the previous solution") while keeping the CasADi
cost small: the path reference enters the OCP as a per-stage parameter instead of a lookup table
inside the solver.

Conventions:
    * Gates are crossed along their local x-axis, so a forward pass goes from ``-x`` to ``+x``.
    * The action is ``[roll, pitch, yaw, collective_thrust]`` in SI units, which is the order
      implemented by [build_action_space][lsy_drone_racing.envs.race_core.build_action_space] and
      expected by the simulator's attitude controller.

.. _lsy_drone_racing_ws26: https://github.com/lihengrongfei/lsy_drone_racing_ws26
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.path_planning import ArcPath, PathConfig, PathPlanner, track_reference

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_EXPORT_DIR = REPO_ROOT / "c_generated_code"
SOLVER_JSON = CODE_EXPORT_DIR / "mpcc_racing.json"
SIGNATURE_FILE = CODE_EXPORT_DIR / "mpcc_racing.signature"

#: State and input dimensions of the OCP.
NX, NU, NP = 15, 5, 7
#: In-process solver cache, so that multi-episode evaluation scripts build the solver only once.
_SOLVER_CACHE: dict[str, tuple[AcadosOcpSolver, AcadosOcp]] = {}


@dataclass(frozen=True)
class MPCCConfig:
    """Tunable parameters of the MPCC controller.

    The weights follow the reference implementation, with the contouring weight split into a
    constant part and a proximity-scaled part, see
    [track_reference][lsy_drone_racing.control.path_planning.track_reference].

    Attributes:
        N: Prediction horizon length.
        dt: Discretization step [s]. Set to ``1 / env.freq`` by the controller.
        q_lag: Lag error weight.
        q_contour: Contour error weight.
        q_lag_peak: Additional lag weight at the gates and obstacles.
        q_contour_peak: Additional contour weight at the gates and obstacles.
        contour_z_weight: Extra emphasis on the vertical contour error. The gates are only 0.4 m
            tall, so altitude accuracy, not lateral accuracy, limits the pass margin.
        q_attitude: Penalty on the absolute roll/pitch/yaw.
        r_rates: Penalty on the rate inputs ``[df, dr, dp, dy]``.
        mu_speed: Weight of the progress reward.
        r_vtheta: Quadratic regularization of the path progress rate.
        w_speed_gate: Proximity-dependent penalty on the path progress rate.
        v_min: Minimum reference path speed [m/s].
        v_max: Maximum reference path speed [m/s] and input bound on the progress rate.
        v_lead: Maximum speed [m/s] by which the path reference may run ahead of the drone. Without
            this limit the reference of the near stages would demand a physically impossible motion
            from a standing start.
        a_lat_max: Lateral acceleration [m/s^2] the path speed is limited to. The reference speed is
            capped by ``sqrt(a_lat_max / curvature)``, which slows the drone down exactly where the
            path bends instead of everywhere.
        curvature_smoothing: Number of samples of the curvature table that are averaged, so that a
            single noisy sample does not throttle the reference for a whole corner.
        thrust_lag: First-order lag of the collective thrust command.
        qp_solver: acados QP solver used inside the RTI iteration.
        condensing_N: Number of condensed stages. Values below ``N`` enable partial condensing,
            which solves the same problem with a smaller dense QP.
        attitude_limit: Bound [rad] on the roll/pitch/yaw setpoints.
        rate_limit: Bound on the attitude and thrust rate inputs.
        pos_low: Lower bounds ``[x, y, z]`` of the position. The lower z bound is deliberately far
            below the floor: the drone starts on the ground with a downward velocity, so any bound
            near z=0 makes the QP infeasible during take-off.
        pos_high: Upper bounds ``[x, y, z]`` of the position.
        mass_adapt: Whether to estimate the drone's mass online and rescale the commanded thrust.
            Level 0 keeps the nominal mass, levels 1-3 randomize it by up to about 12%.
        mass_adapt_gain: Low-pass gain of the mass-ratio estimate per control step.
        mass_adapt_window: Number of control steps the force balance is integrated over. A window
            rather than a single step is used because the thrust and attitude transients of the
            onboard controller would otherwise bias the estimate.
        mass_adapt_range: Admissible mass ratio ``(min, max)`` relative to the nominal mass.
        mass_adapt_deadband: Mass ratios closer to 1 than this are treated as exactly 1.
        mass_adapt_min_thrust: Minimum commanded thrust [N] for which an estimate is taken.
        mass_adapt_min_balance: Minimum integrated vertical balance [m/s] of the window. The window
            integrates ``(a_z + g) * window * dt``, so this rejects windows in which a strong dive
            almost cancels gravity and the estimate would be ill-conditioned.
        mass_adapt_max_speed: Maximum drone speed [m/s] for which an estimate is taken. At higher
            speeds the aerodynamic drag biases the vertical force balance.
        replan_slowdown_time: Time [s] for which the reference speed is reduced after a re-plan.
            A revealed gate can be up to 0.2 m away from its nominal pose, and the drone needs a
            moment to converge onto the corrected approach line before it reaches the gate.
        replan_slowdown_factor: Factor applied to the reference speed during that time.
        name: Name of the acados model and of the generated code.
    """

    N: int = 40
    dt: float = 0.02
    q_lag: float = 100.0
    q_contour: float = 100.0
    q_lag_peak: float = 2000.0
    q_contour_peak: float = 6000.0
    contour_z_weight: float = 2.0
    q_attitude: float = 1.0
    r_rates: tuple[float, float, float, float] = (0.2, 0.3, 0.3, 0.5)
    mu_speed: float = 10.0
    r_vtheta: float = 0.2
    w_speed_gate: float = 18.0
    v_min: float = 0.5
    v_max: float = 2.3
    v_lead: float = 0.6
    a_lat_max: float = 4.0
    curvature_smoothing: int = 9
    thrust_lag: float = 10.0
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    condensing_N: int = 10
    attitude_limit: float = 1.57
    rate_limit: float = 10.0
    pos_low: tuple[float, float, float] = (-2.45, -1.45, -10.0)
    pos_high: tuple[float, float, float] = (2.45, 1.45, 2.45)
    mass_adapt: bool = True
    mass_adapt_gain: float = 0.1
    mass_adapt_window: int = 10
    mass_adapt_range: tuple[float, float] = (0.85, 1.15)
    mass_adapt_deadband: float = 0.01
    mass_adapt_min_thrust: float = 0.3
    mass_adapt_min_balance: float = 1.0
    mass_adapt_max_speed: float = 1.5
    replan_slowdown_time: float = 0.8
    replan_slowdown_factor: float = 0.5
    name: str = "mpcc_racing"

    @property
    def horizon_time(self) -> float:
        """Length of the prediction horizon in seconds."""
        return self.N * self.dt


def _signature(config: MPCCConfig, params: dict) -> str:
    """Compute a signature of everything that defines the generated solver.

    Args:
        config: MPCC configuration.
        params: Drone parameters used by the model.

    Returns:
        A short hexadecimal signature.
    """
    payload = {
        "config": asdict(config),
        "mass": float(params["mass"]),
        "thrust": [float(params["thrust_min"]), float(params["thrust_max"])],
        "lag": [np.asarray(params[k], dtype=float).tolist() for k in _LAG_KEYS],
        "gravity": np.asarray(params["gravity_vec"], dtype=float).tolist(),
    }
    digest = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(digest).hexdigest()[:16]


_LAG_KEYS = ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef")


def _lag_coefficients(params: dict) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Derive the first-order attitude lag from the identified `so_rpy` model.

    Args:
        params: Drone parameters of the `so_rpy` dynamics.

    Returns:
        The decay coefficients ``a`` and the command gains ``b``, both shape ``(3,)``.
    """
    decay = -np.asarray(params["rpy_coef"], dtype=float)
    rates = np.asarray(params["rpy_rates_coef"], dtype=float)
    gain = -np.asarray(params["cmd_rpy_coef"], dtype=float)
    return decay / rates, gain / rates


def build_ocp(config: MPCCConfig, params: dict) -> AcadosOcp:
    """Build the acados optimal control problem of the MPCC.

    Args:
        config: MPCC configuration.
        params: Drone parameters used by the model.

    Returns:
        The fully specified (but not yet generated/solved) OCP.
    """
    mass = float(params["mass"])
    gravity = -float(np.asarray(params["gravity_vec"], dtype=float)[-1])
    lag_a, lag_b = _lag_coefficients(params)
    thrust_min = 4.0 * float(params["thrust_min"])
    thrust_max = 4.0 * float(params["thrust_max"])

    x = cs.SX.sym("x", NX)
    u = cs.SX.sym("u", NU)
    p = cs.SX.sym("p", NP)
    roll, pitch, yaw = x[6], x[7], x[8]
    thrust = x[9]
    thrust_cmd = x[10]
    roll_cmd, pitch_cmd, yaw_cmd = x[11], x[12], x[13]
    v_theta = u[4]

    # Translational and first-order command dynamics, see Eqs. 12-16 of the T-RO22 paper. The
    # attitude and thrust setpoints are states whose derivatives are the inputs, which gives the
    # command smoothing that keeps the solver output usable.
    dynamics = cs.vertcat(
        x[3],
        x[4],
        x[5],
        thrust / mass * (cs.cos(roll) * cs.sin(pitch) * cs.cos(yaw) + cs.sin(roll) * cs.sin(yaw)),
        thrust / mass * (cs.cos(roll) * cs.sin(pitch) * cs.sin(yaw) - cs.sin(roll) * cs.cos(yaw)),
        thrust / mass * cs.cos(roll) * cs.cos(pitch) - gravity,
        lag_a[0] * roll + lag_b[0] * roll_cmd,
        lag_a[1] * pitch + lag_b[1] * pitch_cmd,
        lag_a[2] * yaw + lag_b[2] * yaw_cmd,
        config.thrust_lag * (thrust_cmd - thrust),
        u[0],
        u[1],
        u[2],
        u[3],
        v_theta,
    )

    model = AcadosModel()
    model.name = config.name
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = dynamics
    model.f_impl_expr = None

    # Contouring cost: lag and contour error of the position with respect to the path reference of
    # this stage, weighted by the proximity-aware contouring weights.
    error = x[0:3] - p[0:3]
    tangent = p[3:6] / (cs.norm_2(p[3:6]) + 1e-6)
    lag = cs.dot(tangent, error)
    contour = error - lag * tangent
    # Emphasize the vertical contour error: the drone tends to arrive high at the shortest gates,
    # and the gates are only 0.4 m tall, so altitude accuracy is what limits the pass margin.
    contour = cs.vertcat(contour[0], contour[1], config.contour_z_weight * contour[2])
    q_lag = config.q_lag + config.q_lag_peak * p[6]
    q_contour = config.q_contour + config.q_contour_peak * p[6]
    cost = q_lag * cs.dot(lag, lag) + q_contour * cs.dot(contour, contour)
    cost += config.q_attitude * cs.dot(x[6:9], x[6:9])
    rates = u[0:4]
    cost += cs.dot(rates, cs.mtimes(cs.DM(np.diag(config.r_rates)), rates))
    cost += config.r_vtheta * v_theta * v_theta - config.mu_speed * v_theta
    model.cost_expr_ext_cost = cost
    model.cost_expr_ext_cost_e = cs.SX(0)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_export_directory = str(CODE_EXPORT_DIR)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.parameter_values = np.zeros(NP)

    # Box constraints on position, thrust setpoints, attitude setpoints and path progress.
    ocp.constraints.idxbx = np.array([0, 1, 2, 9, 10, 11, 12, 13, 14])
    ocp.constraints.lbx = np.array(
        [
            *config.pos_low,
            thrust_min,
            thrust_min,
            -config.attitude_limit,
            -config.attitude_limit,
            -config.attitude_limit,
            0.0,
        ]
    )
    ocp.constraints.ubx = np.array(
        [
            *config.pos_high,
            thrust_max,
            thrust_max,
            config.attitude_limit,
            config.attitude_limit,
            config.attitude_limit,
            1.0,
        ]
    )
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array(
        [-config.rate_limit, -config.rate_limit, -config.rate_limit, -config.rate_limit, 0.0]
    )
    ocp.constraints.ubu = np.array(
        [config.rate_limit, config.rate_limit, config.rate_limit, config.rate_limit, config.v_max]
    )
    # The initial state is fixed by a full-state box equality that is overwritten every control
    # step. acados requires the stage-0 box to span the full state when an initial state is given.
    ocp.constraints.idxbx_0 = np.arange(NX)
    ocp.constraints.lbx_0 = np.zeros(NX)
    ocp.constraints.ubx_0 = np.zeros(NX)
    ocp.constraints.x0 = np.zeros(NX)

    ocp.solver_options.N_horizon = config.N
    ocp.solver_options.tf = config.horizon_time
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = config.qp_solver
    ocp.solver_options.qp_solver_cond_N = config.condensing_N
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.print_level = 0
    return ocp


def _solver_artifacts_exist() -> bool:
    """Check whether a generated and compiled solver is on disk."""
    return SOLVER_JSON.exists() and (CODE_EXPORT_DIR / "acados_ocp_solver_mpcc_racing.dll").exists()


def get_solver(
    config: MPCCConfig, params: dict, force_build: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Return the acados solver, reusing cached or previously generated artifacts when possible.

    Generating and compiling the solver takes up to a minute on Windows, while the evaluation
    scripts construct a new controller for every episode. The solver is therefore cached per
    process and the generated artifacts are reused across processes as long as the signature of the
    OCP is unchanged.

    Args:
        config: MPCC configuration.
        params: Drone parameters used by the model.
        force_build: Regenerate and rebuild the solver even if artifacts are present.

    Returns:
        The solver and the OCP it was built from.
    """
    signature = _signature(config, params)
    if not force_build and signature in _SOLVER_CACHE:
        return _SOLVER_CACHE[signature]

    ocp = build_ocp(config, params)
    reuse = (
        not force_build
        and _solver_artifacts_exist()
        and SIGNATURE_FILE.exists()
        and SIGNATURE_FILE.read_text().strip() == signature
    )
    if reuse:
        logger.info("Loading cached MPCC solver from %s", SOLVER_JSON)
        solver = AcadosOcpSolver(
            ocp, json_file=str(SOLVER_JSON), generate=False, build=False, verbose=False
        )
    else:
        logger.info("Generating and building the MPCC solver (this takes a while)...")
        cmake_builder = None
        if os.name == "nt":  # The Windows toolchain requires the CMake builder with MinGW.
            cmake_builder = ocp_get_default_cmake_builder()
            cmake_builder.generator = "MinGW Makefiles"
            cmake_builder.build_dir = "build-mingw-mpcc"
        solver = AcadosOcpSolver(
            ocp,
            json_file=str(SOLVER_JSON),
            generate=True,
            build=True,
            cmake_builder=cmake_builder,
            verbose=False,
        )
        CODE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        SIGNATURE_FILE.write_text(signature)
        logger.info("MPCC solver built and stored in %s", CODE_EXPORT_DIR)
    _SOLVER_CACHE[signature] = (solver, ocp)
    return solver, ocp


def build_default_solver(force: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build the default level-0 solver.

    Args:
        force: Regenerate and rebuild the solver even if artifacts are present.

    Returns:
        The solver and the OCP it was built from.
    """
    params = load_params(Dynamics.so_rpy, "cf21B_500")
    return get_solver(MPCCConfig(), params, force_build=force)


class MPCCController(Controller):
    """Time-optimal drone racing controller based on Model Predictive Contouring Control."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller, plan the path and build the solver.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        # The strict gate handling (straight passages, gate-frame avoidance, slower gate approach,
        # strong gate centring) only pays off when the gate poses are randomized: it costs lap time,
        # and on a perfectly known track the path already passes the gates cleanly. Enable it only
        # for configs that randomize the track.
        randomizations = config.env.get("randomizations") or {}
        randomized_track = any(
            key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos")
        )
        mpcc_overrides = {} if randomized_track else {"w_speed_gate": 9.0, "q_contour_peak": 2000.0}
        self._mpcc = replace(MPCCConfig(dt=1.0 / self._freq), **mpcc_overrides)
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._hover_thrust = float(self._params["mass"]) * (-float(self._params["gravity_vec"][-1]))
        self._thrust_limits = (
            4.0 * float(self._params["thrust_min"]),
            4.0 * float(self._params["thrust_max"]),
        )
        # The strict gate handling (straight passages and gate-frame avoidance) only pays off when
        # the gate poses are randomized: it costs lap time, and on a perfectly known track the path
        # already passes the gates cleanly. Enable it only for configs that randomize the track.
        path_config = (
            PathConfig()
            if randomized_track
            else PathConfig(gate_straight_length=0.0, gate_frame_clearance=0.0)
        )
        self._planner = PathPlanner(path_config)
        self._solver, _ = get_solver(self._mpcc, self._params)
        self._nominal_mass = float(self._params["mass"])
        self._mass_ratio = 1.0
        self._applied_thrust = self._hover_thrust
        self._prev_vel: NDArray[np.floating] | None = None
        self._reset_mass_history()
        self._path: ArcPath | None = None
        self._gate_centers = np.zeros((0, 3))
        self._obstacles = np.zeros((0, 3))
        self._draw_points = np.zeros((0, 3))
        self._horizon_points = np.zeros((0, 3))
        self._needs_plan = True
        self._tick = 0
        self._failures = 0
        self._stall_time = 0.0
        self._stall_reported = False
        self.status = 0
        self.solve_time = 0.0
        self._init_episode(obs)

    # region lifecycle

    def _init_episode(self, obs: dict[str, NDArray[np.floating]]):
        """Plan the path and reset all controller memory for a new episode."""
        self._plan_origin = np.asarray(obs["pos"], dtype=float).copy()
        self._path, info = self._planner.plan(obs, self._plan_origin)
        self._gate_centers = info.gate_centers
        self._obstacles = info.obstacles
        self._draw_points = self._path.pos(np.linspace(0.0, self._path.length, 200))
        self._build_speed_limit()
        self._apply_progress_bound()
        self._seed_visited_flags(obs)
        self._reproject_wide = False
        self._projection_window = (-0.15, 0.6)
        self._slowdown_until = 0
        self.replans = 0
        self._last_f_col = self._hover_thrust
        self._last_f_cmd = self._hover_thrust
        self._last_rpy_cmd = np.zeros(3)
        self._mass_ratio = 1.0
        self._applied_thrust = self._hover_thrust
        self._prev_vel = None
        self._reset_mass_history()
        self._theta = 0.0
        self._tick = 0
        self._failures = 0
        self._stall_time = 0.0
        self._stall_reported = False
        self._needs_plan = False
        self._solver.reset()

    def _seed_visited_flags(self, obs: dict[str, NDArray[np.floating]]):
        """Initialize the reveal latch from the current observation.

        Args:
            obs: The current observation.
        """
        self._last_gate_flags = np.asarray(
            obs.get("gates_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        self._last_obstacle_flags = np.asarray(
            obs.get("obstacles_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

    def _track_info_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return whether the observed gate or obstacle poses changed since the previous step.

        The environment reports nominal poses until the drone comes within the sensor range and the
        true pose afterwards, so a change of the observed pose is exactly the moment at which more
        accurate track information becomes available. Comparing the poses rather than only the
        `visited` flags avoids replanning when a flag rises but the reported pose is the same, which
        is the case on tracks without pose randomisation, and it also reacts to a pose that changes
        more than once.

        Args:
            obs: The current observation.

        Returns:
            ``True`` if at least one observed gate or obstacle pose changed in this step.
        """
        gates = np.asarray(obs["gates_pos"], dtype=float)
        obstacles = np.asarray(obs["obstacles_pos"], dtype=float)
        gates_visited = np.asarray(
            obs.get("gates_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        obstacles_visited = np.asarray(
            obs.get("obstacles_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        previous_gates, previous_obstacles = self._last_gates_pos, self._last_obstacles_pos
        self._last_gates_pos = gates.copy()
        self._last_obstacles_pos = obstacles.copy()
        self._last_gate_flags, self._last_obstacle_flags = gates_visited, obstacles_visited
        if gates.shape != previous_gates.shape or obstacles.shape != previous_obstacles.shape:
            return False
        gate_changed = float(np.max(np.abs(gates - previous_gates), initial=0.0)) > 1e-6
        obstacle_changed = float(np.max(np.abs(obstacles - previous_obstacles), initial=0.0)) > 1e-6
        return bool(gate_changed or obstacle_changed)

    def _replan(self, obs: dict[str, NDArray[np.floating]]):
        """Re-plan the path with the newly revealed gate and obstacle poses.

        Only the path reference changes: the optimal control problem and its solver are untouched,
        because the path enters the cost as per-stage parameters. The speed limit table and the
        progress bound are rebuilt for the new path, and the projection window is widened for one
        step so that the drone re-anchors onto the new arc length immediately.

        Args:
            obs: The current observation.
        """
        path, info = self._planner.plan(obs, self._plan_origin)
        self._path = path
        self._gate_centers = info.gate_centers
        self._obstacles = info.obstacles
        self._draw_points = path.pos(np.linspace(0.0, path.length, 200))
        self._build_speed_limit()
        self._apply_progress_bound()
        self._reproject_wide = True
        self._slowdown_until = self._tick + int(round(self._mpcc.replan_slowdown_time * self._freq))
        self.replans += 1
        logger.info(
            "Re-planned path after sensor update: %.2f m (replan %d)", path.length, self.replans
        )

    def _apply_progress_bound(self):
        """Push the path length into the per-stage upper bound of the theta state.

        Stage 0 is the fixed initial state and the terminal stage carries no box constraint, so only
        the intermediate stages are updated.
        """
        assert self._path is not None
        ubx = np.array(self._stage_ubx())
        ubx[-1] = self._path.length
        for k in range(1, self._mpcc.N):
            self._solver.set(k, "ubx", ubx)

    def _reset_mass_history(self):
        """Clear the rolling window used by the mass estimate."""
        window = self._mpcc.mass_adapt_window
        self._force_hist: deque[float] = deque(maxlen=window)
        self._speed_hist: deque[float] = deque(maxlen=window)
        self._dvz_hist: deque[float] = deque(maxlen=window)

    def _build_speed_limit(self):
        """Precompute the curvature-limited path speed along the arc length.

        The path speed is capped by the lateral acceleration the drone can produce while still
        tracking the path: ``v <= sqrt(a_lat_max / curvature)``. This throttles the reference
        exactly in the tight corners instead of everywhere, which is what keeps the gate passes
        accurate.
        """
        assert self._path is not None
        length = self._path.length
        self._speed_grid = np.linspace(0.0, length, max(32, int(length / 0.02)))
        curvature = self._path.curvature(self._speed_grid)
        window = max(1, self._mpcc.curvature_smoothing)
        if window > 1:
            kernel = np.ones(window) / window
            curvature = np.convolve(np.pad(curvature, window // 2, mode="edge"), kernel, "valid")
        limit = np.sqrt(self._mpcc.a_lat_max / np.maximum(curvature, 1e-3))
        self._speed_limit = np.minimum(limit, self._mpcc.v_max)

    def episode_callback(self):
        """Log the flight statistics of the finished episode."""
        logger.info("MPCC episode finished after %d steps, theta=%.2f m", self._tick, self._theta)

    def episode_reset(self):
        """Mark the controller for a fresh path plan at the next control step."""
        self._needs_plan = True

    def reset(self):
        """Mark the controller for a fresh path plan at the next control step."""
        self._needs_plan = True

    def render_callback(self, sim: Sim):
        """Draw the planned path and the current horizon reference.

        Args:
            sim: The simulator instance used for rendering.
        """
        from crazyflow.sim.visualize import draw_line, draw_points

        if len(self._draw_points):
            draw_line(sim, self._draw_points, rgba=(0.0, 1.0, 0.0, 0.6))
        if len(self._horizon_points):
            draw_points(sim, self._horizon_points, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

    # endregion

    # region helpers

    def _stage_ubx(self) -> list[float]:
        """Return the compact upper state bounds of a regular stage (path length excluded)."""
        cfg = self._mpcc
        _, thrust_max = self._thrust_limits
        return [
            *cfg.pos_high,
            thrust_max,
            thrust_max,
            cfg.attitude_limit,
            cfg.attitude_limit,
            cfg.attitude_limit,
            1.0,
        ]

    def _state_vector(self, obs: dict[str, NDArray[np.floating]], theta: float) -> NDArray:
        """Assemble the full OCP state from an observation.

        Args:
            obs: The current observation.
            theta: Current path progress in meters.

        Returns:
            The state vector of length ``NX = 15``.
        """
        euler = R.from_quat(np.asarray(obs["quat"], dtype=float)).as_euler("xyz")
        return np.concatenate(
            [
                np.asarray(obs["pos"], dtype=float),
                np.asarray(obs["vel"], dtype=float),
                euler,
                [self._last_f_col, self._last_f_cmd],
                self._last_rpy_cmd,
                [theta],
            ]
        )

    def _reference_speed(self, theta: float, speed_actual: float) -> float:
        """Return the reference path speed at a given path progress.

        This is the closed-form minimizer of the progress stage cost ``-mu * v + w * qc * v^2``,
        clipped to the admissible range and to the current drone speed plus a lead. Clipping to the
        measured speed is what makes the per-stage reference realizable: the near stages of the
        horizon are only ever asked to move slightly faster than the drone already does.

        Args:
            theta: Path progress in meters.
            speed_actual: Current drone speed in m/s.

        Returns:
            The reference path speed in m/s.
        """
        cfg = self._mpcc
        assert self._path is not None
        _, _, weight = track_reference(
            self._path, np.array([theta]), self._gate_centers, self._obstacles
        )
        speed = cfg.mu_speed / (2.0 * (cfg.w_speed_gate * float(weight[0]) + cfg.r_vtheta))
        speed = min(speed, speed_actual + cfg.v_lead)
        speed = min(speed, float(np.interp(theta, self._speed_grid, self._speed_limit)))
        if self._tick < self._slowdown_until:
            speed *= cfg.replan_slowdown_factor
        return float(np.clip(speed, cfg.v_min, cfg.v_max))

    def _progress(self, pos: NDArray[np.floating]) -> float:
        """Project the current position onto the path with a monotone guard.

        Args:
            pos: Current drone position, shape ``(3,)``.

        Returns:
            The arc length parameter of the projection.
        """
        assert self._path is not None
        window = getattr(self, "_projection_window", (-0.15, 0.6))
        theta = self._path.closest(np.asarray(pos, dtype=float), self._theta, window=window)
        return float(np.clip(theta, max(0.0, self._theta - 0.05), self._path.length))

    def _command(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Clip an attitude command into the environment's action space.

        Args:
            action: Raw ``[roll, pitch, yaw, thrust]`` command.

        Returns:
            The clipped command as float32.
        """
        low = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, self._thrust_limits[0]])
        high = np.array([np.pi / 2, np.pi / 2, np.pi / 2, self._thrust_limits[1]])
        return np.clip(action, low, high).astype(np.float32)

    # endregion

    def _output(self) -> NDArray[np.floating]:
        """Build the mass-compensated command and remember the thrust that was applied.

        Returns:
            The commanded ``[roll, pitch, yaw, collective_thrust]``.
        """
        thrust = self._last_f_cmd * self._mass_ratio
        command = self._command(np.concatenate((self._last_rpy_cmd, [thrust])))
        self._applied_thrust = float(command[3])
        return command

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude and thrust command.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary. Not used.

        Returns:
            The commanded ``[roll, pitch, yaw, collective_thrust]``.
        """
        if self._needs_plan:
            self._init_episode(obs)
        assert self._path is not None
        self._tick += 1

        pos = np.asarray(obs["pos"], dtype=float)
        self._update_mass_estimate(obs)
        if self._track_info_changed(obs):
            self._replan(obs)
        theta = self._progress(pos)
        speed = self._reference_speed(theta, float(np.linalg.norm(np.asarray(obs["vel"], float))))
        thetas = np.clip(
            theta + np.arange(self._mpcc.N + 1) * self._mpcc.dt * speed, 0.0, self._path.length
        )
        positions, tangents, weights = track_reference(
            self._path, thetas, self._gate_centers, self._obstacles
        )
        self._horizon_points = positions
        for k in range(self._mpcc.N + 1):
            params = np.concatenate((positions[k], tangents[k], [weights[k]]))
            self._solver.set(k, "p", np.ascontiguousarray(params, dtype=np.float64))

        x0 = self._state_vector(obs, theta)
        self._solver.set(0, "lbx", np.ascontiguousarray(x0))
        self._solver.set(0, "ubx", np.ascontiguousarray(x0))

        start = time.perf_counter()
        status = self._solver.solve()
        self.solve_time = time.perf_counter() - start
        self.status = status
        if status != 0:
            self._failures += 1
            if self._failures == 1 or self._failures % 50 == 0:
                logger.warning(
                    "MPCC solver failed with status %d (%d times)", status, self._failures
                )
            if self._failures > 5:
                self._last_f_col = self._hover_thrust
                self._last_f_cmd = self._hover_thrust
                self._last_rpy_cmd = np.zeros(3)
            return self._output()
        self._failures = 0

        state = np.asarray(self._solver.get(1, "x"), dtype=float)
        self._last_f_col = float(state[9])
        self._last_f_cmd = float(state[10])
        self._last_rpy_cmd = np.asarray(state[11:14], dtype=float)
        self._update_watchdog(theta)
        return self._output()

    def _update_mass_estimate(self, obs: dict[str, NDArray[np.floating]]):
        """Estimate the mass ratio from the integrated vertical force balance.

        The OCP always models the nominal drone, but levels 1-3 randomize the mass by up to about
        12%. Over each control interval the vertical momentum balance of the drone reads

        ``m * dv_z = T * cos(roll) * cos(pitch) * dt - m * g * dt``,

        so integrating it over a short window yields a direct estimate of ``m``. Integrating rather
        than evaluating a single step removes the bias that the thrust and attitude transients of
        the onboard controller would otherwise introduce. The estimate is only taken at low speed
        and moderate thrust, where aerodynamic drag and the vertical component of the horizontal
        acceleration are negligible, and it is filtered and clipped to a plausible range.

        The commanded thrust is rescaled by the estimated ratio at the very end of the control step,
        so the drone accelerates like the nominal model the OCP planned with. The OCP state itself
        keeps using nominal-model units.

        Args:
            obs: The current observation.
        """
        cfg = self._mpcc
        velocity = np.asarray(obs["vel"], dtype=float)
        if not cfg.mass_adapt:
            self._prev_vel = velocity
            return
        euler = R.from_quat(np.asarray(obs["quat"], dtype=float)).as_euler("xyz")
        up = float(np.cos(euler[0]) * np.cos(euler[1]))
        if self._prev_vel is not None:
            self._force_hist.append(self._applied_thrust * up)
            self._speed_hist.append(float(np.linalg.norm(velocity)))
            self._dvz_hist.append(float(velocity[2] - self._prev_vel[2]))
        self._prev_vel = velocity
        window = cfg.mass_adapt_window
        if len(self._force_hist) < window:
            return
        impulsion = cfg.dt * float(sum(self._force_hist))
        gravity = float(-np.asarray(self._params["gravity_vec"])[2])
        denominator = float(sum(self._dvz_hist)) + gravity * window * cfg.dt
        if (
            float(np.mean(self._force_hist)) < cfg.mass_adapt_min_thrust
            or float(np.mean(self._speed_hist)) > cfg.mass_adapt_max_speed
            or denominator < cfg.mass_adapt_min_balance
        ):
            return
        ratio = impulsion / (self._nominal_mass * denominator)
        low, high = cfg.mass_adapt_range
        if abs(ratio - 1.0) < cfg.mass_adapt_deadband:
            ratio = 1.0
        ratio = float(np.clip(ratio, low, high))
        self._mass_ratio += cfg.mass_adapt_gain * (ratio - self._mass_ratio)
        self._mass_ratio = float(np.clip(self._mass_ratio, low, high))

    def _update_watchdog(self, theta: float):
        """Warn once if the drone stops making progress along the path.

        Args:
            theta: Current path progress in meters.
        """
        progress = theta - self._theta
        self._theta = theta
        if progress < 0.002:
            self._stall_time += self._mpcc.dt
        else:
            self._stall_time = 0.0
        if self._stall_time > 3.0 and not self._stall_reported:
            self._stall_reported = True
            logger.warning("Drone is not progressing along the path (theta=%.2f m)", theta)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return whether the controller is finished.

        The environment terminates the episode once the last gate in the sequence is passed, so the
        controller never has to finish the episode itself.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.

        Returns:
            Always ``False``.
        """
        return False
