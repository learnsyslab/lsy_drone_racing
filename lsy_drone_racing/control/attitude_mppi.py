"""Sampling-based MPC (MPPI) for drone racing using the collective thrust and attitude interface.

The controller plans over the fitted second-order `so_rpy` dynamics model. At each step it samples
perturbed control sequences around a nominal sequence, simulates them in parallel, and updates the
nominal sequence with a cost-weighted average of the perturbations.

State and input conventions follow `attitude_mpc.py`:

    x = [pos(3), rpy(3), vel(3), drpy(3)]
    u = [roll, pitch, yaw, thrust]  (thrust is the total collective thrust in N)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy.dynamics import dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from jax import Array
    from numpy.typing import NDArray


def dynamics_step(x: Array, u: Array, params: dict, dt: float) -> Array:
    """Advance the so_rpy model by one time step.

    Args:
        x: State [pos, rpy, vel, drpy] with shape (12,).
        u: Input [roll, pitch, yaw, thrust] with shape (4,).
        params: Drone parameters as returned by `crazyflow.dynamics.load_params`.
        dt: Integration step in seconds.

    Returns:
        The next state with shape (12,).
    """
    # TODO Check if RK4 integration is necessary, or if Euler is sufficient for the small dt.
    pos, rpy, vel, drpy = jnp.split(x, 4)
    x_dot = jnp.concatenate(dynamics_euler(
        pos=pos,
        rpy=rpy,
        vel=vel,
        rpy_rates=drpy,
        cmd=u,
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"]
    ))

    # Euler integration step
    x_next = x + dt * x_dot

    return x_next


def rollout(x0: Array, U: Array, params: dict, dt: float) -> Array:
    """Simulate a control sequence starting from x0.

    Args:
        x0: Initial state with shape (12,).
        U: Control sequence with shape (H, 4).
        params: Drone parameters.
        dt: Integration step in seconds.

    Returns:
        The state trajectory with shape (H + 1, 12), including x0.
    """

    # Define a step function for the scan operation
    def step_fn(x, u):
        x_next = dynamics_step(x, u, params, dt)
        return x_next, x_next

    # Use JAX's scan to efficiently compute the trajectory
    _, outputs = jax.lax.scan(step_fn, x0, U)

    # return missing initial state x0 at the beginning of the trajectory
    return jnp.vstack((x0, outputs))


COST_CONSTANTS = {
    "deviation_weight": 1.0,  # Weight for deviation from reference trajectory
    "deviation_tolerance": 0.1,  # Tolerance in meters for an acceptable deviation from the reference trajectory
    "smoothness_weight": 0.1,  # Weight for control smoothness
    "smoothness_tolerance": 1,  # Tolerance for control smoothness in multiples of sigma
    "gate_weight": 2.0,  # Weight for gate pass / collision / miss
    "gate_tolerance": 0.05,  # Tolerance in meters for an acceptable deviation from the gate's center
    "gate_slab": 0.1,  # Half-thickness in meters around a gate's plane in which the gate is checked
    "gate_frame_limit": 0.45,  # Offset in meters from the gate's center beyond which the frame is cleared
    "obstacle_weight": 2.0,  # Weight for coming close to an obstacle
    "obstacle_radius": 0.2,  # Radius in meters of the penalty tube around an obstacle
}


def trajectory_cost(
        X: Array,
        U: Array,
        u_prev: Array,
        ref: Array,
        gates: Array,
        obstacles_pos: Array,
        sigma: Array,
) -> Array:
    """Compute the cost of a single rollout.

    Args:
        X: State trajectory with shape (H + 1, 12).
        U: Control sequence with shape (H, 4).
        u_prev: Previous control input with shape (4,).
        ref: Reference positions and tangential velocities with shape (H + 1, 6).
            The first three columns are positions, the last three are velocities.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
            The first three columns are positions, the last three are normal vectors.
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3). Obstacles are vertical
            poles, so only the x and y coordinates are used.
        sigma: Standard deviation of the control input with shape (4,).

    Returns:
        The scalar cost.
    """
    # Deviation from reference trajectory
    pos = X[:, :3]  # Extract positions from the state trajectory
    ref_pos = ref[:, :3]  # Extract reference positions
    tol = jnp.maximum(1e-6, COST_CONSTANTS["deviation_tolerance"])
    pos_error_per_step = jnp.square((pos - ref_pos) / tol)  # Compute per-step position error
    deviation_cost = jnp.sum(pos_error_per_step)

    # Cost for control smoothness
    U_including_prev = jnp.vstack((u_prev, U))  # Include previous control input for smoothness calculation
    control_diff = jnp.diff(U_including_prev, axis=0)  # Compute differences between consecutive control inputs
    control_diff /= COST_CONSTANTS["smoothness_tolerance"] * jnp.maximum(1e-6, sigma)
    smoothness_cost = jnp.sum(jnp.linalg.norm(control_diff, axis=1))

    # Cost for gate collision: states near a gate's plane must be close to its center, for every gate
    # get gate_pos and gate_normal
    gate_pos = gates[:, :3]  # Extract gate positions
    gate_normal = gates[:, 3:]  # Extract gate normal vectors
    gate_offset = pos[:, None, :] - gate_pos[None, :, :]  # Offset of each state to each gate center
    # a state is near a gate if it is within a thin slab around the gate's plane
    distance = jnp.sum(gate_offset * gate_normal[None, :, :], axis=-1)
    near_gate_mask = jnp.abs(distance) < COST_CONSTANTS["gate_slab"]
    # use maxnorm(to not punish corners) of distance to gate center along gate plane primary axes
    # primary axes are always vertical axis and the cross between normal and vertical axis
    vertical_distance = gate_offset[..., 2]
    horizontal_distance = jnp.sum(gate_offset * jnp.cross(gate_normal, jnp.array([0, 0, 1]))[None, :, :], axis=-1)
    gate_plane_max_distance = jnp.maximum(jnp.abs(vertical_distance), jnp.abs(horizontal_distance))
    # states beside the frame, outside of its outer edge, are free
    frame_mask = near_gate_mask & (gate_plane_max_distance < COST_CONSTANTS["gate_frame_limit"])

    # cost function based on distance to gate center
    distance_further_than_tolerance = jnp.maximum(gate_plane_max_distance / COST_CONSTANTS["gate_tolerance"] - 1, 0)
    gate_cost = jnp.sum(jnp.where(frame_mask, distance_further_than_tolerance ** 2, 0))

    # Cost for coming close to an obstacle: tube around each pole, checked in the x-y plane
    obstacle_xy_distance = jnp.linalg.norm(pos[:, None, :2] - obstacles_pos[None, :, :2], axis=-1)
    # 0 outside the tube radius, rising to 1 at the pole's axis
    tube_penetration = jnp.maximum(1 - obstacle_xy_distance / COST_CONSTANTS["obstacle_radius"], 0)
    obstacle_cost = jnp.sum(tube_penetration ** 2)

    total_cost = (
            COST_CONSTANTS["deviation_weight"] * deviation_cost
            + COST_CONSTANTS["smoothness_weight"] * smoothness_cost
            + COST_CONSTANTS["gate_weight"] * gate_cost
            + COST_CONSTANTS["obstacle_weight"] * obstacle_cost
    )  # Add other cost components if needed

    return total_cost

def adaptive_lambda(costs: Array, target_ess: float, n_iter: int = 30) -> Array:
    """Find the softmin temperature at which the weights reach a target effective sample size.

    The ESS grows monotonically with the temperature, so it is found by bisection in log-space.

    Args:
        costs: Rollout costs with shape (n_samples,).
        target_ess: Desired effective sample size as a fraction of n_samples, in (0, 1].
        n_iter: Number of bisection steps.

    Returns:
        The temperature lambda.
    """
    costs = costs - jnp.min(costs)
    target = target_ess * costs.shape[0]

    def ess(log_lam):
        weights = jax.nn.softmax(-costs / jnp.exp(log_lam))
        return 1 / jnp.sum(weights ** 2)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_peaked = ess(mid) < target
        return jnp.where(too_peaked, mid, lo), jnp.where(too_peaked, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, bisect, (jnp.log(1e-6), jnp.log(1e6)))
    return jnp.exp(0.5 * (lo + hi))


def mppi_update(
        key: Array,
        x0: Array,
        U_nom: Array,
        u_prev: Array,
        ref: Array,
        gates: Array,
        obstacles_pos: Array,
        params: dict,
        dt: float,
        n_samples: int,
        sigma: Array,
        target_ess: float,
) -> tuple[Array, Array, Array]:
    """Perform one MPPI iteration.

    Samples `n_samples` perturbed control sequences around `U_nom`, simulates and scores them, and
    returns the nominal sequence updated with the softmin-weighted perturbations. The softmin
    temperature is chosen every step such that the weights reach the target effective sample size.

    Args:
        key: JAX PRNG key.
        x0: Current state with shape (12,).
        U_nom: Nominal control sequence with shape (H, 4).
        u_prev: Previous control sequence with shape (4,).
        ref: Reference positions with shape (H + 1, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        params: Drone parameters.
        dt: Integration step in seconds.
        n_samples: Number of sampled control sequences.
        sigma: Per-channel standard deviation of the sampling noise with shape (4,).
        target_ess: Desired effective sample size as a fraction of n_samples.

    Returns:
        - U_new: The updated nominal control sequence with shape (H, 4).
        - ESS: Effective sample size (ESS) for debugging and logging.
        - lam: The chosen softmin temperature for debugging and logging.

    """
    samples = jax.random.normal(key, (n_samples, len(U_nom), 4), dtype=jnp.float32)
    samples *= sigma
    sampled_actions = U_nom + samples

    # clip attitude and total thrust
    sampled_actions = jnp.clip(
        sampled_actions,
        jnp.array([-1.0, -1.0, -1.0, 4 * params["thrust_min"]]),
        jnp.array([1.0, 1.0, 1.0, 4 * params["thrust_max"]])
    )

    # define batch functions
    rollout_jit = jax.jit(rollout)
    rollout_batch_of_trajectories = jax.vmap(rollout_jit, in_axes=(None, 0, None, None))

    cost_jit = jax.jit(trajectory_cost)
    trajectory_cost_for_batch = jax.vmap(cost_jit, in_axes=(0, 0, None, None, None, None, None))

    rollouts = rollout_batch_of_trajectories(
        x0,
        sampled_actions,
        params,
        dt,
    )

    costs = trajectory_cost_for_batch(
        rollouts,
        sampled_actions,
        u_prev,
        ref,
        gates,
        obstacles_pos,
        sigma,
    )

    # weighting the runs by costs
    lam = adaptive_lambda(costs, target_ess)
    weights = jax.nn.softmax(-costs / lam)

    # calculate effective sample size (ESS) for debugging and logging
    ess = 1 / jnp.sum(weights ** 2)

    U_new = jnp.sum(weights[:, None, None] * sampled_actions, axis=0)

    return U_new, ess, lam


def build_reference(
        waypoints: NDArray, t_total: float, freq: float, n_extend: int = 0
) -> np.ndarray:
    """Create a time-indexed reference from a cubic spline through the waypoints.

    The waypoints are spaced evenly in time over `t_total`. The spline is sampled at `freq`, so
    row `i` of the result is the reference for time `i / freq`. The reference can be extended past
    the last waypoint with a straight continuation at the final velocity.

    Args:
        waypoints: Positions to pass through with shape (N, 3).
        t_total: Duration of the reference in seconds.
        freq: Sampling frequency of the reference in Hz, i.e. the control rate.
        n_extend: Number of rows appended after the spline ends.

    Returns:
        The reference with shape (T + n_extend, 6) and `T = int(freq * t_total)`. The first three
        columns are positions, the last three are velocities.
    """
    # Spline through the waypoints, with the waypoints evenly spaced in time
    spline = CubicSpline(np.linspace(0, t_total, len(waypoints)), waypoints)
    t = np.arange(int(freq * t_total)) / freq  # one sample time per control step
    ref = np.hstack((spline(t), spline.derivative()(t)))  # [position, velocity] per step
    # Continue straight at the final velocity so the horizon never runs out of reference
    steps = np.arange(1, n_extend + 1)[:, None] / freq
    extension = np.hstack((ref[-1, :3] + steps * ref[-1, 3:], np.tile(ref[-1, 3:], (n_extend, 1))))
    return np.vstack((ref, extension))


class AttitudeMPPI(Controller):
    """Sampling-based MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1 / self._freq
        self._horizon = 25  # planning steps, 0.5 s at 50 Hz
        self._n_samples = 512  # sampled control sequences per step
        self._sigma = jnp.array([0.15, 0.15, 0.0, 0.05]) * 1.5  # noise std: roll, pitch, yaw, thrust
        self._target_ess = 0.05  # fraction of samples that should effectively contribute to the update
        self._lam = float("nan")  # softmin temperature, adapted every step to reach the target ESS
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
        # Hover command, inverted from force = acc_coef + cmd_f_coef * cmd
        self._hover_thrust = float(
            (self._params["mass"] * -self._params["gravity_vec"][-1] - self._params["acc_coef"])
            / self._params["cmd_f_coef"]
        )

        start_pos = obs["pos"]
        waypoints = np.array(
            [
                start_pos,
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.15, -0.2, 0.8],
                [-0.9, 0.1, 1.2],
                [0.0, -0.75, 1.2],
                [1.2, -0.15, 1.2],
                [1.05, 0.75, 1.2],
                [0.25, 1.25, 1.2],
            ]
        )
        self._t_total = 18.75  # time to fly the whole reference
        # The extra horizon + 1 rows keep the last window full once the reference time is over
        self._ref = build_reference(waypoints, self._t_total, self._freq, self._horizon + 1)
        self._tick_max = int(self._t_total * self._freq)

        # n_samples fixes an array shape, so it must be static for jit
        self._update = jax.jit(mppi_update, static_argnames=("n_samples",))
        self._key = jax.random.key(0)
        # Warm start: hold hover thrust with level attitude
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._u_prev = self._U_nom[0]
        self._ess = float("nan")
        self._tick = 0
        self._finished = False

        # Warm-up call so the jit compilation happens here and not in the first control step
        self._update(
            self._key,
            self._state(obs),
            self._U_nom,
            self._u_prev,
            self._ref[: self._horizon + 1],
            self._get_gates_pos_and_normal(obs),
            jnp.asarray(obs["obstacles_pos"]),
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._target_ess,
        )

    def _state(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Convert an observation to the planning state [pos, rpy, vel, drpy]."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        # The model uses roll/pitch/yaw rates, the observation gives the body angular velocity
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        return jnp.asarray(np.concatenate((obs["pos"], rpy, obs["vel"], drpy)))

    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next roll, pitch, yaw and collective thrust command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The command [roll, pitch, yaw, thrust] as a numpy array.
        """
        if self._tick >= self._tick_max:
            self._finished = True
        # Reference window for the current step and the whole horizon
        ref = self._ref[self._tick: self._tick + self._horizon + 1]

        # Fresh noise every step: split the key, keep one half and use the other
        self._key, subkey = jax.random.split(self._key)
        U_new, ess, lam = self._update(
            subkey,
            self._state(obs),
            self._U_nom,
            self._u_prev,
            ref,
            self._get_gates_pos_and_normal(obs),
            jnp.asarray(obs["obstacles_pos"]),
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._target_ess,
        )
        self._ess = float(ess)
        self._lam = float(lam)
        
        self._u_prev = U_new[0]
        # Shift by one step for the next call and repeat the last input to fill the end
        self._U_nom = jnp.concatenate((U_new[1:], U_new[-1:]))
        # Only the first input is applied, the rest is re-planned at the next step
        return np.asarray(U_new[0], dtype=np.float32)

    def step_callback(
            self,
            action: NDArray[np.floating],
            obs: dict[str, NDArray[np.floating]],
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict,
    ) -> bool:
        """Advance the internal step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self._finished = False
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._u_prev = self._U_nom[0]

    def _get_gates_pos_and_normal(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Return the position and the normal vector of every gate with shape (n_gates, 6)."""
        # The gate normal is the gate's local x-axis in the world frame
        gates_normal = R.from_quat(obs["gates_quat"]).apply([1, 0, 0])
        return jnp.hstack([obs["gates_pos"], gates_normal])
