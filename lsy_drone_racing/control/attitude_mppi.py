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
    "smoothness_weight": 0.01,  # Weight for control smoothness
}


def trajectory_cost(X: Array, U: Array, u_prev: Array, ref: Array) -> Array:
    """Compute the cost of a single rollout.

    Args:
        X: State trajectory with shape (H + 1, 12).
        U: Control sequence with shape (H, 4).
        u_prev: Previous control input with shape (4,).
        ref: Reference positions and tangential velocities with shape (H + 1, 6).
            The first three columns are positions, the last three are velocities.

    Returns:
        The scalar cost.
    """
    # Deviation from reference trajectory
    pos = X[:, :3]  # Extract positions from the state trajectory
    ref_pos = ref[:, :3]  # Extract reference positions
    pos_error_per_step = jnp.square(pos - ref_pos)  # Compute per-step position error
    deviation_cost = jnp.sum(pos_error_per_step)

    # Cost for control smoothness
    U_including_prev = jnp.vstack((u_prev, U))  # Include previous control input for smoothness calculation
    control_diff = jnp.diff(U_including_prev, axis=0)  # Compute differences between consecutive control inputs
    smoothness_cost = jnp.sum(jnp.linalg.norm(control_diff, axis=1))

    total_cost = (
            COST_CONSTANTS["deviation_weight"] * deviation_cost
            + COST_CONSTANTS["smoothness_weight"] * smoothness_cost
    )  # Add other cost components if needed

    return total_cost


def mppi_update(
        key: Array,
        x0: Array,
        U_nom: Array,
        u_prev: Array,
        ref: Array,
        params: dict,
        dt: float,
        n_samples: int,
        sigma: Array,
        lam: float,
) -> tuple[Array, Array]:
    """Perform one MPPI iteration.

    Samples `n_samples` perturbed control sequences around `U_nom`, simulates and scores them, and
    returns the nominal sequence updated with the softmin-weighted perturbations.

    Args:
        key: JAX PRNG key.
        x0: Current state with shape (12,).
        U_nom: Nominal control sequence with shape (H, 4).
        u_prev: Previous control sequence with shape (4,).
        ref: Reference positions with shape (H + 1, 3).
        params: Drone parameters.
        dt: Integration step in seconds.
        n_samples: Number of sampled control sequences.
        sigma: Per-channel standard deviation of the sampling noise with shape (4,).
        lam: Temperature of the softmin weighting.

    Returns:
        - U_new: The updated nominal control sequence with shape (H, 4).
        - ESS: Effective sample size (ESS) for debugging and logging.

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
    trajectory_cost_for_batch = jax.vmap(cost_jit, in_axes=(0, 0, None, None))

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
    )

    # weighting the runs by costs
    weights = jax.nn.softmax(-costs / lam)

    # calculate effective sample size (ESS) for debugging and logging
    ess = 1 / jnp.sum(weights ** 2)

    U_new = jnp.sum(weights[:, None, None] * sampled_actions, axis=0)

    return U_new, ess


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
    spline = CubicSpline(np.linspace(0, t_total, len(waypoints)), waypoints)
    t = np.arange(int(freq * t_total)) / freq
    ref = np.hstack((spline(t), spline.derivative()(t)))
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
        self._horizon = 25
        self._n_samples = 512
        self._sigma = jnp.array([0.15, 0.15, 0.0, 0.05])*1.1
        self._lam = 0.05
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
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
        self._t_total = 18.75
        self._ref = build_reference(waypoints, self._t_total, self._freq, self._horizon + 1)
        self._tick_max = int(self._t_total * self._freq)

        self._update = jax.jit(mppi_update, static_argnames=("n_samples",))
        self._key = jax.random.key(0)
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._u_prev = self._U_nom[0]
        self._ess = float("nan")
        self._tick = 0
        self._finished = False

        self._update(
            self._key,
            self._state(obs),
            self._U_nom,
            self._u_prev,
            self._ref[: self._horizon + 1],
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._lam,
        )

    def _state(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Convert an observation to the planning state [pos, rpy, vel, drpy]."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
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
        ref = self._ref[self._tick : self._tick + self._horizon + 1]

        self._key, subkey = jax.random.split(self._key)
        U_new, ess = self._update(
            subkey,
            self._state(obs),
            self._U_nom,
            self._u_prev,
            ref,
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._lam,
        )
        self._ess = float(ess)

        self._u_prev = U_new[0]
        self._U_nom = jnp.concatenate((U_new[1:], U_new[-1:]))
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
