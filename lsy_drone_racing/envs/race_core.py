"""Core environment for drone racing simulations.

This module provides the shared logic for simulating drone racing environments. It defines a core
environment class that wraps our drone simulation, drone control, gate tracking, and collision
detection. The module serves as the base for both single-drone and multi-drone racing environments.

The environment is designed to be configurable, supporting:

* Different control modes (state or attitude)
* Customizable tracks with gates and obstacles
* Various randomization options for robust policy training
* Disturbance modeling for realistic flight conditions
* Vectorized execution for parallel training

This module is primarily used as a base for the higher-level environments in
[drone_race][lsy_drone_racing.envs.drone_race] and
[multi_drone_race][lsy_drone_racing.envs.multi_drone_race], which provide Gymnasium-compatible
interfaces for reinforcement learning, MPC and other control techniques.
"""

from __future__ import annotations

import copy as copy
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import crazyflow.sim.functional as F
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.sim import seed_sim, sync_sim2mjx, use_box_collision
from crazyflow.utils import leaf_replace
from drone_controllers.mellinger.params import ForceTorqueParams
from flax.struct import dataclass
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.randomize import (
    build_full_track_randomization_fn,
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
    randomize_gate_pos_fn,
    randomize_gate_rpy_fn,
    randomize_obstacle_pos_fn,
)
from lsy_drone_racing.envs.utils import gate_passed, load_track

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array, Device
    from ml_collections import ConfigDict
    from mujoco.mjx import Data
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region EnvData / Settings


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment.

    This dataclass stores the dynamic and static state of the environment that is not directly
    part of the physics simulation. It includes information about gate progress, drone status,
    and environment boundaries. Static variables are initialized once and do not change during the
    episode.

    Attributes:
        target_gate: Current target gate index for each drone in each environment
        gates_visited: Boolean flags indicating which gates have been visited by each drone
        obstacles_visited: Boolean flags indicating which obstacles have been detected
        last_drone_pos: Previous positions of drones, used for gate passing detection
        marked_for_reset: Flags indicating which environments need to be reset
        disabled_drones: Flags indicating which drones have crashed or are otherwise disabled
        contact_masks: Masks for contact detection between drones and objects
        pos_limit_low: Lower position limits for the environment
        pos_limit_high: Upper position limits for the environment
        gate_mj_ids: MuJoCo IDs for the gates
        obstacle_mj_ids: MuJoCo IDs for the obstacles
        max_episode_steps: Maximum number of steps per episode
        sensor_range: Range at which drones can detect gates and obstacles
    """

    # Dynamic variables
    target_gate: Array
    gates_visited: Array
    obstacles_visited: Array
    last_drone_pos: Array
    marked_for_reset: Array
    disabled_drones: Array
    steps: Array
    takeoff_pos: Array
    # Track gate pos, quat and obstacle pos outside of mjx_data to allow fusing mjx_data into the
    # contact function
    gates_pos: Array
    gates_quat: Array
    obstacles_pos: Array
    # Nominal positions are dynamic in level 3 (random tracks)
    nominal_gates_pos: Array
    nominal_gates_quat: Array
    nominal_obstacles_pos: Array
    # sim_data is stored in the env data to allow passing a single tree on which we can operate
    sim_data: SimData
    # Static variables
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    max_episode_steps: Array
    sensor_range: Array

    @staticmethod
    def create(
        n_gates: int,
        n_obstacles: int,
        contact_masks: Array,
        max_episode_steps: int,
        sensor_range: float,
        pos_limit_low: Array,
        pos_limit_high: Array,
        nominal_gates_pos: Array,
        nominal_gates_quat: Array,
        nominal_obstacles_pos: Array,
        sim_data: SimData,
        device: Device,
    ) -> EnvData:
        """Create a new environment data struct with default values."""
        n_envs = sim_data.core.n_worlds
        n_drones = sim_data.core.n_drones
        tiled_gates_pos = jax.device_put(
            jp.tile(nominal_gates_pos[None, ...], (n_envs, 1, 1)), device
        )
        tiled_gates_quat = jax.device_put(
            jp.tile(nominal_gates_quat[None, ...], (n_envs, 1, 1)), device
        )
        tiled_obstacles_pos = jax.device_put(
            jp.tile(nominal_obstacles_pos[None, ...], (n_envs, 1, 1)), device
        )
        return EnvData(
            target_gate=jp.zeros((n_envs, n_drones), dtype=int, device=device),
            gates_visited=jp.zeros((n_envs, n_drones, n_gates), dtype=bool, device=device),
            obstacles_visited=jp.zeros((n_envs, n_drones, n_obstacles), dtype=bool, device=device),
            last_drone_pos=jp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            marked_for_reset=jp.zeros(n_envs, dtype=bool, device=device),
            disabled_drones=jp.zeros((n_envs, n_drones), dtype=bool, device=device),
            contact_masks=jp.array(contact_masks, dtype=bool, device=device),
            steps=jp.zeros(n_envs, dtype=int, device=device),
            takeoff_pos=jp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            pos_limit_low=jp.array(pos_limit_low, dtype=np.float32, device=device),
            pos_limit_high=jp.array(pos_limit_high, dtype=np.float32, device=device),
            max_episode_steps=jp.array([max_episode_steps], dtype=int, device=device),
            gates_pos=tiled_gates_pos,
            gates_quat=tiled_gates_quat,
            obstacles_pos=tiled_obstacles_pos,
            nominal_gates_pos=tiled_gates_pos,
            nominal_gates_quat=tiled_gates_quat,
            nominal_obstacles_pos=tiled_obstacles_pos,
            sim_data=sim_data,
            sensor_range=jp.array([sensor_range], dtype=jp.float32, device=device),
        )


@dataclass
class EnvSettings:
    """Struct holding all configuration settings for the environment."""

    freq: int
    max_episode_steps: int
    pos_limit_low: Array
    pos_limit_high: Array
    camera: int | str
    cam_config: dict[str, int | list[float]]
    disturbances: dict[str, Callable[[Array, Array, Array], Array]]
    randomizations: dict[str, Callable[[Array, Array, Array], Array]]
    device: Device
    autoreset: bool = True  # Can be overridden by single env subclasses

    @staticmethod
    def create(
        freq: int,
        max_episode_steps: int,
        pos_limit_low: Array,
        pos_limit_high: Array,
        camera: int | str,
        cam_config: dict[str, int | list[float]],
        disturbances: dict[str, Callable[[Array, Array, Array], Array]],
        randomizations: dict[str, Callable[[Array, Array, Array], Array]],
        device: Device,
        autoreset: bool = True,
    ) -> EnvSettings:
        """Create a new environment settings struct from a configuration dictionary."""
        return EnvSettings(
            freq=freq,
            max_episode_steps=max_episode_steps,
            pos_limit_low=jp.array(pos_limit_low, dtype=jp.float32, device=device),
            pos_limit_high=jp.array(pos_limit_high, dtype=jp.float32, device=device),
            camera=camera,
            cam_config=cam_config,
            disturbances=disturbances,
            randomizations=randomizations,
            device=device,
            autoreset=autoreset,
        )


def build_action_space(control_mode: Literal["state", "attitude"], drone_model: str) -> spaces.Box:
    """Create the action space for the environment.

    Args:
        control_mode: The control mode to use. Either "state" for full-state control
            or "attitude" for attitude control.
        drone_model: Drone model of the environment.

    Returns:
        A Box space representing the action space for the specified control mode.
    """
    if control_mode == "state":
        return spaces.Box(low=-np.inf, high=np.inf, shape=(13,))
    if control_mode == "attitude":
        params = ForceTorqueParams.load(drone_model)
        thrust_min, thrust_max = params.thrust_min * 4, params.thrust_max * 4
        return spaces.Box(
            np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
            np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
        )
    raise ValueError(f"Invalid control mode: {control_mode}")


def build_observation_space(n_gates: int, n_obstacles: int) -> spaces.Dict:
    """Create the observation space for the environment.

    The observation space is a dictionary containing the drone state, gate information,
    and obstacle information.

    Args:
        n_gates: Number of gates in the environment.
        n_obstacles: Number of obstacles in the environment.
    """
    obs_spec = {
        "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "quat": spaces.Box(low=-1, high=1, shape=(4,)),
        "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "target_gate": spaces.Discrete(n_gates, start=-1),
        "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
        "gates_quat": spaces.Box(low=-1, high=1, shape=(n_gates, 4)),
        "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
        "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
        "obstacles_visited": spaces.Box(low=0, high=1, shape=(n_obstacles,), dtype=bool),
    }
    return spaces.Dict(obs_spec)


# region Core Env


class RaceCoreEnv:
    """The core environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It supports various configuration options for
    randomization, disturbances, and physics models.

    The environment provides:

    * A customizable track with gates and obstacles
    * Configurable simulation and control frequencies
    * Support for different physics models (e.g., identified dynamics, analytical dynamics)
    * Randomization of drone properties and initial conditions
    * Disturbance modeling for realistic flight conditions
    * Symbolic expressions for advanced control techniques (optional)

    The environment tracks the drone's progress through the gates and provides termination
    conditions based on gate passages and collisions.

    The observation space is a dictionary with the following keys:

    * pos: Drone position
    * quat: Drone orientation as a quaternion (x, y, z, w)
    * vel: Drone linear velocity
    * ang_vel: Drone angular velocity
    * gates_pos: Positions of the gates
    * gates_quat: Orientations of the gates
    * gates_visited: Flags indicating if the drone already was/ is in the sensor range of the
      gates and the true position is known
    * obstacles_pos: Positions of the obstacles
    * obstacles_visited: Flags indicating if the drone already was/ is in the sensor range of the
      obstacles and the true position is known
    * target_gate: The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller, or a desired collective thrust and attitude command [collective thrust,
    roll, pitch, yaw].
    """

    gate_spec_path = Path(__file__).parent / "assets/gate.xml"
    obstacle_spec_path = Path(__file__).parent / "assets/obstacle.xml"

    def __init__(
        self,
        n_envs: int,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        track: ConfigDict,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int | None = None,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the DroneRacingEnv.

        Args:
            n_envs: Number of worlds in the vectorized environment.
            n_drones: Number of drones.
            freq: Environment step frequency.
            sim_config: Configuration dictionary for the simulation.
            sensor_range: Sensor range for gate and obstacle detection.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: None / -1 for a generated seed or the random seed directly.
            max_episode_steps: Maximum number of steps per episode. Needs to be tracked manually for
                vectorized environments.
            device: Device used for the environment and the simulation.
        """
        super().__init__()
        # 1) Sanitize args
        if sim_config.freq % freq != 0:
            raise ValueError(f"({sim_config.freq=}) is no multiple of ({freq=})")
        assert seed is None or isinstance(seed, int), f"Unexpected seed type: {type(seed)}"

        # 2) Set seeds for reproducibility
        # TOML does not support None values, so we use -1 to indicate that a random seed should be
        # generated. This is equivalent to seed=None, but allows us to use TOML for configuration.
        seed = None if seed == -1 else seed
        # JAX must have an integer key, so we generate one from numpy. If the key was None / -1, we
        # get a randomly-seeded numpy rng, and hence a random JAX rng key. If a seed was given, we
        # get a reproducible numpy rng, which always generates the same JAX key for the same seed.
        rng_key = int(np.random.default_rng(seed).integers(0, 2**32 - 1))

        # 3) Create the simulation
        self.sim = Sim(
            n_worlds=n_envs,
            n_drones=n_drones,
            physics=sim_config.physics,
            drone_model=sim_config.drone_model,
            control=control_mode,
            freq=sim_config.freq,
            state_freq=freq,
            attitude_freq=sim_config.attitude_freq,
            rng_key=rng_key,
            device=device,
            xml_path=Path(p) if (p := getattr(sim_config, "xml_path", None)) else None,
        )
        self._load_track_into_sim(track)
        use_box_collision(self.sim, True)

        # 4) Create the environment data and settings
        self.track = track
        gates, obstacles, drones = load_track(track)
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        contact_masks = _load_contact_masks(self.sim)
        specs = {} if disturbances is None else disturbances
        disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        self.settings = EnvSettings.create(
            freq=freq,
            max_episode_steps=max_episode_steps,
            pos_limit_low=[-3, -3, 0.0],
            pos_limit_high=[3, 3, 2.5],
            camera=sim_config.camera,
            cam_config=sim_config.cam_config[0],
            disturbances=disturbances,
            randomizations=randomizations,
            device=jax.devices(device)[0],
            autoreset=True,
        )
        self.data = EnvData.create(
            n_gates=n_gates,
            n_obstacles=n_obstacles,
            contact_masks=contact_masks,
            max_episode_steps=max_episode_steps,
            sensor_range=sensor_range,
            pos_limit_low=[-3, -3, 0.0],
            pos_limit_high=[3, 3, 2.5],
            nominal_gates_pos=gates.nominal_pos,
            nominal_gates_quat=gates.nominal_quat,
            nominal_obstacles_pos=obstacles.nominal_pos,
            sim_data=self.sim.data,
            device=self.settings.device,
        )

        # 5) Generate functions
        self._setup_sim(randomizations, drones)
        self._reset = self.build_reset_fn()
        self._step = self.build_step_fn()
        self._render_sync = self.build_render_sync_fn()

    @staticmethod
    def _reset(
        data: EnvData, seed: int | None = None, mask: Array | None = None
    ) -> tuple[EnvData, tuple[dict[str, Array], dict]]:
        """Reset the environment.

        Note:
            This function gets generated on initialization by `build_reset_fn`, which compiles the
            reset logic into a single JAX kernel for efficiency. To see the reset logic, check the
            builder function.

        Args:
            data: The environment data.
            seed: Random seed.
            mask: Mask of worlds to reset.

        Returns:
            The environment data, and a tuple of observation and info.
        """

    @staticmethod
    def _step(
        data: EnvData, action: Array
    ) -> tuple[EnvData, tuple[dict[str, Array], float, bool, bool, dict]]:
        """Take one step in the environment.

        Note:
            This function gets generated on initialization by `build_step_fn`, which compiles the
            step logic into a single JAX kernel for efficiency. To see the step logic, check the
            builder function.

        Args:
            data: The environment data.
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """

    def render(self):
        """Render the environment."""
        if not self.data.sim_data.core.mjx_synced:
            self.data, self.sim.mjx_data = self._render_sync(self.data, self.sim.mjx_data)
        self.sim.render(camera=self.settings.camera, cam_config=self.settings.cam_config)

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()

    @property
    def drone_mass(self) -> NDArray[np.floating]:
        """The mass of the drones in the environment."""
        return np.asarray(self.sim.default_data.params.mass[..., 0])

    @property
    def mocap_ids(self) -> tuple[list[int], list[int]]:
        """The MuJoCo mocap IDs for the gates and obstacles."""
        m = self.sim.mj_model
        n_gates, n_obstacles = self.data.gates_pos.shape[1], self.data.obstacles_pos.shape[1]
        gate_ids = [int(m.body(f"gate:{i}").mocapid.squeeze()) for i in range(n_gates)]
        obstacle_ids = [int(m.body(f"obstacle:{i}").mocapid.squeeze()) for i in range(n_obstacles)]
        return gate_ids, obstacle_ids

    def build_reset_fn(
        self,
    ) -> Callable[
        [EnvData, int | None, Array | None], tuple[EnvData, tuple[dict[str, Array], dict]]
    ]:
        """Build a function that resets the environment data and simulation data."""
        sim_reset_fn = self.sim.build_reset_fn()
        default_sim_data = self.sim.default_data
        randomize_track = build_track_randomization_fn(
            self.settings.randomizations, track=self.track
        )

        @jax.jit
        def reset(
            data: EnvData, seed: int | None = None, mask: Array | None = None
        ) -> tuple[EnvData, tuple[dict[str, Array], dict]]:
            sim_data = data.sim_data
            if seed is not None:
                sim_data = seed_sim(sim_data, seed, sim_data.core.device)
            key, subkey = jax.random.split(sim_data.core.rng_key, 2)
            sim_data = sim_data.replace(core=sim_data.core.replace(rng_key=key))
            # Randomization of the drone is compiled into the sim reset pipeline, so we don't need
            # to explicitly do it here
            sim_data = sim_reset_fn(sim_data, default_sim_data, mask)
            data = data.replace(sim_data=sim_data)
            data = randomize_track(data, mask, subkey)
            data = _reset_env_data(data, mask)
            return data, (obs(data), {})

        return reset

    def build_step_fn(self) -> Callable[[EnvData, Array], EnvData]:
        """Build a function that steps the environment."""
        apply_action_fn = self.build_apply_action_fn()
        contact_check_fn = self.build_contact_check_fn()
        sim_step_fn = self.sim.build_step_fn()
        reset_fn = self._reset
        autoreset = self.settings.autoreset
        max_episode_steps = self.settings.max_episode_steps

        @jax.jit
        def step(data: EnvData, action: Array) -> EnvData:
            # 1) Save marked_for_reset before it is updated. Autoresets need to be based on the
            # previous flags, not the ones from the current step
            marked_for_reset = data.marked_for_reset
            # 2) Register the commanded action in the sim controllers
            data = apply_action_fn(action, data)
            # 3) Step the simulation for the number of sim steps per env step
            n_steps = data.sim_data.core.freq // self.settings.freq
            sim_data = sim_step_fn(data.sim_data, n_steps)
            data = data.replace(sim_data=sim_data)
            # 4) Apply environment logic
            data = _update_disabled_drones(data, contact_check_fn(data))
            data = _warp_disabled_drones(data)  # Prevent interference with alive drones
            data = _update_visited_objects(data)
            data = _update_target_gates(data)
            data = _mark_drones_for_reset(data)
            data = data.replace(steps=data.steps + 1)
            # 5) Auto-reset envs if running with autoreset enabled. Disable for single-world envs
            if autoreset:
                # Only run the reset if at least one env is marked for reset
                data, _ = jax.lax.cond(
                    marked_for_reset.any(),
                    reset_fn,
                    lambda data, *_: (data, (obs(data), {})),
                    data,
                    None,
                    marked_for_reset,
                )
            _truncated = truncated(data, max_episode_steps)
            return data, (obs(data), reward(data), terminated(data), _truncated, {})

        return step

    def build_apply_action_fn(self) -> Callable[[Array, EnvData, EnvSettings], EnvData]:
        """Build a function that applies the action to the simulation."""
        action_space = build_action_space(self.sim.control, self.sim.drone_model)
        if self.sim.control == "state":
            ctrl_fn = F.state_control
        elif self.sim.control == "attitude":
            ctrl_fn = F.attitude_control
        else:
            raise ValueError(f"Unsupported control mode: {self.sim.control}")
        disturbances = self.settings.disturbances

        def apply_action(action: Array, data: EnvData) -> None:
            """Apply the commanded state action to the simulation."""
            action = action.reshape((data.sim_data.core.n_worlds, data.sim_data.core.n_drones, -1))
            action = jp.clip(action, action_space.low, action_space.high)
            if "action" in disturbances:
                key, subkey = jax.random.split(data.sim_data.core.rng_key)
                action += disturbances["action"](subkey, action.shape)
                sim_data = data.sim_data.replace(core=data.sim_data.core.replace(rng_key=key))
                data = data.replace(sim_data=sim_data)
            return data.replace(sim_data=ctrl_fn(data.sim_data, action))

        return apply_action

    def build_contact_check_fn(self) -> Callable[[EnvData], Array]:
        """Build a function that checks for contacts between drones and gates/obstacles.

        Note:
            Passing the full mjx_data into jit-compiled functions is expensive because the tree
            contains many elements and is flattened **before** the jit boundary. To avoid this cost,
            we fuse mjx_data into the contact_check function and only sync the gate and obstacle
            poses **inside** the function. This way, we can only pass EnvData, which is faster to
            canonicalize.
        """
        contact_masks = _load_contact_masks(self.sim)
        gate_ids, obstacle_ids = self.mocap_ids
        _mjx_data = self.sim.mjx_data

        def check_contacts(data: EnvData) -> Array:
            """Check for contacts between drones and gates/obstacles."""
            mocap_pos, mocap_quat = _mjx_data.mocap_pos, _mjx_data.mocap_quat
            mocap_pos = mocap_pos.at[..., gate_ids, :].set(data.gates_pos)
            mocap_quat = mocap_quat.at[..., gate_ids, :].set(jp.roll(data.gates_quat, 1, axis=-1))
            mocap_pos = mocap_pos.at[..., obstacle_ids, :].set(data.obstacles_pos)
            mjx_data = _mjx_data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)
            # Sync changes to MuJoCo and perform a collision check
            _, mjx_data = sync_sim2mjx(data.sim_data, mjx_data, self.sim.mjx_model)
            contacts = mjx_data._impl.contact.dist < 0
            return jp.any(contacts[:, None, :] & contact_masks, axis=-1)

        return check_contacts

    def build_render_sync_fn(self) -> Callable[[EnvData, Data], Data]:
        """Build a function that syncs the environment data with the MuJoCo data for rendering."""
        gate_ids, obstacle_ids = self.mocap_ids
        mjx_model = self.sim.mjx_model

        @jax.jit
        def render_sync(data: EnvData, mjx_data: Data) -> tuple[EnvData, Data]:
            """Sync the environment data with the MuJoCo data for rendering."""
            gates_pos = data.gates_pos
            gates_quat = data.gates_quat
            obstacles_pos = data.obstacles_pos
            mocap_pos, mocap_quat = mjx_data.mocap_pos, mjx_data.mocap_quat
            mocap_pos = mocap_pos.at[..., gate_ids, :].set(gates_pos)
            mocap_quat = mocap_quat.at[..., gate_ids, :].set(jp.roll(gates_quat, 1, axis=-1))
            mocap_pos = mocap_pos.at[..., obstacle_ids, :].set(obstacles_pos)
            mjx_data = mjx_data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)
            sim_data, mjx_data = sync_sim2mjx(data.sim_data, mjx_data, mjx_model)
            return data.replace(sim_data=sim_data), mjx_data

        return render_sync

    def _render_sync(self, data: EnvData, mjx_data: Data) -> tuple[EnvData, Data]:
        """Sync the environment data with the MuJoCo data for rendering.

        Note:
            This function is built by `build_render_sync_fn` and compiled into a JAX kernel for
            efficiency. To see the sync logic, check the builder function.
        """

    def _setup_sim(self, randomizations: dict, drones: dict[str, Any]):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        # Set the initial drone states
        pos = self.sim.data.states.pos.at[...].set(drones["pos"])
        quat = self.sim.data.states.quat.at[...].set(drones["quat"])
        vel = self.sim.data.states.vel.at[...].set(drones["vel"])
        ang_vel = self.sim.data.states.ang_vel.at[...].set(drones["ang_vel"])
        states = self.sim.data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
        self.sim.data = self.sim.data.replace(states=states)
        self.sim.build_default_data()
        # Build the reset randomizations and disturbances into the sim itself
        self.sim.reset_pipeline = self.sim.reset_pipeline + (build_drone_reset_fn(randomizations),)
        self.sim.build_reset_fn()
        if dist := self.settings.disturbances.get("dynamics"):
            disturbance_fn = build_dynamics_disturbance_fn(dist)
            self.sim.step_pipeline = (
                self.sim.step_pipeline[:2] + (disturbance_fn,) + self.sim.step_pipeline[2:]
            )
            self.sim.build_step_fn()

    def _load_track_into_sim(self, track: ConfigDict):
        """Load the track into the simulation."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        frame = self.sim.spec.worldbody.add_frame()
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        for i in range(n_gates):
            gate_body = gate_spec.body("gate")
            if gate_body is None:
                raise ValueError("Gate body not found in gate spec")
            gate = frame.attach_body(gate_body, "", f":{i}")
            gate.pos = track.gates[i]["pos"]
            # Convert from scipy order to MuJoCo order
            gate.quat = R.from_euler("xyz", track.gates[i]["rpy"]).as_quat(scalar_first=True)
            gate.mocap = True  # Make mocap to modify the position of static bodies during sim
        for i in range(n_obstacles):
            obstacle_body = obstacle_spec.body("obstacle")
            if obstacle_body is None:
                raise ValueError("Obstacle body not found in obstacle spec")
            obstacle = frame.attach_body(obstacle_body, "", f":{i}")
            obstacle.pos = track.obstacles[i]["pos"]
            obstacle.mocap = True
        self.sim.build_mjx()


# region functional


def obs(data: EnvData) -> dict[str, Array]:
    """Return the observation of the environment."""
    mask = data.gates_visited[..., None]
    sensor_gates_pos = jp.where(mask, data.gates_pos[:, None], data.nominal_gates_pos[:, None])
    sensor_gates_quat = jp.where(mask, data.gates_quat[:, None], data.nominal_gates_quat[:, None])
    mask = data.obstacles_visited[..., None]
    sensor_obstacles_pos = jp.where(
        mask, data.obstacles_pos[:, None], data.nominal_obstacles_pos[:, None]
    )
    return {
        "pos": data.sim_data.states.pos,
        "quat": data.sim_data.states.quat,
        "vel": data.sim_data.states.vel,
        "ang_vel": data.sim_data.states.ang_vel,
        "target_gate": data.target_gate,
        "gates_pos": sensor_gates_pos,
        "gates_quat": sensor_gates_quat,
        "gates_visited": data.gates_visited,
        "obstacles_pos": sensor_obstacles_pos,
        "obstacles_visited": data.obstacles_visited,
    }


def reward(data: EnvData) -> Array:
    """Compute the reward for the current state.

    Note:
        The current sparse reward function will most likely not work directly for training an
        agent. If you want to use reinforcement learning, you will need to define your own
        reward function.

    Returns:
        Reward for the current state.
    """
    return -1.0 * (data.target_gate == -1)  # Implicit float conversion


def terminated(data: EnvData) -> Array:
    """Check if the episode is terminated, i.e., if all drones are disabled."""
    return data.disabled_drones


def truncated(data: EnvData, max_episode_steps: int) -> Array:
    """Array of booleans indicating if the episode is truncated."""
    n_drones = data.sim_data.core.n_drones
    return jp.tile((data.steps >= max_episode_steps)[..., None], (1, n_drones))


@jax.jit
def _reset_env_data(data: EnvData, mask: Array | None = None) -> EnvData:
    """Reset auxiliary variables of the environment data."""
    drone_pos = data.sim_data.states.pos
    mask = jp.ones(data.steps.shape, dtype=bool) if mask is None else mask
    target_gate = jp.where(mask[..., None], 0, data.target_gate)
    last_drone_pos = jp.where(mask[..., None, None], drone_pos, data.last_drone_pos)
    disabled_drones = jp.where(mask[..., None], False, data.disabled_drones)
    steps = jp.where(mask, 0, data.steps)
    # Check which gates are in range of the drone
    dpos = drone_pos[..., None, :2] - data.gates_pos[:, None, :, :2]
    gates_visited = jp.linalg.norm(dpos, axis=-1) < data.sensor_range
    gates_visited = jp.where(mask[..., None, None], gates_visited, data.gates_visited)
    # And which obstacles are in range
    obstacles_pos = data.obstacles_pos
    dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
    obstacles_visited = jp.linalg.norm(dpos, axis=-1) < data.sensor_range
    obstacles_visited = jp.where(mask[..., None, None], obstacles_visited, data.obstacles_visited)
    return data.replace(
        target_gate=target_gate,
        last_drone_pos=last_drone_pos,
        disabled_drones=disabled_drones,
        gates_visited=gates_visited,
        obstacles_visited=obstacles_visited,
        steps=steps,
        takeoff_pos=jp.where(mask[..., None, None], drone_pos, data.takeoff_pos),
        marked_for_reset=jp.where(mask, False, data.marked_for_reset),
    )


def _update_disabled_drones(data: EnvData, contacts: Array) -> EnvData:
    """Update which drones are disabled based on their position and contacts."""
    return data.replace(disabled_drones=_disabled_drones(data.sim_data.states.pos, contacts, data))


def _update_visited_objects(data: EnvData) -> EnvData:
    """Update which gates and obstacles are or have been in range of the drone."""
    drone_pos = data.sim_data.states.pos
    dpos = drone_pos[..., None, :2] - data.gates_pos[:, None, :, :2]
    gates_visited = data.gates_visited | (jp.linalg.norm(dpos, axis=-1) < data.sensor_range)
    dpos = drone_pos[..., None, :2] - data.obstacles_pos[:, None, :, :2]
    obstacles_visited = data.obstacles_visited | (jp.linalg.norm(dpos, axis=-1) < data.sensor_range)
    return data.replace(gates_visited=gates_visited, obstacles_visited=obstacles_visited)


def _update_target_gates(data: EnvData) -> EnvData:
    """Update the target gate index based on the current target gate and the number of gates."""
    n_gates = data.gates_pos.shape[1]
    gates_pos, gates_quat = data.gates_pos, data.gates_quat
    drone_pos = data.sim_data.states.pos
    gate_pos = gates_pos[jp.arange(gates_pos.shape[0])[:, None], data.target_gate % n_gates]
    gate_quat = gates_quat[jp.arange(gates_quat.shape[0])[:, None], data.target_gate % n_gates]
    passed = gate_passed(drone_pos, data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45))
    # Update the target gate index. Increment by one if drones have passed a gate
    target_gate = data.target_gate + passed * ~data.disabled_drones
    target_gate = jp.where(target_gate >= n_gates, -1, target_gate)
    return data.replace(target_gate=target_gate, last_drone_pos=data.sim_data.states.pos)


def _mark_drones_for_reset(data: EnvData) -> EnvData:
    """Mark drones for reset if they are disabled or have reached the max episode steps."""
    truncated = data.steps >= data.max_episode_steps
    marked_for_reset = jp.all(data.disabled_drones | truncated[..., None], axis=-1)
    return data.replace(marked_for_reset=marked_for_reset)


def _load_contact_masks(sim: Sim) -> Array:
    """Load contact masks for the simulation that zero out irrelevant contacts per drone."""
    sim.contacts()  # Trigger initial contact information computation
    contact = sim.mjx_data._impl.contact
    n_contacts = len(contact.geom1[0])
    masks = np.zeros((sim.n_drones, n_contacts), dtype=bool)
    # We only need one world to create the mask
    geom1, geom2 = (contact.geom1[0], contact.geom2[0])
    for i in range(sim.n_drones):
        geom_start = sim.mj_model.body_geomadr[sim.mj_model.body(f"drone:{i}").id]
        geom_count = sim.mj_model.body_geomnum[sim.mj_model.body(f"drone:{i}").id]
        geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
        geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
        masks[i, :] = geom1_valid | geom2_valid
    geom_start = sim.mj_model.body_geomadr[sim.mj_model.body("world").id]
    geom_count = sim.mj_model.body_geomnum[sim.mj_model.body("world").id]
    geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
    geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
    floor_mask = geom1_valid | geom2_valid
    masks = masks & ~floor_mask  # Disable contacts with the floor

    masks = np.tile(masks[None, ...], (sim.n_worlds, 1, 1))
    return masks


def _warp_disabled_drones(data: EnvData) -> EnvData:
    """Warp the disabled drones below the ground."""
    pos = jax.numpy.where(data.disabled_drones[..., None], -1, data.sim_data.states.pos)
    sim_data = data.sim_data.replace(states=data.sim_data.states.replace(pos=pos))
    return data.replace(sim_data=sim_data)


def _disabled_drones(pos: Array, contacts: Array, data: EnvData) -> Array:
    """Check which drones are disabled based on their position and contacts."""
    disabled = data.disabled_drones
    not_in_platform = jp.any(pos[..., :2] < data.takeoff_pos[..., :2] - 0.02, axis=-1)
    not_in_platform |= jp.any(pos[..., :2] > data.takeoff_pos[..., :2] + 0.02, axis=-1)
    disabled = disabled | jp.any(pos < data.pos_limit_low, axis=-1) & not_in_platform
    disabled = disabled | jp.any(pos > data.pos_limit_high, axis=-1)
    disabled = disabled | (data.target_gate == -1)
    contacts = jp.any(contacts[:, :, None] & data.contact_masks, axis=-1)
    disabled = disabled | contacts
    return disabled


# region Factories


def rng_spec2fn(fn_spec: dict) -> Callable:
    """Convert a function spec to a wrapped and scaled function from jax.random."""
    offset, scale = np.array(fn_spec.get("offset", 0)), np.array(fn_spec.get("scale", 1))
    kwargs = fn_spec.get("kwargs", {})
    if "shape" in kwargs:
        raise KeyError("Shape must not be specified for randomization functions.")
    kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    jax_fn = partial(getattr(jax.random, fn_spec["fn"]), **kwargs)

    def random_fn(*args: Any, **kwargs: Any) -> Array:
        return jax_fn(*args, **kwargs) * scale + offset

    return random_fn


def build_drone_reset_fn(randomizations: dict) -> Callable[[SimData, Array], SimData]:
    """Build the reset hook for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "drone_pos":
                randomization_fns += (randomize_drone_pos_fn(rng),)
            case "drone_rpy":
                randomization_fns += (randomize_drone_quat_fn(rng),)
            case "drone_mass":
                randomization_fns += (randomize_drone_mass_fn(rng),)
            case "drone_inertia":
                randomization_fns += (randomize_drone_inertia_fn(rng),)
            case "gate_pos" | "gate_rpy" | "obstacle_pos":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    def reset_fn(data: SimData, mask: Array) -> SimData:
        for randomize_fn in randomization_fns:
            data = randomize_fn(data, mask)
        return data

    return reset_fn


def build_track_randomization_fn(
    randomizations: dict, track: ConfigDict
) -> Callable[[EnvData, Array, jax.random.PRNGKey], EnvData]:
    """Build the track randomization function for the simulation."""
    randomization_fns = ()

    if track.randomize:
        random_layout_fn = build_full_track_randomization_fn(
            [gate["pos"][2] for gate in track.gates],
            [obstacle["pos"][2] for obstacle in track.obstacles],
            track.safety_limits.pos_limit_low,
            track.safety_limits.pos_limit_high,
        )
        randomization_fns += (random_layout_fn,)

    for target, rng in sorted(randomizations.items()):
        match target:
            case "gate_pos":
                randomization_fns += (randomize_gate_pos_fn(rng),)
            case "gate_rpy":
                randomization_fns += (randomize_gate_rpy_fn(rng),)
            case "obstacle_pos":
                randomization_fns += (randomize_obstacle_pos_fn(rng),)
            case "drone_pos" | "drone_rpy" | "drone_mass" | "drone_inertia":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    def track_randomization(data: EnvData, mask: Array, key: jax.random.PRNGKey) -> EnvData:
        # Reset to default track positions first
        data = leaf_replace(
            data,
            mask,
            gates_pos=data.gates_pos.at[...].set(data.nominal_gates_pos),
            gates_quat=data.gates_quat.at[...].set(data.nominal_gates_quat),
            obstacles_pos=data.obstacles_pos.at[...].set(data.nominal_obstacles_pos),
        )
        keys = jax.random.split(key, len(randomization_fns))
        for key, randomize_fn in zip(keys, randomization_fns, strict=True):
            data = randomize_fn(data, mask, key)
        return data

    return track_randomization


def build_dynamics_disturbance_fn(
    fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData], SimData]:
    """Build the dynamics disturbance function for the simulation."""

    def dynamics_disturbance(data: SimData) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        states = data.states
        states = states.replace(force=fn(subkey, states.force.shape))  # World frame
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return dynamics_disturbance
