"""Level 2 observations, actions, rewards, normalization and the SBX VecEnv adapter.

Actor and Critic use the same sensor-limited observation. The official environment
handles physics, randomization, collisions and directional gate crossing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import Array
from jax.scipy.spatial.transform import Rotation
from stable_baselines3.common.vec_env import VecEnv

from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig, build_segment_reset
from lsy_drone_racing.control.rl_single_l2.reward import (
    RewardConfig,
    SparseRewardConfig,
    compute_reward,
    compute_sparse_reward,
    reward_components,
)

OBS_LAYOUTS = ("current", "paper")
GATE_INNER_HALF_WIDTH = 0.2  # Level 2 gates have a 0.4 m square inner opening.


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed feature layout and running-normalization parameters.

    layout="current" is the original feature vector. layout="paper" follows
    Pasumarti et al. (2026): body velocity, attitude, current/next gate corners in
    the body frame and global position; gate_lookahead is then unused and the
    nearest-obstacle block is appended only when paper_obstacles is set.
    """

    gate_lookahead: int = 2
    obstacle_count: int = 4
    clip: float = 10.0
    epsilon: float = 1e-8
    layout: str = "current"
    paper_obstacles: bool = False

    def __post_init__(self) -> None:
        """Reject layouts that cannot represent the target gate."""
        if self.gate_lookahead < 1 or self.obstacle_count < 0:
            raise ValueError("gate_lookahead must be positive and obstacle_count nonnegative")
        if self.clip <= 0 or self.epsilon <= 0:
            raise ValueError("normalization clip and epsilon must be positive")
        if self.layout not in OBS_LAYOUTS:
            raise ValueError(f"layout must be one of {OBS_LAYOUTS}")

    @property
    def size(self) -> int:
        """Return the fixed flattened observation dimension."""
        if self.layout == "paper":
            obstacles = 5 * self.obstacle_count if self.paper_obstacles else 0
            return PAPER_BASE_SIZE + obstacles
        return 23 + 8 * self.gate_lookahead + 5 * self.obstacle_count


# Body velocity (3), rotation (9), current corners (12), next corners (12),
# next-gate valid flag (1), global position (3).
PAPER_BASE_SIZE = 40


@dataclass(frozen=True)
class ActionConfig:
    """Body-frame rotation increment per control step, in radians.

    The increment is ``alpha_max * tanh(|r| * gain / alpha_max)`` along the raw
    rotation r: it saturates at alpha_max_rad and has slope alpha_gain_rad at
    zero. The gain defaults to alpha_max_rad, the original ``alpha * tanh(|r|)``
    mapping, so legacy metadata ({"alpha_max_rad": a}) restores unchanged.
    Raising only alpha_max_rad widens the rate limit (alpha_max / dt) while
    keeping a trained policy's small-signal behaviour.
    """

    alpha_max_rad: float = 0.16
    alpha_gain_rad: float | None = None

    def __post_init__(self) -> None:
        """Check the rotation budget and fill in the default gain."""
        if self.alpha_gain_rad is None:
            object.__setattr__(self, "alpha_gain_rad", self.alpha_max_rad)
        object.__setattr__(self, "alpha_max_rad", float(self.alpha_max_rad))
        object.__setattr__(self, "alpha_gain_rad", float(self.alpha_gain_rad))
        if not 0 < self.alpha_max_rad <= np.pi:
            raise ValueError("alpha_max_rad must be in (0, pi]")
        if not 0 < self.alpha_gain_rad <= np.pi:
            raise ValueError("alpha_gain_rad must be in (0, pi]")


@jax.jit
def _batch_moments(batch: Array) -> tuple[Array, Array]:
    """Reduce one batch on its device without transferring the raw features."""
    return batch.mean(axis=0), batch.var(axis=0)


@jax.jit
def _merge_moments(
    mean: Array, var: Array, count: float, other_mean: Array, other_var: Array, other_count: float
) -> tuple[Array, Array]:
    """Merge parallel Welford moments using device-local float32 arithmetic."""
    total = count + other_count
    delta = other_mean - mean
    moment = var * count + other_var * other_count
    moment += delta**2 * count * other_count / total
    return mean + delta * other_count / total, jnp.maximum(moment / total, 0.0)


@jax.jit
def _normalize(batch: Array, mean: Array, var: Array, epsilon: float, clip: float) -> Array:
    """Apply frozen moments on the device holding the observation batch."""
    values = (batch - mean) * jax.lax.rsqrt(var + epsilon)
    return jnp.clip(values, -clip, clip).astype(jnp.float32)


class RunningNormalizer:
    """Merge raw-observation moments with a numerically stable parallel update.

    The adapter accumulates pending moments separately, then merges them between
    PPO rollouts. No clipped/normalized observations are inverted to estimate stats.
    """

    def __init__(self, size: int, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        """Initialize an identity transform before any samples are available."""
        self.mean = jnp.zeros(size, dtype=jnp.float32)
        self.var = jnp.ones(size, dtype=jnp.float32)
        self.count = 0.0
        self.epsilon = float(epsilon)
        self.clip = float(clip)

    def update(self, batch: Array) -> None:
        """Add a raw batch without storing its individual samples."""
        batch = jnp.asarray(batch, dtype=jnp.float32).reshape(-1, self.mean.size)
        if batch.shape[0] == 0:
            return
        mean, var = _batch_moments(batch)
        self._merge(mean, var, float(batch.shape[0]))

    def _merge(self, mean: Array, var: Array, count: float) -> None:
        if count == 0:
            return
        if self.count == 0:
            self.mean, self.var, self.count = mean, var, float(count)
            return
        self.mean, self.var = _merge_moments(self.mean, self.var, self.count, mean, var, count)
        # Keep the counter exact in Python; it does not read a device result.
        self.count += count

    def normalize(self, batch: Array) -> Array:
        """Apply frozen moments and clip only the final network input."""
        return _normalize(jnp.asarray(batch), self.mean, self.var, self.epsilon, self.clip)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible normalization state for training and deployment."""
        return {
            "mean": np.asarray(self.mean).tolist(),
            "var": np.asarray(self.var).tolist(),
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore validated moments, rejecting incompatible feature dimensions."""
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])
        epsilon = float(state.get("epsilon", self.epsilon))
        clip = float(state.get("clip", self.clip))
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("Normalizer observation dimension does not match this policy")
        if not np.isfinite(mean).all() or not np.isfinite(var).all() or (var < 0).any():
            raise ValueError("Normalizer moments must be finite with nonnegative variance")
        if not np.isfinite([count, epsilon, clip]).all() or count < 0 or min(epsilon, clip) <= 0:
            raise ValueError("Invalid normalizer count, epsilon or clip")
        self.mean = jax.device_put(mean.astype(np.float32), self.mean.device)
        self.var = jax.device_put(var.astype(np.float32), self.var.device)
        self.count = count
        self.epsilon, self.clip = epsilon, clip

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RunningNormalizer:
        """Construct a normalizer from a saved state without an environment."""
        result = cls(len(state["mean"]))
        result.load_state_dict(state)
        return result


def _as_batch(obs: dict[str, Any]) -> tuple[dict[str, Array], bool]:
    single = jnp.asarray(obs["pos"]).ndim == 1
    return {
        key: jnp.asarray(value)[None] if single else jnp.asarray(value)
        for key, value in obs.items()
    }, single


def _gate_corners_body(
    batch: dict[str, Array], slot: Array, world_to_body: Array, pos: Array
) -> tuple[Array, Array]:
    """Return a sequence slot's inner-opening corners in the body frame (n, 12).

    Corners lie at +/-GATE_INNER_HALF_WIDTH along the gate's local y and z axes
    (local x is the crossing normal). For reverse passes the local y axis is
    mirrored so the corner order is identical when seen from the approach side.
    Slots past the end of the sequence are zero padded and flagged invalid.
    """
    count = pos.shape[0]
    rows = jnp.arange(count)
    sequence = batch["gate_sequence"].astype(jnp.int32)
    valid = slot < sequence.shape[1]
    indices = jnp.minimum(slot, sequence.shape[1] - 1)
    gate_ids = sequence[rows, indices]
    direction = batch["gate_sequence_direction"][rows, indices].astype(jnp.float32)
    gate_rot = Rotation.from_quat(batch["gates_quat"][rows, gate_ids]).as_matrix()
    h = GATE_INNER_HALF_WIDTH
    local = jnp.array([[0.0, h, h], [0.0, -h, h], [0.0, -h, -h], [0.0, h, -h]])
    local = (
        local[None]
        * jnp.stack([jnp.ones_like(direction), direction, jnp.ones_like(direction)], axis=-1)[
            :, None
        ]
    )
    corners = jnp.einsum("bij,bkj->bki", gate_rot, local) + batch["gates_pos"][rows, gate_ids, None]
    corners = jnp.einsum("bij,bkj->bki", world_to_body, corners - pos[:, None])
    return jnp.where(valid[:, None], corners.reshape(count, 12), 0), valid


def _obstacle_features(
    batch: dict[str, Array], world_to_body: Array, pos: Array, obstacle_count: int
) -> list[Array]:
    """Nearest obstacles' body-relative position (3), revealed flag (1), valid flag (1)."""
    count = pos.shape[0]
    rows = jnp.arange(count)
    features = []
    obstacles = batch["obstacles_pos"] - pos[:, None]
    order = jnp.argsort(jnp.linalg.norm(obstacles, axis=-1), axis=1, stable=True)
    for offset in range(obstacle_count):
        if offset >= obstacles.shape[1]:
            features.append(jnp.zeros((count, 5)))
            continue
        indices = order[:, offset]
        relative = jnp.einsum("bij,bj->bi", world_to_body, obstacles[rows, indices])
        features.append(
            jnp.concatenate(
                [relative, batch["obstacles_visited"][rows, indices, None], jnp.ones((count, 1))],
                axis=1,
            )
        )
    return features


def _encode_paper(batch: dict[str, Array], cfg: ObservationConfig) -> list[Array]:
    """Pasumarti et al. (2026) actor features, without opponent terms.

    Layout: body-frame linear velocity (3), body-to-world rotation (9), current
    target gate corners in the body frame (12), next gate corners (12), next-gate
    valid flag (1), global position (3); optionally the nearest-obstacle block.
    The valid flag is our addition: one episode is one lap, so the final target
    has no successor (the paper races several laps). Only sensor-masked poses
    are used.
    """
    pos = batch["pos"]
    count = pos.shape[0]
    rotation = Rotation.from_quat(batch["quat"]).as_matrix()
    world_to_body = rotation.transpose(0, 2, 1)
    passed = jnp.asarray(batch["n_gates_passed"], dtype=jnp.int32).reshape(count)
    current, _ = _gate_corners_body(batch, passed, world_to_body, pos)
    following, next_valid = _gate_corners_body(batch, passed + 1, world_to_body, pos)
    features = [
        jnp.einsum("bij,bj->bi", world_to_body, batch["vel"]),
        rotation.reshape(count, 9),
        current,
        following,
        next_valid[:, None].astype(jnp.float32),
        pos,
    ]
    if cfg.paper_obstacles:
        features += _obstacle_features(batch, world_to_body, pos, cfg.obstacle_count)
    return features


@partial(jax.jit, static_argnames=("config",))
def encode_obs(
    obs: dict[str, Any],
    previous_action: Array | None = None,
    config: ObservationConfig | None = None,
) -> Array:
    """Encode single/batched official observations into one shared feature vector.

    Layout "current": position (3), body-to-world rotation matrix (9), body-frame
    linear velocity (3), body angular velocity (3), previous executed attitude/thrust
    command (4), completed sequence fraction (1); then each upcoming gate's
    body-relative center (3), signed crossing normal (3), revealed flag (1),
    valid flag (1); finally nearest obstacles' body-relative positions (3),
    revealed flag (1), valid flag (1). Missing slots are zero padded.
    Layout "paper": see _encode_paper; previous_action is ignored.

    Unlike the reference, there is no privileged critic half, and no target_gate
    sentinel. Official n_gates_passed indexes gate_sequence, supporting repeated
    gates and reverse passes. Only official sensor-masked geometry is used.
    """
    cfg = config or ObservationConfig()
    batch, single = _as_batch(obs)
    pos = batch["pos"]
    count = pos.shape[0]
    if cfg.layout == "paper":
        features = _encode_paper(batch, cfg)
    else:
        features = _encode_current(batch, previous_action, cfg)
    encoded = jnp.concatenate(features, axis=1).astype(jnp.float32)
    if encoded.shape != (count, cfg.size):
        raise ValueError("Official observation produced an invalid feature shape")
    return encoded[0] if single else encoded


def _encode_current(
    batch: dict[str, Array], previous_action: Array | None, cfg: ObservationConfig
) -> list[Array]:
    """Original feature layout; see encode_obs."""
    pos = batch["pos"]
    count = pos.shape[0]
    rotation = Rotation.from_quat(batch["quat"]).as_matrix()
    world_to_body = rotation.transpose(0, 2, 1)
    passed = jnp.asarray(batch["n_gates_passed"], dtype=jnp.int32).reshape(count)
    sequence = batch["gate_sequence"].astype(jnp.int32)
    directions = batch["gate_sequence_direction"]
    previous = jnp.zeros((count, 4)) if previous_action is None else jnp.asarray(previous_action)
    previous = previous.reshape(count, 4)
    features = [
        pos,
        rotation.reshape(count, 9),
        jnp.einsum("bij,bj->bi", world_to_body, batch["vel"]),
        batch["ang_vel"],
        previous,
        (passed / sequence.shape[1])[:, None],
    ]
    rows = jnp.arange(count)
    for offset in range(cfg.gate_lookahead):
        slot = passed + offset
        valid = slot < sequence.shape[1]
        indices = jnp.minimum(slot, sequence.shape[1] - 1)
        gate_ids = sequence[rows, indices]
        centers = batch["gates_pos"][rows, gate_ids] - pos
        gate_rot = Rotation.from_quat(batch["gates_quat"][rows, gate_ids]).as_matrix()
        normals = gate_rot[:, :, 0] * directions[rows, indices, None]
        values = jnp.concatenate(
            [
                jnp.einsum("bij,bj->bi", world_to_body, centers),
                jnp.einsum("bij,bj->bi", world_to_body, normals),
                batch["gates_visited"][rows, gate_ids, None],
                jnp.ones((count, 1)),
            ],
            axis=1,
        )
        features.append(jnp.where(valid[:, None], values, 0))
    return features + _obstacle_features(batch, world_to_body, pos, cfg.obstacle_count)


def _as_euler_xyz(rotation: Rotation) -> Array:
    """Convert on-device while preserving signed pi endpoints before clipping.

    JAX Rotation.as_euler wraps +pi to -pi, unlike SciPy's NumPy path. Those
    equivalent angles produce opposite commands after the official +/-pi/2
    clipping. atan2 preserves the endpoint sign; at gimbal lock, set yaw to zero
    and absorb it into roll, following SciPy's XYZ convention.
    """
    matrix = rotation.as_matrix()
    x, y, z, w = jnp.moveaxis(rotation.as_quat(), -1, 0)
    # Preserve signed zero as well as tiny sine values at exact half turns.
    roll_sine = 2 * jnp.where(y * z == 0, w * x, w * x + y * z)
    yaw_sine = 2 * jnp.where(x * y == 0, w * z, w * z + x * y)
    horizontal = jnp.hypot(matrix[..., 0, 0], yaw_sine)
    pitch = jnp.arctan2(-matrix[..., 2, 0], horizontal)
    singular = horizontal <= 1e-7
    roll = jnp.where(
        singular,
        jnp.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
        jnp.arctan2(roll_sine, matrix[..., 2, 2]),
    )
    yaw = jnp.where(singular, 0, jnp.arctan2(yaw_sine, matrix[..., 0, 0]))
    return jnp.stack([roll, pitch, yaw], axis=-1)


def _rotation_increment(raw: Array, config: ActionConfig) -> Array:
    """Radially tanh-bound raw rotation components to a body-frame rotation vector."""
    magnitude = jnp.linalg.norm(raw[:, 1:], axis=1, keepdims=True)
    ratio = config.alpha_gain_rad / config.alpha_max_rad
    # Same expression as the original alpha * tanh(|r|) / |r| when ratio == 1.
    scale = jnp.tanh(magnitude * ratio) / jnp.maximum(magnitude * ratio, 1e-12)
    return raw[:, 1:] * scale * config.alpha_gain_rad


@partial(jax.jit, static_argnames=("config",))
def rotation_increment(raw_actions: Array, config: ActionConfig | None = None) -> Array:
    """Return the per-step body-frame rotation vector that decode_action applies.

    Dividing by the control period gives the implied commanded body rate, used by
    the sparse reward's command penalty. Shape (n, 3) for batches, (3,) otherwise.
    """
    cfg = config or ActionConfig()
    raw = jnp.asarray(raw_actions, dtype=jnp.float32)
    delta = _rotation_increment(raw.reshape(-1, 4), cfg)
    return delta[0] if raw.ndim == 1 else delta


@partial(jax.jit, static_argnames=("config",))
def decode_action(
    raw_actions: Array,
    quat: Array,
    thrust_min: float,
    thrust_max: float,
    config: ActionConfig | None = None,
) -> Array:
    """Map Gaussian [thrust_raw, rotation_x, rotation_y, rotation_z] to commands.

    A sigmoid maps thrust to the model's collective thrust limits (newtons).
    A radial tanh maps the rotation vector inside alpha_max_rad; right-multiplying
    the current xyzw quaternion applies that increment in the drone body frame.
    XYZ Euler angles are then clipped to the official attitude action space.
    The returned order is [roll, pitch, yaw, total_thrust], all angles in radians.
    The policy's Gaussian log probability always refers to its original raw draw;
    this physical action mapping is part of the environment, as in the reference.
    """
    cfg = config or ActionConfig()
    raw = jnp.asarray(raw_actions, dtype=jnp.float32)
    single = raw.ndim == 1
    raw = raw.reshape(-1, 4)
    orientation = jnp.asarray(quat).reshape(-1, 4)
    if orientation.shape[0] != raw.shape[0]:
        raise ValueError("Actions and orientations must have matching batches")
    delta = _rotation_increment(raw, cfg)
    desired = Rotation.from_quat(orientation) * Rotation.from_rotvec(delta)
    # The official attitude space limits all three Euler coordinates to +/- pi/2.
    angles = jnp.clip(_as_euler_xyz(desired), -jnp.pi / 2, jnp.pi / 2)
    sigmoid = jax.nn.sigmoid(raw[:, 0])
    thrust = thrust_min + (thrust_max - thrust_min) * sigmoid
    commands = jnp.column_stack([angles, thrust]).astype(jnp.float32)
    return commands[0] if single else commands


@partial(jax.jit, static_argnames=("config",))
def _observe_batch(
    obs: dict[str, Array],
    previous_action: Array,
    mean: Array,
    var: Array,
    config: ObservationConfig,
) -> tuple[Array, Array, Array]:
    """Fuse features, frozen normalization and raw batch moments on the device."""
    features = encode_obs(obs, previous_action, config)
    batch_mean, batch_var = _batch_moments(features)
    return (_normalize(features, mean, var, config.epsilon, config.clip), batch_mean, batch_var)


def _zero_reward_stats(components: tuple[str, ...], n_envs: int) -> dict[str, Any]:
    """Device-resident reward accumulators, drained once per rollout."""
    return {
        "sums": {key: jnp.zeros(()) for key in components},
        "episode": {key: jnp.zeros(n_envs) for key in components},
        "episode_totals": {key: jnp.zeros(()) for key in components},
        "episodes": jnp.zeros((), dtype=jnp.int32),
    }


@partial(jax.jit, static_argnames=("config", "action_config", "dt"))
def _transition_reward(
    previous: dict[str, Array],
    current: dict[str, Array],
    terminated: Array,
    truncated: Array,
    raw_actions: Array,
    stats: dict[str, Any],
    config: RewardConfig | SparseRewardConfig,
    action_config: ActionConfig,
    dt: float,
) -> tuple[Array, Array, dict[str, Any], dict[str, Array]]:
    """Compute reward, end flags and on-device metric totals before resetting worlds.

    Besides per-transition component sums, each world's running per-episode
    component sums are kept so completed episodes' totals (e.g. the summed sparse
    command penalty) can be logged without per-step host transfers.
    """
    passed = current["n_gates_passed"].reshape(-1)
    finished = passed >= current["gate_sequence"].shape[1]
    terminated = jnp.asarray(terminated, dtype=bool) | finished
    truncated = jnp.asarray(truncated, dtype=bool)
    if isinstance(config, SparseRewardConfig):
        delta = rotation_increment(raw_actions, action_config)
        reward, components = compute_sparse_reward(
            previous, current, terminated, truncated, delta, dt, None, config
        )
    else:
        reward, components = compute_reward(previous, current, terminated, truncated, config)
    done = terminated | truncated
    running = {key: stats["episode"][key] + value for key, value in components.items()}
    stats = {
        "sums": {key: stats["sums"][key] + value.sum() for key, value in components.items()},
        "episode": {key: jnp.where(done, 0.0, value) for key, value in running.items()},
        "episode_totals": {
            key: stats["episode_totals"][key] + jnp.where(done, value, 0.0).sum()
            for key, value in running.items()
        },
        "episodes": stats["episodes"] + done.sum(dtype=jnp.int32),
    }
    metrics = {
        "terminated": terminated,
        "truncated": truncated,
        "finished": finished,
        "passed": passed,
    }
    return reward, done, stats, metrics


@jax.jit
def _reset_previous_action(commands: Array, done: Array) -> Array:
    """Clear only completed worlds' action histories without a host round trip."""
    return jnp.where(done[:, None], 0, commands)


class Level2SBXEnv(VecEnv):
    """Adapt official vectorized simulation to SB3/SBX's immediate-reset protocol."""

    def __init__(
        self,
        config_path: str | Path,
        n_envs: int = 18432,
        seed: int = 42,
        device: str = "cpu",
        max_episode_steps: int = 1500,
        observation_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | SparseRewardConfig | None = None,
        reset_config: ResetConfig | None = None,
    ) -> None:
        """Build an attitude-mode Level 2 simulation and initialize adapter state."""
        from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
        from lsy_drone_racing.utils import load_config

        if n_envs < 1 or max_episode_steps < 1:
            raise ValueError("n_envs and max_episode_steps must be positive")
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_config(self.config_path)
        self.config.env.control_mode = "attitude"
        self.config.sim.render = False
        self.observation_config = observation_config or ObservationConfig()
        self.action_config = action_config or ActionConfig()
        self.reward_config = reward_config or RewardConfig()
        self.reset_config = reset_config or ResetConfig()
        self.max_episode_steps = int(max_episode_steps)
        self.seed_value = int(seed)
        self.render_mode = None
        self.jax_env = VecDroneRaceEnv(
            num_envs=n_envs,
            freq=self.config.env.freq,
            sim_config=self.config.sim,
            track=self.config.env.track,
            sensor_range=self.config.env.sensor_range,
            control_mode="attitude",
            disturbances=self.config.env.get("disturbances"),
            randomizations=self.config.env.get("randomizations"),
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.jax_env.settings = self.jax_env.settings.replace(autoreset=False)
        self.jax_env._step = self.jax_env.build_step_fn()
        self._segment_reset = (
            build_segment_reset(self.reset_config, self.jax_env.build_contact_check_fn())
            if self.reset_config.segment_init_prob
            else None
        )
        self.thrust_min = float(self.jax_env.single_action_space.low[3])
        self.thrust_max = float(self.jax_env.single_action_space.high[3])
        self.normalizer = RunningNormalizer(
            self.observation_config.size,
            epsilon=self.observation_config.epsilon,
            clip=self.observation_config.clip,
        )
        self._pending_normalizer = RunningNormalizer(self.observation_config.size)
        self._current_obs: dict[str, Array] | None = None
        self._previous_action = jnp.zeros((n_envs, 4), dtype=jnp.float32)
        self._pending_actions: np.ndarray | None = None
        self._episode_returns = np.zeros(n_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int64)
        self._episode_start_gate = np.zeros(n_envs, dtype=np.int32)
        self._episode_is_segment = np.zeros(n_envs, dtype=bool)
        self._closed = False
        self._restored_pending: dict[str, Any] | None = None
        self._completed_episodes: deque = deque(maxlen=1000)
        self._completed_segment_episodes: deque = deque(maxlen=1000)
        self._reward_stats = _zero_reward_stats(reward_components(self.reward_config), n_envs)
        self._reward_transitions = 0
        observation_space = spaces.Box(
            -self.observation_config.clip,
            self.observation_config.clip,
            shape=(self.observation_config.size,),
            dtype=np.float32,
        )
        # Finite bounds satisfy SB3's Box-space contract while preserving every
        # practical Gaussian draw; smooth physical bounds live in decode_action.
        action_space = spaces.Box(-1e6, 1e6, shape=(4,), dtype=np.float32)
        super().__init__(n_envs, observation_space, action_space)

    def _reset_worlds(
        self, mask: np.ndarray | None = None, seed: int | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Apply the official reset, then optionally replace only reset worlds."""
        from lsy_drone_racing.envs.race_core import obs as race_obs

        if mask is None:
            raw, info = self.jax_env.reset(seed=seed)
            mask = np.ones(self.num_envs, dtype=bool)
        else:
            self.jax_env.data, (raw, info) = self.jax_env._reset(
                self.jax_env.data, seed=None, mask=jnp.asarray(mask)
            )
            raw = {key: value[:, 0] for key, value in raw.items()}
        self._episode_start_gate[mask] = 0
        self._episode_is_segment[mask] = False
        if self._segment_reset is not None:
            self.jax_env.data, segment, starts = self._segment_reset(
                self.jax_env.data, jnp.asarray(mask)
            )
            segment, starts = jax.device_get((segment, starts))
            self._episode_is_segment[mask] = segment[mask]
            self._episode_start_gate[mask] = starts[mask]
            raw = {key: value[:, 0] for key, value in race_obs(self.jax_env.data).items()}
        return raw, info

    def reset(self) -> np.ndarray:
        """Reset all worlds, preserving previously learned normalization moments."""
        # Seed only explicitly requested full resets; subsequent resets advance RNGs.
        reset_seed = self._seeds[0]
        raw, info = self._reset_worlds(seed=reset_seed)
        self._current_obs = raw
        self._previous_action = jnp.zeros_like(self._previous_action)
        self._pending_actions = None
        self._episode_returns.fill(0)
        self._episode_lengths.fill(0)
        self._completed_episodes.clear()
        self._completed_segment_episodes.clear()
        self._clear_reward_stats()
        self._pending_normalizer = RunningNormalizer(self.observation_config.size)
        if self._restored_pending is not None:
            self._pending_normalizer.load_state_dict(self._restored_pending)
            self._restored_pending = None
        self.reset_infos = [dict(info) for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return self._observe(accumulate=True)

    def _observe(self, *, accumulate: bool) -> np.ndarray:
        if self._current_obs is None:
            raise RuntimeError("Call reset() before using the adapter")
        if accumulate:
            normalized, mean, var = _observe_batch(
                self._current_obs,
                self._previous_action,
                self.normalizer.mean,
                self.normalizer.var,
                self.observation_config,
            )
            self._pending_normalizer._merge(mean, var, float(self.num_envs))
        else:
            raw = encode_obs(self._current_obs, self._previous_action, self.observation_config)
            normalized = self.normalizer.normalize(raw)
        # This is the observation boundary required by SBX's NumPy rollout buffer.
        result = np.asarray(normalized)
        if not np.isfinite(result).all():
            raise ValueError("Official observation produced non-finite policy features")
        return result

    @property
    def reward_mode(self) -> str:
        """Return "sparse" or "dense" according to the configured reward."""
        return "sparse" if isinstance(self.reward_config, SparseRewardConfig) else "dense"

    def _clear_reward_stats(self) -> None:
        self._reward_stats = _zero_reward_stats(
            reward_components(self.reward_config), self.num_envs
        )
        self._reward_transitions = 0

    def drain_metrics(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Transfer reward means and completed episodes once per rollout.

        Keys are per-transition component means; ``episode_<component>`` keys are
        mean per-episode component sums over episodes completed this rollout.
        Per-episode accumulators of still-running worlds are kept.
        """
        stats = jax.device_get(
            {key: self._reward_stats[key] for key in ("sums", "episode_totals", "episodes")}
        )
        transitions = max(1, self._reward_transitions)
        episodes_done = max(1, int(stats["episodes"]))
        means = {key: float(value) / transitions for key, value in stats["sums"].items()}
        for key, value in stats["episode_totals"].items():
            means[f"episode_{key}"] = float(value) / episodes_done
        episodes = list(self._completed_episodes) + list(self._completed_segment_episodes)
        self._completed_episodes.clear()
        self._completed_segment_episodes.clear()
        running = self._reward_stats["episode"]
        self._clear_reward_stats()
        self._reward_stats["episode"] = running
        return means, episodes

    def get_observation(self) -> np.ndarray:
        """Re-encode the current observation after changing frozen statistics."""
        return self._observe(accumulate=False)

    def update_normalizer(self) -> None:
        """Commit pending raw moments at a rollout boundary, then clear them."""
        pending = self._pending_normalizer
        self.normalizer._merge(pending.mean, pending.var, pending.count)
        self._pending_normalizer = RunningNormalizer(self.observation_config.size)

    def set_normalizer(self, normalizer: RunningNormalizer | dict[str, Any]) -> None:
        """Restore or freeze inference statistics before resetting a training env."""
        state = normalizer if isinstance(normalizer, dict) else normalizer.state_dict()
        self.normalizer.load_state_dict(state)
        self._pending_normalizer = RunningNormalizer(self.observation_config.size)

    def restore_pending_normalizer(self, state: dict[str, Any]) -> None:
        """Defer pending-stat restore until SBX's fresh-episode reset has completed."""
        pending = RunningNormalizer(self.observation_config.size)
        pending.load_state_dict(state)
        self._restored_pending = pending.state_dict()

    def step_async(self, actions: np.ndarray) -> None:
        """Stage a batch of raw Gaussian actions for the next simulation step."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions shaped {(self.num_envs, 4)}")
        if self._pending_actions is not None:
            raise RuntimeError("A step is already pending; call step_wait() first")
        self._pending_actions = actions.copy()

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step, retain terminal observations, and immediately reset only done worlds."""
        if self._pending_actions is None or self._current_obs is None:
            raise RuntimeError("Call reset() and step_async() before step_wait()")
        actions, self._pending_actions = self._pending_actions, None
        actions = jnp.asarray(actions)
        previous = self._current_obs
        commands = decode_action(
            actions, previous["quat"], self.thrust_min, self.thrust_max, self.action_config
        )
        raw, _, terminated, truncated, _ = self.jax_env.step(commands)
        current = raw
        reward_device, done_device, self._reward_stats, metrics = _transition_reward(
            previous,
            current,
            terminated,
            truncated,
            actions,
            self._reward_stats,
            self.reward_config,
            self.action_config,
            1.0 / float(self.config.env.freq),
        )
        self._reward_transitions += self.num_envs
        # SBX consumes only these arrays and the final normalized observation on CPU.
        reward, done = jax.device_get((reward_device, done_device))
        self._episode_returns += reward
        self._episode_lengths += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if done.any():
            terminal_features = encode_obs(current, commands, self.observation_config)
            terminal_observations = np.asarray(self.normalizer.normalize(terminal_features))
            metrics = jax.device_get(metrics)
            terminated, truncated = metrics["terminated"], metrics["truncated"]
            finished, passed = metrics["finished"], metrics["passed"]
            sequence_length = current["gate_sequence"].shape[1]
            for index in np.flatnonzero(done):
                start_gate = int(self._episode_start_gate[index])
                segment_start = bool(self._episode_is_segment[index])
                info = {
                    "TimeLimit.truncated": bool(truncated[index] and not terminated[index]),
                    "finished": bool(finished[index]),
                    "is_success": bool(finished[index] and not segment_start),
                    "crash": bool(terminated[index] and not finished[index]),
                    "timeout": bool(truncated[index] and not terminated[index]),
                    "n_gates_passed": int(passed[index]),
                    "gate_progress": float(
                        (passed[index] - start_gate) / (sequence_length - start_gate)
                    ),
                    "gates_passed": int(passed[index] - start_gate),
                    "start_gate": start_gate,
                    "segment_start": segment_start,
                }
                info["terminal_observation"] = terminal_observations[index].copy()
                info["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_lengths[index]),
                    "t": float(self._episode_lengths[index] / self.config.env.freq),
                }
                info["success_time"] = (
                    self._episode_lengths[index] / self.config.env.freq if finished[index] else None
                )
                infos[index] = info
                if segment_start:
                    self._completed_segment_episodes.append(info)
                else:
                    self._completed_episodes.append(info)
            current, _ = self._reset_worlds(mask=done)
            self._episode_returns[done] = 0
            self._episode_lengths[done] = 0
            for index in np.flatnonzero(done):
                self.reset_infos[index] = {}
        self._current_obs = current
        self._previous_action = _reset_previous_action(commands, done_device)
        # SBX bootstraps timeout rewards in-place, so return writable NumPy memory.
        return self._observe(accumulate=True), reward.copy(), done.copy(), infos

    def get_config(self) -> dict[str, Any]:
        """Describe every observation/action/reward setting needed to reload the policy."""
        return {
            "observation": asdict(self.observation_config),
            "action": asdict(self.action_config),
            "reward_mode": self.reward_mode,
            "reward": asdict(self.reward_config),
            "reset": asdict(self.reset_config),
            "thrust_min": self.thrust_min,
            "thrust_max": self.thrust_max,
            "observation_dim": self.observation_space.shape[0],
            "action_dim": 4,
            "max_episode_steps": self.max_episode_steps,
            "freq": self.config.env.freq,
            "config_path": str(self.config_path),
            "race_config": self.config.to_dict(),
        }

    def close(self) -> None:
        """Release the official simulation resources once."""
        if not self._closed:
            self.jax_env.close()
            self._closed = True

    def seed(self, seed: int | None = None) -> list[int]:
        """Schedule one common vectorized simulation seed for the next full reset."""
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self.seed_value = int(seed)
        self._seeds = [self.seed_value] * self.num_envs
        return self._seeds.copy()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """Return adapter attributes for the requested vector slots."""
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Set global adapter attributes, rejecting misleading per-world changes."""
        if list(self._get_indices(indices)) != list(range(self.num_envs)):
            raise ValueError("Adapter attributes are shared; set them for every world")
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        """Call a shared adapter method once and expose its result for selected slots."""
        selected = list(self._get_indices(indices))
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result for _ in selected]

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        """Report that vector slots contain no individual Gymnasium wrappers."""
        return [False for _ in self._get_indices(indices)]
