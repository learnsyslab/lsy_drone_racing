"""Six-term JAX reward for single-drone Level 2 PPO training.

This adapts the reference's distance progress, L2 angular-speed penalty and
terminal rewards, and explicitly includes a fixed bonus per official gate pass.
It uses sensor-masked observations and the official gate-sequence counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients at the default 50 Hz control frequency.

    Progress is measured in meters per step and angular speed in radians per
    second. Gate and terminal bonuses are paid once on the corresponding event;
    the time penalty is charged on every control step, including the last.
    """

    progress: float = 15.0
    gate_pass: float = 10.0
    completion: float = 100.0
    crash: float = 15.0
    angular_velocity: float = 0.01
    time: float = 0.10


@partial(jax.jit, static_argnames=("config",))
def compute_reward(
    previous_obs: dict[str, Any],
    current_obs: dict[str, Any],
    terminated: Array,
    truncated: Array,
    config: RewardConfig | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute all six components before the adapter resets completed worlds.

    Reward = 15 * distance_progress + 10 * gate_passes + 100 * finished
             - 15 * crash - 0.01 * angular_speed_L2 - 0.10.

    Both distances use the pre-step target and its pre-step observed center, so
    target changes and newly revealed geometry do not create artificial progress.
    Crash-step progress is zero because the simulator warps crashed positions.
    Gate events come from n_gates_passed, which already accounts for required
    direction and repeated gates; no separate geometric pass detector is added.
    The last pass earns both the gate bonus and the one-off completion bonus.
    Timeouts receive no extra crash penalty. Single observations and batches are
    accepted; rewards and component arrays always have shape (n_envs,).

    Unlike the reference's experimental branches, every component returned here
    is included in the sum. There is no per-index gate-bonus scaling, guidance
    field, path construction, obstacle barrier or action-smoothness term.
    """
    del truncated  # SBX handles timeout bootstrapping separately.
    cfg = config or RewardConfig()
    previous_pos = jnp.asarray(previous_obs["pos"]).reshape(-1, 3)
    pos = jnp.asarray(current_obs["pos"]).reshape(-1, 3)
    count = previous_pos.shape[0]
    sequence = jnp.asarray(previous_obs["gate_sequence"], dtype=jnp.int32).reshape(count, -1)
    sequence_length = sequence.shape[1]
    previous_passed = jnp.asarray(previous_obs["n_gates_passed"], dtype=jnp.int32).reshape(count)
    passed = jnp.asarray(current_obs["n_gates_passed"], dtype=jnp.int32).reshape(count)
    previous_passed = jnp.clip(previous_passed, 0, sequence_length)
    passed = jnp.clip(passed, 0, sequence_length)
    active = previous_passed < sequence_length
    finished = active & (passed >= sequence_length)
    crashed = jnp.asarray(terminated, dtype=bool).reshape(count) & ~finished & active
    gate_passes = jnp.maximum(passed - previous_passed, 0)
    gate_passes = jnp.where(crashed, 0, gate_passes)

    rows = jnp.arange(count)
    gate_ids = sequence[rows, jnp.minimum(previous_passed, sequence_length - 1)]
    gates_pos = jnp.asarray(previous_obs["gates_pos"]).reshape(count, -1, 3)
    centers = gates_pos[rows, gate_ids]
    progress = jnp.linalg.norm(previous_pos - centers, axis=-1)
    progress -= jnp.linalg.norm(pos - centers, axis=-1)
    progress = jnp.where(crashed | ~active, 0.0, progress)
    angular_speed = jnp.linalg.norm(jnp.asarray(current_obs["ang_vel"]).reshape(count, 3), axis=-1)

    components = {
        "progress": cfg.progress * progress,
        "gate_pass": cfg.gate_pass * gate_passes.astype(jnp.float32),
        "completion": cfg.completion * finished.astype(jnp.float32),
        "crash": -cfg.crash * crashed.astype(jnp.float32),
        "angular_velocity": -cfg.angular_velocity * angular_speed,
        "time": jnp.full(count, -cfg.time, dtype=jnp.float32),
    }
    components = {key: value.astype(jnp.float32) for key, value in components.items()}
    reward = jnp.sum(jnp.stack(list(components.values())), axis=0)
    return reward, components


REWARD_MODES = ("dense", "sparse")
SPARSE_COMPONENTS = ("gate_pass", "lap", "crash", "cmd")


@dataclass(frozen=True)
class SparseRewardConfig:
    """Task-level reward of Pasumarti et al. (2026), Eq. 2-6, adapted to Level 2.

    gate_pass_* is paid per official gate pass and lap_leading once when the last
    gate_order entry is passed (one episode is one lap here, so the paper's lap
    bonus is our completion event). Every crash is terminal in this environment,
    so the paper's non-terminal 0.1 contact penalty has no counterpart and all
    crashes cost ``crash``. The command penalty uses the implied commanded body
    rate delta / dt of the attitude-increment action; with cmd_scale_by_dt it is
    multiplied by dt (i.e. it penalizes the squared rate integrated over a step).
    Without an opponent, single_agent_leading decides whether the agent counts as
    leading (10 per gate + 50 lap bonus) or trailing (5 per gate, no lap bonus).
    """

    gate_pass_leading: float = 10.0
    gate_pass_trailing: float = 5.0
    lap_leading: float = 50.0
    crash: float = 2.0
    cmd_roll_pitch: float = 0.15
    cmd_yaw: float = 0.05
    cmd_scale_by_dt: bool = True
    single_agent_leading: bool = True


def reward_components(config: RewardConfig | SparseRewardConfig) -> tuple[str, ...]:
    """Return the logged component names produced by the matching reward function."""
    if isinstance(config, SparseRewardConfig):
        return SPARSE_COMPONENTS
    return tuple(field for field in RewardConfig.__dataclass_fields__)


def reward_config_from_dict(mode: str, values: dict[str, Any]) -> RewardConfig | SparseRewardConfig:
    """Rebuild a saved reward configuration from checkpoint metadata."""
    if mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
    return SparseRewardConfig(**values) if mode == "sparse" else RewardConfig(**values)


@partial(jax.jit, static_argnames=("config",))
def compute_sparse_reward(
    previous_obs: dict[str, Any],
    current_obs: dict[str, Any],
    terminated: Array,
    truncated: Array,
    rotation_increment: Array,
    dt: float,
    leading: Array | None = None,
    config: SparseRewardConfig | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute r = r_pass + r_lap - r_cmd - r_crash per agent before resets.

    rotation_increment is the body-frame rotation vector (radians) that the
    action applied this step, so omega_cmd = rotation_increment / dt with x, y, z
    corresponding to roll, pitch and yaw rates. ``leading`` is an optional
    per-agent boolean mask (shape (n,)), e.g. from a multi-agent race ranking;
    when omitted, config.single_agent_leading applies to every agent.

    Gate and lap events come from the official n_gates_passed counter, exactly
    as in the dense reward. A crash step earns no gate reward. Timeouts receive
    no crash penalty. There is no progress term and no time penalty.
    """
    del truncated  # SBX handles timeout bootstrapping separately.
    cfg = config or SparseRewardConfig()
    count = jnp.asarray(previous_obs["pos"]).reshape(-1, 3).shape[0]
    sequence_length = jnp.asarray(previous_obs["gate_sequence"]).reshape(count, -1).shape[1]
    previous_passed = jnp.asarray(previous_obs["n_gates_passed"], dtype=jnp.int32).reshape(count)
    passed = jnp.asarray(current_obs["n_gates_passed"], dtype=jnp.int32).reshape(count)
    previous_passed = jnp.clip(previous_passed, 0, sequence_length)
    passed = jnp.clip(passed, 0, sequence_length)
    active = previous_passed < sequence_length
    finished = active & (passed >= sequence_length)
    crashed = jnp.asarray(terminated, dtype=bool).reshape(count) & ~finished & active
    gate_passes = jnp.where(crashed, 0, jnp.maximum(passed - previous_passed, 0))
    if leading is None:
        leading = jnp.full(count, cfg.single_agent_leading, dtype=bool)
    leading = jnp.asarray(leading, dtype=bool).reshape(count)
    gate_value = jnp.where(leading, cfg.gate_pass_leading, cfg.gate_pass_trailing)
    lap_value = jnp.where(leading, cfg.lap_leading, 0.0)
    rate = jnp.asarray(rotation_increment, dtype=jnp.float32).reshape(count, 3) / dt
    cmd = cfg.cmd_roll_pitch * (rate[:, 0] ** 2 + rate[:, 1] ** 2) + cfg.cmd_yaw * rate[:, 2] ** 2
    if cfg.cmd_scale_by_dt:
        cmd = cmd * dt
    components = {
        "gate_pass": gate_value * gate_passes.astype(jnp.float32),
        "lap": lap_value * finished.astype(jnp.float32),
        "crash": -cfg.crash * crashed.astype(jnp.float32),
        "cmd": -cmd,
    }
    components = {key: value.astype(jnp.float32) for key, value in components.items()}
    reward = jnp.sum(jnp.stack(list(components.values())), axis=0)
    return reward, components
