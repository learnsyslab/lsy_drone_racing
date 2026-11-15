"""Batched Brax environment over the course's functional race core (attitude mode).

The adapter is natively batched: ``reset`` takes (n, 2) keys and ``step`` (n, 4) actions, so it is
used with ``wrap_env=False`` in Brax PPO. It performs its own auto-reset (official reset, then path
starts for a share of the done worlds) and maintains the fields Brax expects in ``State.info``:
``truncation``, ``steps``, ``episode_done`` and ``episode_metrics``.

Episode ends: crash/out of bounds (terminated), lap finished (on the step the last gate is
passed) or the time limit (truncation, bootstrapped by Brax's truncation masking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from jax import Array

from lsy_drone_racing.control.rl_race.env_wrapper import path, rewards
from lsy_drone_racing.control.rl_race.env_wrapper.actions import (
    ACTION_SIZE,
    ActionConfig,
    to_command,
)
from lsy_drone_racing.control.rl_race.env_wrapper.observations import (
    ObsConfig,
    encode,
    squeeze_drone_axis,
)
from lsy_drone_racing.control.rl_race.env_wrapper.starts import StartConfig, build_path_start

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "config" / "level2.toml"

# Per-step metrics; Brax sums them over an episode (training logs and evaluation).
METRICS = (
    "gates",
    "finished",
    "crashed",
    "timeout",
    "official_done",
    "success",
    "lap_time",
    *(f"r_{name}" for name in rewards.COMPONENTS),
)


@dataclass(frozen=True)
class EnvConfig:
    """Everything that defines the training problem besides the track config file."""

    max_episode_steps: int = 750
    action: ActionConfig = field(default_factory=ActionConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: rewards.RewardConfig = field(default_factory=rewards.RewardConfig)
    start: StartConfig = field(default_factory=StartConfig)


class RaceEnv(Env):
    """Single-drone Level 2 race as a batched Brax environment."""

    def __init__(
        self,
        n_envs: int,
        cfg: EnvConfig | None = None,
        config_path: str | Path | None = None,
        device: str = "gpu",
        seed: int = 0,
    ) -> None:
        """Build the course simulator for ``n_envs`` parallel worlds."""
        from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
        from lsy_drone_racing.utils import load_config

        self.cfg = cfg or EnvConfig()
        self.n_envs = int(n_envs)
        self.config_path = Path(config_path or DEFAULT_CONFIG).expanduser().resolve()
        config = load_config(self.config_path)
        config.sim.render = False
        self.freq = int(config.env.freq)
        core = VecDroneRaceEnv(
            num_envs=self.n_envs,
            freq=config.env.freq,
            sim_config=config.sim,
            track=config.env.track,
            sensor_range=config.env.sensor_range,
            control_mode="attitude",
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=seed,
            max_episode_steps=self.cfg.max_episode_steps,
            device=device,
        )
        # Resets are done here (with path starts), not by the core.
        core.settings = core.settings.replace(autoreset=False)
        core._step = core.build_step_fn()
        self._core = core
        self._template = core.data
        low, high = core.single_action_space.low, core.single_action_space.high
        action_cfg = self.cfg.action
        self.action_cfg = ActionConfig(
            roll_limit_deg=action_cfg.roll_limit_deg,
            pitch_limit_deg=action_cfg.pitch_limit_deg,
            yaw_limit_deg=action_cfg.yaw_limit_deg,
            thrust_min=float(low[3] + (high[3] - low[3]) * action_cfg.thrust_min_fraction),
            thrust_max=float(low[3] + (high[3] - low[3]) * action_cfg.thrust_max_fraction),
        )
        self.start_pos = jnp.asarray(np.asarray(config.env.track.drones[0]["pos"], np.float32))
        self._path_start = build_path_start(
            self.cfg.start, core.build_contact_check_fn(), self.start_pos
        )
        self.n_seq = int(core.data.gate_sequence.shape[0])

    # Brax Env interface -------------------------------------------------------------------------

    @property
    def observation_size(self) -> int:
        """Feature dimension."""
        return self.cfg.obs.size

    @property
    def action_size(self) -> int:
        """[roll, pitch, yaw, thrust]."""
        return ACTION_SIZE

    @property
    def backend(self) -> str:
        """Physics comes from crazyflow, not a Brax pipeline."""
        return "crazyflow"

    def reset(self, rng: Array) -> State:
        """Reset all worlds; ``rng`` has shape (n_envs, 2)."""
        from lsy_drone_racing.envs.race_core import obs as race_obs

        key = rng[0]
        key, sim_key, start_key = jax.random.split(key, 3)
        data = self._template
        # Brax passes raw uint32 keys; crazyflow stores typed keys (jax.random.key).
        sim_key = jax.random.wrap_key_data(sim_key)
        data = data.replace(
            sim_data=data.sim_data.replace(core=data.sim_data.core.replace(rng_key=sim_key))
        )
        data, _ = self._core._reset(data, None, None)
        everyone = jnp.ones(self.n_envs, dtype=bool)
        data, official = self._apply_starts(data, everyone, start_key)
        previous_action = jnp.zeros((self.n_envs, ACTION_SIZE), jnp.float32)
        raw = squeeze_drone_axis(race_obs(data))
        zeros = jnp.zeros(self.n_envs, jnp.float32)
        metrics = {name: zeros for name in METRICS}
        info = {
            "rng": key,
            "previous_action": previous_action,
            "official": official.astype(jnp.float32),
            "truncation": zeros,
            "steps": zeros,
            "episode_done": zeros,
            "episode_metrics": {"sum_reward": zeros, "length": zeros, **metrics},
        }
        return State(
            pipeline_state=data,
            obs=encode(raw, previous_action, self.cfg.obs),
            reward=zeros,
            done=zeros,
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: Array) -> State:
        """Advance all worlds by one control step and auto-reset finished episodes."""
        from lsy_drone_racing.envs.race_core import obs as race_obs

        data = state.pipeline_state
        prev = squeeze_drone_axis(race_obs(data))
        prev_passed = prev["n_gates_passed"].astype(jnp.int32)
        data, (raw, _, terminated, truncated, _) = self._core._step(
            data, to_command(action, self.action_cfg)
        )
        raw = squeeze_drone_axis(raw)
        terminated, truncated = terminated[:, 0], truncated[:, 0]
        passed = raw["n_gates_passed"].astype(jnp.int32)
        event = rewards.events(prev_passed, passed, terminated, self.n_seq)
        progress = path.arc_progress(
            data.gates_pos,
            data.gates_quat,
            data.gate_sequence,
            data.gate_sequence_direction,
            prev_passed,
            self.start_pos,
            prev["pos"],
            raw["pos"],
            lateral_weight=self.cfg.reward.path_lateral,
        )
        official = state.info["official"]
        seconds = data.steps.astype(jnp.float32) / self.freq
        lap_seconds = jnp.where(official > 0, seconds, self.cfg.reward.lap_ref)
        reward, components = rewards.compute(
            self.cfg.reward, event, raw["ang_vel"], progress, lap_seconds
        )

        done = terminated | truncated | event["finished"]
        timeout = truncated & ~terminated & ~event["finished"]
        donef = done.astype(jnp.float32)
        finishedf = event["finished"].astype(jnp.float32)
        steps_taken = data.steps.astype(jnp.float32)
        metrics = {
            "gates": event["gates"].astype(jnp.float32),
            "finished": finishedf,
            "crashed": event["crashed"].astype(jnp.float32),
            "timeout": timeout.astype(jnp.float32),
            "official_done": donef * official,
            "success": finishedf * official,
            "lap_time": finishedf * official * seconds,
            **{f"r_{name}": value for name, value in components.items()},
        }
        episode = state.info["episode_metrics"]
        keep = 1.0 - state.info["episode_done"]
        episode = {name: value * keep for name, value in episode.items()}
        episode["sum_reward"] = episode["sum_reward"] + reward
        episode["length"] = episode["length"] + 1.0
        for name, value in metrics.items():
            episode[name] = episode[name] + value

        # Auto-reset finished worlds: official reset, then path starts for a share of them.
        key, start_key = jax.random.split(state.info["rng"])
        data, _ = self._core._reset(data, None, done)
        data, new_official = self._apply_starts(data, done, start_key)
        official = jnp.where(done, new_official.astype(jnp.float32), official)
        previous_action = jnp.where(done[:, None], 0.0, jnp.clip(action, -1.0, 1.0))
        next_raw = squeeze_drone_axis(race_obs(data))
        info = {
            "rng": key,
            "previous_action": previous_action,
            "official": official,
            "truncation": timeout.astype(jnp.float32),
            "steps": steps_taken,  # episode length including this step (read by Brax eval)
            "episode_done": donef,
            "episode_metrics": episode,
        }
        return State(
            pipeline_state=data,
            obs=encode(next_raw, previous_action, self.cfg.obs),
            reward=reward,
            done=donef,
            metrics=metrics,
            info=info,
        )

    # Helpers ------------------------------------------------------------------------------------

    def _apply_starts(self, data: Any, mask: Array, key: Array) -> tuple[Any, Array]:
        """Turn a share of the freshly reset worlds in ``mask`` into path starts.

        Returns the data and a (n,) flag that is True where the world uses an official start.
        """
        choose_key, path_key = jax.random.split(key)
        want_path = mask & (
            jax.random.uniform(choose_key, (self.n_envs,)) >= self.cfg.start.official_prob
        )
        data, accepted = self._path_start(data, want_path, path_key)
        return data, ~accepted
