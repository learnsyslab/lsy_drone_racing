"""Actor-only deterministic inference through the official controller interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import serialization

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ActionConfig,
    ObservationConfig,
    RunningNormalizer,
    decode_action,
    encode_obs,
)
from lsy_drone_racing.control.rl_single_l2.policy import network_from_config


class Level2RLController(Controller):
    """Load a fixed actor export, freeze normalization, and emit attitude commands."""

    def __init__(self, obs: dict, info: dict, config: Any) -> None:
        """Load the package-relative deploy_model (or controller.model_dir override)."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("Level2RLController requires env.control_mode = 'attitude'")
        package_dir = Path(__file__).resolve().parent
        model_dir = Path(config.controller.get("model_dir", "deploy_model")).expanduser()
        if not model_dir.is_absolute():
            model_dir = package_dir / model_dir
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"No actor export at {model_dir}. Export a trained checkpoint with "
                "python -m lsy_drone_racing.control.rl_single_l2.checkpoint --checkpoint=..."
            )
        bundle = json.loads((model_dir / "config.json").read_text())
        network = network_from_config(bundle["network"])
        environment = bundle["environment"]
        if config.env.freq != environment["freq"]:
            raise ValueError("Inference control frequency differs from training")
        from lsy_drone_racing.envs.race_core import build_action_space

        space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
        self.thrust_min, self.thrust_max = environment["thrust_min"], environment["thrust_max"]
        if not np.allclose(space.low[3], self.thrust_min) or not np.allclose(
            space.high[3], self.thrust_max
        ):
            raise ValueError("Inference drone thrust limits differ from training")
        self.observation_config = ObservationConfig(**environment["observation"])
        self.action_config = ActionConfig(**environment["action"])
        self.normalizer = RunningNormalizer(self.observation_config.size)
        self.normalizer.load_state_dict(bundle["normalizer"])
        actor = network.actor()
        params = serialization.msgpack_restore((model_dir / "actor.msgpack").read_bytes())

        def predict(observation: dict, previous_action: jax.Array) -> jax.Array:
            features = encode_obs(observation, previous_action, self.observation_config)
            normalized = self.normalizer.normalize(features)
            raw = actor.apply(params, normalized[None]).mode()[0]
            return decode_action(
                raw, observation["quat"], self.thrust_min, self.thrust_max, self.action_config
            )

        self._predict = jax.jit(predict)
        self.previous_action = np.zeros(4, dtype=np.float32)
        # Compile during initialization, outside the timed competition control loop.
        self._predict(obs, self.previous_action).block_until_ready()

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Encode exactly as in training and transform the Gaussian's mean action."""
        command = np.array(self._predict(obs, self.previous_action), dtype=np.float32)
        self.previous_action = command.copy()
        return command

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Continue until the official environment ends the episode."""
        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        return False

    def reset(self) -> None:
        """Clear action history between episodes while retaining frozen moments."""
        self.previous_action.fill(0)

    def episode_reset(self) -> None:
        """Support the competition runner's episode-reset hook."""
        self.reset()
