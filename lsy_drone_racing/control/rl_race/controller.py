"""Competition controller for exported ``rl_race`` policies (four-action absolute attitude).

Loads ``deploy_model/`` next to this file (or ``controller.model_dir`` from the level config),
encodes the official observation exactly as in training, and applies the deterministic action.
Requires only jax/flax, not Brax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_race.env_wrapper.actions import (
    ACTION_ORDER,
    ACTION_SIZE,
    ActionConfig,
    to_command,
)
from lsy_drone_racing.control.rl_race.env_wrapper.observations import ObsConfig, encode

ACTIVATIONS = {"swish": jax.nn.swish, "elu": jax.nn.elu, "relu": jax.nn.relu, "tanh": jnp.tanh}


def policy_mean_action(policy: dict[str, Any], features: jax.Array, activation: str) -> jax.Array:
    """Brax ``tanh_normal`` mode: normalize, MLP (no final activation), tanh of the mean half."""
    act = ACTIVATIONS[activation]
    x = (features - policy["mean"]) / policy["std"]
    layers = policy["layers"]
    for index, layer in enumerate(layers):
        x = x @ layer["kernel"] + layer["bias"]
        if index < len(layers) - 1:
            x = act(x)
    loc, _ = jnp.split(x, 2, axis=-1)
    return jnp.tanh(loc)


def load_policy(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the export's config and its arrays (layers as a list of dicts)."""
    config = json.loads((model_dir / "config.json").read_text())
    if (
        config.get("format_version") != 2
        or config.get("action_order") != list(ACTION_ORDER)
        or config.get("action_dim") != 4
    ):
        raise ValueError("Incompatible export: retrain and export the four-action format v2")
    arrays = serialization.msgpack_restore((model_dir / "policy.msgpack").read_bytes())
    layers = arrays["layers"]
    if isinstance(layers, dict):  # some serializers store lists as {"0": ..., "1": ...}
        layers = [layers[key] for key in sorted(layers, key=int)]
    arrays["layers"] = list(layers)
    if arrays["mean"].shape != (config["observation_dim"],) or layers[-1]["bias"].shape != (8,):
        raise ValueError("Export observation or action dimensions do not match format v2")
    return config, arrays


class RaceController(Controller):
    """Deterministic exported policy behind the official controller interface."""

    def __init__(self, obs: dict, info: dict, config: Any) -> None:
        """Load the export and compile the policy before the timed control loop."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("RaceController requires env.control_mode = 'attitude'")
        model_dir = Path(config.controller.get("model_dir", "deploy_model")).expanduser()
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parent / model_dir
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"No export at {model_dir}; run python -m lsy_drone_racing.control.rl_race.export"
            )
        bundle, arrays = load_policy(model_dir)
        if int(config.env.freq) != int(bundle["freq"]):
            raise ValueError("Control frequency differs from training")
        obs_cfg = ObsConfig(**bundle["obs"])
        action_cfg = ActionConfig(**bundle["action"])
        activation = bundle["network"]["activation"]
        policy = jax.tree.map(jnp.asarray, arrays)

        def predict(observation: dict, previous: jax.Array) -> tuple[jax.Array, jax.Array]:
            batch = {key: jnp.asarray(value)[None] for key, value in observation.items()}
            action = policy_mean_action(policy, encode(batch, previous[None], obs_cfg), activation)
            return action[0], to_command(action, action_cfg)[0]

        self._predict = jax.jit(predict)
        self._previous = jnp.zeros(ACTION_SIZE, jnp.float32)
        self._predict(obs, self._previous)[1].block_until_ready()  # compile outside the loop

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return [roll, pitch, yaw, thrust]; the policy action is kept for the next features."""
        action, command = self._predict(obs, self._previous)
        self._previous = action
        return np.asarray(command, dtype=np.float32)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Never end the episode early; the base class returns True (finish after one step)."""
        return False

    def episode_reset(self) -> None:
        """Clear the previous action between episodes."""
        self._previous = jnp.zeros(ACTION_SIZE, jnp.float32)

    def reset(self) -> None:
        """Alias used by some runners."""
        self.episode_reset()
