"""Brax Actor/Critic construction; action physics live in env_wrapper.actions."""

import functools
import math
from dataclasses import dataclass
from typing import Any

import jax


def make_networks(
    observation_size: Any, action_size: int, noise_scale: float = 1.0, **kwargs: Any
) -> Any:
    """Brax PPO networks whose exploration std is multiplied by ``noise_scale``.

    The learned std parameters are unchanged (checkpoints stay compatible); only the sampling
    distribution is scaled, consistently for actions, log-probabilities and entropy. The deployed
    controller uses the mean action and is unaffected.
    """
    import dataclasses

    from brax.training import distribution
    from brax.training.agents.ppo import networks as ppo_networks

    nets = ppo_networks.make_ppo_networks(observation_size, action_size, **kwargs)
    if noise_scale == 1.0:
        return nets
    scaled = distribution.NormalTanhDistribution(event_size=action_size, var_scale=noise_scale)
    return dataclasses.replace(nets, parametric_action_distribution=scaled)


@dataclass(frozen=True)
class NetworkConfig:
    """Actor/Critic architecture and distribution, independent of environment physics."""

    policy_layers: tuple[int, ...] = (256, 256, 256)
    value_layers: tuple[int, ...] = (256, 256, 256, 256)
    activation: str = "swish"
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid network shapes before allocating the simulator."""
        if not self.policy_layers or not self.value_layers:
            raise ValueError("Actor and Critic need nonempty hidden layers")
        if any(w < 1 for w in (*self.policy_layers, *self.value_layers)):
            raise ValueError("Hidden layer widths must be positive")
        if self.activation not in ("swish", "elu", "relu", "tanh"):
            raise ValueError("Unsupported network activation")
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive and finite")

    def metadata(self) -> dict[str, Any]:
        """Return the architecture contract used by checkpoints and exports."""
        return dict(
            policy_layers=list(self.policy_layers),
            value_layers=list(self.value_layers),
            activation=self.activation,
            distribution="tanh_normal",
        )

    def factory(self) -> Any:
        """Build the Brax network factory with all architecture choices in one place."""
        return functools.partial(
            make_networks,
            noise_scale=self.noise_scale,
            policy_hidden_layer_sizes=self.policy_layers,
            value_hidden_layer_sizes=self.value_layers,
            activation=getattr(jax.nn, self.activation),
            distribution_type="tanh_normal",
        )
