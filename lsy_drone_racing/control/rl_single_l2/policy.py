"""Flax Actor/Critic networks and their SBX PPO policy for Level 2 racing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from sbx.ppo.policies import PPOPolicy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

NET_ARCH = (256, 256)
LOG_STD_INIT = -0.5
LOG_STD_MIN = -2.5
LOG_STD_MAX = 1.0


class Actor(nn.Module):
    """Gaussian over raw thrust/rotation with a bounded mean and state-dependent std."""

    action_dim: int = 4
    net_arch: Sequence[int] = NET_ARCH
    log_std_init: float = LOG_STD_INIT
    activation_fn: Callable = nn.leaky_relu
    # Mean = b * tanh(z / b): slope 1 at zero for every b, saturating at +/-b.
    mean_bound: float = 1.0
    # Accepted by SBX's constructor; Level2Policy fixes this to False.
    # The network uses the explicit, fixed initializers below.
    ortho_init: bool = False
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Any:
        """Return a Gaussian with a soft-bounded mean and state-dependent clipped log std.

        The mean is ``b * tanh(z / b)``. b = 1 is the original tanh mean, which caps
        the deterministic thrust at sigmoid(1) = 73 % of the range; larger b lets
        the mean reach the actuator limits while keeping the small-signal slope,
        so warm starts from b = 1 change only saturated actions.
        """
        x = obs
        for width in self.net_arch:
            x = self.activation_fn(nn.Dense(width)(x))
        pre_mean = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), name="mean"
        )(x)
        mean = self.mean_bound * nn.tanh(pre_mean / self.mean_bound)
        # A small nonzero kernel makes exploration state-dependent from initialization.
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.constant(self.log_std_init),
            name="log_std_head",
        )(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))

    def get_std(self) -> jnp.ndarray:
        """Provide SBX API parity; actual standard deviations require observations."""
        return jnp.exp(jnp.array(self.log_std_init))


class Critic(nn.Module):
    """Estimate V(o) from the same features using entirely separate parameters."""

    net_arch: Sequence[int] = NET_ARCH
    activation_fn: Callable = nn.leaky_relu
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Return one scalar per observation, with no output activation."""
        x = obs
        for width in self.net_arch:
            x = self.activation_fn(nn.Dense(width)(x))
        return nn.Dense(1, name="value")(x)


ACTIVATIONS: dict[str, Callable] = {
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "relu": nn.relu,
    "tanh": nn.tanh,
}


@dataclass(frozen=True)
class NetworkConfig:
    """Actor/critic hidden layers and their shared activation.

    SBX's PPOPolicy applies one activation to both networks, so the activation
    is shared. The Gaussian head (state-dependent std and its bounds) is fixed;
    ``mean_bound`` sets the soft bound b of the actor mean b * tanh(z / b). The
    class default 1.0 is the original head, so legacy pickles and metadata
    restore unchanged; train.py defaults new runs to a wider bound.
    """

    actor_arch: tuple[int, ...] = NET_ARCH
    critic_arch: tuple[int, ...] = NET_ARCH
    activation: str = "leaky_relu"
    mean_bound: float = 1.0

    def __post_init__(self) -> None:
        """Normalize lists to tuples and validate widths and activation."""
        object.__setattr__(self, "actor_arch", tuple(int(w) for w in self.actor_arch))
        object.__setattr__(self, "critic_arch", tuple(int(w) for w in self.critic_arch))
        if not self.actor_arch or not self.critic_arch:
            raise ValueError("Actor and critic need at least one hidden layer")
        if min(self.actor_arch + self.critic_arch) < 1:
            raise ValueError("Layer widths must be positive")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        object.__setattr__(self, "mean_bound", float(self.mean_bound))
        if not 0 < self.mean_bound < float("inf"):
            raise ValueError("mean_bound must be positive and finite")

    @property
    def activation_fn(self) -> Callable:
        """Return the Flax activation function."""
        return ACTIVATIONS[self.activation]

    def actor(self) -> Actor:
        """Build the inference actor described by this configuration."""
        return Actor(
            net_arch=self.actor_arch, activation_fn=self.activation_fn, mean_bound=self.mean_bound
        )

    def with_mean_bound(self, mean_bound: float) -> NetworkConfig:
        """Return the same layers with another mean bound (same parameter shapes)."""
        return replace(self, mean_bound=mean_bound)


ARCH_PRESETS = {
    "default": NetworkConfig(),
    # Pasumarti et al. (2026): deeper critic than actor, ELU activations.
    "paper": NetworkConfig((512, 512, 256, 128), (512, 512, 256, 256, 128, 128), "elu"),
}


def network_config(config: NetworkConfig | None = None) -> dict[str, Any]:
    """Describe the inference network without serializing Python callables.

    The default network keeps the original single ``net_arch`` description so
    earlier checkpoints and exports compare equal.
    """
    cfg = config or NetworkConfig()
    if cfg.actor_arch == cfg.critic_arch:
        layers: dict[str, Any] = {"net_arch": list(cfg.actor_arch)}
    else:
        layers = {"actor_arch": list(cfg.actor_arch), "critic_arch": list(cfg.critic_arch)}
    return {
        **layers,
        "activation": cfg.activation,
        "action_dim": 4,
        "std_type": "state_dependent",
        "log_std_init": LOG_STD_INIT,
        "log_std_min": LOG_STD_MIN,
        "log_std_max": LOG_STD_MAX,
        # Omitted for the original head so earlier metadata compares equal.
        **({} if cfg.mean_bound == 1.0 else {"mean_bound": cfg.mean_bound}),
    }


def network_from_config(description: dict[str, Any]) -> NetworkConfig:
    """Rebuild a NetworkConfig from saved metadata, rejecting other Gaussian heads."""
    if "net_arch" in description:
        actor = critic = description["net_arch"]
    else:
        actor, critic = description["actor_arch"], description["critic_arch"]
    result = NetworkConfig(
        actor, critic, description["activation"], description.get("mean_bound", 1.0)
    )
    if network_config(result) != description:
        raise ValueError("Saved network/std configuration is incompatible with this code")
    return result


class Level2Policy(PPOPolicy):
    """Use SBX's native optimizers and PPO loop with the Level 2 networks."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable,
        network: NetworkConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate flat observations and the four-dimensional raw action contract.

        ``network`` (a NetworkConfig, passed through SBX policy_kwargs) selects the
        actor/critic layers and activation; SBX pickles it into model.zip.
        """
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError("Level2Policy requires a flat Box observation")
        if not isinstance(action_space, gym.spaces.Box) or action_space.shape != (4,):
            raise ValueError("Level2Policy requires four continuous raw actions")
        if kwargs.get("use_sde", False):
            raise ValueError("State-dependent Gaussian heads do not use gSDE")
        if kwargs.get("ortho_init", False):
            raise ValueError("Level2Policy fixes ortho_init=False and uses fixed initializers")
        kwargs["ortho_init"] = False
        network = network or NetworkConfig()
        fixed = {
            # SBX only calls actor_class, so a partial carries the mean bound.
            "actor_class": partial(Actor, mean_bound=network.mean_bound),
            "critic_class": Critic,
            "net_arch": {"pi": list(network.actor_arch), "vf": list(network.critic_arch)},
            "activation_fn": network.activation_fn,
            "log_std_init": LOG_STD_INIT,
        }
        if fixed.keys() & kwargs.keys():
            raise ValueError(
                "Network architecture and distribution settings are fixed by Level2Policy"
            )
        super().__init__(observation_space, action_space, lr_schedule, **fixed, **kwargs)
        self.network = network
