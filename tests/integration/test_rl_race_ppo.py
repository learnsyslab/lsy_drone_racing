"""The vendored Brax PPO equals upstream without the hook, and the hook freezes the policy."""

from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as upstream

from lsy_drone_racing.control.rl_race import brax_ppo_train as vendored
from lsy_drone_racing.control.rl_race.optim import iteration_updates, separate_lr


class PointMass(Env):
    """2-D point mass pulled toward the origin; reward = -distance."""

    def reset(self, rng: jax.Array) -> State:
        """Random position, zero velocity."""
        pos = jax.random.uniform(rng, (2,), minval=-1, maxval=1)
        obs = jnp.concatenate([pos, jnp.zeros(2)])
        zero = jnp.zeros(())
        return State(pipeline_state=None, obs=obs, reward=zero, done=zero, metrics={}, info={})

    def step(self, state: State, action: jax.Array) -> State:
        """Damped velocity integration."""
        pos, vel = state.obs[:2], 0.9 * state.obs[2:] + 0.1 * action
        pos = pos + 0.1 * vel
        obs = jnp.concatenate([pos, vel])
        return state.replace(obs=obs, reward=-jnp.linalg.norm(pos), done=jnp.zeros(()))

    @property
    def observation_size(self) -> int:
        """Position and velocity."""
        return 4

    @property
    def action_size(self) -> int:
        """Planar acceleration."""
        return 2

    @property
    def backend(self) -> str:
        """No Brax physics pipeline."""
        return "none"


KWARGS = dict(
    num_timesteps=40_960,
    episode_length=50,
    num_envs=64,
    batch_size=64,
    num_minibatches=4,
    unroll_length=8,
    num_updates_per_batch=2,
    num_evals=2,
    learning_rate=3e-4,
    seed=3,
    normalize_observations=True,
)


def _train(module: ModuleType, **extra: Any) -> tuple:
    return module.train(environment=PointMass(), **{**KWARGS, **extra})


def _assert_trees_equal(a: Any, b: Any) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.integration
def test_vendored_train_matches_upstream_without_hook():
    _, ours, _ = _train(vendored)
    _, theirs, _ = _train(upstream)
    _assert_trees_equal(ours, theirs)


@pytest.mark.integration
def test_hook_freezes_policy_but_trains_value():
    steps = iteration_updates(KWARGS["num_updates_per_batch"], KWARGS["num_minibatches"]) * 10_000
    frozen = separate_lr(policy_lr=3e-4, value_lr=3e-4, freeze_policy_steps=steps)
    _, ours, _ = _train(vendored, optimizer_override=frozen)
    _, initial, _ = _train(upstream, learning_rate=0.0)  # nothing updated: initial parameters
    _assert_trees_equal(ours[1], initial[1])  # policy unchanged
    changed = [
        not np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(ours[2]), jax.tree.leaves(initial[2]), strict=True)
    ]
    assert any(changed)  # value network trained


@pytest.mark.integration
def test_hook_rejects_adaptive_kl():
    with pytest.raises(ValueError, match="adaptive KL"):
        _train(
            vendored,
            optimizer_override=separate_lr(1e-4, 1e-4),
            learning_rate_schedule="ADAPTIVE_KL",
        )
