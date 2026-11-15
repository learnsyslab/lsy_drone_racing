"""Actor mean bound, rotation gain and their legacy defaults."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ActionConfig,
    decode_action,
    rotation_increment,
)
from lsy_drone_racing.control.rl_single_l2.policy import (
    Actor,
    NetworkConfig,
    network_config,
    network_from_config,
)


def test_legacy_action_config_fills_gain_and_keeps_mapping():
    legacy = ActionConfig(alpha_max_rad=0.16)
    assert legacy.alpha_gain_rad == 0.16
    assert ActionConfig(**{"alpha_max_rad": 0.16}) == legacy
    raw = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32) * 2
    delta = np.asarray(rotation_increment(raw, legacy))
    norm = np.linalg.norm(raw[:, 1:], axis=1, keepdims=True)
    np.testing.assert_allclose(delta, raw[:, 1:] * np.tanh(norm) / norm * 0.16, rtol=1e-5)


def test_wider_rotation_limit_keeps_small_signal_gain():
    wide = ActionConfig(alpha_max_rad=0.30, alpha_gain_rad=0.16)
    legacy = ActionConfig(alpha_max_rad=0.16)
    small = np.array([[0.0, 1e-3, -2e-3, 5e-4]], np.float32)
    np.testing.assert_allclose(
        rotation_increment(small, wide), rotation_increment(small, legacy), rtol=1e-4
    )
    huge = np.array([[0.0, 50.0, 0.0, 0.0]], np.float32)
    assert np.linalg.norm(rotation_increment(huge, wide)) == pytest.approx(0.30, rel=1e-5)
    assert np.linalg.norm(rotation_increment(huge, legacy)) == pytest.approx(0.16, rel=1e-5)
    with pytest.raises(ValueError):
        ActionConfig(alpha_max_rad=0.3, alpha_gain_rad=0.0)


def test_mean_bound_widens_deterministic_thrust_range():
    obs = jnp.ones((1, 8))
    for bound, expected in ((1.0, 1.0), (3.0, 3.0)):
        actor = Actor(net_arch=(4,), mean_bound=bound)
        params = actor.init(jax.random.PRNGKey(0), obs)
        params = jax.tree_util.tree_map(lambda x: x, params)
        params["params"]["mean"]["bias"] = jnp.full(4, 100.0)
        mean = actor.apply(params, obs).mode()[0]
        np.testing.assert_allclose(mean, expected, rtol=1e-5)
        thrust = decode_action(mean, np.array([0, 0, 0, 1.0]), 0.0, 1.0)[3]
        assert float(thrust) == pytest.approx(1 / (1 + np.exp(-expected)), rel=1e-5)


def test_mean_bound_metadata_round_trip_and_legacy_default():
    legacy = network_config(NetworkConfig())
    assert "mean_bound" not in legacy
    assert network_from_config(legacy).mean_bound == 1.0
    wide = NetworkConfig(mean_bound=3.0)
    assert network_from_config(network_config(wide)) == wide
    assert wide.with_mean_bound(1.0) == NetworkConfig()
    assert wide.actor().mean_bound == 3.0
    with pytest.raises(ValueError):
        NetworkConfig(mean_bound=0.0)
