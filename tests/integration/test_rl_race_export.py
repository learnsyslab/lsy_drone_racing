"""Exported rl_race policies reproduce Brax's deterministic action and run in the controller."""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from lsy_drone_racing.control.rl_race.controller import load_policy, policy_mean_action
from lsy_drone_racing.control.rl_race.export import policy_arrays
from lsy_drone_racing.control.rl_race.optim import separate_lr

OBS, ACT = 49, 4


def _random_params(seed: int = 0) -> tuple:
    """Brax PPO parameters with a non-trivial normalizer, as saved by train.py."""
    network = ppo_networks.make_ppo_networks(
        OBS,
        ACT,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32,),
        activation=jax.nn.swish,
        distribution_type="tanh_normal",
    )
    key = jax.random.PRNGKey(seed)
    normalizer = running_statistics.init_state(jax.ShapeDtypeStruct((OBS,), jnp.float32))
    batch = jax.random.normal(key, (256, OBS)) * 3 + 1
    normalizer = running_statistics.update(normalizer, batch)
    policy = network.policy_network.init(key)
    value = network.value_network.init(key)
    return network, (normalizer, policy, value)


@pytest.mark.integration
def test_controller_forward_matches_brax_deterministic_policy(tmp_path: Path):
    network, params = _random_params()
    inference = ppo_networks.make_inference_fn(network)(params, deterministic=True)
    obs = jax.random.normal(jax.random.PRNGKey(7), (16, OBS)) * 2
    expected, _ = inference(obs, jax.random.PRNGKey(0))
    from flax import serialization

    (tmp_path / "policy.msgpack").write_bytes(
        serialization.msgpack_serialize(policy_arrays(params))
    )
    (tmp_path / "config.json").write_text(
        '{"format_version": 2, "action_order": ["roll", "pitch", "yaw", "thrust"], "action_dim": 4, "observation_dim": 49}'
    )
    _, arrays = load_policy(tmp_path)
    ours = policy_mean_action(jax.tree.map(jnp.asarray, arrays), obs, "swish")
    np.testing.assert_allclose(np.asarray(ours), np.asarray(expected), atol=1e-5)


@pytest.mark.integration
def test_saved_params_roundtrip_through_pickle(tmp_path: Path):
    _, params = _random_params(1)
    path = tmp_path / "p.pkl"
    path.write_bytes(pickle.dumps(jax.device_get(params)))
    loaded = pickle.loads(path.read_bytes())
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert separate_lr(1e-4, 1e-4) is not None


def test_controller_never_ends_episode_early():
    """The base Controller.step_callback returns True (finish after one step); ours must not."""
    from lsy_drone_racing.control.rl_race.controller import RaceController

    assert RaceController.step_callback(None, None, {}, 0.0, False, False, {}) is False


def test_old_export_rejected_explicitly(tmp_path):
    (tmp_path / "config.json").write_text('{"action_dim": 3}')
    with pytest.raises(ValueError, match="Incompatible export"):
        load_policy(tmp_path)
