"""Sparse reward, paper observations and configurable networks end to end."""

from pathlib import Path

import numpy as np
import pytest
from sbx import PPO
from stable_baselines3.common.logger import configure

from lsy_drone_racing.control.rl_single_l2.callbacks import TrainingCallback
from lsy_drone_racing.control.rl_single_l2.checkpoint import (
    export_actor,
    load_checkpoint,
    read_metadata,
)
from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    Level2SBXEnv,
    ObservationConfig,
    decode_action,
    encode_obs,
)
from lsy_drone_racing.control.rl_single_l2.policy import ARCH_PRESETS, Level2Policy
from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig
from lsy_drone_racing.control.rl_single_l2.reward import SparseRewardConfig
from lsy_drone_racing.utils import load_config, load_controller

CONFIG = Path(__file__).parents[2] / "config/level2.toml"
CONTROLLER = CONFIG.parents[1] / "lsy_drone_racing/control/rl_single_level2_controller.py"


@pytest.mark.integration
def test_sparse_paper_training_checkpoint_and_deployment(tmp_path: Path):
    observation = ObservationConfig(layout="paper", paper_obstacles=True)
    network = ARCH_PRESETS["paper"]
    env = Level2SBXEnv(
        CONFIG,
        n_envs=2,
        max_episode_steps=3,
        observation_config=observation,
        reward_config=SparseRewardConfig(),
        reset_config=ResetConfig(segment_init_prob=0.5),
    )
    logger = configure(str(tmp_path / "logs"), ["csv"])
    try:
        assert env.observation_space.shape == (60,)
        assert env.get_config()["reward_mode"] == "sparse"
        kwargs = dict(n_steps=4, batch_size=4, n_epochs=1, seed=3, device="cpu")
        model = PPO(Level2Policy, env, policy_kwargs={"network": network}, **kwargs)
        model.set_logger(logger)
        params = model.policy.actor_state.params["params"]
        assert params["Dense_0"]["kernel"].shape == (60, 512)
        assert params["Dense_3"]["kernel"].shape == (256, 128)
        critic = model.policy.vf_state.params["params"]
        assert critic["Dense_5"]["kernel"].shape == (128, 128)
        callback = TrainingCallback(env, tmp_path / "checkpoints", {"ppo": kwargs}, save_freq=8)
        model.learn(16, callback=callback)
        log = (tmp_path / "logs/progress.csv").read_text()
        for key in ("reward/cmd", "reward/gate_pass", "reward/lap", "reward/episode_cmd"):
            assert key in log
        metadata = read_metadata(callback.last_checkpoint)
        assert metadata["network"]["actor_arch"] == [512, 512, 256, 128]
        assert metadata["environment"]["observation"]["layout"] == "paper"
        restored = load_checkpoint(callback.last_checkpoint, env)
        assert restored.policy.network == network
        restored.set_logger(logger)
        restored.learn(
            8, callback=TrainingCallback(env, tmp_path / "r", {}), reset_num_timesteps=False
        )
        fresh = load_checkpoint(callback.last_checkpoint, env, mode="warm_start", ppo_kwargs=kwargs)
        assert fresh.num_timesteps == 0 and fresh.policy.network == network
        with pytest.raises(ValueError, match="network"):
            load_checkpoint(callback.last_checkpoint, env, network=ARCH_PRESETS["default"])

        export = Path(export_actor(str(callback.last_checkpoint), str(tmp_path / "deploy")))
        config = load_config(CONFIG)
        config.controller.model_dir = str(export)
        ctrl = load_controller(CONTROLLER)
        env.reset()
        raw = {key: value[0] for key, value in env._current_obs.items()}
        controller = ctrl(raw, {}, config)
        features = encode_obs(raw, np.zeros(4), observation)
        expected_raw, _ = model.predict(env.normalizer.normalize(features), deterministic=True)
        expected = decode_action(expected_raw, raw["quat"], env.thrust_min, env.thrust_max)
        np.testing.assert_allclose(controller.compute_control(raw), expected, atol=1e-5)
    finally:
        env.close()
        logger.close()


@pytest.mark.integration
def test_sparse_reward_matches_events_in_real_simulation():
    env = Level2SBXEnv(
        CONFIG,
        n_envs=4,
        max_episode_steps=40,
        reward_config=SparseRewardConfig(),
        reset_config=ResetConfig(segment_init_prob=0),
    )
    try:
        env.seed(1)
        env.reset()
        returns = np.zeros(4)
        rng = np.random.default_rng(0)
        for _ in range(80):
            _, rewards, done, infos = env.step(rng.normal(size=(4, 4)).astype(np.float32))
            assert np.isfinite(rewards).all()
            returns += rewards
        means, episodes = env.drain_metrics()
        assert set(means) >= {"gate_pass", "lap", "crash", "cmd", "episode_cmd"}
        assert episodes and means["cmd"] < 0
        for row in episodes:
            if row["crash"]:
                assert row["episode"]["r"] < 0
    finally:
        env.close()


@pytest.mark.integration
def test_legacy_deploy_model_still_loads():
    deploy = CONFIG.parents[1] / "lsy_drone_racing/control/rl_single_l2/deploy_model"
    if not (deploy / "config.json").is_file():
        pytest.skip("No committed deploy_model")
    config = load_config(CONFIG)
    env = Level2SBXEnv(CONFIG, n_envs=1, max_episode_steps=3)
    try:
        env.reset()
        raw = {key: value[0] for key, value in env._current_obs.items()}
        controller = load_controller(CONTROLLER)(raw, {}, config)
        assert np.isfinite(controller.compute_control(raw)).all()
    finally:
        env.close()


@pytest.mark.integration
def test_eval_paper_reports_every_episode(tmp_path: Path):
    from lsy_drone_racing.control.rl_single_l2.eval_paper import evaluate

    deploy = CONFIG.parents[1] / "lsy_drone_racing/control/rl_single_l2/deploy_model"
    if not (deploy / "config.json").is_file():
        pytest.skip("No committed deploy_model")
    summary = evaluate(
        str(deploy), episodes=3, max_episode_steps=20, csv_path=str(tmp_path / "e.csv")
    )
    ends = sum(summary["crashes"].values()) + summary["timeouts"]
    assert ends + round(summary["success_rate"] * 3) == 3
    assert summary["crashes"]["other"] == 0
    assert len((tmp_path / "e.csv").read_text().splitlines()) == 4
