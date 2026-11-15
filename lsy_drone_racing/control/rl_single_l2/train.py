"""Train Level 2 PPO with SBX; all output paths are owned by this entry point."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import fire
from sbx import PPO
from stable_baselines3.common.logger import configure

from lsy_drone_racing.control.rl_single_l2.callbacks import TrainingCallback
from lsy_drone_racing.control.rl_single_l2.checkpoint import load_checkpoint, read_metadata
from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ActionConfig,
    Level2SBXEnv,
    ObservationConfig,
)
from lsy_drone_racing.control.rl_single_l2.policy import (
    ARCH_PRESETS,
    Level2Policy,
    NetworkConfig,
    network_config,
    network_from_config,
)
from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig
from lsy_drone_racing.control.rl_single_l2.reward import (
    REWARD_MODES,
    RewardConfig,
    SparseRewardConfig,
    reward_config_from_dict,
)

RL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = RL_DIR.parents[2] / "config" / "level2.toml"


def train(
    run_name: str | None = None,
    total_timesteps: int = 300_000_000,
    n_envs: int = 18432,
    n_steps: int = 256,
    batch_size: int = 18432,
    learning_rate: float = 3e-4,
    n_epochs: int = 3,
    clip_range: float = 0.2,
    ent_coef: float = 0.005,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    seed: int = 42,
    device: str = "cpu",
    config: str | None = None,
    max_episode_steps: int = 1500,
    save_freq: int = 20_000_000,
    keep: int = 3,
    checkpoint: str | None = None,
    mode: str = "new",
    output_root: str | None = None,
    eval_freq: int = 0,
    eval_episodes: int = 5,
    eval_seed: int = 10_000,
    gate_lookahead: int = 2,
    obstacle_count: int = 4,
    alpha_max_rad: float = 0.30,
    alpha_gain_rad: float | None = 0.16,
    mean_bound: float = 3.0,
    start_timesteps: int = 0,
    segment_init_prob: float = 0.40,
    segment_init_offset_m: float = 0.50,
    segment_init_vel_mps: float = 2.50,
    segment_init_perturb_m: float = 0.10,
    reward_mode: str = "dense",
    gate_pass_leading: float = 10.0,
    gate_pass_trailing: float = 5.0,
    lap_leading: float = 50.0,
    crash_penalty: float = 2.0,
    cmd_roll_pitch: float = 0.15,
    cmd_yaw: float = 0.05,
    cmd_scale_by_dt: bool = True,
    single_agent_leading: bool = True,
    obs_layout: str = "current",
    paper_obs_obstacles: bool = False,
    arch: str = "default",
    actor_arch: tuple[int, ...] | None = None,
    critic_arch: tuple[int, ...] | None = None,
    activation: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str = "online",
) -> str:
    """Run PPO and return the final complete checkpoint directory.

    total_timesteps is an absolute target (including restored steps for resume).
    PPO rounds it up to a whole rollout. device selects the JAX simulator/policy
    backend; SBX's rollout-buffer tensors stay on CPU. Resume reuses the saved
    PPO/environment/network settings; warm_start uses the supplied new settings
    (the network must match the source checkpoint).

    reward_mode="dense" is the six-term DS baseline (unchanged). "sparse" is the
    task-level reward of Pasumarti et al. (2026); the sparse coefficients are the
    gate_pass_*, lap_leading, crash_penalty, cmd_* and single_agent_leading flags
    (ignored in dense mode). With a sparse reward there is no time penalty, so
    the discount ``gamma`` is the key speed knob: a gate reached k steps later is
    worth gamma**k less.

    obs_layout="paper" selects body-frame gate-corner observations, optionally
    with paper_obs_obstacles. arch selects a NetworkConfig preset ("default":
    (256, 256) leaky ReLU; "paper": actor 512-512-256-128, critic
    512-512-256-256-128-128, ELU); actor_arch/critic_arch/activation override it.

    Action limits: the actor mean is mean_bound * tanh(z / mean_bound) (1.0 is
    the original tanh mean, which caps deterministic thrust at 73 % of the
    range; 3.0 reaches 95 %). The per-step rotation saturates at alpha_max_rad
    (rate limit alpha_max_rad * freq) with small-signal gain alpha_gain_rad
    (None means equal to alpha_max_rad, the original mapping). A warm start may
    change all three; parameters are reused unchanged.

    start_timesteps (warm_start only) continues the step counter from a
    lineage's cumulative steps, e.g. a 200M-step model warm-started from a
    100M checkpoint: logging, W&B, checkpoint names and the absolute
    total_timesteps then count from there.

    wandb_project enables live Weights & Biases logging of everything written to
    progress.csv (requires ``wandb login`` or WANDB_API_KEY); wandb_mode may be
    "online", "offline" or "disabled".
    """
    if mode not in {"new", "resume", "warm_start"}:
        raise ValueError("mode must be new, resume or warm_start")
    if start_timesteps and mode != "warm_start":
        raise ValueError("start_timesteps applies only to warm_start")
    if start_timesteps < 0 or (start_timesteps and total_timesteps <= start_timesteps):
        raise ValueError("total_timesteps must exceed a nonnegative start_timesteps")
    if (checkpoint is None) != (mode == "new"):
        raise ValueError("Provide a checkpoint exactly when mode is resume or warm_start")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if (
        total_timesteps < 1
        or min(n_envs, n_steps, n_epochs, max_episode_steps, keep, eval_episodes) < 1
    ):
        raise ValueError("Training counts and episode limits must be positive")
    if save_freq < 0 or eval_freq < 0 or learning_rate <= 0 or not 0 < clip_range < 1:
        raise ValueError("Invalid save/eval interval, learning rate or PPO clipping range")
    config_path = Path(config).expanduser().resolve() if config else DEFAULT_CONFIG
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
    if arch not in ARCH_PRESETS:
        raise ValueError(f"arch must be one of {sorted(ARCH_PRESETS)}")
    preset = ARCH_PRESETS[arch]
    network = NetworkConfig(
        actor_arch=preset.actor_arch if actor_arch is None else tuple(actor_arch),
        critic_arch=preset.critic_arch if critic_arch is None else tuple(critic_arch),
        activation=activation or preset.activation,
        mean_bound=mean_bound,
    )
    observation = ObservationConfig(
        gate_lookahead=gate_lookahead,
        obstacle_count=obstacle_count,
        layout=obs_layout,
        paper_obstacles=bool(paper_obs_obstacles),
    )
    action = ActionConfig(alpha_max_rad=alpha_max_rad, alpha_gain_rad=alpha_gain_rad)
    reward: RewardConfig | SparseRewardConfig = (
        SparseRewardConfig(
            gate_pass_leading=gate_pass_leading,
            gate_pass_trailing=gate_pass_trailing,
            lap_leading=lap_leading,
            crash=crash_penalty,
            cmd_roll_pitch=cmd_roll_pitch,
            cmd_yaw=cmd_yaw,
            cmd_scale_by_dt=bool(cmd_scale_by_dt),
            single_agent_leading=bool(single_agent_leading),
        )
        if reward_mode == "sparse"
        else RewardConfig()
    )
    reset = ResetConfig(
        segment_init_prob=segment_init_prob,
        segment_init_offset_m=segment_init_offset_m,
        segment_init_vel_mps=segment_init_vel_mps,
        segment_init_perturb_m=segment_init_perturb_m,
    )
    ppo_kwargs = dict(
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device="cpu",
        verbose=1,
    )
    if mode == "resume":
        saved = read_metadata(checkpoint)
        ppo_kwargs = saved["config"]["ppo"]
        old_env = saved["environment"]
        observation = ObservationConfig(**old_env["observation"])
        action = ActionConfig(**old_env["action"])
        reward = reward_config_from_dict(old_env.get("reward_mode", "dense"), old_env["reward"])
        network = network_from_config(saved["network"])
        reset = (
            ResetConfig(**old_env["reset"])
            if "reset" in old_env
            else ResetConfig(segment_init_prob=0.0)
        )
        max_episode_steps = old_env["max_episode_steps"]
        n_envs = saved["config"]["n_envs"]
        seed = ppo_kwargs["seed"]
        if config is None:
            config_path = Path(old_env["config_path"])
        if total_timesteps <= saved["num_timesteps"]:
            raise ValueError("Resume total_timesteps must exceed the restored cumulative steps")
    rollout_size = n_envs * ppo_kwargs["n_steps"]
    if ppo_kwargs["batch_size"] < 2 or rollout_size % ppo_kwargs["batch_size"]:
        raise ValueError("batch_size must be >= 2 and divide n_envs * n_steps exactly")
    # Name the effective network (resume restores it; overrides make it custom).
    arch = next(
        (
            name
            for name, preset in ARCH_PRESETS.items()
            if preset == network.with_mean_bound(preset.mean_bound)
        ),
        "custom",
    )
    if run_name is None:
        # Default names record the compared settings: reward, observations, network, seed.
        tags = ["l2", "ss" if isinstance(reward, SparseRewardConfig) else "ds"]
        tags += [f"obs-{observation.layout}", f"arch-{arch}", f"s{seed}"]
        run_name = "_".join(tags) + datetime.now().strftime("_%Y%m%d_%H%M%S")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    root = Path(output_root).expanduser().resolve() if output_root else RL_DIR / "outputs"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir, checkpoint_dir = run_dir / "logs", run_dir / "checkpoints"
    log_dir.mkdir()
    checkpoint_dir.mkdir()
    env = eval_env = logger = wandb_run = None
    # JAX policy operations and simulator allocations use the selected backend.
    import jax

    try:
        with jax.default_device(jax.devices(device)[0]):
            env_kwargs = dict(
                config_path=config_path,
                device=device,
                max_episode_steps=max_episode_steps,
                observation_config=observation,
                action_config=action,
                reward_config=reward,
            )
            env = Level2SBXEnv(n_envs=n_envs, seed=seed, reset_config=reset, **env_kwargs)
            model = (
                PPO(Level2Policy, env, policy_kwargs={"network": network}, **ppo_kwargs)
                if mode == "new"
                else load_checkpoint(
                    checkpoint, env, mode=mode, ppo_kwargs=ppo_kwargs, network=network
                )
            )
            effective = {
                "run_name": run_name,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "device": device,
                "mode": mode,
                "source_checkpoint": checkpoint,
                "start_timesteps": start_timesteps,
                "ppo": ppo_kwargs,
                "environment": env.get_config(),
                "network": network_config(network),
                "arch": arch,
                "save_freq": save_freq,
                "keep": keep,
                "eval_freq": eval_freq,
                "eval_episodes": eval_episodes,
                "eval_seed": eval_seed,
            }
            (run_dir / "config.json").write_text(json.dumps(effective, indent=2) + "\n")
            logger = configure(str(log_dir), ["stdout", "log", "csv", "json"])
            if wandb_project:
                from lsy_drone_racing.control.rl_single_l2.wandb_logger import (
                    WandbOutputFormat,
                    init_wandb,
                )

                wandb_run = init_wandb(
                    wandb_project,
                    run_name,
                    effective,
                    str(run_dir),
                    entity=wandb_entity,
                    group=wandb_group,
                    tags=[env.reward_mode, f"obs-{observation.layout}", f"arch-{arch}"],
                    mode=wandb_mode,
                )
                logger.output_formats.append(WandbOutputFormat(wandb_run))
            model.set_logger(logger)
            if eval_freq:
                eval_env = Level2SBXEnv(
                    n_envs=1,
                    seed=eval_seed,
                    reset_config=ResetConfig(segment_init_prob=0.0),
                    **env_kwargs,
                )
            callback = TrainingCallback(
                env,
                checkpoint_dir,
                effective,
                save_freq,
                keep,
                eval_env,
                eval_freq,
                eval_episodes,
                eval_seed,
            )
            if start_timesteps:
                # Continue the lineage's counter; SB3 then resets only the env.
                model.num_timesteps = start_timesteps
            remaining = total_timesteps - model.num_timesteps
            continue_counter = mode == "resume" or bool(start_timesteps)
            model.learn(remaining, callback=callback, reset_num_timesteps=not continue_counter)
            if wandb_run is not None:
                wandb_run.summary["last_checkpoint"] = str(callback.last_checkpoint)
                wandb_run.finish()
                wandb_run = None
            return str(callback.last_checkpoint)
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    fire.Fire(train)
