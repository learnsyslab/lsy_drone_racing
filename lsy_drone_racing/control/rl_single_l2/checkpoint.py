"""Atomic SBX checkpoints and actor-only competition exports.

Resume restarts physical episodes: simulator state is intentionally not serialized.
Only load model.zip/runtime.pkl from trusted local training runs (Python pickle).
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import serialization


def read_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata and verify that all required files are present."""
    path = Path(checkpoint_dir).expanduser().resolve()
    metadata = json.loads((path / "metadata.json").read_text())
    for name in ("model.zip", "runtime.pkl", "actor.msgpack"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return metadata


def save_checkpoint(
    checkpoint_dir: str | Path, model: Any, env: Any, config: dict[str, Any], *, keep: int = 3
) -> Path:
    """Publish a complete step directory atomically, then prune old siblings.

    SBX's native model stores both TrainStates (parameters and Adam moments), PPO
    settings and counters. Runtime state supplements RNGs that native loading
    reseeds, plus frozen/pending observation moments and callback state.
    """
    from lsy_drone_racing.control.rl_single_l2.policy import network_config

    if keep < 1:
        raise ValueError("keep must be positive")
    root = Path(checkpoint_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{model.num_timesteps:012d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
    try:
        model.save(temporary / "model.zip")
        (temporary / "actor.msgpack").write_bytes(
            serialization.to_bytes(model.policy.actor_state.params)
        )
        runtime = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "jax_key": np.asarray(model.key),
            "policy_key": np.asarray(model.policy.key),
            "noise_key": np.asarray(model.policy.noise_key),
        }
        import torch

        runtime["torch_rng"] = torch.get_rng_state()
        (temporary / "runtime.pkl").write_bytes(pickle.dumps(runtime))
        metadata = {
            "num_timesteps": model.num_timesteps,
            "n_updates": model._n_updates,
            "progress_remaining": model._current_progress_remaining,
            "config": config,
            "environment": env.get_config(),
            "network": network_config(getattr(model.policy, "network", None)),
            "normalizer": env.normalizer.state_dict(),
            "pending_normalizer": env._pending_normalizer.state_dict(),
            "callback_state": getattr(model, "l2_callback_state", {}),
            "physical_state_saved": False,
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    for old in sorted(root.glob("step_[0-9]*"))[:-keep]:
        if old.is_dir():
            shutil.rmtree(old)
    return target


def load_checkpoint(
    checkpoint_dir: str | Path,
    env: Any,
    *,
    mode: str = "resume",
    ppo_kwargs: dict[str, Any] | None = None,
    network: Any = None,
) -> Any:
    """Restore a run or copy its networks into freshly initialized optimizers.

    ``network`` (a NetworkConfig) defaults to the checkpoint's own network and
    must match it: parameters are copied layer by layer. A warm start may change
    the actor's mean bound and the action mapping (ActionConfig), which reuse the
    same parameter shapes; a resume must match exactly.
    """
    from sbx import PPO

    from lsy_drone_racing.control.rl_single_l2.policy import Level2Policy, network_from_config

    if mode not in {"resume", "warm_start"}:
        raise ValueError("mode must be resume or warm_start")
    path = Path(checkpoint_dir).expanduser().resolve()
    metadata = read_metadata(path)
    saved_network = network_from_config(metadata["network"])
    if network is not None and network != saved_network:
        if mode == "resume" or network.with_mean_bound(1.0) != saved_network.with_mean_bound(1.0):
            raise ValueError("Checkpoint network differs from the requested network")
    target_network = saved_network if network is None else network
    current = env.get_config()
    keys = ("observation", "thrust_min", "thrust_max", "observation_dim", "action_dim", "freq")
    from lsy_drone_racing.control.rl_single_l2.env_wrapper import ActionConfig

    saved_env = dict(metadata["environment"])
    # Earlier checkpoints store only alpha_max_rad; the gain defaults to it.
    saved_env["action"] = asdict(ActionConfig(**saved_env["action"]))
    # Earlier checkpoints predate reward modes and observation layouts.
    saved_env.setdefault("reward_mode", "dense")
    saved_env["observation"] = {"layout": "current", "paper_obstacles": False} | saved_env[
        "observation"
    ]
    if mode == "resume":
        keys += ("action", "reward_mode", "reward", "max_episode_steps", "race_config")
    for key in keys:
        if saved_env[key] != current[key]:
            raise ValueError(f"Checkpoint environment setting differs: {key}")
    if mode == "resume":
        from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig

        # Earlier checkpoints used only official starts; absence means probability zero.
        saved_reset = metadata["environment"].get(
            "reset", asdict(ResetConfig(segment_init_prob=0.0))
        )
        if saved_reset != current["reset"]:
            raise ValueError("Checkpoint reset settings differ; use warm_start to change them")
    model = PPO.load(path / "model.zip", env=env, device="cpu", force_reset=True)
    if mode == "warm_start":
        fresh = PPO(
            Level2Policy,
            env,
            policy_kwargs={"network": target_network},
            **(ppo_kwargs or metadata["config"]["ppo"]),
        )
        fresh.policy.actor_state = fresh.policy.actor_state.replace(
            params=model.policy.actor_state.params
        )
        fresh.policy.vf_state = fresh.policy.vf_state.replace(params=model.policy.vf_state.params)
        model = fresh
    else:
        runtime = pickle.loads((path / "runtime.pkl").read_bytes())
        model.key = jax.numpy.asarray(runtime["jax_key"])
        model.policy.key = jax.numpy.asarray(runtime["policy_key"])
        model.policy.noise_key = jax.numpy.asarray(runtime["noise_key"])
        random.setstate(runtime["python_rng"])
        np.random.set_state(runtime["numpy_rng"])
        import torch

        torch.set_rng_state(runtime["torch_rng"])
        model.l2_callback_state = metadata["callback_state"]
        if getattr(model.policy, "network", saved_network) != saved_network:
            raise ValueError("Restored policy network differs from checkpoint metadata")
        if (
            model.num_timesteps != metadata["num_timesteps"]
            or model._n_updates != metadata["n_updates"]
        ):
            raise ValueError("Native SBX restore did not preserve training counters")
        env.restore_pending_normalizer(metadata["pending_normalizer"])
    env.set_normalizer(metadata["normalizer"])
    return model


def export_actor(checkpoint: str, destination: str | None = None) -> str:
    """Export actor parameters, frozen moments and JSON configs, without a critic.

    Run with ``python -m lsy_drone_racing.control.rl_single_l2.checkpoint
    --checkpoint=... --destination=...``. Existing exports are preserved; choose
    a new destination to compare submissions.
    """
    path = Path(checkpoint).expanduser().resolve()
    metadata = read_metadata(path)
    target = (
        Path(destination).expanduser().resolve()
        if destination
        else Path(__file__).resolve().parent / "deploy_model"
    )
    if target.exists():
        raise FileExistsError(f"Export destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=target.parent))
    try:
        shutil.copyfile(path / "actor.msgpack", temporary / "actor.msgpack")
        bundle = {
            key: metadata[key] for key in ("environment", "network", "normalizer", "num_timesteps")
        }
        (temporary / "config.json").write_text(json.dumps(bundle, indent=2) + "\n")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target)


if __name__ == "__main__":
    import fire

    fire.Fire(export_actor)
