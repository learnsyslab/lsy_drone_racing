"""Export a trained policy to plain arrays for the competition controller (no Brax needed).

    python -m lsy_drone_racing.control.rl_race.export --run_dir=outputs/<run> [--stage=race]
        [--checkpoint=final|latest] [--destination=deploy_model]

The export holds the observation normalizer (mean, std), the policy MLP layers and a JSON config
with the observation/action layout. ``controller.policy_mean_action`` reproduces Brax's
deterministic ``tanh_normal`` action: normalize, run the MLP, take tanh of the mean half.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from flax import serialization

from .env_wrapper.actions import ACTION_ORDER, ACTION_SIZE

RL_DIR = Path(__file__).resolve().parent


def policy_arrays(params: Any) -> dict[str, Any]:
    """Extract (normalizer, policy) from Brax params ``(normalizer, policy, value)`` as numpy."""
    normalizer, policy, _ = params
    layers = policy["params"]
    names = sorted(layers, key=lambda name: int(name.split("_")[-1]))
    return {
        "mean": np.asarray(normalizer.mean, np.float32),
        "std": np.asarray(normalizer.std, np.float32),
        "layers": [
            {
                "kernel": np.asarray(layers[n]["kernel"], np.float32),
                "bias": np.asarray(layers[n]["bias"], np.float32),
            }
            for n in names
        ],
    }


def export(
    run_dir: str,
    stage: str = "race",
    checkpoint: str = "final",
    destination: str | None = None,
    overwrite: bool = False,
) -> str:
    """Write ``config.json`` and ``policy.msgpack`` for a run's checkpoint; return the path."""
    run = Path(run_dir).expanduser().resolve()
    target = Path(destination).expanduser().resolve() if destination else RL_DIR / "deploy_model"
    if target == run or target in run.parents:
        raise ValueError("Export destination must not contain the source run directory")
    config = json.loads((run / f"{stage}_config.json").read_text())
    params = pickle.loads((run / "checkpoints" / f"{stage}_{checkpoint}.pkl").read_bytes())
    arrays = policy_arrays(params)
    from .env_wrapper.observations import ObsConfig

    if arrays["mean"].shape != (ObsConfig(**config["env"]["obs"]).size,) or arrays["layers"][-1][
        "bias"
    ].shape != (8,):
        raise ValueError("Checkpoint must use four actions and the matching observation layout")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite=True to replace it")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    bundle = {
        "format_version": 2,
        "action_order": list(ACTION_ORDER),
        "action_dim": ACTION_SIZE,
        "observation_dim": int(arrays["mean"].shape[0]),
        "obs": config["env"]["obs"],
        "action": {
            **{
                k: config["env"]["action"][k]
                for k in ("roll_limit_deg", "pitch_limit_deg", "yaw_limit_deg")
            },
            "thrust_min": config["thrust_min"],
            "thrust_max": config["thrust_max"],
        },
        "network": config["network"],
        "freq": config["freq"],
        "source": {"run_dir": str(run), "stage": stage, "checkpoint": checkpoint},
    }
    (target / "config.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (target / "policy.msgpack").write_bytes(serialization.msgpack_serialize(arrays))
    return str(target)


if __name__ == "__main__":
    fire.Fire(export)
