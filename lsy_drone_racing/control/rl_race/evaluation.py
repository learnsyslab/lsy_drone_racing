"""Independent final evaluation: count the first episode per world, no best selection."""

import json
import pickle
from pathlib import Path
from typing import Any

import fire
import jax
import jax.numpy as jnp
import numpy as np

from .controller import policy_mean_action
from .env_wrapper.actions import ActionConfig
from .env_wrapper.env import EnvConfig, RaceEnv
from .env_wrapper.observations import ObsConfig
from .env_wrapper.rewards import RewardConfig
from .env_wrapper.starts import StartConfig
from .export import policy_arrays


def ranking_score(
    success_rate: float, lap_time: float | None, minimum_success: float = 0.5
) -> float:
    """Minimize time among eligible models; penalize ineligible or zero-success models."""
    if success_rate >= minimum_success and lap_time is not None:
        return float(lap_time)
    return 100.0 + 100.0 * max(minimum_success - success_rate, 0.0)


def evaluate(
    run_dir: str,
    stage: str = "race",
    episodes: int = 1024,
    seeds: tuple[int, ...] = (100000, 200000, 300000),
    batch_size: int = 256,
    device: str = "gpu",
    report_name: str = "final",
) -> dict:
    """Evaluate `episodes` official-start episodes per seed; write JSON and episode rows."""
    if episodes < 1 or batch_size < 1 or not seeds:
        raise ValueError("Positive episode count, batch size and nonempty seeds required")
    if not report_name or Path(report_name).name != report_name or report_name in {".", ".."}:
        raise ValueError("report_name must be a simple filename prefix")
    run = Path(run_dir).resolve()
    config = json.loads((run / f"{stage}_config.json").read_text())
    c = config["env"]
    cfg = EnvConfig(
        max_episode_steps=c["max_episode_steps"],
        action=ActionConfig(**c["action"]),
        obs=ObsConfig(**c["obs"]),
        reward=RewardConfig(**c["reward"]),
        start=StartConfig(official_prob=1.0),
    )
    arrays = jax.tree.map(
        jnp.asarray,
        policy_arrays(pickle.loads((run / "checkpoints" / f"{stage}_final.pkl").read_bytes())),
    )
    count = min(batch_size, episodes)
    env = RaceEnv(count, cfg, device=device)

    @jax.jit
    def batch(key: jax.Array) -> tuple:
        state = env.reset(jax.random.split(key, count))
        initial = (
            state,
            jnp.zeros(count, bool),
            jnp.zeros(count),
            jnp.zeros(count),
            jnp.zeros(count),
        )

        def step(carry: Any, _: Any) -> tuple:
            state, seen, success, times, gates = carry
            action = policy_mean_action(arrays, state.obs, config["network"]["activation"])
            state = env.step(state, action)
            first = (~seen) & (state.done > 0)
            success = jnp.where(first, state.metrics["success"], success)
            times = jnp.where(first, state.info["steps"] / env.freq, times)
            gates = gates + jnp.where(~seen, state.metrics["gates"], 0)
            return (state, seen | first, success, times, gates), None

        (_, seen, success, times, gates), _ = jax.lax.scan(
            step, initial, None, length=cfg.max_episode_steps
        )
        return seen, success, times, gates

    rows = []
    for seed in seeds:
        for offset in range(0, episodes, count):
            key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), offset // count)
            seen, success, times, gates = map(np.asarray, batch(key))
            n = min(count, episodes - offset)
            if not seen[:n].all():
                raise RuntimeError("Final evaluation failed to finish all first episodes")
            rows.extend(
                dict(
                    seed=int(seed),
                    episode=offset + i,
                    success=bool(success[i]),
                    time=float(times[i]),
                    gates=float(gates[i]),
                )
                for i in range(n)
            )
    successful = [r["time"] for r in rows if r["success"]]
    report = dict(
        stage=stage,
        report_name=report_name,
        batch_size=count,
        episodes=len(rows),
        episodes_per_seed=episodes,
        seeds=list(seeds),
        success_rate=sum(r["success"] for r in rows) / len(rows),
        mean_success_time=float(np.mean(successful)) if successful else None,
        std_success_time=float(np.std(successful)) if successful else None,
        mean_gates=float(np.mean([r["gates"] for r in rows])),
        checkpoint=str(run / "checkpoints" / f"{stage}_final.pkl"),
        per_seed={
            str(s): dict(
                success_rate=np.mean([r["success"] for r in rows if r["seed"] == s]).item()
            )
            for s in seeds
        },
    )
    # Match the local official evaluator's >=50% eligibility; keep reliability visible separately.
    rate = report["success_rate"]
    lap = report["mean_success_time"]
    report["leaderboard_score"] = ranking_score(rate, lap)
    report["balanced_score"] = ranking_score(rate, lap, 0.8)
    (run / f"{report_name}_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (run / f"{report_name}_episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    fire.Fire(evaluate)
