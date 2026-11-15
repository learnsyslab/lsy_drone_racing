"""Forward SB3 logger dumps to Weights & Biases for live monitoring.

Every ``logger.dump`` (once per PPO rollout, plus evaluations) becomes one
``wandb.log`` call at step = total environment transitions, so W&B shows the
same reward components, episode statistics, evaluation metrics, action stds
and PPO losses as progress.csv, with no extra host transfers in the step loop.
"""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
from stable_baselines3.common.logger import KVWriter


class WandbOutputFormat(KVWriter):
    """SB3 KVWriter that logs numeric scalars to an active wandb run."""

    def __init__(self, run: Any) -> None:
        """Wrap an initialized ``wandb.Run``."""
        self.run = run

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0
    ) -> None:
        """Log scalar values; strings (e.g. checkpoint paths) go to the run summary."""
        data: dict[str, float] = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key) or ()
            if "wandb" in excluded:
                continue
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            if isinstance(value, Number) and not isinstance(value, bool):
                data[key] = float(value)
            elif isinstance(value, str):
                self.run.summary[key] = value
        if data:
            self.run.log(data, step=int(step))

    def close(self) -> None:
        """The run is finished by train.py, which owns its lifetime."""


def init_wandb(
    project: str,
    run_name: str,
    config: dict[str, Any],
    run_dir: str,
    *,
    entity: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
) -> Any:
    """Start a wandb run that records the full effective training configuration."""
    import wandb

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        config=config,
        dir=run_dir,
        mode=mode,
    )
