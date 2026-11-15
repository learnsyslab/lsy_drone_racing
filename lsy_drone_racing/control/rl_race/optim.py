"""Optimizer with separate policy/value learning rates and an optional critic-only phase.

Brax PPO trains both networks with one Adam and one learning rate. After the switch from the
pre-training reward to the race reward the critic is re-initialized; it is then trained alone for
a few PPO iterations (policy learning rate 0) before the policy resumes with a lower rate.
"""

from __future__ import annotations

from typing import Any

import jax
import optax


def iteration_updates(num_updates_per_batch: int, num_minibatches: int) -> int:
    """Optimizer steps per PPO iteration (epochs x minibatches)."""
    return num_updates_per_batch * num_minibatches


def _labels(params: Any) -> Any:
    """Label every leaf of PPONetworkParams(policy=..., value=...) with its network."""
    return type(params)(
        policy=jax.tree.map(lambda _: "policy", params.policy),
        value=jax.tree.map(lambda _: "value", params.value),
    )


def lr_schedule(
    lr: float,
    freeze_steps: int = 0,
    warmup_steps: int = 0,
    decay_steps: int = 0,
    final_lr_factor: float = 1.0,
) -> optax.Schedule:
    """Schedule freeze, warmup and decay per optimizer step.

    After warmup, apply constant rates or linear decay to ``final_lr_factor * lr``
    over ``decay_steps`` (constant afterwards).
    """
    parts: list[Any] = []
    boundaries: list[int] = []
    if freeze_steps > 0:
        parts.append(optax.constant_schedule(0.0))
        boundaries.append(freeze_steps)
    if warmup_steps > 0:
        parts.append(optax.linear_schedule(0.0, lr, warmup_steps))
        boundaries.append(freeze_steps + warmup_steps)
    if decay_steps > 0 and final_lr_factor != 1.0:
        parts.append(optax.linear_schedule(lr, lr * final_lr_factor, decay_steps))
    else:
        parts.append(optax.constant_schedule(lr))
    if len(parts) == 1:
        return parts[0]
    return optax.join_schedules(parts, boundaries=boundaries)


def lr_schedules(
    policy_lr: float,
    value_lr: float,
    freeze_policy_steps: int = 0,
    decay_steps: int = 0,
    final_lr_factor: float = 1.0,
    warmup_steps: int = 0,
) -> tuple[optax.Schedule, optax.Schedule]:
    """Policy and value learning-rate schedules of ``separate_lr`` (also used for logging)."""
    policy = lr_schedule(policy_lr, freeze_policy_steps, warmup_steps, decay_steps, final_lr_factor)
    value = lr_schedule(value_lr, 0, warmup_steps, decay_steps, final_lr_factor)
    return policy, value


def separate_lr(
    policy_lr: float,
    value_lr: float,
    max_grad_norm: float | None = 0.5,
    freeze_policy_steps: int = 0,
    decay_steps: int = 0,
    final_lr_factor: float = 1.0,
    warmup_steps: int = 0,
) -> optax.GradientTransformation:
    """Global-norm clipping, then Adam per network.

    The policy is frozen (learning rate 0) for the first ``freeze_policy_steps`` optimizer steps.
    Both learning rates then ramp up linearly from 0 over ``warmup_steps`` (a fresh Adam after a
    restart has no moment estimates) and finally, with ``decay_steps > 0``, decay linearly to
    ``final_lr_factor`` times their initial value over ``decay_steps`` optimizer steps.
    """
    policy_schedule, value_schedule = lr_schedules(
        policy_lr, value_lr, freeze_policy_steps, decay_steps, final_lr_factor, warmup_steps
    )
    per_network = optax.multi_transform(
        {"policy": optax.adam(policy_schedule), "value": optax.adam(value_schedule)}, _labels
    )
    if max_grad_norm is None:
        return per_network
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), per_network)
