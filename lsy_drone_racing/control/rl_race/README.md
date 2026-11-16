# Racing with Brax PPO

Run commands from the repository root. The policy outputs four normalized actions
`[roll, pitch, yaw, thrust]`. Observations have 49 values with two gates, or 56 with three.

## Single run

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false SCIPY_ARRAY_API=1
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.train \
  --run_name=my_run --seed=1 \
  --pretrain_max_steps=500000000 --pretrain_time=0.2 --pretrain_crash=20 \
  --race_steps=2500000000 --race_gate=5 --race_lap=10 --race_time=0.3 --race_crash=20 \
  --roll_limit_deg=75 --pitch_limit_deg=75 --yaw_limit_deg=80
```

Pretrain uses dense path rewards until `switch_success` is reached, then race resets
and trains the critic before unfreezing the actor. Add `--anneal_steps=400000000` for
learning-rate decay after race. If pretrain fails to reach the threshold, training stops.
See `train.py` or append `--help` for all parameters and their defaults.

Results are in `lsy_drone_racing/control/rl_race/outputs/RUN/`:
stage configs, `stages.json`, `checkpoints/{stage}_{latest,final}.pkl`,
`final_evaluation.json` and `final_episodes.json`. Brax rounds budgets up to complete
training/evaluation intervals. Final evaluation defaults to 1024 official-start episodes
for each of three seeds.

For a continuation, use a new run name and the same network/environment parameters:
`--start_stage=race --continue_race=True --start_params=PATH/race_final.pkl`.
This restores weights and normalization; Adam and rollout state start fresh.
For anneal only, use `--start_stage=anneal --start_params=PATH/race_final.pkl --anneal_steps=400000000`.

## W&B sweep

Create once:

```bash
pixi run -e gpu wandb login
pixi run -e gpu python lsy_drone_racing/control/rl_race/sweep_agent.py \
  --machine 9-2-3 --create lsy_drone_racing/control/rl_race/sweeps/overnight.yaml
```

Join the returned sweep ID from each machine, with a unique machine label:

```bash
pixi run -e gpu python lsy_drone_racing/control/rl_race/sweep_agent.py \
  --machine 9-2-3 --sweep ENTITY/lsy-drone-racing/SWEEP_ID
```

Run in tmux to survive terminal disconnects. `--count N` limits trials on that agent;
`--hours H` interrupts the current trial at the deadline. Without limits, agents work
until the sweep finishes. Each agent runs one trial at a time. Keep the shared sweep ID
and use the installed Pixi environment on all hosts.

`overnight.yaml` contains the 72-setting grid; `example.yaml` is a smaller random-search
template. `tilt_limit_deg` sets roll and pitch together. Agent logs/cache are isolated
under `sweeps/wandb-MACHINE/`; training outputs are under
`outputs/sweeps/SWEEP_ID/runs/RUN_ID_sSEED/`.

Completed trials attach the final checkpoint, config, stage log and evaluation as a W&B
model artifact. `upload_checkpoint` can disable weight uploads; `sweep_trial_count` and
`cloud_checkpoint_budget_gb` limit the estimated total weight storage (defaults: 1 and 50 GB).
These three options and `tilt_limit_deg` belong to the sweep adapter, not `train.py`.
Sort `final/leaderboard_score` ascending (at least 50% success, then successful mean time);
`final/balanced_score` uses an 80% success threshold. Check `final/success_rate` as well.

Brax `batch_size` counts sequences per minibatch. One PPO iteration samples
`batch_size * num_minibatches * unroll_length` transitions. Current defaults are
32768 environments and batch 8192 (about 24.3 GiB peak in a short RTX 5090 probe).
The historical batch was 1024: increasing it changes optimization and has not been
shown to reproduce the old success rate. Memory-fit checks do not establish learning quality.

## Evaluate and export

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.evaluation \
  --run_dir=PATH_TO_RUN --stage=race --episodes=1024 \
  --seeds='(900001,900002,900003)' --report_name=holdout
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.export \
  --run_dir=PATH_TO_RUN --stage=race --destination=PATH_TO_RUN/exported
```

Use `--stage=anneal` for an annealed model. Evaluation writes separate
`holdout_evaluation.json` and `holdout_episodes.json`; times are measured in seconds.
Export writes `config.json` and `policy.msgpack` for `controller.py`.
To deploy, set the destination to `lsy_drone_racing/control/rl_race/deploy_model` and pass
`--overwrite=True` to replace the current deployment. Recording instructions are in
[personal_tools/README.record_replay.md](../../../personal_tools/README.record_replay.md).

## Implementation and checks

`env_wrapper/` adapts the simulator and defines observations, actions, rewards and starts.
`network.py` defines the actor/critic; `optim.py` provides separate learning rates.
`brax_ppo_train.py` is the pinned Brax trainer with the optimizer override required by
`train.py`. `sweep.py` adapts W&B configs; `sweep_agent.py` launches machine-isolated agents.

```bash
SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu WANDB_MODE=disabled \
pixi run -e gpu-tests python -m pytest -q \
  tests/unit/control/test_rl_race*.py tests/integration/test_rl_race*.py
```
