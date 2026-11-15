# Four-action racing with Brax PPO

Run commands from the repository root. Actions are normalized `[roll, pitch, yaw, thrust]`; attitude commands are absolute and independently bounded. The default observation has 49 values (two gates, four obstacles, previous four actions). Three gates gives 56 values. Export metadata carries the observation size and action limits; no controller rewrite is needed. Old three-action weights are incompatible.

Install: `pixi install --locked -e gpu -e gpu-tests -e default`.

## Single training run

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false SCIPY_ARRAY_API=1
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.train \
  --run_name=my_run --seed=1 --n_gates=2 \
  --pretrain_max_steps=500000000 --pretrain_time=0.2 --pretrain_crash=20 \
  --race_steps=2500000000 --race_gate=5 --race_lap=10 --race_time=0.3 --race_crash=20 \
  --roll_limit_deg=75 --pitch_limit_deg=75 --yaw_limit_deg=80 \
  --anneal_steps=400000000
```

Pretrain adds dense path progress; success >= switch_success switches to race. If pretraining exhausts its budget before switching, training stops without claiming a final race result. The race switch resets the critic and initially freezes the actor. Optional anneal keeps both networks, starts fresh Adam, warms up and decays learning rates. Starts do not automatically become all-official during anneal.

Outputs: `outputs/RUN/`, stage configs, `stages.json`, `checkpoints/{stage}_latest.pkl` (overwritten), `{stage}_final.pkl` (stage end), `final_evaluation.json`, `final_episodes.json`. No best-checkpoint selection. Budgets round up to Brax iterations. Final evaluation counts the first episode per world from official starts, default 3 x 1024 episodes.

Weight continuation: add `--start_stage=race --continue_race=True --start_params=PATH/race_final.pkl --warmup_iterations=5`, using a NEW run name and the parent's architecture, observation, action and reward settings. This restores parameters/normalizer, not Adam, RNG or the physical rollout. `start_step` only offsets logs. For anneal-only continuation use `--start_stage=anneal --anneal_steps=400000000` instead.

## Shared-filesystem, multi-machine sweep

Each machine runs independent trials on its own GPU. W&B assigns configurations from ONE shared cloud sweep; weights are not synchronized across machines within a trial. Use one agent per 5090 and a unique machine label. The wrapper creates directories automatically, including for sweep creation; do not use a bare wandb agent if you want this isolation.

Create the grid ONCE on this machine:

```bash
pixi run -e gpu wandb login
pixi run -e gpu python lsy_drone_racing/control/rl_race/sweep_agent.py \
  --machine 9-3-2 --create lsy_drone_racing/control/rl_race/sweeps/overnight.yaml
```

Copy the returned `ENTITY/lsy-drone-racing/SWEEP_ID`. On this machine:

```bash
tmux new -s drone-sweep
pixi run -e gpu python lsy_drone_racing/control/rl_race/sweep_agent.py \
  --machine 9-3-2 --sweep ENTITY/lsy-drone-racing/SWEEP_ID --hours 10
```

On every other machine, use the SAME sweep ID and a DIFFERENT label:

```bash
cd /u/halle/zshiy/home_at/projects/lsy_drone_racing
pixi run -e gpu python lsy_drone_racing/control/rl_race/sweep_agent.py \
  --machine 9-3-3 --sweep ENTITY/lsy-drone-racing/SWEEP_ID --hours 10
```

Use the already-installed shared Pixi environment; avoid concurrent environment installation. Authenticate on another host if necessary. Ctrl-b then d detaches tmux; `tmux attach -t drone-sweep` returns. `--hours` is optional: it interrupts the current trial at the deadline (exit124), which can leave only latest weights and no final report. `--count N` optionally limits trials on this agent. Without these limits agents work until the cloud grid is exhausted. Do not create a new sweep on each host.

Local SDK and agent files: `sweeps/wandb-MACHINE/`, with separate cache/staging/artifact directories. Trial files: `outputs/sweeps/SWEEP_ID/runs/RUN_ID_sSEED/`, independent of host. The final trial manifest records machine, seed and resolved training parameters. Machine identity is launcher metadata, not a grid search dimension. Concurrent agents need different labels even on one host.

### Grid and storage

`overnight.yaml` has 72 configurations: tied roll/pitch 70/75/80 degrees x yaw70/80/85 x n_gates2/3 x anneal0/400M x race_time0.3/0.4. One seed (1); repeat promising configurations with other seeds later. Actor256x3 and critic256x4 are fixed, using the architecture with a completed full-budget result; the wide-network comparison was unfinished, so this is a practical choice, not proof of superiority. LR3e-4, race actorLR1e-4, entropy0.005, official starts0.5, gate5/lap10/crash20 are fixed. Each trial gets up to500M pretrain +2.5B race, plus optional anneal. Grid completion depends on machine count and throughput; 72 full runs are not guaranteed in one night.

`tilt_limit_deg` is a sweep-only alias expanding to roll_limit_deg and pitch_limit_deg; do not combine it with separate axis limits. `upload_checkpoint`, `sweep_trial_count`, `cloud_checkpoint_budget_gb` are sweep-only controls (default true,1,50). The grid explicitly supplies72 for its storage estimate. If you enlarge the grid, update this count as well.

Only the final stage checkpoint is uploaded once per completed trial, with its config, stage log and evaluation as a W&B model artifact. No rollout, optimizer or repeated latest snapshots are uploaded. The adapter records actual bytes and estimated whole-sweep GB, skipping checkpoint upload above the50GB budget. This estimate excludes existing account artifacts and logs. Local weights remain available even if upload is disabled. Before deleting local runs, verify successful cloud artifact upload or export weights you want to keep.

Sort `final/leaderboard_score` ascending (success >=50%, then successful mean time); also inspect success rate. `final/balanced_score` uses an80% threshold. These scores only rank final evaluations; they do not change rewards or automatically select a best training checkpoint. Local large evaluation is a leaderboard proxy, not an official submission.

## Evaluation, export and recording

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.evaluation \
  --run_dir=PATH_TO_RUN --stage=anneal --episodes=2048 \
  --seeds='(900001,900002,900003)' --report_name=holdout
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.export \
  --run_dir=PATH_TO_RUN --stage=anneal --destination=PATH_TO_RUN/exported
```

Use `--stage=race` when anneal=0. Independent holdout writes separate reports. Export contains config.json and policy.msgpack (actor and normalizer). To use the default controller, export to `lsy_drone_racing/control/rl_race/deploy_model` with `--overwrite=True` after selecting a model. This replaces the deployed model.

```bash
pixi run -e default python scripts/sim.py --config=level2.toml \
  --controller=rl_race_controller.py --n_runs=20 --render=False
pixi run -e default evaluate
pixi run -e default python personal_tools/record_run.py --config level2.toml \
  --controller rl_race_controller.py --output-dir runs/my_run --n_runs 1 \
  --width 640 --height 360 --fps 30
pixi run -e default python personal_tools/replay_run.py \
  runs/my_run/run_0001/trajectory.npz --camera fpv
```

Recording needs ffmpeg/libx264 and EGL. See personal_tools/README.record_replay.md.

## Layout and design

| Files | Responsibility |
| --- | --- |
| env_wrapper/env.py | Official simulator adapter, Brax state, resets and metrics |
| env_wrapper/actions.py, observations.py | Four-action mapping and sensor-masked inputs |
| env_wrapper/rewards.py | Race reward, dense pretrain potential, optional lap-time bonus |
| env_wrapper/path.py, starts.py | Reference path, projection and safe middle starts |
| network.py | Actor/critic factories and exploration distribution |
| train.py, optim.py | Stage orchestration, Adam groups, freeze/warmup/decay |
| brax_ppo_train.py | Pinned Brax PPO with optimizer override hook |
| evaluation.py | Multi-seed first-episode final evaluation |
| sweep.py, sweep_agent.py, sweeps/ | Cloud trials, machine isolation, finite grids |
| export.py, controller.py | Portable array export and deterministic inference |
| tools/min_time_bound.py | Approximate dynamics-constrained time bound |
| research.py, tools/summarize_research.py | Optional finite local experiments and summaries |

Compared with rl_single_l2: Brax keeps sampling and optimization on GPU; absolute attitude replaces relative rotation increments. Path pretraining/middle starts, critic reset and optional anneal remain separate from deployment. Path geometry is a training reference, not an obstacle-avoidance or feasibility guarantee. Deployment uses masked sensors and no planner. Reward potentials penalize path distance to discourage off-course arc progress; failure/time costs can still produce degenerate policies. Control frequency is50Hz, physics normally500Hz.

## All training arguments

These are train.py signature defaults; the grid explicitly overrides its recipe and memory settings. Reward overrides of None use rewards.py presets: gate10, lap50, time0.05, angular velocity0.005, crash20, progress0 (15 in dense_path), path_lateral1, lap_bonus0, lap_ref4 seconds. Set explicit reward coefficients for reproducible comparisons. CLI lists use e.g. `--policy_layers='(256,256,256)'`.

| Argument | Default |
| --- | --- |
| `run_name` | `None` |
| `output_root` | `None` |
| `seed` | `0` |
| `device` | `'gpu'` |
| `roll_limit_deg` | `70.0` |
| `pitch_limit_deg` | `70.0` |
| `yaw_limit_deg` | `70.0` |
| `thrust_min_fraction` | `0.0` |
| `thrust_max_fraction` | `1.0` |
| `n_gates` | `2` |
| `official_start_prob` | `0.5` |
| `max_episode_steps` | `750` |
| `pretrain_max_steps` | `500000000` |
| `pretrain_crash` | `None` |
| `pretrain_time` | `None` |
| `pretrain_progress` | `None` |
| `pretrain_path_lateral` | `None` |
| `race_steps` | `1000000000` |
| `switch_success` | `0.3` |
| `critic_only_iterations` | `10` |
| `start_params` | `None` |
| `start_stage` | `'pretrain'` |
| `continue_race` | `False` |
| `start_step` | `0` |
| `lr_decay_to` | `1.0` |
| `warmup_iterations` | `0` |
| `anneal_steps` | `0` |
| `anneal_to` | `0.1` |
| `anneal_warmup_iterations` | `5` |
| `race_time` | `None` |
| `race_crash` | `None` |
| `race_gate` | `None` |
| `race_lap` | `None` |
| `race_lap_bonus` | `None` |
| `race_lap_ref` | `None` |
| `num_envs` | `32768` |
| `unroll_length` | `32` |
| `batch_size` | `8192` |
| `num_minibatches` | `32` |
| `num_updates_per_batch` | `4` |
| `learning_rate` | `0.0003` |
| `policy_lr_after_switch` | `0.0001` |
| `entropy_cost` | `0.005` |
| `noise_scale` | `1.0` |
| `discounting` | `0.99` |
| `gae_lambda` | `0.95` |
| `clipping_epsilon` | `0.2` |
| `max_grad_norm` | `1.0` |
| `policy_layers` | `(256, 256, 256)` |
| `value_layers` | `(256, 256, 256, 256)` |
| `activation` | `'swish'` |
| `eval_every_steps` | `20000000` |
| `num_eval_envs` | `512` |
| `final_eval_episodes` | `1024` |
| `final_eval_seeds` | `(100000, 200000, 300000)` |
| `final_eval_batch_size` | `256` |
| `wandb_project` | `None` |
| `wandb_entity` | `None` |
| `wandb_group` | `None` |

Brax `batch_size` counts sequences per optimizer minibatch, not individual SBX transitions. Samples per iteration = batch_size * num_minibatches * unroll_length; batch_size * num_minibatches must be divisible by num_envs. Larger batches change optimization, not just VRAM. SBX's old18432-env/18432-batch settings cannot be copied directly. The 5090 defaults are32768 environments,8192 sequences per minibatch,32 minibatches and32-step unrolls. Peak sampled process VRAM was24.3GiB; larger batches hit temporary-allocation OOM, so28–30GiB occupancy was not forced. See GPU_PROFILE.md for measured memory and smoke results.

## Checks and history

```bash
pixi run -e gpu-tests python -m pytest -q tests/unit/control/test_rl_race.py \
  tests/unit/control/test_rl_race_sweep.py tests/integration/test_rl_race_export.py
```

Previous baseline: seed1,70-degree limits, approximately2.684B actual steps; final3072-episode success90.30%, successful mean3.544s. Short continuation experiments suggested85-degree tilt could reach3.456s at90.23% on1536 validation episodes, but no independent confirmation was finished. These are historical local results, not new leaderboard claims. Old outputs and local W&B state were removed at the user's request; deploy_model was preserved. No commits or pushes were made.
