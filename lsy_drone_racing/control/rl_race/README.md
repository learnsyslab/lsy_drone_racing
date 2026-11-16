# Drone racing with Brax PPO

Four-action policy `[roll, pitch, yaw, thrust]`. Training runs **pretrain → race → optional anneal**.
Run all commands from the repository root.

## Setup

```bash
pixi install --locked -e gpu -e gpu-tests -e default
export XLA_PYTHON_CLIENT_PREALLOCATE=false SCIPY_ARRAY_API=1
```

The current defaults peaked at **12.75 GiB** in a short staged RTX 5090 test.
This checks memory and execution, not final training quality or compatibility with every 16 GB GPU.

## 1. Train one model

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.train \
  --run_name=race_s1 --seed=1
```

All other options use the defaults below. Override any option with `--name=value`:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.train \
  --run_name=race_custom --seed=2 --race_steps=10000000000 \
  --policy_layers='(256,256,256)' --anneal_steps=400000000
```

Pretrain stops early when success reaches the switching threshold; if the threshold is not
reached within its budget, race is skipped. Weight continuation requires a new run name,
matching network/environment settings and
`--start_stage=race --continue_race=True --start_params=PATH/race_final.pkl`.
This restores weights/normalization; Adam and rollout state start fresh.

<details>
<summary><strong>Training arguments: meaning and actual defaults</strong></summary>

Defaults come from `train.py`; `None` reward overrides use the preset described in the table.
Boolean/list examples: `--continue_race=True`, `--policy_layers='(256,256)'`.

| Argument | Default when omitted | Meaning |
| --- | --- | --- |
| `--run_name` | `None` | Output folder name; generated from the current time when omitted. |
| `--output_root` | `None` | Parent output folder; None uses rl_race/outputs. |
| `--seed` | `0` | Training random seed. |
| `--device` | `gpu` | Simulator device: gpu or cpu. |
| `--roll_limit_deg` | `70.0` | Absolute target roll limit, degrees (plus/minus). |
| `--pitch_limit_deg` | `70.0` | Absolute target pitch limit, degrees (plus/minus). |
| `--yaw_limit_deg` | `70.0` | Absolute target yaw limit, degrees (plus/minus). |
| `--thrust_min_fraction` | `0.0` | Lower fraction of the official thrust range, between 0 and 1. |
| `--thrust_max_fraction` | `1.0` | Upper fraction of the official thrust range, between 0 and 1. |
| `--n_gates` | `2` | Upcoming gates encoded in the observation: 2 gives 49D; 3 gives 56D. |
| `--official_start_prob` | `0.5` | Official-start probability; remaining starts are sampled along the path. |
| `--max_episode_steps` | `750` | Maximum control steps per episode (50 Hz). |
| `--pretrain_max_steps` | `2000000000` | Maximum pretrain environment transitions; may switch earlier. |
| `--pretrain_crash` | `20.0` | Pretrain crash penalty. |
| `--pretrain_time` | `0.2` | Pretrain time penalty per control step. |
| `--pretrain_progress` | `None` | Path-progress coefficient; None uses dense_path preset (15). |
| `--pretrain_path_lateral` | `None` | Path-distance weight; None uses preset (1). |
| `--race_steps` | `12000000000` | Requested race environment transitions, rounded up by Brax. |
| `--switch_success` | `0.3` | Pretrain official-start success threshold for switching to race. |
| `--critic_only_iterations` | `10` | PPO iterations freezing the actor after resetting the critic. |
| `--start_params` | `None` | Checkpoint path for weight continuation; None starts from scratch. |
| `--start_stage` | `pretrain` | Starting stage: pretrain, race or anneal. |
| `--continue_race` | `False` | Keep the restored critic for race continuation; otherwise reset it. |
| `--start_step` | `0` | Offset for cumulative step logs; does not advance optimizer state. |
| `--lr_decay_to` | `1.0` | Final learning-rate multiplier for race continuation. |
| `--warmup_iterations` | `0` | Learning-rate warmup iterations for race continuation. |
| `--anneal_steps` | `0` | Anneal transitions after race; 0 disables anneal. |
| `--anneal_to` | `0.1` | Final learning-rate multiplier during anneal. |
| `--anneal_warmup_iterations` | `5` | Learning-rate warmup iterations during anneal. |
| `--race_time` | `0.3` | Race time penalty per control step. |
| `--race_crash` | `20.0` | Race crash penalty. |
| `--race_gate` | `5.0` | Reward per gate passed. |
| `--race_lap` | `10.0` | Reward for finishing a lap. |
| `--race_lap_bonus` | `None` | Optional lap-time bonus coefficient; None uses preset (0). |
| `--race_lap_ref` | `None` | Reference lap time in seconds for that bonus; None uses preset (4). |
| `--num_envs` | `16384` | Parallel simulator worlds. |
| `--unroll_length` | `32` | Consecutive steps in each trajectory segment / GAE window. |
| `--batch_size` | `4096` | Trajectory segments per SGD minibatch (not individual transitions). |
| `--num_minibatches` | `32` | Minibatches covering one collected rollout per epoch. |
| `--num_updates_per_batch` | `4` | Epochs over the same collected rollout. |
| `--learning_rate` | `0.0003` | Pretrain learning rate; critic learning rate in race/anneal. |
| `--policy_lr_after_switch` | `0.0001` | Actor learning rate in race/anneal. |
| `--entropy_cost` | `0.005` | Entropy regularization coefficient. |
| `--noise_scale` | `1.0` | Exploration standard-deviation multiplier; deterministic export is unaffected. |
| `--discounting` | `0.99` | Discount factor gamma. |
| `--gae_lambda` | `0.95` | GAE lambda. |
| `--clipping_epsilon` | `0.2` | PPO ratio clipping threshold. |
| `--max_grad_norm` | `1.0` | Global gradient norm limit. |
| `--policy_layers` | `(256, 256, 256)` | Actor hidden-layer widths. |
| `--value_layers` | `(256, 256, 256, 256)` | Critic hidden-layer widths. |
| `--activation` | `swish` | swish, elu, relu or tanh. |
| `--eval_every_steps` | `80000000` | Requested transition spacing for intermediate evaluation. |
| `--num_eval_envs` | `512` | Intermediate evaluation worlds. |
| `--final_eval_episodes` | `1024` | Final official-start episodes PER evaluation seed. |
| `--final_eval_seeds` | `(100000, 200000, 300000)` | Seeds for final evaluation, independent of the training seed. |
| `--final_eval_batch_size` | `256` | Worlds per final evaluation batch. |
| `--wandb_project` | `None` | Standalone-run logging project; None disables standalone W&B initialization. |
| `--wandb_entity` | `None` | Standalone W&B user/team; None uses account default. |
| `--wandb_group` | `None` | Optional standalone W&B run group. |

</details>

**PPO counts:** one rollout has `batch_size × num_minibatches × unroll_length` transitions;
one iteration makes `num_minibatches × num_updates_per_batch` SGD updates. With the defaults,
12B requested race steps round to **3,000 PPO iterations / 384,000 SGD updates**.
Changing batch size changes these counts; it is not only a VRAM setting.

## 2. Run a sweep on one or more machines

Create a sweep **once**:

```bash
pixi run -e gpu wandb login
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.sweep \
  --machine 9-2-3 --create lsy_drone_racing/control/rl_race/sweeps/L2_sweep.yaml
```

Copy the returned `ENTITY/PROJECT/SWEEP_ID`. Start an agent in tmux on each machine,
using the **same sweep ID and a different machine label**:

```bash
# Machine 9-2-3
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.sweep \
  --machine 9-2-3 --sweep ENTITY/PROJECT/SWEEP_ID --hours 10

# Machine 9-1-2, from the same shared project directory
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.sweep \
  --machine 9-1-2 --sweep ENTITY/PROJECT/SWEEP_ID --hours 10
```

Each agent runs one trial at a time. Directories are created automatically.
Without arguments, `sweep.py` executes one agent-assigned trial; users normally use the launcher flags below.

| Launcher argument | Default when omitted | Meaning |
| --- | --- | --- |
| `--machine` | Required with create/join | Unique machine/agent label; isolates local W&B state. |
| `--create` | None | YAML path; create a cloud sweep. Mutually exclusive with `--sweep`. |
| `--sweep` | None | Existing `ENTITY/PROJECT/SWEEP_ID` to join. |
| `--project` | lsy-drone-racing | Project for sweep creation. |
| `--count` | Unlimited | Maximum trials for this agent, until the sweep is exhausted. |
| `--hours` | Unlimited | Wall-time limit; interrupts the current trial at the deadline. |
| `--dry-run` | False | Create local directories and print the command without starting W&B. |

**Choose a YAML:**

| File | Trials | Parameters varied |
| --- | ---: | --- |
| [sweeps/L2_sweep.yaml](sweeps/L2_sweep.yaml) | 25 | 16 attitude pairs, original-PPO control, gates, anneal, budgets and a seed repeat |
| [sweeps/example.yaml](sweeps/example.yaml) | 2 | seed: 1 / 2; race_steps fixed at 12B |

`L2_sweep.yaml` lists 25 explicit experiments under `parameters.experiment.values`.
Each entry merges the `&baseline` settings with its overrides via standard YAML `<<`.
W&B grid runs each complete entry once; it does not multiply every field into a full factorial.
Roll/pitch are tied at 70/75/80/85 degrees; yaw independently takes all four values (16 pairs).
Additional entries compare n_gates=3, anneal=1.6B transitions (400 PPO iterations),
2,500/3,500 race iterations, and seed2 against the 3,000-iteration seed1 baseline.
The original-PPO control uses 8192 worlds/batch1024/race2.5B/eval20M/pretrain500M;
it retains the current four-action interface, so it is not an exact copy of the teammate's
old three-action policy. Fixed architecture, learning rates and rewards match the baseline.
The first YAML entry is the moderate baseline; cloud scheduling order is not guaranteed.
Two GPUs may finish only part of the25 experiments in10 hours; continue the same sweep ID later.
Training options go inside YAML `parameters`; omitted options inherit `train.py` defaults.
Both YAMLs explicitly set final_eval_seeds to `(410000,420000,430000)`.
The adapter expands the selected experiment into training arguments, records its name in W&B and the trial manifest, and sets run names, output subdirectories and W&B ownership automatically.

| Additional YAML parameter | Default when omitted | Meaning |
| --- | --- | --- |
| `experiment` | None | Named dictionary of training overrides, selected as one complete grid point. |
| `tilt_limit_deg` | None | Set roll and pitch limits together; cannot combine with separate roll/pitch limits. |
| `upload_checkpoint` | True | Include the final checkpoint in the model artifact. |
| `sweep_trial_count` | 1 | Total planned trials for estimating cloud weight storage; YAMLs set this to 25 / 2. |
| `cloud_checkpoint_budget_gb` | 50 | Skip weight uploads if checkpoint bytes × trial count exceed this many GB. |

Changing a local YAML does **not** update an existing cloud sweep: create a new sweep.
An interrupted trial keeps its saved checkpoints but does not automatically resume.
For a 16 GB GPU, if allocator limits cause OOM, prefix the agent command with
`XLA_CLIENT_MEM_FRACTION=0.9`; do not also set the legacy fraction variable.

## 3. Find results

```text
outputs/RUN/                              # standalone training
outputs/sweeps/SWEEP_ID/runs/RUN_ID_sSEED/ # sweep trial
  checkpoints/{stage}_latest.pkl          # rolling checkpoint
  checkpoints/{stage}_final.pkl           # stage-end checkpoint
  {stage}_config.json, stages.json        # settings and actual update counts
  final_evaluation.json, final_episodes.json
sweeps/wandb-MACHINE/                     # machine-specific agent logs/cache
```

Final evaluation uses official starts; default is 3 seeds × 1,024 episodes.
No intermediate best-checkpoint selection. Completed sweep trials attach the final model,
config and evaluation to W&B. Confirm the cloud artifact is available before deleting local weights.
Sort `final/leaderboard_score` ascending (success ≥50%, then successful mean time), or
`final/balanced_score` for an 80% threshold; always inspect `final/success_rate` too.

## 4. Evaluate, export and record

```bash
# Independent evaluation; adds holdout reports without replacing the original final report.
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.evaluation \
  --run_dir=PATH_TO_RUN --stage=race --episodes=1024 \
  --seeds='(900001,900002,900003)' --report_name=holdout

# Portable actor + normalizer. Use --stage=anneal when that was the final stage.
pixi run -e gpu python -m lsy_drone_racing.control.rl_race.export \
  --run_dir=PATH_TO_RUN --stage=race --destination=PATH_TO_RUN/exported
```

Export defaults: `checkpoint=final`, `destination=rl_race/deploy_model`, `overwrite=False`.
To replace the deployed policy, explicitly export there with `--overwrite=True`. Then:

```bash
pixi run -e default python scripts/sim.py --config=level2.toml \
  --controller=rl_race_controller.py --n_runs=20 --render=False
pixi run -e default python personal_tools/record_run.py --config level2.toml \
  --controller rl_race_controller.py --output-dir runs/my_run --n_runs 1 \
  --width 640 --height 360 --fps 30
```

Recording/replay details: [personal_tools guide](../../../personal_tools/README.record_replay.md).
Implementation: [train](train.py), [networks](network.py), [environment](env_wrapper/), [sweep](sweep.py).
