# Sparse-reward single-agent PPO (SS), phase 1

This reproduces the **sparse-reward single-agent (SS)** baseline of
*Agile Flight Emerges from Multi-Agent Competitive Racing* (Pasumarti, Bianchi,
Loquercio, arXiv:2512.11781v2) inside our Level 2 SBX/JAX pipeline. Dense (DS)
and sparse runs share one code path. Every addition sits behind a flag whose
default reproduces the previous behaviour bit for bit.

## What was implemented

| Piece | Where | Flag (train.py) | Default |
| --- | --- | --- | --- |
| Sparse task-level reward | `reward.py`: `SparseRewardConfig`, `compute_sparse_reward` | `--reward_mode sparse` | `dense` |
| Sparse coefficients | same | `--gate_pass_leading 10 --gate_pass_trailing 5 --lap_leading 50 --crash_penalty 2 --cmd_roll_pitch 0.15 --cmd_yaw 0.05 --cmd_scale_by_dt True --single_agent_leading True` | paper values |
| Paper observation layout | `env_wrapper.py`: `ObservationConfig(layout="paper")`, `_encode_paper` | `--obs_layout paper [--paper_obs_obstacles True]` | `current` |
| Configurable networks | `policy.py`: `NetworkConfig`, `ARCH_PRESETS` | `--arch paper` or `--actor_arch/--critic_arch/--activation` | `(256,256)` leaky ReLU |
| Paper evaluation protocol | `eval_paper.py` | n/a | 50 episodes |
| Live monitoring | `wandb_logger.py` | `--wandb_project ...` | off |
| Experiment runner | `run_experiments.sh` | n/a | n/a |

Details:

- **Reward**: `r = r_pass + r_lap - r_cmd - r_crash`.
  - `r_pass` is 10 per gate when leading and 5 otherwise. `r_lap` is 50 when the lap is completed while leading.
  - `r_crash` is 2 for every crash.
  - `r_cmd = (0.15 (w_x² + w_y²) + 0.05 w_z²) · dt`, where `w = delta / dt` is the commanded body rate implied by the action's body-frame rotation increment `delta`. The same `rotation_increment()` that `decode_action` uses computes `delta`.
  - Gate and lap events come from the official `n_gates_passed`. On the final pass, both the gate reward and the lap bonus are paid. A crash step pays no gate reward, and a timeout pays no crash penalty.
  - `compute_sparse_reward(..., leading=mask)` takes an optional per-agent `leading` mask for phase 2. Without a mask, `single_agent_leading` applies.
- **Logging**: each reward component is logged per transition (`reward/<c>`) and as a mean per-episode sum (`reward/episode_<c>`). For example, `reward/episode_cmd` is the summed command penalty per episode. The sums are accumulated on the device and read back once per rollout.
- **Paper observation (40-D, +20 with obstacles)**:
  - body-frame velocity (3)
  - rotation matrix (9)
  - current-gate inner-opening corners in the body frame (12)
  - next-gate corners (12)
  - next-gate valid flag (1)
  - global position (3)
  - optionally, the existing 4 nearest obstacles (5 each)

  Corners are at ±0.2 m along the gate's local y and z axes. For reverse passes, local y is mirrored so the corner order is the same as seen from the approach side. Only sensor-masked poses are used. Running normalization, checkpoints and the deployed controller all rebuild the layout from metadata.
- **Networks**: the `paper` preset is actor `[512,512,256,128]` and critic `[512,512,256,256,128,128]`, both ELU. The `NetworkConfig` is passed through SBX `policy_kwargs`, so it is pickled into `model.zip`. It is also stored in the checkpoint and export metadata. Legacy `net_arch` metadata still loads, including the committed `deploy_model`.
- **Evaluation** (`eval_paper.py`): N parallel worlds start from the official starts (segment starts off) and act on the deterministic Gaussian mean. It reports:
  - success rate
  - mean time to complete
  - mean speed
  - mean gates passed
  - crashes split into gate contact, obstacle contact, out of bounds/ground, and timeouts

  The env moves a disabled drone below the ground. To attribute crashes exactly, the evaluator replays the action and physics substeps on the identical pre-step state and RNG key, without that move.
- **Tests**: `tests/unit/control/test_rl_single_l2_sparse.py` covers the reward and paper observations. `tests/integration/test_rl_single_l2_sparse.py` covers training → checkpoint → resume/warm start → export → controller, the sparse reward in the real simulator, the legacy deploy model, and the evaluation script. Dense mode was additionally checked to be bit-identical to `main`: observations, rewards, dones and reward means over 150 steps including resets.

## Deviations from the paper (and why)

| Topic | Paper | Here | Reason |
| --- | --- | --- | --- |
| Action | Collective thrust + body rates, inner 500 Hz rate PID | Our attitude-increment action: body-frame rotation `delta` (≤ 0.16 rad/step) on the current attitude, plus sigmoid thrust | The environment only offers `state` and `attitude` control. `delta/dt` is used as the commanded rate for `r_cmd` |
| Contact | Non-terminal contact costs 0.1, terminal crash costs 2.0 | Every contact is terminal, every crash costs 2.0 | `_disabled_drones` disables a drone on any gate or obstacle contact |
| Laps | 3 laps | 1 lap = 1 episode, so the lap bonus is paid on completing the course | Level 2 ends the episode after the last `gate_order` entry |
| "Leading" without an opponent | Not specified | Flag `single_agent_leading` (default `True`: 10 per gate + 50 per lap) | Ambiguous in the paper |
| `r_cmd` scale | Coefficients given, no dt stated | Flag `cmd_scale_by_dt` (default `True`) | Without dt, `w` up to ~8 rad/s gives up to ~10 penalty per step, which swamps the +10 gate reward |
| Eq. 5 sign | Leading minus and subtracted again | Treated as a penalty | Sign typo |
| Gates | 1 m gates | 0.72 m outer, 0.4 m inner opening | Level 2 track |
| Obstacles | Static; policy infers them from global position | Randomized ±0.15 m per episode, so the nearest-obstacle block is optionally added (`--paper_obs_obstacles`) | Global position cannot reveal randomized obstacles |
| Next-gate valid flag | n/a (multi-lap) | +1 dim (40-D instead of 39-D) | The final target has no successor in a single lap |
| Critic | Privileged joint state | Same observation as the actor (no asymmetric critic) | See below |
| Starts | Initial positions beyond a selected gate | Existing segment starts (prob 0.40, 0.5 m before a gate, 2.5 m/s) during training; official starts for evaluation | Same idea; already validated in our DS pipeline |
| Budget | ~1.5·10⁹ agent steps | 100M steps per run, 3 seeds | Course compute budget |

**Asymmetric critic: not implemented.** SBX's PPO feeds one observation tensor to both the actor and the critic. A privileged critic would need extra true-pose features appended to the observation, an actor that slices them off, and a deploy-time normalizer reduced to the actor slice. That is doable without hacking SBX. Phase 1 has no opponent, so the paper's privileged signal (the joint state of both agents) does not exist here, so it was left out. It becomes relevant in phase 2.

**Command-penalty sanity check.** I measured this under the freshly initialized policy, with 512 worlds and official starts.

| `cmd_scale_by_dt` | Mean episode length | Command penalty per step | Command penalty per episode |
| --- | --- | --- | --- |
| `True` | 108 steps | −0.075 | **−7.5** |
| `False` | 108 steps | −3.76 | −374 |

With dt scaling, the per-episode penalty is about 75 % of one gate reward, not "well below" it. Most of it is caused by exploration noise, because the penalty applies to the sampled action. Since a crash costs only 2, a policy that finds no gates pays less by crashing early than by hovering. In a 1M-step smoke run, the policy's rotation std fell from 0.57 to 0.30. We kept the paper values (× dt) for the main runs, to stay faithful. Training did not take off, so we applied the fallbacks; see Results.

## Reproducing

Training uses identical PPO settings for every config: 8192 envs × 128 steps, about 1M transitions per rollout (about 100 updates per 100M steps), batch 16384, 3 epochs, lr 3e-4, gamma 0.99, GAE 0.95, entropy coefficient 0.005, clip 0.2.

```bash
# One tmux session per seed; each runs DS, SS-current and SS-paper for 100M steps
# and logs to W&B (project lsy-drone-racing, group phase1-100M).
tmux new -d -s seed1 "bash lsy_drone_racing/control/rl_single_l2/run_experiments.sh 1"
# Individual configs (same as the runner):
python -m lsy_drone_racing.control.rl_single_l2.train --reward_mode=dense ...                # DS
python -m lsy_drone_racing.control.rl_single_l2.train --reward_mode=sparse ...               # SS-current
python -m lsy_drone_racing.control.rl_single_l2.train --reward_mode=sparse --obs_layout=paper \
    --paper_obs_obstacles=True --arch=paper ...                                               # SS-paper
# Paper-protocol evaluation (checkpoint step dir or exported actor):
python -m lsy_drone_racing.control.rl_single_l2.eval_paper --model=<dir> --episodes=50 --csv_path=...
# Official evaluation (20 runs) with an exported actor:
python -m lsy_drone_racing.control.rl_single_l2.checkpoint --checkpoint=<dir>  # -> deploy_model
pixi run evaluate
```

Fallback variants use the runner's `EXTRA`/`TAG` hooks. Each fallback stacks on the previous ones:

```bash
F1="--segment_init_prob=0.8"
F2="$F1 --gamma=0.995"
F3="$F2 --cmd_roll_pitch=0.015 --cmd_yaw=0.005"
EXTRA="$F3" TAG=f3 WANDB_GROUP=phase1-fallbacks bash .../run_experiments.sh <seed> ss_current
EXTRA="$F3 --mode=warm_start --checkpoint=outputs/p1_ds_s<seed>/checkpoints/step_000100663296" \
    TAG=f4 WANDB_GROUP=phase1-fallbacks bash .../run_experiments.sh <seed> ss_current
# Equal-budget control: DS continued from the same checkpoint with the dense reward
EXTRA="--mode=warm_start --checkpoint=outputs/p1_ds_s<seed>/checkpoints/step_000100663296" \
    TAG=cont WANDB_GROUP=phase1-fallbacks bash .../run_experiments.sh <seed> ds
```

## Results

All runs use 100M steps per stage and 3 seeds. Each run was evaluated with `eval_paper.py`: 50 episodes from the official starts, deterministic actions, final checkpoint. Values are mean ± std over seeds. Crash counts are per 50 episodes, averaged over seeds. "Time" is the mean time to complete over successful episodes. W&B: project `lsy-drone-racing`, groups `phase1-100M` and `phase1-fallbacks`.

| Config | Total steps | Success | Time [s] | Gates passed (of 5) | Gate / obstacle / out-of-bounds crashes | Timeouts |
| --- | --- | --- | --- | --- | --- | --- |
| DS (dense baseline) | 100M | 14.0 ± 4.9 % | 12.6 ± 1.4 | 1.91 ± 0.14 | 29.0 / 9.3 / 0.0 | 4.7 |
| SS-current | 100M | 0 % | n/a | 0.00 | 0.0 / 0.0 / 46.7 | 3.3 |
| SS-paper (paper obs + paper nets) | 100M | 0 % | n/a | 0.00 | 1.0 / 2.3 / 5.0 | 41.7 |
| SS-current + F1 (`segment_init_prob` 0.8) | 100M | 0 % | n/a | 0.00 | 0 / 0 / 50 | 0 |
| + F2 (gamma 0.995) | 100M | 0 % | n/a | 0.00 | 0 / 0 / 50 | 0 |
| + F3 (cmd weights ÷10) | 100M | 0 % | n/a | 0.71 ± 0.52 | 26.3 / 18.3 / 1.3 | 4.0 |
| + F4 (warm start from DS) | 100M DS + 100M SS | 38.7 ± 2.5 % | 11.6 ± 1.5 | 2.67 ± 0.11 | 20.3 / 8.3 / 0.0 | 2.0 |
| DS continued (equal-budget control for F4) | 100M + 100M DS | 40.0 ± 4.3 % | 11.1 ± 1.4 | 2.81 ± 0.07 | 18.0 / 11.0 / 0.0 | 1.0 |

Official `pixi run evaluate` (20 runs; `deploy_model` was swapped in temporarily and then restored):

| Actor | Successful runs |
| --- | --- |
| SS + F1–F4, seed 3 | 8 / 20 (40 %) |
| DS, seed 1 (100M) | 4 / 20 (20 %) |

Both fall below the script's 50 % threshold.

### Interpretation

- **Sparse reward from scratch does not take off within 100M steps.** This holds for both observation and network variants.
  - Both SS variants learn to exploit the segment starts. A start 0.5 m in front of a gate at 2.5 m/s yields a nearly free +10 (and +50 before the last gate). About 11 % of training episodes "finish" this way, and they average about 0.6 gates per episode.
  - From the official start they pass no gates. SS-current dives out of bounds (−2 crash) and SS-paper hovers until the timeout.
  - Exploration collapses: the rotation std falls to 0.09–0.18, versus 0.65–0.84 for DS.
- **Fallbacks needed: all four (F4, warm start from DS).**
  - F1 and F2 change nothing: the agent still crashes immediately.
  - F3 is the first change in behaviour: the agent flies toward gates (0.7 gates on average, crashes now at gates and obstacles instead of out of bounds), but it never completes the course.
  - With F4, sparse fine-tuning reaches 39 % success.
- **The F4 gain comes from the extra training steps, not from the sparse reward.** The equal-budget DS control reaches the same success (40 ± 4 % vs 39 ± 3 %, within noise) and a similar lap time. Sparse fine-tuning keeps the DS skill but, at this budget, does not improve it.
- The DS baseline itself has not converged at 100M steps: it goes from 14 % to 40 % with another 100M. The paper trains about 1.5·10⁹ agent steps, roughly 15× our per-run budget. A fair test of SS-from-scratch therefore needs a much larger budget, or a curriculum that removes the segment-start exploit, for example starts in the middle between gates instead of 0.5 m before them. We consider this the main open item before phase 2.

## Continuation to 1B steps with wider actuator limits

**Why:** the original actor head was `tanh(mean)`, then a sigmoid (thrust) or a radial tanh (rotation). Evaluation and deployment use the mean, so the deterministic policy could reach only:
- 73 % of the thrust range: thrust-to-weight 1.43 instead of 1.88, and never below 0.65 g;
- 6.1 rad/s per rotation axis instead of 8.

The noisy actions during training could go beyond these limits, so training and evaluation saw different limits. Since commit `032d1a6`:

| Setting | Flag | Value for new runs | Value for legacy checkpoints/exports |
| --- | --- | --- | --- |
| Soft mean bound b in `b·tanh(z/b)` | `--mean_bound` | 3.0 (95 % of the thrust range) | 1.0 |
| Rotation limit per step | `--alpha_max_rad` | 0.30 (15 rad/s) | 0.16 (8 rad/s) |
| Small-signal rotation gain | `--alpha_gain_rad` | 0.16 | equal to `alpha_max_rad` |

Both mappings keep their slope at zero, so a warm-started policy changes only where it used to saturate. `--start_timesteps` continues a lineage's step counter on warm starts, so logs, W&B and checkpoint names count cumulative steps.

**Runs** (W&B group `phase1-1B`). Both lineages continue from 201M cumulative steps to 1B with the new limits, 3 seeds each:
- **SS**: the F4 checkpoints, trained with the sparse reward and F1–F3 settings (sparse reward with gamma 0.995).
- **DS**: the equal-budget DS control checkpoints, trained with the dense reward (gamma 0.99).

Applying the wider limits to the SS checkpoint before any further training made it faster (8.8 s vs about 11.6 s) but less reliable (14 % success instead of about 40 %). PPO had to re-tune.

| Config (1B cumulative steps) | Success | Time [s] | Speed, all episodes [m/s] | Gates passed (of 5) | Gate / obstacle / out-of-bounds crashes | Timeouts |
| --- | --- | --- | --- | --- | --- | --- |
| DS: 1B dense | 52.0 ± 4.3 % | **6.0 ± 0.5** | 1.91 ± 0.07 | 3.48 ± 0.16 | 10.0 / 13.7 / 0.0 | 0.3 |
| SS: 100M dense, then 900M sparse | **71.3 ± 11.6 %** | 7.6 ± 1.2 | 1.66 ± 0.05 | 4.30 ± 0.28 | 4.3 / 8.0 / 0.0 | 2.0 |

Per seed, success was 48 / 58 / 50 % for DS and 56 / 74 / 84 % for SS.

Official `pixi run evaluate` (20 runs) on the best seed of each config:

| Actor | Success | Average time (successful runs) |
| --- | --- | --- |
| SS, seed 3 | 15 / 20 (75 %) | 6.76 s |
| DS, seed 2 | 13 / 20 (65 %) | 5.72 s |

Both pass the official 50 % threshold. The exports are in `outputs/exports/ss_1B_s3` and `outputs/exports/ds_1B_s2`; `deploy_model` is unchanged.

**Interpretation:**
- Given a DS warm start and enough steps, the sparse reward gives the more **reliable** policy: success 71 % vs 52 %, and about 2.5× fewer gate contacts.
- DS gives the **faster** one: 6.0 s vs 7.6 s mean lap time.
- This matches the reward structure:
  - The sparse reward has no time penalty, so gamma is its only pressure to fly fast, and it pays only on events.
  - The dense reward pays per metre of progress and every step costs 0.1, which pushes speed at the cost of risk.
- **Confounds:**
  - The SS lineage uses gamma 0.995 and segment-start probability 0.8; DS uses gamma 0.99 and 0.4.
  - The 100M-step DS warm start is part of the SS result.
  - Seed variance in SS is large (56–84 %).
- The wider actuator limits made both lineages much faster than at 200M steps (DS 11.1 → 6.0 s, SS 11.6 → 7.6 s), with success still rising. We did not run an ablation that would separate the effect of the limits from the effect of the extra steps.
