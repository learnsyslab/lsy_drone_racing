#!/usr/bin/env bash
# Phase 1 comparison (SPARSE_REWARD.md): DS baseline vs. SS-current vs. SS-paper.
# Usage: run_experiments.sh <seed> [config ...]   configs: ds ss_current ss_paper
# Fallback variants: EXTRA="--segment_init_prob=0.8" TAG=f1 run_experiments.sh 1 ss_current
# appends EXTRA to every config and names runs p1_<config>_<TAG>_s<seed>.
# Runs the given configs sequentially for one seed on the GPU. Start one tmux
# session per seed to run seeds in parallel, e.g.
#   tmux new -d -s seed1 "bash lsy_drone_racing/control/rl_single_l2/run_experiments.sh 1"
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # the GPU is shared between runs/users

SEED="$1"
shift
CONFIGS=("${@:-ds ss_current ss_paper}")
WANDB_GROUP="${WANDB_GROUP:-phase1-100M}"
STEPS="${STEPS:-100000000}"
OUT=lsy_drone_racing/control/rl_single_l2/outputs
mkdir -p "$OUT/logs"

# Identical PPO settings for every config: ~1M-transition rollouts (~100 updates
# per 100M steps); everything else stays at train.py defaults.
COMMON=(--total_timesteps="$STEPS" --n_envs=8192 --n_steps=128 --batch_size=16384
  --seed="$SEED" --device=gpu --save_freq=10000000 --eval_freq=10000000 --eval_episodes=10)
# Live monitoring (needs `wandb login` once); set WANDB_PROJECT= to disable.
WANDB_PROJECT="${WANDB_PROJECT-lsy-drone-racing}"
if [ -n "$WANDB_PROJECT" ]; then
  COMMON+=(--wandb_project="$WANDB_PROJECT" --wandb_group="$WANDB_GROUP")
  [ -n "${WANDB_ENTITY:-}" ] && COMMON+=(--wandb_entity="$WANDB_ENTITY")
fi

for config in ${CONFIGS[@]}; do
  case "$config" in
    ds) extra=(--reward_mode=dense) ;;
    ss_current) extra=(--reward_mode=sparse) ;;
    ss_paper) extra=(--reward_mode=sparse --obs_layout=paper --paper_obs_obstacles=True --arch=paper) ;;
    *) echo "unknown config $config" >&2; exit 1 ;;
  esac
  name="p1_${config}${TAG:+_$TAG}_s${SEED}"
  read -r -a user_extra <<< "${EXTRA:-}"
  extra+=("${user_extra[@]}")
  if [ -d "$OUT/$name" ]; then
    echo "skip $name (exists)"
    continue
  fi
  echo "$(date) start $name"
  pixi run -e gpu python -m lsy_drone_racing.control.rl_single_l2.train \
    --run_name="$name" "${COMMON[@]}" "${extra[@]}" > "$OUT/logs/$name.log" 2>&1 \
    && echo "$(date) done $name" || echo "$(date) FAILED $name"
done
