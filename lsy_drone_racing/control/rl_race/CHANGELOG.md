# Change log

## Shared-lab grid sweep (2026-11-16)

- Added sweep_agent.py: one machine label creates isolated W&B agent/SDK logs, cache, staging and download directories under sweeps/wandb-MACHINE. Multiple hosts join the same cloud sweep; each trial uses a separate run ID on the shared filesystem.
- Replaced the overnight random search with a finite72-trial grid: tied roll/pitch70/75/80, yaw70/80/85, two/three observed gates, anneal0/400M, time cost0.3/0.4. Fixed actor256x3 and critic256x4 to limit cost; wide-network superiority was never established. Start with seed1, then repeat promising settings across seeds.
- Added tilt_limit_deg alias, machine metadata and cloud checkpoint budget controls. Upload only the final checkpoint plus metadata; skip weights when measured size times declared trial count exceeds50GB. Local checkpoints remain available.
- Benchmarked larger Brax batches on5090 (9216/10240 OOM;4096 and8192 passed); selected defaults32768 environments/8192 batch; exact settings, measured memory and timings are recorded in GPU_PROFILE.md. Larger batches change PPO updates as well as memory use.
- Rewrote README in English with single-run, shared multi-host sweep, every training argument, evaluation, export and recording instructions.
- Removed old rl_race outputs and generated W&B state at the user's request. Preserved source YAML files and deploy_model. No commits, pushes or official leaderboard submissions.

## Four-action refactor (2026-11-15)

- Split environment actions, observations, rewards, starts and reference path into env_wrapper; moved actor/critic configuration into network.py.
- Replaced fixed-yaw three-action policies with four normalized actions [roll,pitch,yaw,thrust], configurable physical bounds, four-action history and export formatv2. Default observation49D; old3D exports rejected.
- Kept dense pretrain -> race -> optional anneal; critic reset/freeze on reward switch, separate learning rates and fresh-Adam continuation with warmup. No exact optimizer/environment resume or best-checkpoint search.
- Added final multi-seed first-episode evaluation, separate holdout report prefixes, W&B trial adapter/artifacts and finite local research scheduling. Final ranking scores do not affect training rewards.
- Fixed evaluation/export to load checkpoint arrays without importing the long-running training module. Verified a fresh staged training process can reach automatic final evaluation.
- Baseline seed1: approximately2.684B actual steps,3072-episode success90.30%, successful mean3.544s; official local20-run evaluation18/20,3.566s; personal_tools recordings3/3 successful. Portable baseline deployment retained.
- Local continuation screening completed five configurations. Best validation3.456s/90.23% on1536 episodes using85-degree tilt,time0.4,crash30. Other trials were cancelled after the user paused/ended research; no independent final holdout was completed. Historical outputs were subsequently removed as requested.
