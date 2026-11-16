# Change log

## Unified sweep entrypoint (2026-11-16)

- Merged the machine-isolated agent launcher into sweep.py and removed sweep_agent.py. Use --machine with --create or --sweep; no arguments still execute one assigned trial, preserving existing cloud YAML commands.
- Kept count/hour limits, directory isolation and trial behavior; training imports are deferred until a trial starts. Updated usage and added launcher/dispatch regression tests.

## Cleanup (2026-11-16)

- Removed the standalone local research scheduler, research summaries, minimum-time solver and their unused tests. Training and search use train.py and W&B sweeps.
- Shortened README to training, sweeps, evaluation and export instructions; folded the useful GPU memory/batch notes into README.
- Fixed W&B summary.update to accept a mapping: keyword arguments previously crashed completed trials before artifact upload. Explicit failure exit codes keep W&B run status consistent with process failures.
- Export rejects overwrite destinations that contain the source run and extracts checkpoint arrays once.
- Brax rollout/evaluation rounding and training schedules are unchanged.
- Validation: 30 unit/integration tests passed on CPU; training/evaluation/export CLI help, both sweep YAML configs and export/load of an existing checkpoint passed. Ruff and git diff --check passed.

## Historical model results (2026-11-15)

- Four-action baseline, seed 1, 70-degree limits: approximately 2.684B actual steps; 3072-episode success 90.30%, successful mean 3.544 s. Official local evaluation: 18/20, 3.566 s. The baseline deployment remains in deploy_model.
- Continuation screening: best validation 90.23% / 3.456 s on 1536 episodes with 85-degree tilt, time 0.4 and crash 30. No independent final holdout was completed; original research outputs were previously removed.
- Later defaults increased batch from 1024 to 8192 after short memory checks. This changes PPO optimization; it was not a demonstrated learning improvement.
