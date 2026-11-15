# RTX 5090 batch validation

Measured on the local32GB RTX5090 with XLA_PYTHON_CLIENT_PREALLOCATE=false. One real PPO iteration per profile, using32768 environments,32 minibatches,32-step unrolls,4 optimization epochs, the default256-wide actor/critic and three observed gates (56 inputs). Each run starts a fresh process. The requested pretrain budget was1 step, rounded to a complete iteration; switch_success=1 prevents the deliberately untrained probe from entering race.

| Batch sequences | Actual transitions | Peak process MiB | Peak whole GPU MiB | Process seconds | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 4,194,304 | 17168 | 17658 | 220.9 | PASS |
| 8192 | 8,388,608 | 24840 | 26247 | 127.0 | PASS |
| 9216 | 9,437,184 | 17076 | 17566 | 92.8 | OOM |
| 10240 | 10,485,760 | 17088 | 18527 | 91.8 | OOM |

Selected default: num_envs32768, batch_size8192, num_minibatches32, unroll_length32. Successful sampled process peak24840MiB =24.26GiB.10240 failed allocating13.59GiB;9216 failed allocating12.23GiB. Failed-allocation peaks are not the memory required to succeed. Polling every2 seconds can miss short peaks; do not force28–30GB occupancy based only on nvidia-smi. Other local processes and a small staged smoke account for differences between process and whole-card peaks.

These are compile-and-first-iteration checks, not steady-state throughput comparisons or evidence of better learning. The selected iteration has8388608 transitions, eight times the old1024x32x32 rollout, with the same128 SGD steps; this changes PPO optimization. The old SBX single_l2 used18432 environments,256 steps and an18432-transition minibatch; Brax batch_size counts32-step sequences and is not directly comparable. If full staged training encounters higher memory pressure, reduce batch_size to4096 consistently across the sweep and create a new sweep.

Additional smoke:8 worlds, small32x2 networks, n_gates3,64 requested steps each of pretrain/race/anneal,2 evaluation seeds x3 episodes (batch2). All stages and automatic final evaluation completed with exit0. Export metadata reports56 observations and4 actions; controller policy inference returned a finite(1,4) action. This is a mechanics check; zero success after tiny training is expected, not a performance result.

Cloud storage estimate: full default actor+critic+normalizer checkpoint measured1444006 bytes.72 final checkpoints are approximately0.104GB, plus small metadata files; repeated latest checkpoints are local only. Actual upload decisions are rechecked per trial against the declared50GB budget.

Validation:22 related unit/export tests passed; the three sweep-adapter tests were rerun after adding alias expansion and budget handling. Ruff passed for rl_race; git diff --check passed. Temporary benchmark models are deleted after recording these measurements. No formal training or cloud sweep was started by this change.
