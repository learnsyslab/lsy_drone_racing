"""Machine-local W&B agent launcher; run this file from any working directory."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
ROOT = RL_DIR.parents[2]


def machine_environment(machine: str) -> dict[str, str]:
    """Create machine-specific SDK, agent, cache and artifact directories."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", machine):
        raise ValueError("Machine name must contain only letters, digits, _, - or .")
    base = RL_DIR / "sweeps" / f"wandb-{machine}"
    env = dict(os.environ)
    for name, path in {
        "WANDB_DIR": base,
        "WANDB_CACHE_DIR": base / "cache",
        "WANDB_DATA_DIR": base / "staging",
        "WANDB_ARTIFACT_DIR": base / "artifacts",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(
        RL_RACE_MACHINE=machine,
        WANDB_MODE="online",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        SCIPY_ARRAY_API="1",
    )
    return env


def main() -> None:
    """Join one existing cloud sweep or create its configuration exactly once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--sweep", help="ENTITY/PROJECT/SWEEP_ID shared by all machines")
    action.add_argument("--create", help="YAML file; create once, then copy its sweep ID")
    parser.add_argument("--project", default="lsy-drone-racing")
    parser.add_argument(
        "--count", type=int, help="Maximum trials on this agent, not the whole sweep"
    )
    parser.add_argument("--hours", type=float, help="Wall time limit; interrupts the last trial")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.count is not None and args.count < 1) or (args.hours is not None and args.hours <= 0):
        parser.error("count and hours must be positive")
    env = machine_environment(args.machine)
    cmd = [sys.executable, "-m", "wandb"]
    if args.create:
        cmd += ["sweep", "--project", args.project, str(Path(args.create).resolve())]
    else:
        cmd += ["agent"]
        if args.count:
            cmd += ["--count", str(args.count)]
        cmd += [args.sweep]
        if args.hours:
            cmd = ["timeout", "--signal=INT", "--kill-after=120s", f"{args.hours}h", *cmd]
    print(f"Machine: {args.machine}; W&B directory: {env['WANDB_DIR']}", flush=True)
    print("Command:", " ".join(cmd), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
