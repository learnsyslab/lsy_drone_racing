"""Local budgeted sweep scheduler; each trial runs in an isolated Python/GPU process.

This is also usable without W&B credentials: workers record offline W&B runs. The plan,
config, stdout, lineage and final evaluation remain together under outputs/sweeps/STUDY/PHASE.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def write_json(path: Path, value: object) -> None:
    """Atomically publish a small machine-readable research record."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(config_path: str) -> None:
    """Run exactly one explicit configuration (no hidden parameter mutation)."""
    from .train import train

    config = json.loads(Path(config_path).read_text())
    unknown = set(config) - set(inspect.signature(train).parameters)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    for name in ("policy_layers", "value_layers", "final_eval_seeds"):
        if name in config:
            config[name] = tuple(config[name])
    train(**config)


def schedule(plan_path: str, parallel: int = 2, hours: float = 2.8) -> None:
    """Execute a finite plan, keeping every run and failure; never select checkpoints mid-run."""
    if parallel < 1 or hours <= 0:
        raise ValueError("parallel and hours must be positive")
    plan_file = Path(plan_path).resolve()
    plan = json.loads(plan_file.read_text())
    ids = [trial["id"] for trial in plan["trials"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trial IDs would collide in outputs")
    directory = plan_file.parent
    for name in ("configs", "logs", "runs"):
        (directory / name).mkdir(exist_ok=True)
    states = []
    for trial in plan["trials"]:
        name = trial["id"]
        if not name or Path(name).name != name:
            raise ValueError("Trial id must be a simple directory name")
        config = dict(plan.get("common", {})) | trial["parameters"]
        config.update(output_root=str(directory / "runs"), run_name=name)
        if (directory / "runs" / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing run {name}")
        write_json(directory / "configs" / f"{name}.json", config)
        states.append(
            dict(
                id=name,
                status="pending",
                parameters=config,
                output_dir=str(directory / "runs" / name),
            )
        )
    write_json(directory / "manifest.json", states)
    started = time.time()
    active = {}
    pending = list(states)
    try:
        while pending or active:
            # Stop launching at the deadline; an already-started finite trial may finish.
            while pending and len(active) < parallel and time.time() - started < hours * 3600:
                trial = pending.pop(0)
                name = trial["id"]
                stream = (directory / "logs" / f"{name}.log").open("w")
                env = os.environ.copy()
                env.update(
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    SCIPY_ARRAY_API="1",
                    WANDB_MODE="offline",
                    WANDB_SILENT="true",
                )
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "lsy_drone_racing.control.rl_race.research",
                    "--worker",
                    str(directory / "configs" / f"{name}.json"),
                ]
                process = subprocess.Popen(
                    cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                trial.update(status="running", pid=process.pid, started_utc=time.time())
                active[name] = (process, stream, trial)
                print(f"START {name} pid={process.pid}", flush=True)
            for name, (process, stream, trial) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                trial.update(exit_code=code, ended_utc=time.time())
                report = Path(trial["output_dir"]) / "final_evaluation.json"
                trial["status"] = "complete" if code == 0 and report.exists() else "failed"
                if report.exists():
                    trial["evaluation"] = json.loads(report.read_text())
                print(f"END {name} {trial['status']} {trial.get('evaluation', {})}", flush=True)
                del active[name]
            write_json(directory / "manifest.json", states)
            if pending and not active and time.time() - started >= hours * 3600:
                for trial in pending:
                    trial["status"] = "not_started_budget"
                pending.clear()
                write_json(directory / "manifest.json", states)
            if pending or active:
                time.sleep(10)
    finally:
        write_json(directory / "manifest.json", states)


def main() -> None:
    """CLI for the finite local scheduler and its trial worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--plan")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--hours", type=float, default=2.8)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.plan:
        schedule(args.plan, args.parallel, args.hours)
    else:
        parser.error("--worker or --plan required")


if __name__ == "__main__":
    main()
