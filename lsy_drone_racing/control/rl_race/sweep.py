"""Create/join machine-isolated W&B sweeps, or execute an agent-assigned trial."""

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import wandb

RL_DIR = Path(__file__).resolve().parent
ROOT = RL_DIR.parents[2]


def run_trial() -> None:
    """Train one configuration assigned by the cloud agent and publish its results."""
    from lsy_drone_racing.control.rl_race.train import train

    directory = Path(os.environ.get("WANDB_DIR", str(RL_DIR / "outputs")))
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(dir=str(directory))
    exit_code = 1
    try:
        run.summary["machine"] = os.environ.get("RL_RACE_MACHINE", "unlabelled")
        args = dict(run.config)
        experiment = dict(args.pop("experiment", {}))
        experiment_name = experiment.pop("name", None)
        args.update(experiment)
        if experiment_name:
            run.summary["experiment"] = experiment_name
        tilt = args.pop("tilt_limit_deg", None)
        if tilt is not None:
            if "roll_limit_deg" in args or "pitch_limit_deg" in args:
                raise ValueError("Use tilt_limit_deg or separate roll/pitch limits, not both")
            args.update(roll_limit_deg=tilt, pitch_limit_deg=tilt)
        upload_checkpoint = args.pop("upload_checkpoint", True)
        trial_count = int(args.pop("sweep_trial_count", 1))
        cloud_budget = float(args.pop("cloud_checkpoint_budget_gb", 50))
        if trial_count < 1 or cloud_budget <= 0:
            raise ValueError("Sweep trial count and cloud budget must be positive")
        unknown = set(args) - set(inspect.signature(train).parameters)
        if unknown:
            raise ValueError(f"Unknown train parameters: {sorted(unknown)}")
        for name in ("policy_layers", "value_layers", "final_eval_seeds"):
            if name in args:
                args[name] = tuple(args[name])
        sweep_id = getattr(run, "sweep_id", None) or "standalone"
        parent = Path(args.pop("output_root", None) or RL_DIR / "outputs")
        args["output_root"] = str(parent / "sweeps" / sweep_id / "runs")
        args["run_name"] = f"{run.id}_s{args.get('seed', 0)}"
        args["wandb_project"] = None  # This entrypoint owns the single W&B run.
        output = Path(train(**args))
        manifest = dict(
            sweep_id=sweep_id,
            experiment=experiment_name,
            run_id=run.id,
            seed=args.get("seed", 0),
            machine=os.environ.get("RL_RACE_MACHINE", "unlabelled"),
            config=args,
            output_dir=str(output),
        )
        (output / "trial_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report_file = output / "final_evaluation.json"
        if not report_file.exists():
            run.summary["training_complete"] = False
            raise RuntimeError("Pretraining did not reach race; no final race evaluation")
        report = json.loads(report_file.read_text())
        run.log({f"final/{k}": v for k, v in report.items() if isinstance(v, (int, float))})
        run.summary.update(
            {
                "output_dir": str(output),
                "checkpoint": report["checkpoint"],
                "training_complete": True,
            }
        )
        checkpoint_bytes = Path(report["checkpoint"]).stat().st_size
        estimated_gb = checkpoint_bytes * trial_count / 1e9
        run.summary.update(
            {"checkpoint_bytes": checkpoint_bytes, "estimated_sweep_checkpoint_gb": estimated_gb}
        )
        artifact = wandb.Artifact(f"final-{run.id}", type="model")
        included = upload_checkpoint and estimated_gb <= cloud_budget
        if included:
            artifact.add_file(report["checkpoint"])
        run.summary["checkpoint_uploaded"] = included
        artifact.add_file(str(report_file))
        artifact.add_file(str(output / f"{report['stage']}_config.json"))
        artifact.add_file(str(output / "stages.json"))
        run.log_artifact(artifact)
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)


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


def main(argv: list[str] | None = None) -> None:
    """Join one existing cloud sweep or create its configuration exactly once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", help="Unique machine/agent label, e.g. 9-3-2")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--sweep", help="ENTITY/PROJECT/SWEEP_ID shared by all machines")
    action.add_argument("--create", help="YAML file; create once, then copy its sweep ID")
    parser.add_argument("--project", default="lsy-drone-racing")
    parser.add_argument(
        "--count", type=int, help="Maximum trials on this agent, not the whole sweep"
    )
    parser.add_argument("--hours", type=float, help="Wall time limit; interrupts the last trial")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.create and not args.sweep:
        if args.machine or args.count is not None or args.hours is not None or args.dry_run:
            parser.error("Use --create or --sweep with launcher options")
        run_trial()  # Existing sweep YAMLs invoke this module without arguments.
        return
    if not args.machine:
        parser.error("--machine is required with --create or --sweep")
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
