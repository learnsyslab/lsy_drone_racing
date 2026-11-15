"""W&B trial entrypoint. Launch each trial as a separate process using sweeps/example.yaml."""

import inspect
import json
import os
from pathlib import Path

import wandb

from .train import RL_DIR, train


def main() -> None:
    """Run a W&B-configured trial and publish final metrics and checkpoint."""
    directory = Path(os.environ.get("WANDB_DIR", str(RL_DIR / "outputs")))
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(dir=str(directory))
    try:
        run.summary["machine"] = os.environ.get("RL_RACE_MACHINE", "unlabelled")
        args = dict(run.config)
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
            checkpoint_bytes=checkpoint_bytes, estimated_sweep_checkpoint_gb=estimated_gb
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
    finally:
        run.finish()


if __name__ == "__main__":
    main()
