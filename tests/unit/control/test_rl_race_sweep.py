"""Sweep adapter runs one trial and attaches reproducible outputs without network calls."""

import json
from types import SimpleNamespace

import pytest

from lsy_drone_racing.control.rl_race import sweep
from lsy_drone_racing.control.rl_race import train as training


class WandbSummary(dict):
    """Match W&B's mapping-only update signature."""

    def update(self, values):
        super().update(values)


@pytest.mark.parametrize("budget,uploaded", [(50, True), (1e-12, False)])
@pytest.mark.parametrize("grouped", [False, True])
def test_trial_forwards_config_and_publishes_final_files(
    monkeypatch, tmp_path, budget, uploaded, grouped
):
    calls = {}
    checkpoint = tmp_path / "race_final.pkl"
    checkpoint.write_bytes(b"test")
    report = {
        "stage": "race",
        "success_rate": 0.8,
        "mean_success_time": 4.0,
        "checkpoint": str(checkpoint),
    }
    (tmp_path / "final_evaluation.json").write_text(json.dumps(report))
    (tmp_path / "race_config.json").write_text("{}")
    (tmp_path / "stages.json").write_text("[]")

    def train(
        seed=0,
        policy_layers=(256,),
        run_name=None,
        wandb_project=None,
        output_root=None,
        roll_limit_deg=None,
        pitch_limit_deg=None,
    ):
        calls.update(seed=seed, policy_layers=policy_layers, run_name=run_name)
        assert roll_limit_deg == pitch_limit_deg == 75
        assert output_root.endswith("sweeps/standalone/runs")
        return str(tmp_path)

    logs, attachments, finished = [], [], []
    run = SimpleNamespace(
        config={
            "seed": 3,
            "policy_layers": [32, 32],
            "cloud_checkpoint_budget_gb": budget,
            "tilt_limit_deg": 75,
        },
        id="offline-test",
        log=lambda x: logs.append(x),
        summary=WandbSummary(),
        log_artifact=lambda x: None,
        finish=lambda exit_code: finished.append(exit_code),
    )
    if grouped:
        run.config["experiment"] = {
            "name": "tilt75",
            "tilt_limit_deg": run.config.pop("tilt_limit_deg"),
        }
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(sweep.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(
        sweep.wandb,
        "Artifact",
        lambda *a, **k: SimpleNamespace(add_file=lambda p: attachments.append(p)),
    )
    sweep.run_trial()
    assert calls == {"seed": 3, "policy_layers": (32, 32), "run_name": "offline-test_s3"}
    manifest = json.loads((tmp_path / "trial_manifest.json").read_text())
    assert manifest["experiment"] == ("tilt75" if grouped else None)
    if grouped:
        assert run.summary["experiment"] == "tilt75"
    assert logs[0]["final/success_rate"] == 0.8
    assert (str(checkpoint) in attachments) == uploaded
    assert len(attachments) == (4 if uploaded else 3)
    assert run.summary["checkpoint_uploaded"] == uploaded
    assert run.summary["checkpoint_bytes"] == len(b"test")
    assert run.summary["estimated_sweep_checkpoint_gb"] == len(b"test") / 1e9
    assert run.summary["training_complete"] and finished == [0]


def test_trial_failure_is_reported_to_wandb(monkeypatch, tmp_path):
    finished = []
    run = SimpleNamespace(
        config={},
        id="failed",
        summary=WandbSummary(),
        finish=lambda exit_code: finished.append(exit_code),
    )
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setattr(sweep.wandb, "init", lambda **kwargs: run)

    def broken_train(output_root=None, run_name=None, wandb_project=None):
        raise RuntimeError("training or evaluation failed")

    monkeypatch.setattr(training, "train", broken_train)
    with pytest.raises(RuntimeError, match="training or evaluation failed"):
        sweep.run_trial()
    assert finished == [1]


def test_machine_directories_are_isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "RL_DIR", tmp_path)
    a = sweep.machine_environment("9-3-2")
    b = sweep.machine_environment("9-3-3")
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        from pathlib import Path

        assert a[key] != b[key]
        assert Path(a[key]).is_dir()
    assert a["RL_RACE_MACHINE"] == "9-3-2"
    with pytest.raises(ValueError):
        sweep.machine_environment("../../escape")


def test_default_entrypoint_runs_assigned_trial(monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "run_trial", lambda: calls.append(True))
    sweep.main([])
    assert calls == [True]


def test_launcher_preserves_machine_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "RL_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(sweep.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    with pytest.raises(SystemExit) as result:
        sweep.main(
            ["--machine", "9-3-2", "--sweep", "team/project/id", "--hours", "10", "--count", "3"]
        )
    assert result.value.code == 0
    cmd, options = calls[0]
    assert cmd[:4] == ["timeout", "--signal=INT", "--kill-after=120s", "10.0h"]
    assert cmd[-4:] == ["agent", "--count", "3", "team/project/id"]
    assert options["env"]["RL_RACE_MACHINE"] == "9-3-2"


def test_create_sweep_resolves_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "RL_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(sweep.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    with pytest.raises(SystemExit) as result:
        sweep.main(["--machine", "9-3-2", "--create", "grid.yaml"])
    assert result.value.code == 0
    assert calls[0][-4:] == [
        "sweep",
        "--project",
        "lsy-drone-racing",
        str(sweep.Path("grid.yaml").resolve()),
    ]


@pytest.mark.parametrize("args", [["--machine", "9-3-2"], ["--sweep", "team/project/id"]])
def test_incomplete_launcher_cannot_start_a_trial(args):
    with pytest.raises(SystemExit) as result:
        sweep.main(args)
    assert result.value.code == 2
