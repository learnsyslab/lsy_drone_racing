"""Sweep adapter runs one trial and attaches reproducible outputs without network calls."""

import json
from types import SimpleNamespace

import pytest

from lsy_drone_racing.control.rl_race import sweep


@pytest.mark.parametrize("budget,uploaded", [(50, True), (1e-12, False)])
def test_trial_forwards_config_and_publishes_final_files(monkeypatch, tmp_path, budget, uploaded):
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
        summary={},
        log_artifact=lambda x: None,
        finish=lambda: finished.append(True),
    )
    monkeypatch.setattr(sweep, "train", train)
    monkeypatch.setattr(sweep.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(
        sweep.wandb,
        "Artifact",
        lambda *a, **k: SimpleNamespace(add_file=lambda p: attachments.append(p)),
    )
    sweep.main()
    assert calls == {"seed": 3, "policy_layers": (32, 32), "run_name": "offline-test_s3"}
    assert logs[0]["final/success_rate"] == 0.8
    assert (str(checkpoint) in attachments) == uploaded
    assert len(attachments) == (4 if uploaded else 3)
    assert run.summary["checkpoint_uploaded"] == uploaded
    assert run.summary["training_complete"] and finished


def test_machine_directories_are_isolated(monkeypatch, tmp_path):
    from lsy_drone_racing.control.rl_race import sweep_agent

    monkeypatch.setattr(sweep_agent, "RL_DIR", tmp_path)
    a = sweep_agent.machine_environment("9-3-2")
    b = sweep_agent.machine_environment("9-3-3")
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        from pathlib import Path

        assert a[key] != b[key]
        assert Path(a[key]).is_dir()
    assert a["RL_RACE_MACHINE"] == "9-3-2"
    with pytest.raises(ValueError):
        sweep_agent.machine_environment("../../escape")
