"""Local sweep state and output isolation without launching GPU jobs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsy_drone_racing.control.rl_race import research


def test_local_sweep_separates_trials_and_refuses_overwrite(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                common=dict(seed=1),
                trials=[dict(id="a", parameters={}), dict(id="b", parameters=dict(seed=2))],
            )
        )
    )

    def launch(command, **kwargs):
        assert kwargs["env"]["WANDB_MODE"] == "offline"
        config = json.loads(Path(command[-1]).read_text())
        output = Path(config["output_root"]) / config["run_name"]
        output.mkdir()
        (output / "final_evaluation.json").write_text('{"success_rate": 0.8}')
        return SimpleNamespace(pid=123, poll=lambda: 0)

    monkeypatch.setattr(research.subprocess, "Popen", launch)
    research.schedule(str(plan), parallel=2)
    state = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["status"] for row in state] == ["complete", "complete"]
    assert state[0]["output_dir"] != state[1]["output_dir"]
    assert state[1]["parameters"]["seed"] == 2
    assert (tmp_path / "configs/a.json").exists() and (tmp_path / "logs/b.log").exists()
    with pytest.raises(FileExistsError):
        research.schedule(str(plan))


def test_local_sweep_rejects_colliding_ids(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trials": [{"id": "same"}, {"id": "same"}]}))
    with pytest.raises(ValueError, match="Duplicate"):
        research.schedule(str(plan))
    with pytest.raises(ValueError, match="positive"):
        research.schedule(str(plan), parallel=0)
