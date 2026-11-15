"""Summarize all trials, retaining failures and distinguishing inherited budgets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(directory: str) -> list[dict]:
    """Write an auditable CSV/Markdown overview; ranking here is validation only."""
    root = Path(directory).resolve()
    rows = []
    for manifest in sorted(root.glob("*/manifest.json")):
        for trial in json.loads(manifest.read_text()):
            p = trial["parameters"]
            run = Path(trial["output_dir"])
            evaluation = trial.get("evaluation", {})
            if (run / "final_evaluation.json").exists():
                evaluation = json.loads((run / "final_evaluation.json").read_text())
            holdout = {}
            if (run / "holdout_evaluation.json").exists():
                holdout = json.loads((run / "holdout_evaluation.json").read_text())
            stages = (
                json.loads((run / "stages.json").read_text())
                if (run / "stages.json").exists()
                else []
            )
            row = dict(
                phase=manifest.parent.name,
                trial=trial["id"],
                status=trial["status"],
                seed=p.get("seed", 0),
                parent=p.get("start_params", "scratch"),
                stage=p.get("start_stage", "pretrain"),
                actual_steps=sum(s["steps"] for s in stages),
                budget=p.get("anneal_steps", 0)
                if p.get("start_stage") == "anneal"
                else p.get("race_steps"),
                roll=p.get("roll_limit_deg", 70),
                pitch=p.get("pitch_limit_deg", 70),
                yaw=p.get("yaw_limit_deg", 70),
                actor=str(p.get("policy_layers", [256, 256, 256])),
                critic=str(p.get("value_layers", [256] * 4)),
                official_prob=p.get("official_start_prob", 0.5),
                race_time=p.get("race_time", 0.05),
                crash=p.get("race_crash", 20),
                entropy=p.get("entropy_cost", 0.005),
                success=evaluation.get("success_rate"),
                time=evaluation.get("mean_success_time"),
                episodes=evaluation.get("episodes"),
                holdout_success=holdout.get("success_rate"),
                holdout_time=holdout.get("mean_success_time"),
                output=str(run),
            )
            rows.append(row)
    if rows:
        with (root / "results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Research results (validation, not official leaderboard submissions)",
        "",
        "All trials retained. Continuations and scratch runs have different inherited budgets; "
        "compare within each phase.",
        "",
        "| Phase | Trial | Status | Seed | New steps | Success | Successful mean [s] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        rate = "—" if r["success"] is None else f"{r['success']:.2%}"
        lap = "—" if r["time"] is None else f"{r['time']:.4f}"
        lines.append(
            f"| {r['phase']} | {r['trial']} | {r['status']} | {r['seed']} | "
            f"{r['actual_steps']:,} | {rate} | {lap} |"
        )
    eligible = [
        r
        for r in rows
        if r["status"] == "complete"
        and r["success"] is not None
        and r["success"] >= 0.8
        and r["time"] is not None
    ]
    speed_eligible = [
        r
        for r in rows
        if r["status"] == "complete"
        and r["success"] is not None
        and r["success"] >= 0.5
        and r["time"] is not None
    ]
    if speed_eligible:
        fastest = min(speed_eligible, key=lambda r: r["time"])
        lines.extend(
            [
                "",
                f"Validation fastest at >=50% success: {fastest['phase']}/{fastest['trial']} "
                f"({fastest['time']:.4f}s, {fastest['success']:.2%}).",
            ]
        )
    if eligible:
        best = min(eligible, key=lambda r: r["time"])
        lines.extend(
            [
                "",
                f"Validation fastest at >=80% success: {best['phase']}/{best['trial']} "
                f"({best['time']:.4f}s, {best['success']:.2%}).",
                "This choice must be checked on reserved holdout seeds before recommendation.",
            ]
        )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    summarize(parser.parse_args().directory)
