"""Evaluation ranking is a reporting objective, never the training reward."""

import pytest
from lsy_drone_racing.control.rl_race.evaluation import ranking_score, evaluate


def test_ranking_eligibility_and_no_success():
    assert ranking_score(0.5, 3.4) == 3.4
    assert ranking_score(0.49, 3.0) > 100
    assert ranking_score(0, None) == 150
    assert ranking_score(0.79, 3.2, 0.8) > 100
    assert ranking_score(0.9, 3.5, 0.8) == 3.5


def test_report_prefix_cannot_overwrite_other_paths():
    with pytest.raises(ValueError, match="report_name"):
        evaluate("nonexistent", report_name="../report")
