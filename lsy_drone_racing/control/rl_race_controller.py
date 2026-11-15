"""Competition loader entry point for exported rl_race policies (see rl_race/README.md)."""

from lsy_drone_racing.control.rl_race.controller import RaceController


class SubmissionController(RaceController):
    """Expose one locally defined controller class to the official loader."""
