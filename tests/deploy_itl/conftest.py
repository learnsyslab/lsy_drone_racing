"""Fixtures for the crazyflow_itl deployment tests."""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Iterator

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("crazyflow_itl")

from tests.deploy_itl.harness import DeployStack  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

DOMAIN_IDS = itertools.count(random.randint(100, 150))


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[DeployStack]:
    """Provide a deployment stack whose processes are all killed after the test."""
    stack = DeployStack(tmp_path, next(DOMAIN_IDS))
    yield stack
    print(stack.close())
