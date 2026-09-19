"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

import os
import tempfile
from pathlib import Path

from crazyflow.utils import enable_cache

import lsy_drone_racing.envs  # noqa: F401, register environments with gymnasium

if os.name == "nt":
    # Use a Windows-compatible cache path because the default relies on os.getuid().
    enable_cache(cache_path=Path(tempfile.gettempdir()) / "lsy_drone_racing_jax")
else:
    enable_cache()  # Enable persistent caching of JAX functions.
