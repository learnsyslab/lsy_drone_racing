import os

os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import pytest

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Do not enable XLA caches, crashes PyTest
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

skip_if_headless = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None,
    reason="DISPLAY is not set, skipping test in headless environment",
)
