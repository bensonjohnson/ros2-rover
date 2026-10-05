"""Compatibility shim — re-exports from rover_sim.core.batched.

Kept so existing `from pnn_sim.batched.env import ...` paths keep working;
the implementation moved to rover_sim.core.batched in the SIM_PLATFORM
stage 1 relocation. `import *` skips underscore names, so the module-private
helpers are listed explicitly.
"""
from rover_sim.core.batched import (  # noqa: F401
    BatchedEnv, BatchedGate, batched_preprocess,
    _bin_index, _DUMMY_SEG, _BIN_CACHE)
