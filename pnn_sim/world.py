"""Compatibility shim — re-exports from rover_sim.core.world.

Kept so existing `from pnn_sim.world import ...` paths keep working; the
implementation moved to rover_sim.core.world in the SIM_PLATFORM stage 1
relocation. `import *` skips underscore names, so they are listed here.
"""
from rover_sim.core.world import (  # noqa: F401
    World, make_house, _box_segments, _wall_with_door)
