"""Compatibility shim — re-exports from rover_sim.core.buildings.

Kept so existing `from pnn_sim.buildings import ...` paths keep working;
the implementation moved to rover_sim.core.buildings in the SIM_PLATFORM
stage 1 relocation.
"""
from rover_sim.core.buildings import (  # noqa: F401
    Building, make_level, make_building, house_to_building,
    LEVELS, FAMILIES, MIX_WEIGHTS,
    U, ROBOT_R, FREE_MARGIN, RASTER, MAX_SEGMENTS)
