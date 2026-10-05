"""Compatibility shim — re-exports from rover_sim.core.rover.

Kept so existing `from pnn_sim.rover import ...` paths keep working; the
implementation moved to rover_sim.core.rover in the SIM_PLATFORM stage 1
relocation.
"""
from rover_sim.core.rover import RoverConfig, SimRover  # noqa: F401
