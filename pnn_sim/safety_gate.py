"""Compatibility shim — re-exports from rover_sim.core.gate.

Kept so existing `from pnn_sim.safety_gate import ...` paths keep working;
the implementation (GateConfig, SimSafetyGate) moved to rover_sim.core.gate
in the SIM_PLATFORM stage 1 relocation. `import *` skips underscore names,
so the sector constants are listed explicitly.
"""
from rover_sim.core.gate import (  # noqa: F401
    GateConfig, SimSafetyGate, _SIDE_MIN, _SIDE_MAX, _REAR_MIN)
