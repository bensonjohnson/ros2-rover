"""Compatibility shim — re-exports from rover_sim.core.rayscan.

Kept so existing `from evo.fused_scan import fused_scan` paths and
`python3 -m evo.fused_scan --help` keep working; the fused Triton kernel,
`fused_scan()` and the CLI moved to rover_sim.core.rayscan in the
SIM_PLATFORM stage 1 relocation.
"""
from __future__ import annotations

from rover_sim.core.rayscan import fused_scan, main  # noqa: F401


if __name__ == "__main__":
    main()
