"""rover_sim — the rover simulation as a clean, importable building block.

Stage 1 of the SIM_PLATFORM migration: the sim core (worlds, rover body,
safety gate, building generator, batched torch env, graph-capturable fast
paths, fused Triton raycast) lives under `rover_sim.core`. Legacy import
paths (`pnn_sim.*`, `evo.arena`, `evo.fused_scan`) are kept alive by
explicit re-export shims. No behavior change. See docs/SIM_PLATFORM.md.
"""
