"""Core simulation modules — one source of truth, no training code.

    world.py      World, make_house
    rover.py      RoverConfig, SimRover
    gate.py       GateConfig, SimSafetyGate   (from pnn_sim/safety_gate.py)
    buildings.py  Building families, levels, rooms + arena support helpers
    batched.py    BatchedEnv, BatchedGate, batched_preprocess
    fast.py       Fp32/Fp16 envs, NoSyncGate, gate presets (from evo/arena.py)
    rayscan.py    fused Triton lidar scan      (from evo/fused_scan.py)
"""
