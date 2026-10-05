"""Core simulation modules — one source of truth, no training code.

    world.py      World, make_house
    rover.py      RoverConfig, SimRover
    geom.py       BodyConfig, HW_BODY, mounts, rect perimeter sampling
    gate.py       GateConfig, SimSafetyGate, GATE_PRESETS (from pnn_sim/safety_gate.py)
    buildings.py  Building families, levels, rooms + arena support helpers
    batched.py    BatchedEnv, BatchedGate, batched_preprocess
    fast.py       Fp32/Fp16 envs, NoSyncGate, gate presets (from evo/arena.py)
    rayscan.py    fused Triton lidar scan      (from evo/fused_scan.py)
"""

from .geom import (  # noqa: F401  (public re-exports, stage 4a)
    BodyConfig,
    HW_BODY,
    HW_LIDAR_MOUNT_X,
    HW_CAMERA_MOUNT_X,
    RectBody,
    rect_perimeter_offsets,
    rect_max_sample_spacing,
)

__all__ = [
    "BodyConfig",
    "HW_BODY",
    "HW_LIDAR_MOUNT_X",
    "HW_CAMERA_MOUNT_X",
    "RectBody",
    "rect_perimeter_offsets",
    "rect_max_sample_spacing",
]
