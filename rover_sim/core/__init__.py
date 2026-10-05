"""Core simulation modules — one source of truth, no training code.

    world.py      World, make_house
    rover.py      RoverConfig, SimRover
    geom.py       BodyConfig, HW_BODY, mounts, rect perimeter sampling
    gate.py       GateConfig, SimSafetyGate, GATE_PRESETS (from pnn_sim/safety_gate.py)
    buildings.py  Building families, levels, rooms + arena support helpers
    batched.py    BatchedEnv, BatchedGate, batched_preprocess
    fast.py       Fp32/Fp16 envs, NoSyncGate, gate presets (from evo/arena.py)
    rayscan.py    fused Triton lidar scan      (from evo/fused_scan.py)
    render.py     egocentric 2D camera renderer (NEW — stage 4b)
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

# Stage-4b render names are exported lazily (PEP 562): importing `render`
# eagerly here would make `python -m rover_sim.core.render --bench` re-import
# itself as a submodule of its own package and emit a spurious
# "found in sys.modules" RuntimeWarning. `from rover_sim.core import
# CameraConfig` still works via __getattr__.
_RENDER_EXPORTS = ("CameraConfig", "CameraRenderer", "column_angles",
                   "render_np")


def __getattr__(name):
    if name in _RENDER_EXPORTS:
        from . import render as _render
        return getattr(_render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BodyConfig",
    "HW_BODY",
    "HW_LIDAR_MOUNT_X",
    "HW_CAMERA_MOUNT_X",
    "RectBody",
    "rect_perimeter_offsets",
    "rect_max_sample_spacing",
    "CameraConfig",
    "CameraRenderer",
    "column_angles",
    "render_np",
]
