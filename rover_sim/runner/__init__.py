"""rover_sim.runner — the rollout engine + model plug-in contract.

Stage 2 of the SIM_PLATFORM migration (docs/SIM_PLATFORM.md §3): the evo
`Arena` becomes `RolloutEngine`, obs assembly becomes an explicit `ObsSpec`
+ channel presentation layer, and the `Policy` protocol is the centerpiece
plug-in contract. Explicit re-exports below are the public surface.
"""

from .engine import Arena, CONTROL_HZ, RolloutEngine
from .obs import (DEFAULT_OBS_SPEC, MAX_RANGE, N_LAST_ACT, N_PROPRIO,
                  NUM_BINS, OBS_DIM, ObsSpec, channel_views, flat)
from .policy import FunctionPolicy, Policy, check_graph_safe
from ..core.render import CameraConfig, CameraRenderer, render_np

__all__ = [
    "RolloutEngine", "Arena", "ObsSpec", "DEFAULT_OBS_SPEC",
    "channel_views", "flat", "Policy", "FunctionPolicy",
    "check_graph_safe", "OBS_DIM", "NUM_BINS", "MAX_RANGE", "N_PROPRIO",
    "N_LAST_ACT", "CONTROL_HZ",
    "CameraConfig", "CameraRenderer", "render_np",
]
