"""rover_sim — the rover simulation as a clean, importable building block.

Stage 1 of the SIM_PLATFORM migration: the sim core (worlds, rover body,
safety gate, building generator, batched torch env, graph-capturable fast
paths, fused Triton raycast) lives under `rover_sim.core`. Legacy import
paths (`pnn_sim.*`, `evo.arena`, `evo.fused_scan`) are kept alive by
explicit re-export shims. No behavior change. See docs/SIM_PLATFORM.md.

Stage 5 makes it consumable: `rover_sim.suites` (named scenario registry +
val/holdout selection discipline), `rover_sim.bench` (standard report,
scripted baselines, acceptance gates), `rover_sim.smoke` (fast loop). The
public names are exported lazily (PEP 562) so `import rover_sim` stays
cheap and free of torch/evo side effects — the same pattern
`rover_sim.core.__init__` uses for the render names.
"""

from __future__ import annotations

import importlib

# explicit public-name -> lazy loader; keeps package import side-effect free
_LAZY = {
    # rover_sim.suites
    "suites": ("rover_sim.suites", None),
    "get_suite": ("rover_sim.suites", "get_suite"),
    "build": ("rover_sim.suites", "build"),
    "summarize": ("rover_sim.suites", "summarize"),
    "fingerprint": ("rover_sim.suites", "fingerprint"),
    "require_selectable": ("rover_sim.suites", "require_selectable"),
    # rover_sim.bench
    "bench": ("rover_sim.bench", None),
    "report": ("rover_sim.bench", "report"),
    "run_baselines": ("rover_sim.bench", "run_baselines"),
    "worst_trim_multi_draw_pick": ("rover_sim.bench",
                                   "worst_trim_multi_draw_pick"),
    "trim_envelope": ("rover_sim.bench", "trim_envelope"),
    "pivot_cap": ("rover_sim.bench", "pivot_cap"),
    # rover_sim.smoke
    "smoke": ("rover_sim.smoke", None),
    "run_smoke": ("rover_sim.smoke", "run_smoke"),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        m = importlib.import_module(mod)
        return m if attr is None else getattr(m, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
