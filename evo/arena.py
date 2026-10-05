"""Compatibility shim — the arena moved to rover_sim.runner.engine.

Stage 2 of the SIM_PLATFORM migration (docs/SIM_PLATFORM.md §6): `Arena`
is now `rover_sim.runner.engine.RolloutEngine` (the same object under the
alias `Arena`), obs constants + `_proprio` live in rover_sim.runner.obs,
`fitness`/`rev_frac` in rover_sim.scoring, and the fast paths / batched sim
/ building helpers in rover_sim.core. This module re-exports every name
legacy consumers import, so existing scripts keep working unchanged.
"""

from __future__ import annotations

from rover_sim.core.batched import (  # noqa: F401
    BatchedEnv, BatchedGate, batched_preprocess)
from rover_sim.core.buildings import (  # noqa: F401
    _PaddedWorld, world_caps, cell_ceiling)
from rover_sim.core.fast import (  # noqa: F401
    _DefaultRNGMixin, Fp32Env, Fp16Env, GATE_PRESETS, gate_config, NoSyncGate)
from rover_sim.runner.engine import (  # noqa: F401
    Arena, RolloutEngine, CONTROL_HZ)
from rover_sim.runner.obs import (  # noqa: F401
    NUM_BINS, MAX_RANGE, N_PROPRIO, N_LAST_ACT, OBS_DIM, ObsSpec,
    DEFAULT_OBS_SPEC, channel_views, flat, _proprio)
from rover_sim.runner.policy import (  # noqa: F401
    Policy, FunctionPolicy, check_graph_safe)
from rover_sim.scoring import fitness, rev_frac  # noqa: F401

# Identity the migration guarantees: evo.arena.Arena IS the engine class.
assert Arena is RolloutEngine
