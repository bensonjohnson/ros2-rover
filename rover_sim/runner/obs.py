"""Observation presentation layer — ObsSpec + zero-copy channel views.

Stage 2 of the SIM_PLATFORM migration (docs/SIM_PLATFORM.md §3). The obs
constants (NUM_BINS/MAX_RANGE/N_PROPRIO/N_LAST_ACT/OBS_DIM) and the
`_proprio` assembler move verbatim out of evo/arena.py; ObsSpec names the
channel layout the engine presents to a Policy, and `channel_views()`
hands out VIEWS into the engine's flat obs buffer — Python slicing only,
no copies, no GPU ops, because these run inside the CUDA-graph capture
region.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rover_sim.core.batched import BatchedEnv

NUM_BINS = 72
MAX_RANGE = 5.0
N_PROPRIO = 8
N_LAST_ACT = 2
OBS_DIM = NUM_BINS + N_PROPRIO + N_LAST_ACT


def _proprio(env: BatchedEnv) -> torch.Tensor:
    """Same 8-channel proprio and normalization the PC trainer used — the
    sensors exist, the policy just has to make something of them."""
    e = env
    return torch.stack([
        (0.5 + 0.5 * e.wheel_l / 8.0).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.wheel_r / 8.0).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.noise(e.cfg.gyro_noise_std) / 2.5).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.noise(e.cfg.gyro_noise_std) / 2.5).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.yaw_rate / 2.5).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.accel[:, 0] / 19.6).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.accel[:, 1] / 19.6).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.accel[:, 2] / 19.6).clamp(0.0, 1.0),
    ], dim=1)


@dataclass(frozen=True)
class ObsSpec:
    """Which channels the engine presents to a Policy. Stage-2 layout is
    fixed and byte-compatible with the pre-platform obs: [scan_bins(72) |
    proprio(8) | prev_act(2)]. proprio_version selects the SEMANTICS of
    proprio channels 2/3: 'v1' = gyro noise, 'v2' = gate latched front/rear
    blocked flags (engine gate_obs=True). Adding channels is a future stage;
    existing layouts never move."""
    proprio_version: str = "v1"

    @property
    def dim(self) -> int:
        return OBS_DIM

    def slices(self) -> dict:  # channel name -> (start, stop)
        return {"scan_bins": (0, NUM_BINS),
                "proprio": (NUM_BINS, NUM_BINS + N_PROPRIO),
                "prev_act": (NUM_BINS + N_PROPRIO, OBS_DIM)}


DEFAULT_OBS_SPEC = ObsSpec()


def channel_views(obs_flat: torch.Tensor, spec: ObsSpec = DEFAULT_OBS_SPEC) -> dict:
    """Plain dict of VIEWS into the same storage: no copies, no GPU ops.
    `_flat` is the reserved key back to the whole concatenated obs."""
    views = {name: obs_flat[:, sl[0]:sl[1]]
             for name, sl in spec.slices().items()}
    views["_flat"] = obs_flat
    return views


def flat(obs: dict) -> torch.Tensor:
    """The concatenated obs back out of a channel_views() dict."""
    return obs["_flat"]
