"""Policy plug-in contract for the rollout engine (docs/SIM_PLATFORM.md §3).

    Policy          runtime_checkable Protocol: obs_spec, graph_safe,
                    bind(P, G), reset(), step(obs, prev_act) -> raw cmd [B, 2]
    FunctionPolicy  adapts a legacy policy_step(obs_flat[82], prev_act)
                    closure (graph_safe=False by default)
    check_graph_safe
                    sandbox capture+replay of one step; CUDA-only
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .obs import DEFAULT_OBS_SPEC, OBS_DIM, ObsSpec, channel_views, flat


@runtime_checkable
class Policy(Protocol):
    """RolloutEngine plug-in contract (docs/SIM_PLATFORM.md §3).
    - obs_spec: ObsSpec (channels the engine presents).
    - graph_safe: True only when step() is capture-safe: static buffers,
      in-place updates, no host sync (.item()/int()/.cpu()), default-RNG
      only, no rebinding of state tensors.
    - bind(P,G): allocate for P individuals x G games (idempotent/cheap on repeat).
    - reset(): zero recurrent state between games; idempotent. The engine
      registers it as a reset hook when present.
    - step(obs, prev_act) -> raw cmd [B,2]; the engine clamps."""
    obs_spec: "ObsSpec"
    graph_safe: bool

    def bind(self, P: int, G: int) -> None: ...

    def reset(self) -> None: ...

    def step(self, obs: dict, prev_act: torch.Tensor) -> torch.Tensor: ...


class FunctionPolicy:
    """Adapts a legacy policy_step(obs_flat[82], prev_act) closure.
    obs_spec=None means 'accept whatever layout the engine produces'."""

    def __init__(self, fn, *, graph_safe: bool = False, obs_spec=None):
        self._fn = fn
        self.graph_safe = graph_safe
        self.obs_spec = obs_spec

    def bind(self, P, G):
        pass

    def reset(self):
        pass

    def step(self, obs, prev_act):
        return self._fn(flat(obs), prev_act)


def check_graph_safe(policy, *, P: int = 8, G: int = 4,
                     device: str = "cuda") -> dict:
    """Sandbox capture+replay of one step: warmup x3 on a side stream,
    capture policy.step into a CUDA graph, then (a) replay twice, compare
    outputs bitwise; (b) write fresh values into the obs buffer, run eager
    step on those values, replay the graph, compare eager vs replay.
    Returns {"captured": bool, "match": bool, "error": str | None}.
    Catches capture exceptions (does not raise). CUDA-only."""
    result = {"captured": False, "match": False, "error": None}
    if not str(device).startswith("cuda"):
        result["error"] = "cuda-only: device is not cuda"
        return result
    try:
        import torch.cuda
        spec = getattr(policy, "obs_spec", None) or DEFAULT_OBS_SPEC
        B = P * G
        dev = torch.device(device)
        obs_flat = torch.rand(B, OBS_DIM, device=dev)
        prev_act = torch.zeros(B, 2, device=dev)
        views = channel_views(obs_flat, spec)

        # warmup on a side stream (allocs settle, kernels pick)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                policy.step(views, prev_act)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_cap = policy.step(views, prev_act)
        g.replay()
        torch.cuda.synchronize()
        r1 = out_cap.clone()
        g.replay()
        torch.cuda.synchronize()
        r2 = out_cap.clone()
        # fresh obs values -> eager step vs graph replay
        obs_flat.copy_(torch.rand(B, OBS_DIM, device=dev))
        eager = policy.step(views, prev_act).clone()
        g.replay()
        torch.cuda.synchronize()
        r3 = out_cap.clone()

        result["captured"] = True
        result["match"] = bool(torch.equal(r1, r2) and torch.equal(eager, r3))
    except Exception as ex:                          # noqa: BLE001
        result["captured"] = False
        result["match"] = False
        result["error"] = f"{type(ex).__name__}: {ex}"
    return result
