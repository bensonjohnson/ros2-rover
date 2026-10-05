"""Contract tests for the stage-2 runner (docs/SIM_PLATFORM.md §3).

    python3 -m rover_sim.tests.test_contract [--device cpu]

C1  channel_views: correct channel shapes, views SHARE storage with the
    flat tensor, flat() returns the same tensor object.
C2  ObsSpec dim == 82; slices() covers [0, 82) exactly once.
C3  FunctionPolicy(closure).step(views, prev) == closure(flat, prev).
C4  a native Policy (bind/reset/step) runs via RolloutEngine.run() eager on
    CPU; reset called, metrics carry the usual keys.
C5  check_graph_safe (CUDA only): capture-safe policy -> captured+match;
    a policy that syncs (.item()) -> captured False with an error, no raise.
"""

from __future__ import annotations

import argparse

import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:
    # torch built with CUDA support but no usable driver (e.g. the Hermes
    # SBC): ANY torch.cuda.* call aborts. On CPU rollouts this call is
    # semantically False — shim it (same as evo.equivalence_probe).
    torch.cuda.is_current_stream_capturing = lambda: False

from ..runner import (DEFAULT_OBS_SPEC, FunctionPolicy, ObsSpec,
                      RolloutEngine, channel_views, check_graph_safe, flat)
from ..runner import NUM_BINS, N_PROPRIO, N_LAST_ACT, OBS_DIM


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # C1: channel views — shapes + shared storage + flat() identity
    obs = torch.rand(5, OBS_DIM)
    views = channel_views(obs)
    assert views["scan_bins"].shape == (5, NUM_BINS)
    assert views["proprio"].shape == (5, N_PROPRIO)
    assert views["prev_act"].shape == (5, N_LAST_ACT)
    assert views["scan_bins"].data_ptr() == obs[:, 0:NUM_BINS].data_ptr()
    assert views["_flat"] is obs
    for name in ("scan_bins", "proprio", "prev_act"):
        assert views[name].data_ptr() == obs.data_ptr() + \
            DEFAULT_OBS_SPEC.slices()[name][0] * obs.element_size()
    for name in ("scan_bins", "proprio", "prev_act"):
        v = views[name]
        v += 1.0                                    # in-place writes through
    assert torch.equal(obs[:, :NUM_BINS], views["scan_bins"])
    assert flat(views) is obs
    print("C1 ok: channel_views shapes + shared storage + flat() identity")

    # C2: ObsSpec dim + slices cover [0, 82) exactly once
    spec = ObsSpec()
    assert spec.dim == OBS_DIM == 82
    sl = spec.slices()
    assert sl == {"scan_bins": (0, 72), "proprio": (72, 80),
                  "prev_act": (80, 82)}
    covered = [i for a, b in sl.values() for i in range(a, b)]
    assert sorted(covered) == list(range(OBS_DIM))
    assert len(covered) == len(set(covered))
    print(f"C2 ok: dim {spec.dim}; slices cover [0, {OBS_DIM}) exactly once")

    # C3: FunctionPolicy == the raw closure
    seen = {}

    def closure(obs_flat, prev):
        seen["obs"], seen["prev"] = obs_flat, prev
        return obs_flat[:, :2] * 0.5

    prev = torch.zeros(5, 2)
    out = FunctionPolicy(closure).step(views, prev)
    assert seen["obs"] is obs and seen["prev"] is prev
    assert torch.equal(out, closure(obs, prev))
    print("C3 ok: FunctionPolicy.step == closure(flat, prev)")

    # C4: native Policy through RolloutEngine.run() (eager, CPU)
    class Native:
        def __init__(self):
            self.obs_spec = ObsSpec()
            self.graph_safe = False
            self.binds = self.resets = self.steps = 0

        def bind(self, P, G):
            self.binds += 1

        def reset(self):
            self.resets += 1

        def step(self, obs, prev_act):
            self.steps += 1
            return torch.tanh(flat(obs)[:, :2])

    eng = RolloutEngine(1, 2, seed=4321, device="cpu", noise_seed=99,
                        cells=False)
    pol = Native()
    m = eng.run(pol, 200)
    assert pol.binds == 1, pol.binds
    assert pol.resets >= 1, pol.resets
    assert pol.steps == 200, pol.steps
    for k in ("rooms", "rooms_total", "dist_m", "collisions", "cells",
              "cells_total", "stops", "fwd_m", "back_m", "net_m",
              "range_m", "spin_t", "spin_frac"):
        assert k in m, k
    assert m["rooms"].shape == (2,)
    print(f"C4 ok: native policy via run() — bind={pol.binds} "
          f"reset={pol.resets} step={pol.steps}")

    # C5: graph-safety checker (CUDA only)
    if not str(args.device).startswith("cuda"):
        print("C5 skip: check_graph_safe is CUDA-only (--device cuda)")
    else:
        safe = FunctionPolicy(lambda o, p: torch.tanh(o[:, :2]),
                              graph_safe=True)
        r = check_graph_safe(safe, P=4, G=2, device=args.device)
        assert r["captured"] and r["match"], r
        print(f"C5a ok: capture-safe policy captured+matched ({r})")

        def unsafe(o, p):
            _ = o[:, 0].sum().item()             # host sync -> capture error
            return o[:, :2]

        r2 = check_graph_safe(FunctionPolicy(unsafe, graph_safe=True),
                              P=4, G=2, device=args.device)
        assert not r2["captured"] and r2["error"], r2
        print(f"C5b ok: host-sync policy not captured, error recorded "
              f"({r2['error'][:60]})")

    print("ALL CONTRACT CHECKS PASSED")


if __name__ == "__main__":
    main()
