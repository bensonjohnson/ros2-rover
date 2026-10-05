"""Runner equivalence tests (docs/SIM_PLATFORM.md §6 stage 2).

    python3 -m rover_sim.tests.test_runner_equiv [--device cpu]

E1  legacy run_games(closure) vs protocol run(FunctionPolicy(closure)) on
    CPU: every metric tensor bit-identical (legacy world).
E2  same for building mode (small pool).
E3  graph gating: run(graph=True) with a non-graph-safe policy runs eager on
    CPU without exception and equals E1.
E4  (CUDA only) capture path: a graph-safe FunctionPolicy captures, replays
    and matches the EAGER run of the SAME policy (coarse metrics exact,
    dist within replay-jitter tolerance). NOTE one live graph per process —
    E4 is the only capture in this process; run with --device cuda.

E1-E3 are CPU-pinned by design: bit-exactness is a CPU-determinism claim
(CUDA eager determinism is not guaranteed by the stack); E4 owns the GPU.
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

from evo.baselines import wall_follower
from ..runner import FunctionPolicy, RolloutEngine


def _equal_metrics(m1: dict, m2: dict):
    assert set(m1) == set(m2), (set(m1) ^ set(m2))
    for k in m1:
        assert torch.equal(m1[k], m2[k]), k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device
    ticks = 900
    cpu = "cpu"   # E1-E3 pin to CPU on purpose (see module docstring)

    # E1: legacy vs protocol, legacy world
    kw = dict(seed=777_000, noise_seed=12345, device=cpu, fp16=False)
    a1 = RolloutEngine(1, 8, **kw)
    a2 = RolloutEngine(1, 8, **kw)
    m1 = a1.run_games(wall_follower, ticks)
    m2 = a2.run(FunctionPolicy(wall_follower), ticks)
    _equal_metrics(m1, m2)
    print(f"E1 ok: legacy vs protocol bit-exact (rooms "
          f"{float(m1['rooms'].mean()):.3f}, dist_m "
          f"{float(m1['dist_m'].mean()):.3f})")

    # E2: building mode
    from evo.worlds import build_pool
    pool = build_pool(3, 8, 778_000)
    bkw = dict(seed=778_000, noise_seed=12345, device=cpu, fp16=False,
               worlds=pool)
    b1 = RolloutEngine(1, 8, **bkw)
    b2 = RolloutEngine(1, 8, **bkw)
    mb1 = b1.run_games(wall_follower, ticks)
    mb2 = b2.run(FunctionPolicy(wall_follower), ticks)
    _equal_metrics(mb1, mb2)
    print(f"E2 ok: buildings mode bit-exact (rooms "
          f"{float(mb1['rooms'].mean()):.3f}, door_x "
          f"{float(mb1['door_crossings'].mean()):.3f})")

    # E3: graph=True with a non-graph-safe policy -> eager, same numbers
    a3 = RolloutEngine(1, 8, **kw)
    m3 = a3.run(FunctionPolicy(wall_follower, graph_safe=False), ticks,
                graph=True)
    _equal_metrics(m1, m3)
    print("E3 ok: graph=True on non-graph-safe policy ran eager and matched")

    # E4: capture path (CUDA only). Baseline = the EAGER run of the SAME
    # policy (E1's wall_follower is a different controller — not comparable).
    if not str(dev).startswith("cuda"):
        print("E4 skip: capture path is CUDA-only (--device cuda)")
    else:
        ckw = dict(seed=777_000, noise_seed=12345, device=dev, fp16=False)
        eager_eng = RolloutEngine(1, 8, **ckw)
        m_eager = eager_eng.run(
            FunctionPolicy(lambda o, p: torch.tanh(o[:, :2])), ticks)
        cap_eng = RolloutEngine(1, 8, **ckw)
        mc = cap_eng.run(
            FunctionPolicy(lambda o, p: torch.tanh(o[:, :2]), graph_safe=True),
            ticks, graph=True)
        assert not getattr(cap_eng, "_graph_broken", False), "capture failed"
        assert getattr(cap_eng, "_graphs", None), "no graph captured"
        assert torch.equal(mc["rooms"], m_eager["rooms"]), \
            f"rooms differ: cap={mc['rooms'].tolist()} " \
            f"eager={m_eager['rooms'].tolist()}"
        assert torch.equal(mc["collisions"], m_eager["collisions"]), \
            f"collisions differ: cap={mc['collisions'].tolist()} " \
            f"eager={m_eager['collisions'].tolist()}"
        assert (mc["dist_m"] - m_eager["dist_m"]).abs().max() < 2.0, \
            f"dist_m outside replay-jitter: cap={mc['dist_m'].tolist()} " \
            f"eager={m_eager['dist_m'].tolist()}"
        cap_eng.drop_graphs()
        print("E4 ok: graph-safe capture matched the eager run (same policy)")

    print("ALL RUNNER EQUIVALENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
