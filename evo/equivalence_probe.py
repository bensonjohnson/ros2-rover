#!/usr/bin/env python3
"""Fixed-genome equivalence probe: bit-exact CPU rollout check.

SIM_PLATFORM migration instrument (see docs/SIM_PLATFORM.md). Evaluates
ONE genome on the CPU with pinned seeds — noise_seed makes every game
replay one shared noise stream, and a single torch thread removes any
scheduling variance — so the rollout metrics are bit-reproducible. Run
it against two trees (old checkout vs new) and the RESULT lines must
match EXACTLY if the change is behavior-preserving. Verified SIM_PLATFORM
stage 1 (rover_sim/ relocation): legacy and buildings modes both matched
to full float precision.

    cd <tree> && PYTHONPATH=. python3 -m evo.equivalence_probe \
        --genome /abs/path/genome.npz --world legacy
    cd <tree> && PYTHONPATH=. python3 -m evo.equivalence_probe \
        --genome /abs/path/genome.npz --world buildings --level 3

Why CPU: CUDA-graph replay is NOT bit-stable for continuous metrics
(~0.8 m dist over 16 envs, cells +-2) — the Spark smoke covers "does the
CUDA path run and capture", this probe covers "is the math identical".
Also note: a fixed-seed 3-gen EVOLVE smoke is NOT a tight equivalence
instrument — rollout jitter changes selection within 1-2 generations and
downstream metrics diverge between identical-code runs. Compare gen-0
metrics and fixed-genome rollouts instead.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.cuda.is_current_stream_capturing()
except Exception:
    # torch built with CUDA support but no usable driver (e.g. the Hermes
    # SBC): ANY torch.cuda.* call aborts. On CPU rollouts this call is
    # semantically False — shim it. No-op on healthy machines.
    torch.cuda.is_current_stream_capturing = lambda: False

import evo.arena                            # noqa: E402
from evo.arena import Arena                 # noqa: E402
from evo.baselines import run_genome        # noqa: E402
from evo.policy import read_meta            # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=777_000)
    ap.add_argument("--noise-seed", type=int, default=12345)
    ap.add_argument("--world", choices=("legacy", "buildings"),
                    default="legacy")
    ap.add_argument("--level", type=int, default=3)
    a = ap.parse_args()

    t0 = time.monotonic()
    bkw = {}
    if a.world == "buildings":
        from evo.worlds import HOLDOUT_SEED, build_pool
        bkw = dict(worlds=build_pool(a.level, a.games, HOLDOUT_SEED))

    d = np.load(a.genome)
    obs_g = read_meta(d)[0]
    arena = Arena(1, a.games, seed=a.seed, device="cpu", fp16=False,
                  noise_seed=a.noise_seed, gate_obs=(obs_g == "v2"), **bkw)
    r = run_genome(a.genome, arena, a.ticks)
    print(f"elapsed_s={time.monotonic() - t0:.1f}", file=sys.stderr)
    print("TREE=" + evo.arena.__file__)
    print("RESULT=" + json.dumps(r, sort_keys=True))


if __name__ == "__main__":
    main()
