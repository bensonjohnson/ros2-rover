"""ES adapter equivalence tests (docs/SIM_PLATFORM.md §6 stage 3a).

    python3 -m rover_sim.tests.test_es_adapter [--device cpu]

A1  BIT-EXACT: rover_sim.evaluate.evaluate (adapter + engine.run) vs the
    legacy evo.evolve.evaluate closure, on CPU. Identity over the fitness
    vector AND every metric tensor, merged-houses production config, for
    the mem=0 and mem>0 paths.
A2  protocol shape: obs_spec, graph_safe, idempotent bind(), reset() zeroes
    hidden + memory, set_thetas() writes the static buffer, step() output
    shape/dtype/range.
A3  (CUDA only) real-adapter capture equivalence: graph=True replay of the
    SAME adapter/weights as an eager adapter run (coarse metrics exact,
    dist within replay-jitter tolerance). NOTE one live graph per process —
    A3 is the only capture here; run with --device cuda.
A4  hook hygiene: repeated one-shot evaluate() calls do not leak reset hooks.
A5  shim identity: evo.policy re-exports the same objects as
    rover_sim.policies.es_genome (including the private _param_shapes).

A1-A2, A4 are CPU-pinned: bit-exactness is a CPU-determinism claim.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:
    # torch built with CUDA support but no usable driver (e.g. the Hermes
    # SBC): ANY torch.cuda.* call aborts. On CPU rollouts this call is
    # semantically False — shim it (same as evo.equivalence_probe).
    torch.cuda.is_current_stream_capturing = lambda: False

from evo.evolve import evaluate as legacy_evaluate
from rover_sim.adapters.es import ESGenomePolicy
from rover_sim.evaluate import evaluate as new_evaluate
from rover_sim.policies.es_genome import sample_population
from rover_sim.runner import Arena, DEFAULT_OBS_SPEC, OBS_DIM, channel_views


def _equal_metrics(m1: dict, m2: dict):
    assert set(m1) == set(m2), (set(m1) ^ set(m2))
    for k in m1:
        assert torch.equal(m1[k], m2[k]), k


def _a1_case(P, G, hidden, mem, merged, seed, noise_seed, rng_seed, ticks,
             label):
    kw = dict(seed=seed, noise_seed=noise_seed, merged_houses=merged,
              device="cpu", fp16=False)
    engA = Arena(P, G, **kw)
    engB = Arena(P, G, **kw)
    thetas = torch.as_tensor(
        sample_population(P, OBS_DIM, hidden, np.random.default_rng(rng_seed),
                          mem=mem))
    fit_A, m_A = legacy_evaluate(engA, thetas, hidden, ticks, mem=mem)
    fit_B, m_B = new_evaluate(engB, thetas, hidden, ticks, mem=mem)
    assert torch.equal(fit_A, fit_B), \
        f"[{label}] fitness differs: legacy={fit_A.tolist()} " \
        f"new={fit_B.tolist()}"
    _equal_metrics(m_A, m_B)
    print(f"A1 ok [{label}]: legacy-vs-adapter bit-exact "
          f"(P={P} G={G} merged={merged} mem={mem} B={P * G * merged}; "
          f"fit[0]={float(fit_A[0]):.6f}, dist_m mean "
          f"{float(m_A['dist_m'].mean()):.4f})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device

    # A1: bit-exact legacy vs adapter -------------------------------------
    # merged_houses=2 is the production config (MUST be covered).
    _a1_case(3, 4, 16, 0, 2, 777_000, 12345, 7, 400, "merged mem=0")
    _a1_case(3, 4, 16, 4, 1, 778_000, 54321, 11, 400, "mem=4")

    # A2: protocol shape --------------------------------------------------
    pol = ESGenomePolicy(2, 8, merged_houses=2)
    assert pol.obs_spec.proprio_version == "v1", pol.obs_spec
    assert pol.graph_safe is True
    assert ESGenomePolicy(2, 8, obs="v2").obs_spec.proprio_version == "v2"
    pol.bind(2, 3)
    assert pol._G_eff == 6, pol._G_eff
    pol.bind(2, 3)                      # idempotent: no error, same layout
    assert pol._G_eff == 6 and tuple(pol.h.shape) == (2, 6, 8)
    # set_thetas writes the static buffer
    fresh = torch.randn(2, pol.theta_buf.shape[1])
    pol.set_thetas(fresh)
    assert torch.equal(pol.theta_buf, fresh)
    # reset(): hidden zeroed, then memory (mem>0) zeroed. Order h-then-m
    # matches the legacy hook registration; test the mem path.
    pm = ESGenomePolicy(2, 8, mem=4, merged_houses=2)
    pm.bind(2, 3)
    pm.h.fill_(0.7)
    pm.net.m.fill_(0.9)
    pm.reset()
    assert pm.h.abs().max() == 0, pm.h.abs().max()
    assert pm.net.m is not None and pm.net.m.abs().max() == 0
    # step() shape / finiteness / tanh range
    B = 2 * 6
    obs = channel_views(torch.rand(B, OBS_DIM), DEFAULT_OBS_SPEC)
    out = pol.step(obs, torch.zeros(B, 2))
    assert out.shape == (B, 2), out.shape
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0, out.abs().max()
    print(f"A2 ok: obs_spec/graph_safe/bind idempotent/reset order/"
          f"set_thetas/step shape (B={B}, |a|max={float(out.abs().max()):.4f})")

    # A3: real-adapter capture equivalence (CUDA only) --------------------
    if not str(dev).startswith("cuda"):
        print("A3 skip: real-adapter capture is CUDA-only (--device cuda)")
    else:
        ckw = dict(seed=777_000, noise_seed=12345, merged_houses=2,
                   device=dev, fp16=False)
        thetas = torch.as_tensor(
            sample_population(3, OBS_DIM, 16, np.random.default_rng(7)),
            device=dev)
        eager_eng = Arena(3, 4, **ckw)
        eager_pol = ESGenomePolicy(3, 16, merged_houses=2, device=dev)
        eager_pol.set_thetas(thetas)
        m_eager = eager_eng.run(eager_pol, 900)
        cap_eng = Arena(3, 4, **ckw)
        cap_pol = ESGenomePolicy(3, 16, merged_houses=2, device=dev)
        cap_pol.set_thetas(thetas)
        mc = cap_eng.run(cap_pol, 900, graph=True)
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
        print("A3 ok: real-adapter capture matched the eager adapter run")

    # A4: hook hygiene ----------------------------------------------------
    kw = dict(seed=777_000, noise_seed=12345, device="cpu", fp16=False)
    eng = Arena(2, 3, **kw)
    n0 = len(eng._reset_hooks)
    thetas = torch.as_tensor(
        sample_population(2, OBS_DIM, 16, np.random.default_rng(3)))
    new_evaluate(eng, thetas, 16, ticks=50)
    assert len(eng._reset_hooks) == n0, \
        f"hook leaked after 1st evaluate: {len(eng._reset_hooks)} != {n0}"
    new_evaluate(eng, thetas, 16, ticks=50)
    assert len(eng._reset_hooks) == n0, \
        f"hook leaked after 2nd evaluate: {len(eng._reset_hooks)} != {n0}"
    print(f"A4 ok: repeated evaluate() left _reset_hooks at {n0}")

    # A5: shim identity ---------------------------------------------------
    import evo.policy as shim
    import rover_sim.policies.es_genome as g
    for name in ("PopulationNet", "genome_meta", "genome_size",
                 "per_gene_scale", "read_meta", "sample_population",
                 "OBS_MODES", "ACTION_MODES", "_param_shapes"):
        assert getattr(shim, name) is getattr(g, name), name
    print("A5 ok: evo.policy shim re-exports the es_genome objects")

    print("ALL ES ADAPTER CHECKS PASSED")


if __name__ == "__main__":
    main()
