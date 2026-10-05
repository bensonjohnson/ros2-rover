"""Fast smoke loop for new model/strategy ideas (SIM_PLATFORM §5).

    python3 -m rover_sim.smoke [--suite legacy|buildings] [--pop 8]
                               [--gens 2] [--ticks 200] [--seed 4242]

A small-population, few-generation run of the rover_sim GA strategy
(`rover_sim.strategies.reproduce`) over a subset of a named suite, printing
the standard bench table per generation plus the final best genome. This is
the "does this idea behave" loop before committing to a real run: it must
finish in seconds on CPU.

Determinism (CPU): `--seed` pins `torch.manual_seed`, the numpy population
rng (`seed + 99`, the evolve layout), and the engine noise seed, so the
rollout is replay-stable and the smoke is bit-reproducible across runs on
CPU. This is a self-contained pin, not the sitecustomize/SEEDDIR recipe
that the MINI-EVOLVE equivalence gate uses. CUDA is NOT bit-stable for
continuous metrics (docs/SIM_PLATFORM.md §3) — the smoke still runs there
but its numbers are only tolerance-stable.

The full evaluation primitive is `rover_sim.evaluate.evaluate` and the
report table is `rover_sim.bench.format_table`, so the smoke prints exactly
the metrics evolve/baselines print.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:                       # noqa: BLE001
    torch.cuda.is_current_stream_capturing = lambda: False

from rover_sim import bench, suites as _suites
from rover_sim.evaluate import evaluate
from rover_sim.policies.es_genome import (genome_size, per_gene_scale,
                                          sample_population)
from rover_sim.runner import Arena, OBS_DIM
from rover_sim.strategies import reproduce

__all__ = ["run_smoke", "main"]


def run_smoke(suite="legacy", *, pop=8, gens=2, ticks=200, seed=4242,
              device="cpu", hidden=16, games=4, level=None, train_seed=None,
              w_dist=0.03, w_coll=0.25, cache=True, quiet=False) -> dict:
    """Small GA run over `suite`; returns {gen rows, best row, best_theta}."""
    if str(device) == "cpu":
        torch.set_num_threads(1)       # single-thread CPU -> replay-stable
    s = _suites.get_suite(suite)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 99)
    N = genome_size(OBS_DIM, hidden)
    scale = torch.as_tensor(per_gene_scale(OBS_DIM, hidden), device=device)
    thetas = torch.as_tensor(sample_population(pop, OBS_DIM, hidden, rng),
                             device=device)
    sigma = torch.full((pop, 1), 0.03, device=device)
    seed_engine = s.engine_seed(train_seed)
    worlds = (s.build(n=games, level=level, train_seed=train_seed,
                      cache=cache) if s.external_worlds else None)
    kw: dict = dict(seed=seed_engine, device=device, noise_seed=seed,
                    gate_obs=False)
    if s.external_worlds:
        kw["worlds"] = list(worlds)[:games]
    eng = Arena(pop, games, **kw)
    if not quiet:
        if s.external_worlds:
            print(f"[smoke] suite={s.name} kind={s.kind} pop={pop} "
                  f"games={games} ticks={ticks} seed={seed} | "
                  f"{_suites.summarize(worlds)}")
        else:
            print(f"[smoke] suite={s.name} kind={s.kind} pop={pop} "
                  f"games={games} ticks={ticks} seed={seed} "
                  f"engine_seed={seed_engine}")

    n_elite = max(2, pop // 4)
    n_immigrant = max(1, pop // 10)
    tau = 1.0 / np.sqrt(N)
    rows = []
    best_fit, best_theta = -np.inf, thetas[0].cpu().clone()
    t0 = time.time()
    for gen in range(gens):
        fit, m = evaluate(eng, thetas, hidden, ticks, w_dist=w_dist,
                          w_coll=w_coll)
        fit_np = fit.cpu().numpy()
        order = np.argsort(-fit_np)
        if fit_np[order[0]] > best_fit:
            best_fit = float(fit_np[order[0]])
            best_theta = thetas[order[0]].cpu().clone()
        row = bench._row(f"gen{gen}", fit, m)
        rows.append(row)
        if not quiet:
            print(f"  g{gen:>3d} best={row['fitness']:.4f} "
                  f"med={float(np.median(fit_np)):.4f} rooms={row['rooms']:.2f} "
                  f"cells={row['cells']:.4f} dist={row['dist_m']:.1f} "
                  f"coll={row['collisions']:.1f} ({time.time() - t0:.0f}s)")
        thetas, sigma = reproduce(thetas, sigma, fit_np, order, scale, rng,
                                  n_elite=n_elite, n_immigrant=n_immigrant,
                                  tau=tau, sigma_max=0.15, sigma0=0.03,
                                  hidden=hidden)

    # final best: the standard table row for the champion genome (P=1)
    best_rows = bench.report(
        s, [bench.spec(best_theta, hidden, name="best")], games=games,
        ticks=ticks, level=level, train_seed=train_seed, device=device,
        noise_seed=seed, seed=seed_engine, cache=cache,
        w_dist=w_dist, w_coll=w_coll, quiet=True, worlds=worlds)
    best = best_rows[0]
    if not quiet:
        print("final best genome:")
        print(bench.format_table([best], building=s.external_worlds))
    assert np.isfinite(best["fitness"]), "smoke produced non-finite fitness"
    if not quiet:
        print(f"SMOKE OK: {gens} gens, pop {pop}, best fitness "
              f"{best_fit:.4f} -> {best['fitness']:.4f}")
    return {"rows": rows, "best": best, "best_fitness": best_fit,
            "best_theta": best_theta}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="legacy", choices=("legacy", "buildings",
                                                          "val", "holdout"))
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--train-seed", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    run_smoke(a.suite, pop=a.pop, gens=a.gens, ticks=a.ticks, seed=a.seed,
              device=a.device, hidden=a.hidden, games=a.games, level=a.level,
              train_seed=a.train_seed, cache=not a.no_cache)


if __name__ == "__main__":
    main()
