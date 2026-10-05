"""Bench tests (docs/SIM_PLATFORM.md §6 stage 5).

    python3 -m rover_sim.tests.test_bench [--device cpu]

B1  worst-trim multi-draw pick EQUIVALENCE: the new pure function reproduces
    the legacy evo/deploy_pick.py computation bit-for-bit on identical fixed
    inputs (legacy algorithm transcribed with the evo.arena/evo.policy
    imports, which are the same objects the platform wraps).
B2  trim envelope EQUIVALENCE: same for evo/trim_eval.py (rows + ratio).
B3  pivot cap: the metric definitions are ported verbatim — spin_frac and
    arc/net of the new gate equal a transcription of the engine spin
    accumulator + evo/orbit_audit.py on identical inputs.
B4  report table: standard metric keys present, deterministic under fixed
    seeds (two runs identical), baselines runnable.
B5  smoke: tiny run completes with finite fitness and is reproducible.
B6  selection discipline is enforced by the gates (holdout raises).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:                       # noqa: BLE001
    torch.cuda.is_current_stream_capturing = lambda: False

# legacy imports (the tools the gates port) — evo.arena / evo.policy shims
from evo.arena import OBS_DIM, Arena as LegacyArena, fitness, rev_frac
from evo.policy import PopulationNet as LegacyNet
from rover_sim import bench, suites
from rover_sim.policies.es_genome import sample_population
from rover_sim.smoke import run_smoke

G = 4
TICKS = 120
H = 16
SEED = 7
WORLDS = None       # filled in main(); one build reused by both sides


def _thetas(P):
    return torch.as_tensor(sample_population(
        P, OBS_DIM, H, np.random.default_rng(SEED)), device="cpu")


# ------------------------------------------------------------------ legacy
def _legacy_pick(thetas, worlds, games, ticks, reset_draws):
    """Literal transcription of evo/deploy_pick.main()'s computation."""
    P = thetas.shape[0]
    arena = LegacyArena(P, games, seed=777_000, device="cpu", fp16=False,
                        fused=False, worlds=worlds, trim_rand=True,
                        gate_obs=False)
    net = LegacyNet(thetas, OBS_DIM, H)
    net.bind(P, games)
    per_trim = []
    for lt, rt in bench.TRIMS:
        arena.set_trims([lt] * games, [rt] * games)
        draws = []
        for k in range(reset_draws):
            state = {"h": torch.zeros(P, games, H)}

            def step(o, prev, _s=state):
                a, _s["h"] = net.step(o.view(P, games, OBS_DIM), _s["h"])
                return a.view(P * games, 2)

            torch.manual_seed(3 + 1000 * k)
            m = arena.run_games(step, ticks)
            fit = fitness(m, w_dist=0.005, w_coll=1.0,
                          w_cov=0.3).view(P, games).mean(1)
            rooms = m["rooms"].view(P, games)
            draws.append({
                "fit": fit.cpu().numpy(),
                "cross": (rooms >= 2).float().mean(1).cpu().numpy(),
                "rooms": rooms.mean(1).cpu().numpy(),
                "coll": m["collisions"].view(P, games).mean(1).cpu().numpy(),
                "rev": rev_frac(m).view(P, games).mean(1).cpu().numpy()})
        per_trim.append({
            "coll": np.maximum.reduce([d["coll"] for d in draws]),
            "fit": np.mean([d["fit"] for d in draws], axis=0),
            "cross": np.mean([d["cross"] for d in draws], axis=0),
            "rooms": np.mean([d["rooms"] for d in draws], axis=0)})
    cross = np.stack([t["cross"] for t in per_trim])
    coll = np.stack([t["coll"] for t in per_trim])
    rooms = np.stack([t["rooms"] for t in per_trim])
    fitn = per_trim[0]["fit"]
    worst_cross, worst_coll, worst_rooms = cross.min(0), coll.max(0), rooms.min(0)
    ok = worst_coll <= 20.0
    order = np.lexsort((-fitn, -worst_cross))
    ranked = [int(i) for i in order if ok[int(i)]]
    return {"worst_cross": worst_cross, "worst_coll": worst_coll,
            "worst_rooms": worst_rooms, "nominal_fit": fitn, "ranked": ranked}


def _legacy_envelope(thetas, worlds, games, ticks):
    """Literal transcription of evo/trim_eval.main()'s computation."""
    arena = LegacyArena(1, games, seed=1, device="cpu", fp16=False, fused=False,
                        worlds=worlds, trim_rand=True, gate_obs=False)
    net = LegacyNet(thetas, OBS_DIM, H)
    net.bind(1, games)
    net.reset_memory()
    rows = []
    for lt, rt in bench.TRIMS:
        arena.set_trims([lt] * games, [rt] * games)
        state = {"h": torch.zeros(1, games, H)}

        def step(o, prev, _s=state):
            a, _s["h"] = net.step(o.view(1, games, OBS_DIM), _s["h"])
            return a.view(games, 2)

        torch.manual_seed(3)
        m = arena.run_games(step, ticks)
        rows.append((float(fitness(m).mean()), float(m["rooms"].mean()),
                     float((m["rooms"] >= 2).float().mean()),
                     float(m["collisions"].mean()),
                     float(rev_frac(m).mean())))
    return rows


def _legacy_pivot(thetas, worlds, games, ticks):
    """Transcription of the engine spin accumulator + evo/orbit_audit.py."""
    P = thetas.shape[0]
    arena = LegacyArena(P, games, seed=777_000, device="cpu", fp16=False,
                        worlds=worlds, noise_seed=3, gate_obs=False)
    net = LegacyNet(thetas, OBS_DIM, H)
    net.bind(P, games)
    net.reset_memory()
    state = {"h": torch.zeros(P, games, H)}

    def step(o, prev):
        a, state["h"] = net.step(o.view(P, games, OBS_DIM), state["h"])
        return a.view(P * games, 2)

    m = arena.run_games(step, ticks)
    return {"spin": m["spin_frac"].view(P, games).mean(1).cpu().numpy(),
            "arc_net": m["dist_m"].div(m["net_m"].clamp(min=0.05)).view(
                P, games).mean(1).cpu().numpy(),
            "dist": m["dist_m"].view(P, games).mean(1).cpu().numpy()}


# -------------------------------------------------------------------- B1
def b1_trim_pick():
    P, draws = 2, 2
    th = _thetas(P)
    got = bench.worst_trim_multi_draw_pick(
        th, hidden=H, suite="holdout", worlds=WORLDS, games=G, ticks=TICKS,
        reset_draws=draws, final=True)
    ref = _legacy_pick(th, WORLDS, G, TICKS, draws)
    for k in ("worst_cross", "worst_coll", "worst_rooms", "nominal_fit"):
        assert np.array_equal(getattr(got, k), ref[k]), \
            f"B1 {k}: {getattr(got, k)} != {ref[k]}"
    assert got.ranked == ref["ranked"], (got.ranked, ref["ranked"])
    print(f"B1 ok: worst-trim pick == evo/deploy_pick transcription "
          f"(P={P} T={len(bench.TRIMS)} draws={draws} G={G}; "
          f"worst_cross={got.worst_cross.tolist()} "
          f"worst_coll={got.worst_coll.tolist()} ranked={got.ranked})")


# -------------------------------------------------------------------- B2
def b2_trim_envelope():
    th = _thetas(1)
    got = bench.trim_envelope(th, hidden=H, suite="holdout", worlds=WORLDS,
                              games=G, ticks=TICKS, final=True)
    ref = _legacy_envelope(th, WORLDS, G, TICKS)
    for row, r in zip(got.rows, ref):
        assert np.allclose([row["fit"], row["rooms"], row["cross"],
                            row["coll"], row["rev"]], r, rtol=0, atol=0), \
            (row, r)
    nom = ref[0][1]
    ratio = min(r[1] for r in ref) / max(nom, 1e-6)
    assert abs(got.ratio - ratio) < 1e-12, (got.ratio, ratio)
    print(f"B2 ok: trim envelope == evo/trim_eval transcription "
          f"(T={len(got.rows)} rows bit-equal; ratio={got.ratio:.4f})")


# -------------------------------------------------------------------- B3
def b3_pivot_cap():
    th = _thetas(3)
    got = bench.pivot_cap(th, hidden=H, suite="holdout", worlds=WORLDS,
                          games=G, ticks=TICKS, final=True)
    ref = _legacy_pivot(th, WORLDS, G, TICKS)
    assert np.array_equal(got.spin_frac, ref["spin"]), \
        (got.spin_frac, ref["spin"])
    assert np.array_equal(got.arc_net, ref["arc_net"]), \
        (got.arc_net, ref["arc_net"])
    assert np.array_equal(got.dist_m, ref["dist"]), (got.dist_m, ref["dist"])
    assert (got.ok == ((got.spin_frac <= bench.PIVOT_CAP)
                       & (got.arc_net <= bench.ORBIT_CAP))).all()
    print(f"B3 ok: pivot_cap spin_frac/arc-net == engine+orbit_audit "
          f"transcription (P=3; spin={np.round(got.spin_frac, 3).tolist()} "
          f"arc/net={np.round(got.arc_net, 2).tolist()} "
          f"ok={got.ok.tolist()})")


# -------------------------------------------------------------------- B4
def b4_report_baselines():
    r1 = bench.report("legacy", genomes=["deploy/genomes/champ21f_idx103.npz"],
                      games=G, ticks=TICKS, quiet=True)
    r2 = bench.report("legacy", genomes=["deploy/genomes/champ21f_idx103.npz"],
                      games=G, ticks=TICKS, quiet=True)
    assert r1 == r2, "report not deterministic under fixed seeds"
    keys = {"name", "fitness", "rooms", "rooms_total", "dist_m", "cells",
            "collisions"}
    assert keys <= set(r1[0]), set(r1[0])
    assert np.isfinite(r1[0]["fitness"])
    # building suite carries the door columns
    rb = bench.report("holdout", genomes=["deploy/genomes/champ21f_idx103.npz"],
                      games=2, ticks=TICKS, level=3, cache=False, quiet=True,
                      worlds=WORLDS[:2])
    assert {"cross_rate", "door_x", "rev"} <= set(rb[0]), set(rb[0])
    base = bench.run_baselines("legacy", games=G, ticks=TICKS, quiet=True)
    assert [b["name"] for b in base] == ["random_walk", "go_forward",
                                         "wall_follower"], base
    assert all(np.isfinite(b["fitness"]) for b in base)
    print(f"B4 ok: report deterministic + keys {sorted(keys)}; building "
          f"adds {sorted({'cross_rate','door_x','rev'})}; baselines "
          f"{[b['name'] for b in base]} runnable")


# -------------------------------------------------------------------- B5
def b5_smoke():
    a = run_smoke("legacy", pop=4, gens=1, ticks=100, seed=4242, quiet=True)
    b = run_smoke("legacy", pop=4, gens=1, ticks=100, seed=4242, quiet=True)
    assert np.isfinite(a["best"]["fitness"]) and \
        np.isfinite(a["best_fitness"])
    assert a["best_fitness"] == b["best_fitness"], \
        (a["best_fitness"], b["best_fitness"])
    assert a["rows"] == b["rows"], "smoke rows not reproducible"
    assert len(a["rows"]) == 1
    print(f"B5 ok: smoke completes finite + reproducible "
          f"(best_fitness={a['best_fitness']:.6f})")


# -------------------------------------------------------------------- B6
def b6_discipline():
    th = _thetas(1)
    try:
        bench.trim_envelope(th, hidden=H, suite="holdout", worlds=WORLDS,
                            games=G, ticks=TICKS)
        raise AssertionError("B6: holdout envelope did not raise")
    except PermissionError:
        pass
    try:
        bench.pivot_cap(th, hidden=H, suite="holdout", worlds=WORLDS, games=G,
                        ticks=TICKS)
        raise AssertionError("B6: holdout pivot did not raise")
    except PermissionError:
        pass
    print("B6 ok: gates refuse holdout selection without final=True")


def main():
    global WORLDS
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.parse_args()

    # one cache-free build shared by both sides of every equivalence check
    WORLDS = suites.get_suite("holdout").build(n=G, level=3, workers=1,
                                               cache=False)
    b1_trim_pick()
    b2_trim_envelope()
    b3_pivot_cap()
    b4_report_baselines()
    b5_smoke()
    b6_discipline()
    print("ALL BENCH CHECKS PASSED")


if __name__ == "__main__":
    main()
