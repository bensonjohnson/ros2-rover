"""Bench: standard report, scripted baselines, and the acceptance gates.

SIM_PLATFORM stage 5 (docs/SIM_PLATFORM.md §5/§6). Three kinds of
"consumable" tool over a named suite:

  report()       -- the standard metric table for one or more genomes
                    (evo.evolve / evo.baselines metric names: fitness,
                    rooms, rooms_total, dist_m, cells, collisions, and for
                    building suites cross_rate/door_x/rev).
  run_baselines()-- the SCRIPTED baselines from evo/baselines.py
                    (random_walk, go_forward, wall_follower), same sensors,
                    same gate, same houses; "if evolution can't beat a
                    20-line wall-follower, say so in the report."
  gates          -- pure functions porting the acceptance tools:
                    * worst_trim_multi_draw_pick  <- evo/deploy_pick.py
                    * trim_envelope               <- evo/trim_eval.py
                    * pivot_cap                   <- engine spin_frac +
                                                    evo/orbit_audit.py

Selection discipline: the gates PICK genomes, so on a holdout-kind suite
they call `suites.require_selectable(..., final=False)` and RAISE unless
the caller passes final=True (or selects on the 'val' suite). See
rover_sim/suites.py.

Legacy source map (exact metrics ported):
  worst_trim_multi_draw_pick:
    TRIMS + Arena(P,G,seed=777_000,trim_rand=True) + per-trim reset draws
    `torch.manual_seed(3+1000*k)`; per-draw per-genome means; across draws
    coll = max, others = mean; worst_cross=min over trims, worst_coll=max
    over trims; rank lexsort(-fitn, -worst_cross) filtered by
    worst_coll <= coll_guard (deploy_pick.main).
  trim_envelope:
    TRIMS, single reset draw `torch.manual_seed(seed=3)`, per-trim rows
    (fit, rooms, cross, coll, rev); ratio = worst-rooms / nominal-rooms
    (trim_eval.main).
  pivot_cap:
    metric spin_frac = fraction of ticks with opposite-sign gated wheels
    of meaningful magnitude (engine._rollout_body / RolloutEngine metric
    "spin_frac"); orbit ratio arc/net = dist_m / net-displacement
    (orbit_audit.py). Pre-registered cap: spin <= 0.30 (docs/HANDOFF +
    docs/MEMORY_GENOME validation chain).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:                       # noqa: BLE001
    # torch built with CUDA support but no usable driver (the Hermes SBC):
    # ANY torch.cuda.* call aborts. On CPU rollouts the call is
    # semantically False — shim it (same as evo.equivalence_probe).
    torch.cuda.is_current_stream_capturing = lambda: False

from rover_sim import suites as _suites
from rover_sim.evaluate import evaluate
from rover_sim.policies.es_genome import PopulationNet, read_meta
from rover_sim.runner import Arena, NUM_BINS, OBS_DIM
from rover_sim.scoring import fitness, rev_frac

TRIMS = [(0.8, 1.0), (1.0, 1.0), (1.0, 0.8), (0.9, 1.0), (1.0, 0.9)]
PIVOT_CAP = 0.30        # pre-registered: spin <= 30% (docs/HANDOFF)
ORBIT_CAP = 8.0         # orbit_audit: arc/net > 8 == orbiting

__all__ = [
    "GenomeSpec", "load_genome", "spec",
    "report", "run_baselines", "BASELINES",
    "worst_trim_multi_draw_pick", "trim_envelope", "pivot_cap", "TRIMS",
    "PIVOT_CAP", "ORBIT_CAP", "TrimPickResult", "TrimEnvelopeResult",
    "PivotCapResult", "format_table", "main",
]


# ==================================================== genome specs + loading
@dataclass
class GenomeSpec:
    """A genome (or population) ready to evaluate: thetas [P, N] + modes."""
    name: str
    thetas: torch.Tensor
    hidden: int
    obs: str = "v1"
    action: str = "lr"
    mem: int = 0

    @property
    def P(self) -> int:
        return int(self.thetas.shape[0])


def load_genome(path, *, name=None) -> GenomeSpec:
    d = np.load(path)
    obs, action, mem = read_meta(d)
    th = torch.as_tensor(d["thetas"])
    if th.ndim == 1:
        th = th.unsqueeze(0)
    return GenomeSpec(name or os.path.basename(str(path)), th,
                      int(d["hidden"]), obs, action, mem)


def spec(thetas, hidden, *, name="genome", obs="v1", action="lr", mem=0):
    th = torch.as_tensor(thetas)
    if th.ndim == 1:
        th = th.unsqueeze(0)
    return GenomeSpec(name, th, int(hidden), obs, action, int(mem))


def _as_spec(genome, *, name=None, hidden=None, obs="v1", action="lr", mem=0):
    if isinstance(genome, GenomeSpec):
        return genome
    if isinstance(genome, (str, os.PathLike)):
        return load_genome(genome, name=name)
    return spec(genome, hidden, name=name or "genome", obs=obs, action=action,
                mem=mem)


# ==================================================== scripted baselines
# Ported VERBATIM from evo/baselines.py (same functions, same metric defs).
def _scan(obs: torch.Tensor) -> torch.Tensor:
    return obs[:, :NUM_BINS]


def random_walk(obs, prev):
    B = obs.shape[0]
    a = torch.rand(B, 2, device=obs.device) * 2 - 1
    return torch.where((prev.abs().sum(1, keepdim=True) < 0.05)
                       | (torch.rand(B, 1, device=obs.device) < 0.05),
                       a, prev)


def go_forward(obs, prev):
    """Trust the gate: full stick forward, let the safety monitor deal."""
    B = obs.shape[0]
    return torch.ones(B, 2, device=obs.device)


def wall_follower(obs, prev):
    """Reactive wall-hug: steer toward the more open hemisphere, modulated
    by front clearance. ~30 lines, no state beyond persistence."""
    s = _scan(obs)                       # [B, 72], bin 0 = forward
    B = s.shape[0]
    front = torch.cat([s[:, :8], s[:, -8:]], dim=1).min(dim=1).values
    left = s[:, 9:36].mean(dim=1)        # 45..180 deg
    right = s[:, 36:63].mean(dim=1)      # 180..315 deg
    open_diff = (left - right).clamp(-1, 1)
    fwd = (0.3 + 0.7 * front.clamp(0.0, 1.0)) * 0.8
    turn = 0.6 * open_diff + 0.25 * (s[:, :8].min(dim=1).values < 0.15) \
        * torch.where(prev[:, 0] >= prev[:, 1], 1.0, -1.0)
    l = (fwd - turn).clamp(-1, 1)
    r = (fwd + turn).clamp(-1, 1)
    wedged = (front < 0.06) & (torch.maximum(left, right) < 0.12)
    back = torch.stack([-0.6 * torch.sign(prev[:, 0] + prev[:, 1] + 0.01),
                        0.6 * torch.sign(prev[:, 0] + prev[:, 1] + 0.01)],
                       dim=1)
    return torch.where(wedged[:, None], back, torch.stack([l, r], dim=1))


BASELINES = {"random_walk": random_walk, "go_forward": go_forward,
             "wall_follower": wall_follower}


# ==================================================== metrics + table
def _door_cols(m) -> dict:
    if "door_crossings" not in m:
        return {}
    return {"cross_rate": float((m["rooms"] >= 2).float().mean()),
            "door_x": float(m["door_crossings"].mean()),
            "rev": float(rev_frac(m).mean())}


def _row(name, fit, m) -> dict:
    """Standard metric row — the evo.baselines / evo.evolve metric names."""
    return {"name": name,
            "fitness": float(fit.mean()),
            "rooms": float(m["rooms"].mean()),
            "rooms_total": float(m["rooms_total"].mean()),
            "dist_m": float(m["dist_m"].mean()),
            "cells": float((m["cells"] / m["cells_total"].clamp(min=1)
                            ).mean()),
            "collisions": float(m["collisions"].mean()),
            **_door_cols(m)}


def format_table(rows, building=False) -> str:
    hdr = (f"{'controller':<28} {'fitness':>8} {'rooms':>6} {'/tot':>5} "
           f"{'cells':>6} {'dist_m':>7} {'coll':>6}"
           + (f" {'cross':>6} {'door_x':>6} {'rev':>5}" if building else ""))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        extra = (f" {r['cross_rate']:>6.3f} {r['door_x']:>6.2f}"
                 f" {r['rev']:>5.2f}" if "cross_rate" in r else "")
        lines.append(f"{r['name']:<28} {r['fitness']:>8.4f} {r['rooms']:>6.2f} "
                     f"{r['rooms_total']:>5.1f} {r['cells']:>6.3f} "
                     f"{r['dist_m']:>7.1f} {r['collisions']:>6.1f}{extra}")
    return "\n".join(lines)


# ==================================================== arena construction
def _make_arena(suite, worlds, P, G, *, device, fp16, noise_seed, seed,
                gate_obs=False, trim_rand=False):
    """Build a RolloutEngine for `suite`.

    Building suites pass `worlds` (G houses, shared across the P paired
    individuals); the legacy suite passes a seed and lets the arena build
    its own make_house draw (`Arena(worlds=None, seed=...)`), matching
    `evo.evolve --world legacy`."""
    kw: dict = dict(seed=seed, device=device,
                    fp16=bool(fp16 and str(device).startswith("cuda")),
                    noise_seed=noise_seed, gate_obs=gate_obs, trim_rand=trim_rand)
    if suite.external_worlds:
        kw["worlds"] = list(worlds)[:G]
    return Arena(P, G, **kw)


def _suite_worlds(suite, n, *, level, train_seed, workers, cache):
    return suite.build(n=n, level=level, train_seed=train_seed,
                       workers=workers, cache=cache)


# ==================================================== report
def report(suite="legacy", genomes=(), *, population=None, games=8, ticks=3600,
           level=None, train_seed=None, device="cpu", noise_seed=12345,
           seed=None, fp16=False, w_dist=0.03, w_coll=0.25, w_cov=0.0,
           w_rev=0.0, w_net=0.0, w_spin=0.0, workers=None, cache=True,
           final=False, quiet=False, worlds=None) -> list:
    """Evaluate genomes over a named suite -> standard metric rows.

    `genomes` is a list of npz paths / GenomeSpecs / thetas tensors; a
    `population` npz adds every row as one spec (scored as P paired
    individuals). Deterministic under fixed `noise_seed` + `seed` (CPU is
    replay-stable; CUDA is not bit-stable for continuous metrics).
    Reporting is not selection, so holdout suites are allowed here; the
    PICKING gates enforce the discipline.
    """
    s = _suites.get_suite(suite)
    specs = [_as_spec(g) for g in genomes]
    if population:
        specs.append(load_genome(population, name=f"population({os.path.basename(population)})"))
    if not specs:
        raise ValueError("report: pass at least one genome or a population")
    seed = s.engine_seed(train_seed) if seed is None else int(seed)
    worlds = (list(worlds)[:games] if worlds is not None else
              (_suite_worlds(s, games, level=level, train_seed=train_seed,
                             workers=workers, cache=cache)
               if s.external_worlds else None))
    rows = []
    for sp in specs:
        P = sp.P
        eng = _make_arena(s, worlds, P, games, device=device, fp16=fp16,
                          noise_seed=noise_seed, seed=seed,
                          gate_obs=sp.obs == "v2")
        fit, m = evaluate(eng, sp.thetas.to(device), sp.hidden, ticks,
                          w_dist=w_dist, w_coll=w_coll, w_cov=w_cov,
                          w_rev=w_rev, w_net=w_net, w_spin=w_spin,
                          obs=sp.obs, action=sp.action, mem=sp.mem)
        rows.append(_row(sp.name, fit, m))
    if not quiet:
        if s.external_worlds:
            print(f"[bench] suite={s.name} kind={s.kind} games={games} "
                  f"ticks={ticks} | {_suites.summarize(worlds)}")
        else:
            print(f"[bench] suite={s.name} kind={s.kind} games={games} "
                  f"ticks={ticks} seed={seed}")
        print(format_table(rows, building=s.external_worlds))
    return rows


# ==================================================== baselines
def run_baselines(suite="legacy", *, genomes=(), games=8, ticks=3600, level=None,
                  train_seed=None, device="cpu", noise_seed=12345, seed=None,
                  fp16=False, workers=None, cache=True, quiet=False,
                  worlds=None) -> list:
    """Run the scripted baselines (and any extra genomes) over a suite.

    Same metric definitions as evo.baselines: blocked behind the identical
    safety gate on the identical houses."""
    s = _suites.get_suite(suite)
    seed = s.engine_seed(train_seed) if seed is None else int(seed)
    worlds = (list(worlds)[:games] if worlds is not None else
              (_suite_worlds(s, games, level=level, train_seed=train_seed,
                             workers=workers, cache=cache)
               if s.external_worlds else None))
    eng = _make_arena(s, worlds, 1, games, device=device, fp16=fp16,
                      noise_seed=noise_seed, seed=seed)
    rows = []
    for fn in BASELINES.values():
        torch.manual_seed(noise_seed)
        m = eng.run_games(lambda o, p: fn(o, p), ticks)
        rows.append(_row(fn.__name__, fitness(m), m))
    for g in genomes:
        sp = _as_spec(g)
        eng2 = _make_arena(s, worlds, sp.P, games, device=device, fp16=fp16,
                           noise_seed=noise_seed, seed=seed,
                           gate_obs=sp.obs == "v2")
        fit, m = evaluate(eng2, sp.thetas.to(device), sp.hidden, ticks,
                          obs=sp.obs, action=sp.action, mem=sp.mem)
        rows.append(_row(sp.name, fit, m))
    if not quiet:
        if s.external_worlds:
            print(f"[bench] baselines suite={s.name} kind={s.kind} "
                  f"games={games} ticks={ticks} | {_suites.summarize(worlds)}")
        else:
            print(f"[bench] baselines suite={s.name} kind={s.kind} "
                  f"games={games} ticks={ticks} seed={seed}")
        print(format_table(rows, building=s.external_worlds))
    return rows


# ==================================================== gate: worst-trim pick
@dataclass
class TrimPickResult:
    """Port of evo/deploy_pick.py's selection table (per-genome arrays)."""
    trims: list
    per_trim: list = field(default_factory=list)
    worst_cross: "np.ndarray | None" = None
    worst_coll: "np.ndarray | None" = None
    worst_rooms: "np.ndarray | None" = None
    nominal_fit: "np.ndarray | None" = None
    nominal_rev: "np.ndarray | None" = None
    ok: "np.ndarray | None" = None
    ranked: list = field(default_factory=list)
    coll_guard: float = 20.0
    min_worst_cross: float = 0.0

    def top(self):
        return self.ranked[0] if self.ranked else None

    def selected(self, k=1):
        return self.ranked[:k]


def _require_building_suite(s, what):
    if not s.external_worlds:
        raise ValueError(
            f"{what} needs a building suite (val/holdout); suite {s.name!r} "
            f"is the legacy single-world suite with no door metrics")


def worst_trim_multi_draw_pick(genome, *, suite="holdout", hidden=None, obs="v1",
                               action="lr", mem=0, games=32, ticks=5400,
                               level=None, train_seed=None, device="cpu",
                               cache=True, seed=None, trims=TRIMS,
                               coll_guard=20.0, reset_draws=3,
                               w_dist=0.005, w_coll=1.0, w_cov=0.3,
                               min_worst_cross=0.0, final=False, worlds=None,
                               workers=None) -> TrimPickResult:
    """Worst-trim multi-draw deploy pick (port of evo/deploy_pick.py).

    Scores each genome on the building suite across the trim envelope,
    reset_draws independent initial-pose draws per trim. Per-trim: coll =
    worst (max) across draws, fit/cross/rooms/rev = mean across draws.
    Across trims: worst_cross = min cross, worst_coll = max coll,
    worst_rooms = min rooms. Guard: worst_coll <= coll_guard. Rank =
    lexsort(-nominal_fit, -worst_cross) over genomes passing the guard.

    Selection discipline: holdout-kind suite raises unless final=True.
    """
    s = _suites.require_selectable(suite, final=final)
    _require_building_suite(s, "worst_trim_multi_draw_pick")
    seed = 777_000 if seed is None else int(seed)
    sp = _as_spec(genome, hidden=hidden, obs=obs, action=action, mem=mem)
    th = sp.thetas.to(device)
    P = sp.P
    if worlds is None:
        worlds = _suite_worlds(s, games, level=level, train_seed=train_seed,
                               workers=workers, cache=cache)
    worlds = list(worlds)[:games]
    assert len(worlds) == games, f"need {games} worlds, got {len(worlds)}"
    fused = str(device).startswith("cuda")
    arena = Arena(P, games, seed=seed, device=device, fp16=fused,
                  fused=fused, worlds=worlds, trim_rand=True,
                  gate_obs=sp.obs == "v2")
    net = PopulationNet(th, OBS_DIM, sp.hidden, obs=sp.obs, action=sp.action,
                        mem=sp.mem)
    net.bind(P, games)
    if sp.mem:
        arena._reset_hooks.append(net.reset_memory)

    per_trim = []
    for lt, rt in trims:
        arena.set_trims([lt] * games, [rt] * games)
        draws = []
        for k in range(reset_draws):
            state = {"h": torch.zeros(P, games, sp.hidden, device=device)}

            def step(o, prev, _s=state):
                a, _s["h"] = net.step(o.view(P, games, OBS_DIM), _s["h"])
                return a.view(P * games, 2)

            torch.manual_seed(3 + 1000 * k)
            m = arena.run_games(step, ticks)
            fit = fitness(m, w_dist=w_dist, w_coll=w_coll,
                          w_cov=w_cov).view(P, games).mean(1)
            rooms = m["rooms"].view(P, games)
            draws.append({
                "fit": fit.cpu().numpy(),
                "cross": (rooms >= 2).float().mean(1).cpu().numpy(),
                "rooms": rooms.mean(1).cpu().numpy(),
                "coll": m["collisions"].view(P, games).mean(1).cpu().numpy(),
                "rev": rev_frac(m).view(P, games).mean(1).cpu().numpy(),
            })
        per_trim.append({
            "trim": [lt, rt],
            "coll": np.maximum.reduce([d["coll"] for d in draws]),
            "fit": np.mean([d["fit"] for d in draws], axis=0),
            "cross": np.mean([d["cross"] for d in draws], axis=0),
            "rooms": np.mean([d["rooms"] for d in draws], axis=0),
            "rev": np.mean([d["rev"] for d in draws], axis=0),
        })

    cross = np.stack([t["cross"] for t in per_trim])       # [T, P]
    coll = np.stack([t["coll"] for t in per_trim])
    rooms = np.stack([t["rooms"] for t in per_trim])
    fitn = per_trim[0]["fit"]                              # nominal 0.8/1.0
    worst_cross = cross.min(0)
    worst_coll = coll.max(0)
    worst_rooms = rooms.min(0)
    ok = worst_coll <= coll_guard
    order = np.lexsort((-fitn, -worst_cross))              # cross, then fit
    ranked = [int(i) for i in order if ok[int(i)]]
    return TrimPickResult(trims=list(trims), per_trim=per_trim,
                          worst_cross=worst_cross, worst_coll=worst_coll,
                          worst_rooms=worst_rooms, nominal_fit=fitn,
                          nominal_rev=per_trim[0]["rev"], ok=ok, ranked=ranked,
                          coll_guard=float(coll_guard),
                          min_worst_cross=float(min_worst_cross))


# ==================================================== gate: trim envelope
@dataclass
class TrimEnvelopeResult:
    """Port of evo/trim_eval.py's table (single genome)."""
    trims: list = field(default_factory=list)
    rows: list = field(default_factory=list)   # dict(lt, rt, fit, rooms, cross, coll, rev)
    nominal_rooms: float = 0.0
    worst_rooms: float = 0.0
    ratio: float = 0.0
    ok: bool = False
    min_ratio: float = 0.5


def trim_envelope(genome, *, suite="holdout", hidden=None, obs="v1", action="lr",
                  mem=0, games=32, ticks=5400, level=None, train_seed=None,
                  device="cpu", cache=True, seed=None, reset_seed=3, trims=TRIMS,
                  min_ratio=0.5, final=False, worlds=None,
                  workers=None) -> TrimEnvelopeResult:
    """Trim robustness of one genome (port of evo/trim_eval.py).

    One reset draw per trim (`torch.manual_seed(reset_seed)`), per-trim rows
    (fit, rooms, cross, coll, rev). Nominal = first trim (0.8/1.0);
    worst = min over trims of mean rooms; ratio = worst/nominal. The legacy
    tool prints the ratio; `min_ratio` is the gate's pre-registered pass bar.

    Selection discipline: holdout-kind suite raises unless final=True.
    """
    s = _suites.require_selectable(suite, final=final)
    _require_building_suite(s, "trim_envelope")
    seed = 1 if seed is None else int(seed)
    sp = _as_spec(genome, hidden=hidden, obs=obs, action=action, mem=mem)
    if sp.P != 1:
        raise ValueError(f"trim_envelope evaluates ONE genome (P=1); got "
                         f"P={sp.P} — pass a single-row genome")
    if worlds is None:
        worlds = _suite_worlds(s, games, level=level, train_seed=train_seed,
                               workers=workers, cache=cache)
    worlds = list(worlds)[:games]
    fused = str(device).startswith("cuda")
    arena = Arena(1, games, seed=seed, device=device, fp16=fused, fused=fused,
                  worlds=worlds, trim_rand=True, gate_obs=sp.obs == "v2")
    net = PopulationNet(sp.thetas.to(device), OBS_DIM, sp.hidden, obs=sp.obs,
                        action=sp.action, mem=sp.mem)
    net.bind(1, games)
    rows = []
    for lt, rt in trims:
        arena.set_trims([lt] * games, [rt] * games)
        # legacy tool builds a fresh net per trim (and zeroes memory): hidden
        # clock / memory state must not leak across trims (mem>0 genomes)
        net.reset_memory()
        state = {"h": torch.zeros(1, games, sp.hidden, device=device)}

        def step(o, prev, _s=state):
            a, _s["h"] = net.step(o.view(1, games, OBS_DIM), _s["h"])
            return a.view(games, 2)

        torch.manual_seed(reset_seed)
        m = arena.run_games(step, ticks)
        rooms = m["rooms"]
        rows.append({"trim": [lt, rt], "fit": float(fitness(m).mean()),
                     "rooms": float(rooms.mean()),
                     "cross": float((rooms >= 2).float().mean()),
                     "coll": float(m["collisions"].mean()),
                     "rev": float(rev_frac(m).mean())})
    nominal = rows[0]["rooms"]
    worst = min(r["rooms"] for r in rows)
    ratio = worst / max(nominal, 1e-6)
    return TrimEnvelopeResult(trims=list(trims), rows=rows,
                              nominal_rooms=nominal, worst_rooms=worst,
                              ratio=ratio, ok=ratio >= min_ratio,
                              min_ratio=float(min_ratio))


# ==================================================== gate: pivot cap
@dataclass
class PivotCapResult:
    """Pivot/orbit gate over a genome (or population)."""
    spin_frac: "np.ndarray | None" = None    # per genome, mean over games
    arc_net: "np.ndarray | None" = None      # per genome, mean dist/net
    dist_m: "np.ndarray | None" = None
    net_m: "np.ndarray | None" = None
    rooms: "np.ndarray | None" = None
    ok: "np.ndarray | None" = None
    max_spin_frac: float = PIVOT_CAP
    max_arc_net: float = ORBIT_CAP

    def selected(self):
        return [int(i) for i in np.where(self.ok)[0]]


def pivot_cap(genome, *, suite="holdout", hidden=None, obs="v1", action="lr",
              mem=0, games=32, ticks=5400, level=None, train_seed=None,
              device="cpu", cache=True, seed=None, reset_seed=3,
              max_spin_frac=PIVOT_CAP, max_arc_net=ORBIT_CAP, final=False,
              worlds=None, workers=None) -> PivotCapResult:
    """Pivot/orbit acceptance gate.

    Metric (ported verbatim from the engine): spin_frac = fraction of ticks
    with a genuine pivot — `(gated_l*gated_r < -0.01) & (|gated|.min > 0.15)`
    (engine._rollout_body, exposed as the "spin_frac" metric). The orbit
    ratio arc/net = dist_m / net-displacement is orbit_audit.py's
    "genomes orbiting (arc/net > 8)" test. Pre-registered caps: spin <=
    0.30 (docs/HANDOFF + docs/MEMORY_GENOME: "spin <= 30%"), arc/net <= 8.

    Selection discipline: holdout-kind suite raises unless final=True.
    """
    s = _suites.require_selectable(suite, final=final)
    seed = 777_000 if seed is None else int(seed)
    sp = _as_spec(genome, hidden=hidden, obs=obs, action=action, mem=mem)
    P = sp.P
    if worlds is None:
        worlds = _suite_worlds(s, games, level=level, train_seed=train_seed,
                               workers=workers, cache=cache)
    worlds = list(worlds)[:games]
    if s.external_worlds:
        eng = Arena(P, games, seed=seed, device=device,
                    fp16=bool(str(device).startswith("cuda")), worlds=worlds,
                    noise_seed=reset_seed, gate_obs=sp.obs == "v2")
    else:
        eng = Arena(P, games, seed=seed, device=device,
                    fp16=bool(str(device).startswith("cuda")),
                    noise_seed=reset_seed, gate_obs=sp.obs == "v2")
    fit, m = evaluate(eng, sp.thetas.to(device), sp.hidden, ticks, obs=sp.obs,
                      action=sp.action, mem=sp.mem)
    spin = m["spin_frac"].view(P, games).mean(1).cpu().numpy()
    dist = m["dist_m"].view(P, games).mean(1).cpu().numpy()
    net = m["net_m"].view(P, games).mean(1).cpu().numpy()
    arc_net = m["dist_m"].div(m["net_m"].clamp(min=0.05)).view(
        P, games).mean(1).cpu().numpy()
    ok = (spin <= max_spin_frac) & (arc_net <= max_arc_net)
    return PivotCapResult(spin_frac=spin, arc_net=arc_net, dist_m=dist,
                          net_m=net,
                          rooms=m["rooms"].view(P, games).mean(1).cpu().numpy(),
                          ok=ok, max_spin_frac=float(max_spin_frac),
                          max_arc_net=float(max_arc_net))


# ==================================================== CLI
def _add_suite_args(ap):
    ap.add_argument("--suite", default="legacy", choices=_suites.suite_names())
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--train-seed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-cache", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("report", help="standard metric table for genomes")
    _add_suite_args(rp)
    rp.add_argument("--genome", action="append", default=[])
    rp.add_argument("--population", default="")

    bl = sub.add_parser("baselines", help="scripted baselines over a suite")
    _add_suite_args(bl)
    bl.add_argument("--genome", action="append", default=[])

    tp = sub.add_parser("trim-pick", help="worst-trim multi-draw pick")
    _add_suite_args(tp)
    tp.add_argument("--genome", required=True)
    tp.add_argument("--reset-draws", type=int, default=3)
    tp.add_argument("--coll-guard", type=float, default=20.0)
    tp.add_argument("--final", action="store_true")

    te = sub.add_parser("trim-envelope", help="trim robustness of one genome")
    _add_suite_args(te)
    te.add_argument("--genome", required=True)
    te.add_argument("--min-ratio", type=float, default=0.5)
    te.add_argument("--final", action="store_true")

    pc = sub.add_parser("pivot-cap", help="pivot/orbit acceptance gate")
    _add_suite_args(pc)
    pc.add_argument("--genome", required=True)
    pc.add_argument("--max-spin-frac", type=float, default=PIVOT_CAP)
    pc.add_argument("--max-arc-net", type=float, default=ORBIT_CAP)
    pc.add_argument("--final", action="store_true")

    a = ap.parse_args(argv)
    cache = not a.no_cache
    common = dict(games=a.games, ticks=a.ticks, level=a.level,
                  train_seed=a.train_seed, device=a.device, fp16=a.fp16,
                  seed=a.seed, cache=cache)

    if a.cmd == "report":
        report(a.suite, genomes=a.genome, population=a.population or None,
               **common)
    elif a.cmd == "baselines":
        run_baselines(a.suite, genomes=a.genome, **common)
    elif a.cmd == "trim-pick":
        r = worst_trim_multi_draw_pick(a.genome, suite=a.suite,
                                       reset_draws=a.reset_draws,
                                       coll_guard=a.coll_guard, final=a.final,
                                       **common)
        print(f"trim-pick: {a.genome} suite={a.suite} "
              f"guard(coll<={r.coll_guard:g}) {len(r.ranked)}/{len(r.ok)} pass")
        for rank, i in enumerate(r.ranked[:10]):
            print(f"  {rank:>3} idx={i:>4} worst_cross={r.worst_cross[i]:.3f} "
                  f"worst_rooms={r.worst_rooms[i]:.2f} "
                  f"worst_coll={r.worst_coll[i]:.1f} "
                  f"nom_fit={r.nominal_fit[i]:.3f} "
                  f"nom_rev={r.nominal_rev[i]:.2f}")
    elif a.cmd == "trim-envelope":
        r = trim_envelope(a.genome, suite=a.suite, min_ratio=a.min_ratio,
                          final=a.final, **common)
        print(f"trim-envelope: {a.genome} suite={a.suite}")
        for row in r.rows:
            print(f"  L{row['trim'][0]:.1f}/R{row['trim'][1]:.1f} "
                  f"fit={row['fit']:.3f} rooms={row['rooms']:.2f} "
                  f"cross={row['cross']:.3f} coll={row['coll']:.1f} "
                  f"rev={row['rev']:.2f}")
        print(f"  worst/nominal rooms = {r.ratio:.2f}  "
              f"(min_ratio {r.min_ratio:.2f}: {'PASS' if r.ok else 'FAIL'})")
    elif a.cmd == "pivot-cap":
        r = pivot_cap(a.genome, suite=a.suite, max_spin_frac=a.max_spin_frac,
                      max_arc_net=a.max_arc_net, final=a.final, **common)
        print(f"pivot-cap: {a.genome} suite={a.suite} "
              f"(spin<={r.max_spin_frac:.2f}, arc/net<={r.max_arc_net:.1f})")
        for i in range(len(r.ok)):
            print(f"  idx={i:>4} spin={r.spin_frac[i]:.3f} "
                  f"arc/net={r.arc_net[i]:.1f} rooms={r.rooms[i]:.2f} "
                  f"{'PASS' if r.ok[i] else 'FAIL'}")


if __name__ == "__main__":
    main()
