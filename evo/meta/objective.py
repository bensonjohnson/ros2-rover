"""The honest score: what "a good model" is allowed to mean here.

Three rules, each one paid for by a specific failure in runs 1-25:

1. **Score on `val`, never on `holdout`.** `rover_sim.suites` enforces it for
   picking (`require_selectable`); this module simply never asks for holdout
   except through `holdout_confirm()`, which is called once, at the end, with
   `final=True`. Selecting on holdout is how a search convinces itself of a
   number nobody can reproduce.

2. **Score the POPULATION, over >=2 seeds.** `best_genome` in a run dir is an
   argmax over a per-generation resampled draw; run-11's transfer guard failed
   3 of 4 runs precisely because the tracked champion was a collision
   brute-forcer that happened to win the draw it was scored on. Seed spread
   (~0.10) has repeatedly exceeded the effect under test (0.03-0.08), so a
   single-seed comparison is noise; the manager averages seeds and reports the
   spread.

3. **Constraint before score.** Collisions and reverse-fraction are gates, not
   terms: a config that buys rooms by crashing, or by driving backwards, is
   rejected outright (`feasible=False`, score -inf) rather than ranked. This is
   the mechanical version of the pre-registered guards the runs used by hand
   (`hob_champ_coll <= 20`, `elite coll <= 3`, forward `rev <= 0.15`).

The primary metric is rooms/game — the only one that survived every audit
(door-crossings are gameable by ping-ponging one doorway, and arc length is
gameable by orbiting in place: metric flaws #1 and #2).
"""

from __future__ import annotations

import json
import os

import numpy as np

#: Trailing fraction of generations whose elite metrics are averaged. The last
#: window is what will actually be deployed; early generations only measure
#: search speed.
GEN_WINDOW_FRAC = 0.25
MIN_WINDOW = 3

#: Pre-registered guards (see run 11's "collision guard PASS/FAIL" column).
COLL_CAP = 3.0            # elite train collisions per game
VAL_COLL_CAP = 20.0       # deep-eval collisions on the validation suite
REV_CAP = 0.15            # only enforced when the config asks for forward


def read_gens(out_dir: str) -> list:
    path = os.path.join(out_dir, "gens.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_gens(rows: list) -> dict:
    """Window means of the elite (population) metrics — never the champion."""
    if not rows:
        return {}
    n = max(MIN_WINDOW, int(len(rows) * GEN_WINDOW_FRAC))
    win = rows[-n:]

    def mean(key):
        vals = [float(r[key]) for r in win if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    last = rows[-1]
    return {
        "gens": len(rows),
        "window": n,
        "fit_best": mean("fit_best"),
        "fit_elite": mean("fit_elite"),
        "fit_med": mean("fit_med"),
        "elite_rooms": mean("elite_rooms"),
        "elite_cross_rate": mean("elite_cross_rate"),
        "elite_coll": mean("elite_coll"),
        "elite_cells": mean("elite_cells"),
        "elite_rev_frac": mean("elite_rev_frac"),
        "elite_dist": mean("elite_dist"),
        "final_sigma_med": last.get("sigma_med"),
        "level": last.get("level"),
    }


def val_deep_eval(genome_path: str, cfg: dict, *, device: str = "cpu",
                  games: int = 8, ticks: int = 1800, workers: int | None = 2,
                  seed: int | None = None) -> dict:
    """Re-evaluate a saved genome on the VAL suite.

    Fitness weights are deliberately the bench defaults (not the training
    config's), because the point is a comparable number across trials — the
    same convention run 12 used ("deep-eval keeps default fitness for
    comparability"). Rooms is read from the row regardless of weights.
    """
    from rover_sim import bench
    rows = bench.report("val", [genome_path], games=games, ticks=ticks,
                        device=device, workers=workers, seed=seed, quiet=True)
    if not rows:
        return {}
    r = rows[0]
    return {"val_rooms": float(r.get("rooms", 0.0)),
            "val_rooms_total": float(r.get("rooms_total", 0.0)),
            "val_cross_rate": float(r.get("cross_rate", 0.0)),
            "val_collisions": float(r.get("collisions", 0.0)),
            "val_fitness": float(r.get("fitness", 0.0)),
            "val_dist_m": float(r.get("dist_m", 0.0)),
            "val_rev": float(r.get("rev", 0.0))}


def score_trial(out_dir: str, cfg: dict, *, deep: bool = True,
                device: str = "cpu", val_games: int = 8,
                val_ticks: int = 1800) -> dict:
    """Parse a finished run dir into a ledger-ready scoring record."""
    rows = read_gens(out_dir)
    summary = summarize_gens(rows)
    if not summary:
        return {"status": "failed", "reasons": ["no gens.jsonl rows"],
                "score": float("-inf"), "feasible": False}

    reasons = []
    coll = summary.get("elite_coll")
    if coll is not None and coll > COLL_CAP:
        reasons.append(f"collision brute-forcer: elite_coll {coll:.2f} > "
                       f"{COLL_CAP}")
    if float(cfg.get("w_rev", 0.0)) > 0.0:
        rev = summary.get("elite_rev_frac")
        if rev is not None and rev > REV_CAP:
            reasons.append(f"not forward: elite_rev_frac {rev:.3f} > {REV_CAP}")

    metrics = dict(summary)
    primary = summary.get("elite_rooms") or 0.0
    secondary = summary.get("elite_cross_rate") or 0.0

    # A NaN/∞ metric must never enter the ranking: NaN compares false against
    # everything, so an unguarded NaN would sort unpredictably and poison the
    # mean in promote(). One sampled config really did produce an all-NaN
    # rollout (caught by the first dry run), and the NaN landed in dist/fitness
    # while rooms/cross stayed at 0.0 — so the guard has to sweep EVERY numeric
    # metric, not just the two that get scored.
    bad = [k for k, v in metrics.items()
           if isinstance(v, float) and not np.isfinite(v)]
    if bad:
        return {"status": "done", "metrics": metrics, "feasible": False,
                "reasons": [f"non-finite metrics ({', '.join(sorted(bad))}) — "
                            f"NaN rollout"],
                "score": float("-inf")}

    if deep:
        gpath = os.path.join(out_dir, "val_best_genome.npz")
        if os.path.exists(gpath):
            ev = val_deep_eval(gpath, cfg, device=device, games=val_games,
                               ticks=val_ticks)
            metrics.update(ev)
            if ev:
                primary = ev["val_rooms"]
                secondary = ev["val_cross_rate"]
                if ev["val_collisions"] > VAL_COLL_CAP:
                    reasons.append(f"val collisions {ev['val_collisions']:.1f} "
                                   f"> {VAL_COLL_CAP}")
        else:
            reasons.append("no val_best_genome.npz for deep eval")

    feasible = not reasons
    return {
        "status": "done",
        "metrics": metrics,
        "feasible": bool(feasible),
        "reasons": reasons,
        # rooms dominates; cross-rate is a far tie-break, never a substitute
        "score": float(primary + 1e-3 * secondary) if feasible
        else float("-inf"),
    }


def holdout_confirm(genome_path: str, *, device: str = "cpu", games: int = 32,
                    ticks: int = 3600, seed: int | None = None) -> dict:
    """The ONE holdout touch, for the final champion, with final=True."""
    from rover_sim import bench
    rows = bench.report("holdout", [genome_path], games=games, ticks=ticks,
                        device=device, seed=seed, final=True, quiet=True)
    if not rows:
        return {}
    r = rows[0]
    return {"holdout_rooms": float(r.get("rooms", 0.0)),
            "holdout_rooms_total": float(r.get("rooms_total", 0.0)),
            "holdout_cross_rate": float(r.get("cross_rate", 0.0)),
            "holdout_collisions": float(r.get("collisions", 0.0)),
            "holdout_rev": float(r.get("rev", 0.0))}
