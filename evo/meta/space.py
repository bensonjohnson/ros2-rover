"""The search space: which knobs may be tuned, over what, and which may not.

A knob is declared here ONCE, with (a) the CLI flag it maps to, (b) the domain,
(c) whether it is tunable at all. The last part matters: several parameters in
this project are NOT hyperparameters and must never enter a search, because
tuning them moves the goalposts instead of the result:

* `world` / `level` / the suite split — changing them changes what is being
  measured (and `holdout` is report-only by construction: see
  `rover_sim.suites.require_selectable`).
* `train-rotations` past 4 / `merge`-style tricks that buy fitness by adding
  worlds — that is more compute, not a better model; it belongs in a scale
  decision, not in a hyperparameter search.
* Anything that would select on holdout.

`tune=False` knobs are still recorded in every trial's config (so the ledger is
self-contained and a champion can be rebuilt exactly) — they are simply never
sampled.

Domains are deliberately conservative and anchored on the values this project
actually ran, so the search explores *near the evidence* instead of wandering:
hidden 64/128 (run 6a falsified 128 as a capacity fix, so both stay in range),
sigma0 0.03/0.05 (the measured cliff between 0.1 and 0.25 is above this),
w_coll 0.25/1.0 (the run-11 collision-brute-forcer fix), and lidar-hz 10 vs
off (the multi-rate clock, fidelity item 1 — a causal change, not a tweak).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPACE_VERSION = "2026-10-06.1"


@dataclass(frozen=True)
class Knob:
    """One tunable parameter: its CLI flag, its domain, and whether to sample."""

    name: str
    flag: str
    kind: str                       # choice | int | float | loguniform | flag
    values: tuple = ()
    lo: float = 0.0
    hi: float = 0.0
    default: object = None
    tune: bool = True
    help: str = ""

    def sample(self, rng: np.random.Generator):
        if self.kind == "choice":
            return self.values[int(rng.integers(len(self.values)))]
        if self.kind == "int":
            return int(rng.integers(self.lo, self.hi + 1))
        if self.kind == "flag":                 # store_true style
            return bool(rng.integers(2))
        if self.kind == "float":
            return float(rng.uniform(self.lo, self.hi))
        if self.kind == "loguniform":
            return float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))
        raise ValueError(f"unknown knob kind {self.kind!r}")

    def to_cli(self, value) -> list:
        """['--flag', 'v'] — or just ['--flag'] for store_true knobs set true."""
        if self.kind == "flag":
            return [self.flag] if value else []
        if value is None:
            return []
        return [self.flag, str(value)]


# --------------------------------------------------------------- the knobs
KNOBS = (
    # --- architecture -------------------------------------------------
    Knob("hidden", "--hidden", "choice", values=(64, 128), default=128,
         help="policy width. 512 is deliberately NOT offered: 323k genes is "
              "ES-unsearchable at pop 128 and runs 1-7a were void anyway"),
    Knob("obs", "--obs", "choice", values=("v1", "v2"), default="v1",
         help="v2 = gate latch flags replace the pure-noise gyro channels"),
    Knob("action", "--action", "choice", values=("lr", "vw"), default="lr",
         help="action parameterization (left/right wheel vs v/omega)"),
    Knob("mem", "--mem", "choice", values=(0, 8), default=0,
         help="explicit memory slots (run 25 failed with the write path "
              "closed by construction — see docs/MEMORY_GENOME.md)"),
    # --- ES strategy --------------------------------------------------
    Knob("algo", "--algo", "choice", values=("ga", "oes"), default="ga",
         help="ga = the fixed GA; oes = OpenAI-ES mirrored/rank/Adam"),
    Knob("sigma0", "--sigma0", "choice", values=(0.03, 0.05, 0.08),
         default=0.05,
         help="mutation scale. Measured: 0.1 -> 0.25 is a CLIFF for these "
              "bang-bang controllers (open-loop action change 0.032 -> 0.27)"),
    Knob("sigma_max", "--sigma-max", "choice", values=(0.15, 0.25),
         default=0.15),
    # --- fitness shaping ----------------------------------------------
    Knob("w_coll", "--w-coll", "choice", values=(0.25, 0.5, 1.0), default=1.0,
         help="collision penalty. 1.0 was the fix for the run-11 collision "
              "brute-forcers reappearing on holdouts"),
    Knob("w_cov", "--w-cov", "choice", values=(0.0, 0.3, 0.6), default=0.3,
         help="cell-coverage novelty. Run 7a: raising the PRICE did not move "
              "the plateau (reachability, not reward)"),
    Knob("w_rev", "--w-rev", "choice", values=(0.0, 0.15), default=0.15,
         help="reverse penalty. 0.15 buys forward honesty (~0.51 cross "
              "ceiling); reverse genomes reach 0.87 but do not field-transfer"),
    Knob("w_net", "--w-net", "choice", values=(0.0, 0.15), default=0.0,
         help="straight-line range term (kills the orbit exploit)"),
    Knob("w_spin", "--w-spin", "choice", values=(0.0, 0.3), default=0.3,
         help="pivot-tick penalty. MANDATORY post-hardware: the loaded track "
              "stalls on pivots (live runs 5-8)"),
    Knob("w_dist", "--w-dist", "choice", values=(0.0, 0.005), default=0.005),
    Knob("trim_rand", "--trim-rand", "choice", values=(0.0, 0.75), default=0.75,
         help="per-house track-trim randomization. Champions are extremely "
              "trim-sensitive (mirror-test artifact), so this is a sim->real "
              "insurance knob"),
    # --- the world / the machine --------------------------------------
    Knob("lidar_hz", "--lidar-hz", "choice", values=(None, 10.0), default=10.0,
         help="multi-rate lidar (fidelity item 1). 10 Hz is the rover truth: "
              "the brain acts on a scan up to one period old, and 1 in 3 "
              "decisions re-uses content it already acted on"),
    Knob("driver_hz", "--driver-hz", "choice", values=(None, 25.0),
         default=None,
         help="motor driver command acceptance rate (25 Hz today = no drops "
              "at a 15 Hz brain; the knob exists to make the mismatch "
              "expressible). None and 25.0 are near-equivalent: a no-op check"),
    Knob("gate", "--gate", "choice", values=("legacy", "symmetric"),
         default="legacy",
         help="symmetric was falsified in sim (run 16: the gate asymmetry does "
              "NOT cause the reverse habit) — kept only as a control"),
    # --- budget (tunable within a stage, but visible) ------------------
    Knob("pop", "--pop", "choice", values=(64, 128), default=128),
    Knob("games", "--games", "choice", values=(16, 32), default=16),
    Knob("ticks", "--ticks", "choice", values=(2700, 5400), default=5400,
         help="2700 is a safe early-gen proxy (rank corr 0.93 fit / 0.98 "
              "cells at half length; 1800 was marginal)"),
    Knob("gens", "--gens", "choice", values=(40,), default=40, tune=False,
         help="STAGE-controlled budget — run.py overwrites this per stage "
              "(20 / 40 / 120). Not sampled: it is the halving schedule, and "
              "leaving it out of the CLI silently ran every stage at evolve's "
              "own default (found by the first real dry run, which sat at "
              "gen 4 of 40)"),
    Knob("train_rotations", "--train-rotations", "choice", values=(4,), 
         default=4, tune=False,
         help="fixed: env-column scaling, not a model hyperparameter"),
    Knob("every_cover", "--every-cover", "choice", values=(24,), default=24,
         tune=False),
    Knob("no_cells", "--no-cells", "flag", default=False, tune=False,
         help="coverage accumulator off (fault-bisect mode)"),
    Knob("seed_from", "--seed-from", "choice", values=(None,), default=None,
         tune=False, help="warm start is a stage decision, not a knob"),
)

BY_NAME = {k.name: k for k in KNOBS}
TUNABLE = tuple(k for k in KNOBS if k.tune)

#: Recorded in every trial but never sampled. Changing any of these invalidates
#: comparability across the ledger, so they are pinned here on purpose.
FIXED = {
    "world": "buildings",          # the only world source with rooms/doors
    "level": 3,                    # full-mix buildings
    "holdout_seed": 777_000,       # never selected on; see suites.py
    "valid_seed": 779_000,
    "select_suite": "val",         # promotion metric comes from here
    "space_version": SPACE_VERSION,
}


def default_config() -> dict:
    return {k.name: k.default for k in KNOBS}


def sample_configs(n: int, *, seed: int = 0,
                   overrides: dict | None = None) -> list[dict]:
    """`n` random configurations over the tunable knobs, with `overrides`
    (a partial config, e.g. from --space) forced in every sample."""
    rng = np.random.default_rng(seed)
    overrides = dict(overrides or {})
    out = []
    for _ in range(n):
        cfg = default_config()
        for k in TUNABLE:
            cfg[k.name] = k.sample(rng)
        # A sampled config must not be self-contradictory: the reverse penalty
        # is a forward-honesty lever, so a reverse-allowed search (w_rev 0)
        # must not also carry a reverse-fraction guard later. w_spin without
        # wheels that can pivot is harmless, so only w_rev is coupled.
        cfg["w_rev"] = float(cfg["w_rev"])
        cfg["w_spin"] = float(cfg["w_spin"])
        cfg.update(overrides)
        # -----------------------------------------------------------
        # KNOWN, CONTAINED FAILURE MODE (not a ban — see below): `algo=oes`
        # with a memory genome (mem>0) produced a NaN search step and an
        # all-NaN rollout twice — once in the meta manager, once in a 4-way
        # parallel batch, both in the full sampled corner (obs v2 AND action vw
        # AND mem 8 AND oes AND sigma0 0.08). It has NOT reproduced since: six
        # consecutive attempts at that exact corner, plus one-factor-at-a-time
        # tests (each of v2, vw, sigma0 0.08, ticks 40 clean at h128/mem8/oes),
        # are all finite. Root cause unknown, so the search is NOT fenced off
        # from oes+mem (that would ban an unexplored corner of the memory
        # hypothesis) — it is protected instead by the 1-generation preflight
        # probe and by objective.score_trial's non-finite guard, which catch a
        # NaN for ~6 s or one stage and can never let it rank.
        out.append(cfg)
    return out


def cli_args(cfg: dict) -> list:
    """Config -> evolve CLI flags, in knob declaration order."""
    args: list = []
    for k in KNOBS:
        args += k.to_cli(cfg.get(k.name, k.default))
    return args
