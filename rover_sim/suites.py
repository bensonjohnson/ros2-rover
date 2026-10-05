"""Named scenario suites + the val/holdout selection discipline.

SIM_PLATFORM stage 5 (docs/SIM_PLATFORM.md §5/§6): the suite registry makes
world sources comparable across model families and strategies. It is a thin
WRAPPER over the existing pooling math — the four suites forward to
`evo.worlds` primitives (`build_pool`, `level_seed`, `VAL_SEED`,
`HOLDOUT_SEED`) rather than duplicating them, exactly as `evo/evolve.py`
consumes them today:

  suite      kind      worlds
  legacy     train     make_house draws (RUNS 1-9: Arena(worlds=None,
                       seed=train_seed) builds them internally with
                       seed + 1_000_003*k)
  buildings  train     build_pool(level, n, level_seed(train_seed, level))
  val        val       build_pool(level, n, VAL_SEED)     -- champion set
  holdout    holdout   build_pool(level, n, HOLDOUT_SEED) -- never selected

DISCIPLINE (port of the doc's "VAL_SEED ... champion selection set; never
holdout"): a suite carries its purpose `kind`. `require_selectable(suite,
final=False)` RAISES for a holdout-kind suite unless the caller passes
`final=True` — the explicit "this is the final, committed pick" escape
hatch. Picking/ranking genomes on a holdout suite without it is a bug, and
the registry refuses it.

Legacy worlds are plain `rover_sim.core.world.World` (no `.family`), so
`summarize()` branches: building pools forward to `evo.worlds.summarize`;
legacy pools get a partition-count summary. `fingerprint()` is a stable
structural digest for equality tests (no pickling of the shared cache).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Forward the pooling primitives (do NOT reimplement the math).
from evo.worlds import (CACHE_DIR, HOLDOUT_SEED, POOL_VERSION, VAL_SEED,
                        build_pool, level_seed, summarize as _summarize_pool)
from rover_sim.core.world import make_house

SUITE_KINDS = ("train", "val", "holdout")
TRAIN_SEED = 40_000            # evo.evolve --train-seed default
LEVEL = 3                      # building curriculum L3 (full mix)
DEFAULT_N = 16                 # worlds per suite when the caller does not ask

# Re-exported so callers get the seeds off the registry, not from evo.
__all__ = [
    "Suite", "SUITES", "SUITE_KINDS", "TRAIN_SEED", "LEVEL", "DEFAULT_N",
    "HOLDOUT_SEED", "VAL_SEED", "CACHE_DIR", "POOL_VERSION",
    "get_suite", "suite_names", "build", "summarize", "fingerprint",
    "require_selectable", "level_seed",
]


@dataclass(frozen=True)
class Suite:
    """A named world source + its purpose kind.

    external_worlds=False (legacy) means the arena generates its own houses
    from a seed (`Arena(worlds=None, seed=...)`) — passing make_house Worlds
    into `Arena(worlds=...)` would wrongly enter building mode, so the
    legacy suite exposes a seed instead of a world list to the engine.
    """

    name: str
    kind: str
    external_worlds: bool
    default_level: int | None = None
    description: str = ""

    # ------------------------------------------------------------------ build
    def build(self, *, n: int | None = None, level: int | None = None,
              train_seed: int | None = None, workers: int | None = None,
              cache: bool = True) -> list:
        """Build (or load from the shared cache) this suite's `n` worlds.

        `cache=False` never touches evo_runs/world_cache — use it in tests
        and equality checks. Legacy worlds are make_house draws matching
        `Arena(worlds=None, seed=train_seed)`: rng seed train_seed +
        1_000_003*k, k = 0..n-1.
        """
        n = DEFAULT_N if n is None else int(n)
        ts = TRAIN_SEED if train_seed is None else int(train_seed)
        if self.name == "legacy":
            return [make_house(np.random.default_rng(ts + 1_000_003 * k))
                    for k in range(n)]
        lv = self.default_level if level is None else int(level)
        if lv is None:
            raise ValueError(f"suite {self.name!r} needs a level")
        if self.name == "buildings":
            return build_pool(lv, n, level_seed(ts, lv), workers=workers,
                              cache=cache)
        if self.name == "val":
            return build_pool(lv, n, VAL_SEED, workers=workers, cache=cache)
        if self.name == "holdout":
            return build_pool(lv, n, HOLDOUT_SEED, workers=workers,
                              cache=cache)
        raise KeyError(f"suite {self.name!r} has no builder")  # pragma: no cover

    def engine_seed(self, train_seed: int | None = None) -> int:
        """Seed for `Arena(worlds=None, seed=...)` — the legacy draw."""
        return TRAIN_SEED if train_seed is None else int(train_seed)

    # -------------------------------------------------------------- discipline
    def selectable(self, final: bool = False) -> bool:
        return self.kind != "holdout" or bool(final)

    def require_selectable(self, final: bool = False) -> "Suite":
        return require_selectable(self, final=final)

    # ---------------------------------------------------------------- summary
    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"<suite {self.name} kind={self.kind}>"


SUITES: dict[str, Suite] = {
    "legacy": Suite("legacy", "train", external_worlds=False,
                    description="classic runs 1-9 make_house single-world "
                                "suite (Arena(worlds=None))"),
    "buildings": Suite("buildings", "train", external_worlds=True,
                       default_level=LEVEL,
                       description="L{level} building pool for training "
                                   "(build_pool + level_seed)"),
    "val": Suite("val", "val", external_worlds=True, default_level=LEVEL,
                 description="L{level} VAL_SEED pool: champion selection "
                             "set, never holdout"),
    "holdout": Suite("holdout", "holdout", external_worlds=True,
                     default_level=LEVEL,
                     description="L{level} HOLDOUT_SEED pool: never used "
                                 "for selection"),
}


def get_suite(name) -> Suite:
    """Resolve a suite by name (a Suite passes through unchanged)."""
    if isinstance(name, Suite):
        return name
    try:
        return SUITES[str(name)]
    except KeyError:
        raise KeyError(f"unknown suite {name!r}; known: "
                       f"{sorted(SUITES)}") from None


def suite_names() -> tuple:
    return tuple(SUITES)


def build(name, *, n: int | None = None, level: int | None = None,
          train_seed: int | None = None, workers: int | None = None,
          cache: bool = True) -> list:
    """Module-level convenience: build a named suite's worlds."""
    return get_suite(name).build(n=n, level=level, train_seed=train_seed,
                                 workers=workers, cache=cache)


def summarize(worlds) -> str:
    """Port/forward of `evo.worlds.summarize` (+ a legacy branch).

    Building pools (`.family`/`.n_rooms`/`.doors`) forward verbatim to the
    evo implementation; legacy `World`s (partitions only) get a compact
    wall-count summary so bench/smoke can print something honest.
    """
    worlds = list(worlds)
    if not worlds:
        return "empty"
    w0 = worlds[0]
    if hasattr(w0, "family") and hasattr(w0, "n_rooms"):
        return _summarize_pool(worlds)
    walls = np.array([len(getattr(w, "partitions", ())) for w in worlds])
    return (f"legacy {len(worlds)} houses walls {walls.mean():.1f} "
            f"[{walls.min()}-{walls.max()}]")


def fingerprint(worlds) -> str:
    """Stable structural digest of a world list.

    Used by tests to assert suite pools equal `evo.worlds.build_pool`
    output for the same keys without pickling or touching the cache.
    Includes the fields the arena actually consumes (segments, bounds,
    start pose, partitions; and for buildings: room rects, doors, rooms).
    """
    rows = []
    for w in worlds:
        parts = [np.asarray(w.segments, dtype=np.float64).tobytes(),
                 np.asarray(w.bounds, dtype=np.float64).tobytes(),
                 np.asarray(w.start_pose, dtype=np.float64).tobytes(),
                 repr(getattr(w, "partitions", ()))]
        for attr in ("family", "n_rooms"):
            if hasattr(w, attr):
                parts.append(repr(getattr(w, attr)))
        for attr in ("room_rects", "doors"):
            if hasattr(w, attr):
                parts.append(np.asarray(getattr(w, attr),
                                        dtype=np.float64).tobytes())
        if hasattr(w, "rooms_reachable"):
            parts.append(repr(sorted(w.rooms_reachable)))
        rows.append(b"|".join(
            p if isinstance(p, bytes) else repr(p).encode()
            for p in parts))
    h = hashlib.sha256()
    for r in rows:
        h.update(r)
        h.update(b"\n")
    return h.hexdigest()


def require_selectable(suite, *, final: bool = False) -> Suite:
    """Raise if `suite` is holdout-kind and no explicit final pick is set.

    This is the code form of the doc rule "VAL_SEED ... champion selection
    set; never holdout": selection/ranking tools must call this (gates do)
    so a holdout suite can never silently pick a genome.
    """
    s = get_suite(suite)
    if s.kind == "holdout" and not final:
        raise PermissionError(
            f"suite {s.name!r} is a holdout set (kind='holdout'): "
            f"genome selection/picking on it is forbidden. Pass final=True "
            f"only for the explicit committed deploy pick, or select on the "
            f"'val' suite instead.")
    return s
