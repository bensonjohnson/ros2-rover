"""House pools for building-mode evolution (pnn_sim.buildings).

A pool is N buildings of one curriculum level, generated in parallel
(each index i has its own rng seed = base + i, so a pool is reproducible
and a prefix of a larger pool is identical) and cached as a stripped
pickle: the reachability raster is only needed for the coverage ceiling,
which is precomputed for the arena lattice and kept, so cached houses are
a few KB each.

Seeds: train pools use train_seed + 1_000_003 * (level + 1); the building
holdout uses HOLDOUT_SEED at level 3 and is never used for selection.
"""
from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pnn_sim.buildings import make_level

POOL_VERSION = 1           # bump when generator output changes
HOLDOUT_SEED = 778_000
EXTENT = 16.0              # Arena(extent=) lattice the ceilings are cached for
CACHE_DIR = os.path.join("evo_runs", "world_cache")


def lattice(extent: float = EXTENT):
    g = np.arange(0.4, extent - 0.39, 0.8)
    return g, g


def _one(args):
    level, seed = args
    from .arena import cell_ceiling
    w = make_level(np.random.default_rng(seed), level)
    gx, gy = lattice()
    cell_ceiling(w, gx, gy)                   # cached on the object
    w.reach = None                            # strip the raster
    w.labels = None
    return w


def level_seed(train_seed: int, level: int) -> int:
    return train_seed + 1_000_003 * (level + 1)


def build_pool(level: int, n: int, seed: int, workers: int | None = None,
               cache: bool = True):
    path = os.path.join(CACHE_DIR,
                        f"L{level}_s{seed}_n{n}_v{POOL_VERSION}.pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    jobs = [(level, seed + i) for i in range(n)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers > 1 and n > 8:
        with ProcessPoolExecutor(workers) as ex:
            pool = list(ex.map(_one, jobs, chunksize=8))
    else:
        pool = [_one(j) for j in jobs]
    print(f"[worlds] level {level}: {n} buildings in {time.time() - t0:.0f}s"
          f" ({workers} workers)", flush=True)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(pool, f)
        os.replace(tmp, path)
    return pool


def summarize(pool) -> str:
    fam = {}
    for w in pool:
        fam[w.family] = fam.get(w.family, 0) + 1
    rooms = np.array([w.n_rooms for w in pool])
    doors = np.array([int(((w.doors[:, 4] >= 0) & (w.doors[:, 5] >= 0)).sum())
                      for w in pool])
    return (f"rooms {rooms.mean():.1f} [{rooms.min()}-{rooms.max()}] "
            f"doors {doors.mean():.1f} families {fam}")
