"""Suite registry tests (docs/SIM_PLATFORM.md §6 stage 5).

    python3 -m rover_sim.tests.test_suites [--device cpu]

T1  registry: all four named suites resolve; kinds (train/val/holdout) and
    external_worlds flags are as documented.
T2  pool equality: buildings/val/holdout `Suite.build(cache=False)` produce
    the SAME worlds as `evo.worlds.build_pool` for the same keys (stable
    structural fingerprint equality — never touches the shared cache).
T3  selection discipline: `require_selectable` RAISES for holdout without
    final=True and passes with it / for val. The bench gates enforce it
    (imported here so the discipline is tested at the source).
T4  legacy suite loads: the classic make_house draw builds and summarizes;
    it is a train-kind suite with external_worlds=False.
T5  summarize/fingerprint: summarize forwards to evo.worlds for building
    pools; fingerprint is stable across identical builds.
"""

from __future__ import annotations

import argparse

import numpy as np

from evo.worlds import HOLDOUT_SEED, VAL_SEED, build_pool, level_seed
from rover_sim import bench, suites
from rover_sim.suites import (SUITES, fingerprint, get_suite, require_selectable,
                              suite_names, summarize)

# small + sequential: deterministic and cache-free (never writes evo_runs/)
N = 3
LEVEL = 3
TRAIN_SEED = 63_000


def t1_registry():
    assert suite_names() == ("legacy", "buildings", "val", "holdout"), \
        suite_names()
    kinds = {n: SUITES[n].kind for n in suite_names()}
    assert kinds == {"legacy": "train", "buildings": "train", "val": "val",
                     "holdout": "holdout"}, kinds
    ext = {n: SUITES[n].external_worlds for n in suite_names()}
    assert ext == {"legacy": False, "buildings": True, "val": True,
                   "holdout": True}, ext
    for n in suite_names():
        assert get_suite(n).name == n
    assert get_suite(get_suite("val")) is get_suite("val")
    try:
        get_suite("nope")
        raise AssertionError("unknown suite did not raise")
    except KeyError:
        pass
    print(f"T1 ok: registry {suite_names()} kinds={kinds} "
          f"external_worlds={ext}")


def t2_pool_equality():
    for name, seed in (("buildings", level_seed(TRAIN_SEED, LEVEL)),
                       ("val", VAL_SEED), ("holdout", HOLDOUT_SEED)):
        got = get_suite(name).build(n=N, level=LEVEL, train_seed=TRAIN_SEED,
                                    workers=1, cache=False)
        ref = build_pool(LEVEL, N, seed, workers=1, cache=False)
        fg, fr = fingerprint(got), fingerprint(ref)
        assert fg == fr, f"{name}: fingerprint {fg[:12]} != {fr[:12]}"
        # structural spot-check independent of the digest
        assert [len(w.segments) for w in got] == [len(w.segments) for w in ref]
        assert all(np.array_equal(a.start_pose, b.start_pose)
                   for a, b in zip(got, ref))
        print(f"T2 ok [{name}]: {N} worlds == evo.worlds.build_pool(L{LEVEL}, "
              f"seed={seed}) fingerprint {fg[:12]} | {summarize(got)}")


def t3_discipline():
    try:
        require_selectable("holdout")
        raise AssertionError("holdout: no raise without final=True")
    except PermissionError:
        pass
    assert require_selectable("holdout", final=True).kind == "holdout"
    assert require_selectable("val").kind == "val"
    assert require_selectable("buildings").kind == "train"
    # the pick/pick-like gates also refuse a holdout suite
    for fn in (bench.worst_trim_multi_draw_pick, bench.trim_envelope,
               bench.pivot_cap):
        try:
            fn("deploy/genomes/champ21f_idx103.npz", suite="holdout", games=2,
               ticks=60, cache=False)
            raise AssertionError(f"{fn.__name__}: holdout selection allowed")
        except PermissionError:
            pass
    print("T3 ok: holdout requires final=True (registry + all 3 gates); "
          "val/train pass")


def t4_legacy():
    s = get_suite("legacy")
    assert s.kind == "train" and not s.external_worlds
    w = s.build(n=N, cache=False)
    assert len(w) == N
    assert all(hasattr(x, "segments") and hasattr(x, "start_pose")
               for x in w)
    assert s.engine_seed(TRAIN_SEED) == TRAIN_SEED
    assert s.engine_seed() == suites.TRAIN_SEED
    print(f"T4 ok: legacy suite loads ({N} make_house worlds; "
          f"{summarize(w)})")


def t5_summarize_fingerprint():
    a = get_suite("val").build(n=N, level=LEVEL, workers=1, cache=False)
    b = get_suite("val").build(n=N, level=LEVEL, workers=1, cache=False)
    assert fingerprint(a) == fingerprint(b), "fingerprint unstable"
    assert fingerprint(a) != fingerprint(
        get_suite("holdout").build(n=N, level=LEVEL, workers=1, cache=False)), \
        "val and holdout fingerprints collide"
    txt = summarize(a)
    assert txt.startswith("rooms ") and "families" in txt, txt
    ltxt = summarize(get_suite("legacy").build(n=N, cache=False))
    assert ltxt.startswith("legacy "), ltxt
    print(f"T5 ok: fingerprint stable + val!=holdout; summarize "
          f"building='{txt[:40]}...' legacy='{ltxt}'")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.parse_args()

    t1_registry()
    t2_pool_equality()
    t3_discipline()
    t4_legacy()
    t5_summarize_fingerprint()
    print("ALL SUITES CHECKS PASSED")


if __name__ == "__main__":
    main()
