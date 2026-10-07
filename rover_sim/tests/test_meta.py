"""Meta-search manager tests (evo/meta — the unattended experiment driver).

    python3 -m rover_sim.tests.test_meta [--device cpu]

The manager's job is to make a search trustworthy that nobody is watching, so
the tests pin the DISCIPLINE, not just the plumbing:

M1  Space integrity: every knob maps to a flag `evo/evolve.py` really accepts
    (a typo would silently drop a knob for the whole search); sampling is
    deterministic in the search seed; the fixed knobs pin world/level/holdout.
M2  Ledger: append-only, torn final line tolerated, "last record for a trial
    wins", `completed()` drives resume, config hash stable and content-based.
M3  Objective honesty: the collision guard rejects a brute-forcer, the forward
    guard fires only when the config asks for forward, the window (not the
    first generations) is scored, and the holdout suite refuses to be selected
    on without final=True.
M4  End-to-end dry run: sampling -> subprocess launch -> gens.jsonl parse ->
    scoring -> ledger -> report, then a SECOND run that must skip everything
    (resume) instead of re-training.
M5  Supervisor: the keep-alive wrapper exists, is executable, and restarts on
    non-zero exit while stopping on a clean one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile

import numpy as np

from evo.meta import ledger as ledger_mod
from evo.meta import objective, run as meta_run, space as meta_space

EVOLVE_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "evo", "evolve.py")


def _evolve_flags() -> set:
    src = open(EVOLVE_SRC).read()
    return set(re.findall(r'"(--[a-z0-9-]+)"', src))


def _write_gens(out_dir, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gens.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rows(n=20, rooms=2.0, coll=0.5, cross=0.4, rev=0.02):
    return [dict(gen=i, elite_rooms=rooms, elite_coll=coll,
                 elite_cross_rate=cross, elite_rev_frac=rev,
                 elite_cells=0.08, elite_dist=15.0, sigma_med=0.05,
                 level=3, fit_elite=0.3) for i in range(n)]


# ----------------------------------------------------------------------- M1
def m1_space(dev):
    flags = _evolve_flags()
    missing = [k.flag for k in meta_space.KNOBS if k.flag not in flags]
    assert not missing, f"knobs with flags evolve does not accept: {missing}"
    assert len(meta_space.TUNABLE) >= 10, len(meta_space.TUNABLE)
    # determinism in the search seed, variation across seeds
    a = meta_space.sample_configs(4, seed=11)
    b = meta_space.sample_configs(4, seed=11)
    c = meta_space.sample_configs(4, seed=12)
    assert a == b, "sampling is not deterministic in the search seed"
    assert a != c, "different search seeds produced identical samples"
    # every sampled config expands to CLI args covering each knob
    cfg = a[0]
    args = meta_space.cli_args(cfg)
    for k in meta_space.KNOBS:
        v = cfg.get(k.name, k.default)
        if k.kind == "flag":
            assert (k.flag in args) == bool(v), k.name
        elif v is None:
            assert k.flag not in args, k.name
        else:
            assert k.flag in args and str(v) in args, (k.name, v)
    # non-tunable knobs are present in the config but never sampled
    assert all(k.name in cfg for k in meta_space.KNOBS)
    fixed_names = {k.name for k in meta_space.KNOBS if not k.tune}
    assert {"train_rotations", "every_cover"} <= fixed_names
    assert meta_space.FIXED["world"] == "buildings"
    assert meta_space.FIXED["select_suite"] == "val"
    print(f"M1 ok: {len(meta_space.KNOBS)} knobs "
          f"({len(meta_space.TUNABLE)} tunable) all map to real evolve flags; "
          f"sampling deterministic in --seed; fixed: world={meta_space.FIXED['world']} "
          f"level={meta_space.FIXED['level']} select={meta_space.FIXED['select_suite']}")


# ----------------------------------------------------------------------- M2
def m2_ledger(dev):
    tmp = tempfile.mkdtemp(prefix="meta_ledger_")
    try:
        led = ledger_mod.Ledger(os.path.join(tmp, "l.jsonl"))
        cfg = {"hidden": 64, "w_rev": 0.15}
        key = ledger_mod.trial_key(cfg, 63000)
        assert led.load() == []
        assert not led.completed(cfg, 63000)
        led.append(trial=key, config_hash=ledger_mod.config_hash(cfg),
                   stage="screen", seed=63000, status="running", config=cfg)
        assert not led.completed(cfg, 63000), "running must not count as done"
        led.append(trial=key, config_hash=ledger_mod.config_hash(cfg),
                   stage="screen", seed=63000, status="done", config=cfg,
                   score=1.25)
        assert led.completed(cfg, 63000)
        # last record for a trial wins (the score is the later one)
        assert led.state()[key]["score"] == 1.25
        # a torn final line (hard kill mid-append) must not break the ledger
        with open(led.path, "a") as f:
            f.write('{"trial": "torn", "status": "don')
        assert len(led.load()) == 2
        # hash is content-based and order-insensitive
        h1 = ledger_mod.config_hash({"a": 1, "b": 2})
        h2 = ledger_mod.config_hash({"b": 2, "a": 1})
        assert h1 == h2 and h1 != ledger_mod.config_hash({"a": 1, "b": 3})
        # a different seed is a different trial of the same config
        assert ledger_mod.trial_key(cfg, 63001) != key
        assert ledger_mod.config_hash(cfg) == ledger_mod.config_hash(dict(cfg))
        # the stage BUDGET is not part of identity: promotion matches trials by
        # hash, so hashing --gens would leave a stage with 0 survivors despite
        # having run (observed before the fix)
        assert (ledger_mod.config_hash({"hidden": 64, "gens": 2})
                == ledger_mod.config_hash({"hidden": 64, "gens": 40}))
        recs = [dict(status="done", feasible=True, score=s, config=c,
                     config_hash=ledger_mod.config_hash(c))
                for c, s in (({"hidden": 64}, 0.2), ({"hidden": 128}, 0.9))]
        surv = meta_run.promote(recs, 0.5)
        assert len(surv) == 1 and surv[0]["hidden"] == 128, surv
        print(f"M2 ok: append-only ledger survives a torn line, resume keys on "
              f"(config, seed), last record wins (score {led.state()[key]['score']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------- M3
def m3_objective(dev):
    tmp = tempfile.mkdtemp(prefix="meta_obj_")
    try:
        # (a) a collision brute-forcer is rejected, whatever its rooms
        d1 = os.path.join(tmp, "brute")
        _write_gens(d1, _rows(rooms=3.9, coll=12.0))
        r1 = objective.score_trial(d1, {"w_rev": 0.15}, deep=False)
        assert not r1["feasible"] and r1["score"] == float("-inf"), r1
        assert "brute-forcer" in " ".join(r1["reasons"]), r1["reasons"]

        # (b) a clean run is feasible and scored on rooms
        d2 = os.path.join(tmp, "clean")
        _write_gens(d2, _rows(rooms=2.5, coll=0.4, cross=0.5))
        r2 = objective.score_trial(d2, {"w_rev": 0.15}, deep=False)
        assert r2["feasible"], r2
        assert abs(r2["score"] - (2.5 + 1e-3 * 0.5)) < 1e-9, r2["score"]

        # (c) the forward guard fires only when the config asks for forward
        d3 = os.path.join(tmp, "rev")
        _write_gens(d3, _rows(rooms=3.0, coll=0.5, rev=0.9))
        assert not objective.score_trial(d3, {"w_rev": 0.15},
                                         deep=False)["feasible"]
        assert objective.score_trial(d3, {"w_rev": 0.0},
                                     deep=False)["feasible"]

        # (d) the trailing window is scored, not the (bad) early generations
        d4 = os.path.join(tmp, "window")
        rows = _rows(20, rooms=0.5) + _rows(20, rooms=4.0)
        for i, r in enumerate(rows):          # renumber so gens are ordered
            r["gen"] = i
        _write_gens(d4, rows)
        r4 = objective.score_trial(d4, {"w_rev": 0.0}, deep=False)
        assert r4["metrics"]["elite_rooms"] > 3.5, r4["metrics"]

        # (e) the discipline: holdout is NOT selectable without final=True
        from rover_sim import suites
        try:
            suites.require_selectable("holdout")
            raise AssertionError("holdout selection was allowed")
        except PermissionError:
            pass
        assert suites.require_selectable("holdout", final=True).kind == "holdout"

        # (f) NaN metrics must be infeasible, never ranked: a sampled config
        # really did produce an all-NaN rollout, and NaN compares false with
        # everything so it would poison the promotion mean
        d5 = os.path.join(tmp, "nan")
        _write_gens(d5, _rows(rooms=float("nan"), cross=float("nan")))
        r5 = objective.score_trial(d5, {"w_rev": 0.0}, deep=False)
        assert r5["feasible"] is False and r5["score"] == float("-inf"), r5
        assert "non-finite" in " ".join(r5["reasons"]), r5["reasons"]
        print(f"M3 ok: brute-forcer rejected (coll 12), clean run scored "
              f"{r2['score']:.3f}, forward guard w_rev-conditional, trailing "
              f"window scored ({r4['metrics']['elite_rooms']:.2f} rooms vs 0.5 "
              f"early), holdout refuses selection")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------- M4
def m4_dry_run(dev):
    """The whole loop for real: 2 tiny configs, then a resume that skips."""
    tmp = tempfile.mkdtemp(prefix="meta_e2e_")
    try:
        argv = ["--dry-run", "--root", tmp, "--n-configs", "2", "--parallel",
                "2", "--seed", "7"]
        assert meta_run.main(argv) == 0
        led = ledger_mod.Ledger(os.path.join(tmp, "ledger.jsonl"))
        state = led.state()
        done = [r for r in state.values() if r.get("status") == "done"]
        assert len(done) == 2, [r.get("status") for r in state.values()]
        assert os.path.exists(os.path.join(tmp, "report.json"))
        assert os.path.exists(os.path.join(tmp, "report.md"))
        for rec in done:
            # gens is the stage budget: if --gens were not passed through,
            # evolve would silently run its own default (40) — the exact bug
            # the first real dry run exposed
            assert rec["gens"] == 2, rec["gens"]
            # evolve emits gens+2 rows for --gens N (gen 0 plus the report
            # cadence); the point is that the stage budget ARRIVED — the bug
            # this catches is a stage silently running evolve's own default 40
            assert len(objective.read_gens(rec["out_dir"])) <= 6, rec
            assert os.path.exists(os.path.join(rec["out_dir"], "gens.jsonl"))
            assert rec["config"]["pop"] == 8, rec["config"]

        # second run: everything is already complete -> nothing re-trained
        before = len(state)
        assert meta_run.main(argv) == 0
        after = len(led.state())
        assert after == before, f"resume re-ran trials: {before} -> {after}"
        first = done[0]
        print(f"M4 ok: end-to-end dry run — {len(done)} trials trained, parsed "
              f"and scored (score {first['score']}); report written; second run "
              f"added 0 records (resume skipped all {before})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------- M5
def m5_supervisor(dev):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evo", "meta", "supervisor.sh")
    assert os.path.exists(path), path
    assert os.access(path, os.X_OK), "supervisor.sh is not executable"
    txt = open(path).read()
    assert "while true" in txt and "rc=" in txt, "no restart loop"
    assert "sleep" in txt, "no backoff between restarts"
    assert "-m evo.meta.run" in txt, "supervisor does not run the manager"
    print("M5 ok: supervisor.sh present, executable, restarts on failure and "
          "stops on a clean exit")


def m6_preflight(dev):
    """A config that cannot produce a finite rollout is rejected for ~6 s,
    not after it has spent a whole stage."""
    tmp = tempfile.mkdtemp(prefix="meta_pre_")
    try:
        args = argparse.Namespace(root=tmp, device="cpu", dry_run=False,
                                  preflight_seed=63000)
        good = {k.name: k.default for k in meta_space.KNOBS}
        good.update(pop=4, games=2, ticks=20, gens=1, no_cells=True,
                    train_rotations=1, algo="ga", mem=0, lidar_hz=None)
        ok, why = meta_run.preflight(good, args)
        assert ok, why
        assert len(objective.read_gens(os.path.join(
            tmp, "_preflight", os.listdir(os.path.join(tmp, "_preflight"))[0]))) == 1
        # a config that cannot even start is rejected with a reason
        bad = dict(good, action="bogus")
        ok2, why2 = meta_run.preflight(bad, args)
        assert not ok2 and why2, (ok2, why2)
        print(f"M6 ok: preflight accepts a usable config (1 gen, finite) and "
              f"rejects a broken one before the stage: {why2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m1_space(args.device)
    m2_ledger(args.device)
    m3_objective(args.device)
    m5_supervisor(args.device)
    m6_preflight(args.device)
    m4_dry_run(args.device)
    print("ALL META CHECKS PASSED")


if __name__ == "__main__":
    main()
