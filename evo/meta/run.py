"""Unattended meta-search driver.

    # smoke it (minutes, CPU)
    python3 -m evo.meta.run --dry-run --root /tmp/meta_smoke

    # leave it running (days, Spark)
    nohup evo/meta/supervisor.sh --hours 48 --device cuda --parallel 1 &

Shape of a search: sample `--n-configs` configs, run stage 1 (short, 1 seed),
keep the better half, run stage 2 (longer, 2 seeds) on the survivors, keep the
front third, run stage 3 full-length, then deep-eval and finally confirm the
winner on the holdout suite ONCE. Successive halving, i.e. most configs die
cheap — which is the only way a search over 40-min runs fits in a night.

Everything is driven off the ledger (`evo/meta/ledger.py`): promotions are
recomputed from recorded scores, so `--resume` (the default behaviour) is exact
and a crash mid-stage costs only the trials in flight. Launching is a
subprocess of `python -m evo.evolve` per (config, seed) — the manager never
imports the trainer, so a segfaulting trial cannot take the search down with
it.

LIVE-RUN HAZARD THIS FILE EXISTS TO AVOID: on the Spark, never gate a launch on
`pgrep` from inside the same ssh command — the pattern matches the wrapper's own
cmdline and the launch is silently skipped. The manager launches and then reads
the trial's own log/gens.jsonl, so "did it start" is answered by evidence from
the run, not by process-name guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import objective
from .ledger import Ledger, config_hash, trial_key
from .space import (FIXED, KNOBS, SPACE_VERSION, BY_NAME, cli_args,
                    default_config, sample_configs)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Successive-halving schedule. gens/ticks grow, `keep` is the survivor
#: fraction handed to the next stage. 2700 ticks at stage 2 is a validated
#: early-gen proxy (rank corr 0.93 fit / 0.98 cells vs full length).
STAGES = (
    dict(name="screen", gens=20, ticks=900, seeds=(63000,), keep=0.5),
    dict(name="trim", gens=40, ticks=2700, seeds=(63000, 63001), keep=0.34),
    dict(name="full", gens=120, ticks=5400, seeds=(63000, 63001), keep=1.0),
)
DRY_STAGES = (dict(name="dry", gens=2, ticks=100, seeds=(63000,), keep=1.0),)

#: Pre-flight budget: one generation, tiny. A configuration that produces a
#: non-finite rollout (observed once in a real dry run: `--mem 8` + OES gave an
#: all-NaN rollout and `oes_rel_step: NaN`) must be caught HERE, for ~6 seconds,
#: rather than after it has spent a 40-minute stage. The failure is rare and was
#: not reproducible in isolation, which is exactly why it needs a filter rather
#: than a fix someone remembers to make.
PREFLIGHT = dict(pop=4, games=2, ticks=20, gens=1, no_cells=True,
                 train_rotations=1)

#: Dry mode pins the cheap end of every budget knob so the whole pipeline —
#: sampling, launch, parse, score, promote, ledger, resume — is exercised.
DRY_OVERRIDES = dict(pop=8, games=2, no_cells=True, train_rotations=1,
                     ticks=100)


def build_cmd(cfg: dict, seed: int, out_dir: str, *, device: str,
              dry: bool) -> list:
    """evolve CLI for one (config, seed). All knobs go in explicitly so the
    ledger's config is the run's config."""
    cmd = [sys.executable, "-m", "evo.evolve"]
    cmd += cli_args(cfg)
    cmd += ["--world", FIXED["world"], "--level", str(FIXED["level"]),
            "--holdout-seed", str(FIXED["holdout_seed"]),
            "--seed", str(seed), "--train-seed", str(seed),
            "--device", device, "--out-dir", out_dir,
            "--report-every", "1" if dry else "5"]
    if str(device).startswith("cuda"):
        cmd += ["--fp16", "--fused", "--compile"]
        if not dry:
            cmd += ["--graph"]
    return cmd


def run_trial(cfg: dict, seed: int, stage: dict, args) -> dict:
    """One (config, seed): launch, wait, score, return a ledger record."""
    key = trial_key(cfg, seed)
    out_dir = os.path.join(args.root, key)
    os.makedirs(out_dir, exist_ok=True)
    cmd = build_cmd(cfg, seed, out_dir, device=args.device, dry=args.dry_run)
    t0 = time.time()
    with open(os.path.join(out_dir, "train.log"), "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=f,
                                  stderr=subprocess.STDOUT,
                                  timeout=args.trial_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            f.write(f"\n[trial] TIMEOUT after {args.trial_timeout}s\n")
    elapsed = time.time() - t0
    base = dict(trial=key, config_hash=config_hash(cfg), stage=stage["name"],
                seed=int(seed), config=cfg, out_dir=out_dir,
                gens=int(stage["gens"]), t_start=t0,
                elapsed_s=round(elapsed, 1))
    if rc != 0:
        return dict(base, status="failed", score=float("-inf"),
                    feasible=False, reasons=[f"evolve rc={rc}"],
                    detail=f"see {out_dir}/train.log")
    res = objective.score_trial(out_dir, cfg, deep=args.deep_eval,
                                device=args.device,
                                val_games=args.val_games,
                                val_ticks=args.val_ticks)
    return dict(base, **res)


def preflight(cfg: dict, args) -> tuple:
    """Cheap filter: can this configuration produce a finite rollout at all?

    Returns (ok, reason). Runs ONE tiny generation (~6 s) at the cheap end of
    every budget knob and checks the fitness is finite. A config that fails is
    recorded in the ledger as `rejected-preflight` with its reason, so the
    search never silently loses a budget slot to a NaN.
    """
    pcfg = {**cfg, **PREFLIGHT}
    out = os.path.join(args.root, "_preflight",
                       config_hash(pcfg) + f"-s{args.preflight_seed}")
    os.makedirs(out, exist_ok=True)
    cmd = build_cmd(pcfg, args.preflight_seed, out, device=args.device, dry=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "preflight timeout"
    if proc.returncode != 0:
        return False, f"preflight rc={proc.returncode}"
    rows = objective.read_gens(out)
    if not rows:
        return False, "preflight produced no gens.jsonl"
    fit = rows[-1].get("fit_best")
    if fit is None or not np.isfinite(float(fit)):
        return False, f"preflight non-finite fitness ({fit})"
    return True, ""


def promote(records: list, keep: float) -> list:
    """Survivor configs: mean score across the seeds tried, feasible first."""
    by_hash: dict = {}
    for rec in records:
        if rec.get("status") != "done":
            continue
        h = rec["config_hash"]
        by_hash.setdefault(h, {"cfg": rec["config"], "scores": [],
                               "feasible": True})
        if rec.get("score") is None or rec["score"] == float("-inf"):
            by_hash[h]["feasible"] = False
            by_hash[h]["scores"].append(float("-inf"))
        else:
            by_hash[h]["scores"].append(float(rec["score"]))
    ranked = []
    for h, d in by_hash.items():
        n = len(d["scores"])
        mean = sum(s for s in d["scores"] if s != float("-inf")) / \
            max(1, sum(1 for s in d["scores"] if s != float("-inf")))
        spread = (max(d["scores"]) - min(d["scores"])) if n > 1 else 0.0
        ranked.append((bool(d["feasible"]), mean, spread, h, d["cfg"]))
    # feasible first, then score, then smaller spread (a config whose seeds
    # disagree is not a discovery)
    ranked.sort(key=lambda r: (not r[0], -r[1], r[2]))
    n_keep = max(1, int(round(len(ranked) * keep)))
    return [dict(cfg) for _, _, _, _, cfg in ranked[:n_keep]]


def _stage_records(ledger: Ledger, stage: dict, stage_cfgs: list) -> list:
    hashes = {config_hash(c) for c in stage_cfgs}
    return [rec for rec in ledger.state().values()
            if rec.get("config_hash") in hashes
            and rec.get("stage") == stage["name"]
            and rec.get("status") == "done"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("evo_runs", "meta"),
                    help="trial + ledger root (each trial gets its own dir)")
    ap.add_argument("--n-configs", type=int, default=12,
                    help="stage-1 candidates sampled from the space")
    ap.add_argument("--seed", type=int, default=4242, help="search RNG seed")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="wall-clock budget; 0 = no limit. Overruns stop the "
                         "search cleanly at a stage boundary (resumable)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="trials in flight (GPU box: 1-2, each run is big)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny budgets (pop 8 / games 2 / ticks 100 / gens 2), "
                         "no deep eval, no holdout: proves the whole loop")
    ap.add_argument("--no-deep-eval", action="store_true",
                    help="score from gens.jsonl only (skips the val re-eval)")
    ap.add_argument("--val-games", type=int, default=8)
    ap.add_argument("--val-ticks", type=int, default=1800)
    ap.add_argument("--space", default=None,
                    help="JSON file of forced knobs (merged into every sample)")
    ap.add_argument("--trial-timeout", type=int, default=21600)
    ap.add_argument("--preflight-seed", type=int, default=63000,
                    help="world/ES seed used by the pre-flight probe")
    args = ap.parse_args(argv)

    args.deep_eval = not (args.no_deep_eval or args.dry_run)
    stages = DRY_STAGES if args.dry_run else STAGES
    overrides = {}
    if args.space:
        with open(args.space) as f:
            overrides = json.load(f)
        unknown = set(overrides) - set(BY_NAME)
        if unknown:
            raise SystemExit(f"--space has unknown knobs: {sorted(unknown)}")
    if args.dry_run:
        overrides = {**DRY_OVERRIDES, **overrides}

    os.makedirs(args.root, exist_ok=True)
    ledger = Ledger(os.path.join(args.root, "ledger.jsonl"))
    deadline = time.time() + args.hours * 3600 if args.hours else None

    print(f"[meta] space {SPACE_VERSION} | stages "
          f"{[s['name'] for s in stages]} | root {args.root} | "
          f"device {args.device} | parallel {args.parallel} | "
          f"deep_eval {args.deep_eval}"
          + (f" | budget {args.hours} h" if deadline else ""))

    candidates = sample_configs(args.n_configs, seed=args.seed,
                                overrides=overrides)
    # Pre-flight every candidate once (not in dry mode, where the trial budget
    # IS the pre-flight budget): drop configs that cannot produce a finite
    # rollout before they are allowed to spend a stage.
    if not args.dry_run:
        filtered, rejected = [], []
        for cfg in candidates:
            ok, why = preflight(cfg, args)
            if ok:
                filtered.append(cfg)
            else:
                rejected.append((cfg, why))
                ledger.append(trial=trial_key(cfg, -1),
                              config_hash=config_hash(cfg), stage="preflight",
                              seed=-1, status="rejected-preflight",
                              config=cfg, score=float("-inf"),
                              feasible=False, reasons=[why])
                print(f"[meta] preflight REJECT {config_hash(cfg)}: {why}")
        print(f"[meta] preflight: {len(filtered)}/{len(candidates)} candidates "
              f"usable")
        candidates = filtered
    for stage in stages:
        trials = [(cfg, seed) for cfg in candidates for seed in stage["seeds"]]
        pending = [(c, s) for c, s in trials if not ledger.completed(c, s)]
        skipped = len(trials) - len(pending)
        print(f"[meta] stage {stage['name']}: {len(trials)} trials "
              f"({len(pending)} to run, {skipped} already done) | "
              f"gens {stage['gens']} ticks {stage['ticks']}")
        if deadline and time.time() >= deadline:
            print("[meta] budget exhausted before this stage — stopping "
                  "(resume later with the same command)")
            break

        def one(cs):
            cfg, seed = cs
            cfg = {**cfg, "gens": stage["gens"]}
            return run_trial(cfg, seed, stage, args)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            for rec in ex.map(one, pending):
                ledger.append(**rec)
                print(f"[meta]   {rec['trial']} {rec['status']:6s} "
                      f"score={rec.get('score')} "
                      f"{('; '.join(rec.get('reasons') or []) or 'ok')} "
                      f"({rec.get('elapsed_s', 0)}s)")
        recs = _stage_records(ledger, stage, candidates)
        candidates = promote(recs, stage["keep"])
        print(f"[meta] stage {stage['name']} done in {time.time() - t0:.0f}s "
              f"-> {len(candidates)} survivors")

    # ------------------------------------------------------------- report
    state = ledger.state()
    done = [r for r in state.values() if r.get("status") == "done"]
    ranked = sorted(done, key=lambda r: (not r.get("feasible", False),
                                         -(r.get("score") or float("-inf"))))
    report = {
        "space_version": SPACE_VERSION,
        "stages": [s["name"] for s in stages],
        "trials_total": len(state),
        "trials_done": len(done),
        "trials_failed": sum(1 for r in state.values()
                             if r.get("status") == "failed"),
        "leaders": [
            {"trial": r["trial"], "stage": r["stage"], "seed": r["seed"],
             "score": r.get("score"), "feasible": r.get("feasible"),
             "reasons": r.get("reasons"), "config": r.get("config"),
             "metrics": r.get("metrics"), "out_dir": r.get("out_dir")}
            for r in ranked[:5]],
    }
    rpath = os.path.join(args.root, "report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    lines = [f"# meta search — {args.root}", "",
             f"space {SPACE_VERSION} · {len(state)} trials "
             f"({len(done)} done, {report['trials_failed']} failed)", ""]
    for i, lad in enumerate(report["leaders"]):
        cfg = lad["config"] or {}
        lines += [f"## {i + 1}. {lad['trial']}  (score {lad['score']})",
                  f"stage {lad['stage']} seed {lad['seed']} · feasible "
                  f"{lad['feasible']}"
                  + (f" · {', '.join(lad['reasons'])}" if lad["reasons"] else ""),
                  f"genome: `{os.path.join(lad['out_dir'], 'val_best_genome.npz')}`",
                  "```", json.dumps(cfg, indent=2, sort_keys=True,
                                    default=str), "```", ""]
    with open(os.path.join(args.root, "report.md"), "w") as f:
        f.write("\n".join(lines))

    if ranked and not args.dry_run:
        champ = ranked[0]
        gpath = os.path.join(champ["out_dir"], "val_best_genome.npz")
        if os.path.exists(gpath):
            print("[meta] holdout confirmation for the leader (final=True, "
                  "the ONE holdout touch)")
            hold = objective.holdout_confirm(gpath, device=args.device,
                                             games=32, ticks=3600)
            report["holdout"] = hold
            with open(rpath, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[meta] holdout: {hold}")

    print(f"[meta] report -> {rpath} and "
          f"{os.path.join(args.root, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
