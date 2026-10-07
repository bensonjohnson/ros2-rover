# evo/meta — the unattended ES configuration search

Turns the rover ES training into a background job that samples configurations,
scores them honestly, keeps the best, and can be left alone for days.

```bash
# 1. smoke it on the dev box (minutes, CPU, tiny budgets)
python3 -m evo.meta.run --dry-run --root /tmp/meta_smoke

# 2. real search on the Spark, detached
ssh benson@172.0.0.201
cd ~/projects/ros2-rover && source /home/benson/venv/bin/activate
nohup evo/meta/supervisor.sh --hours 48 --device cuda --parallel 1 \
      --n-configs 16 > /dev/null 2>&1 &

# 3. watch it
tail -f evo_runs/meta/supervisor.log        # each trial as it lands
cat evo_runs/meta/report.md                 # leaders, configs, genome paths
```

## What it does

Successive halving over the space in `space.py`:

| stage  | gens | ticks | seeds | keeps |
|--------|------|-------|-------|-------|
| screen | 20   | 900   | 1     | 50%   |
| trim   | 40   | 2700  | 2     | 34%   |
| full   | 120  | 5400  | 2     | all   |

Then: deep-eval every finalist on the **val** suite, and confirm the winner on
the **holdout** suite exactly once, with `final=True`. Most configs die in the
20-generation screen, which is the only way a search over 40-minute runs fits
in a night.

## The discipline it enforces (do not relax these)

* **Selection on `val` only.** `rover_sim.suites.require_selectable` refuses a
  holdout-kind suite without `final=True`; the manager only asks for holdout in
  `objective.holdout_confirm`, at the very end, for the single leader.
* **Population, not champion.** Scoring reads the *elite* metrics out of the
  trailing window of `gens.jsonl`. `best_genome` is an argmax on a
  per-generation resampled draw and has repeatedly been a collision
  brute-forcer (run 11: the transfer guard failed 3 of 4 runs on exactly this).
* **≥2 seeds before believing anything.** Measured seed spread (~0.10) exceeded
  the effects under test (0.03–0.08) throughout runs 12–14. Stages 2 and 3 run
  two seeds and the promotion ranks smaller spread higher on ties.
* **Constraints before score.** `elite_coll > 3` or, when the config asks for
  forward driving, `elite_rev_frac > 0.15` ⇒ `feasible=False`, score `-inf`.
  A config that buys rooms by crashing is not a candidate.
* **rooms/game is the metric.** Door-crossing counts are gameable by
  ping-ponging one doorway (metric flaw #2) and arc length by orbiting in place
  (flaw #1); rooms is the only one that survived every audit.

## Reading the ledger

`evo_runs/meta/ledger.jsonl` is append-only; one record per
`(config, seed, attempt)`; last record for a trial key wins. Every record
carries the FULL config, so any champion can be rebuilt exactly:

```bash
# best trials so far
python3 -c "
import json
rec=[json.loads(l) for l in open('evo_runs/meta/ledger.jsonl')]
done=[r for r in rec if r.get('status')=='done']
done.sort(key=lambda r:-(r.get('score') or -1e9))
for r in done[:8]: print(r['score'], r['trial'], r['stage'], r['reasons'])"
```

Resume is automatic and exact: re-running the same command skips every finished
trial (that is what makes the supervisor wrapper safe).

## Knobs

`space.py` is the single source of truth: each knob declares its CLI flag, its
domain, and whether it is tunable. Knobs with `tune=False` (`world`, `level`,
`train-rotations`, `every-cover`, …) are recorded in every trial but never
sampled — changing them moves the goalposts rather than the result. Override
anything per-run with `--space file.json`, e.g.:

```json
{"lidar-hz": 10.0, "w_rev": 0.15, "hidden": 128}
```

Adding a knob is three lines in `space.py`; the CLI mapping and the ledger come
along for free. `test_meta.py` M1 fails loudly if a knob's flag is not one that
`evo/evolve.py` actually accepts.

## Known limits

* The pivot/spin constraint is enforced through `w_spin` in training; the full
  `rover_sim.bench` gates (`worst_trim_multi_draw_pick`, `trim_envelope`,
  `pivot_cap`) are NOT yet part of automatic promotion — run them on the top
  few genomes before any field test:
  `python3 -m rover_sim.bench --suite val --genome <path>` (see `bench.py`).
* Parallel >1 on the Spark is allowed but each run is large; start at 1 and
  watch GPU memory / thermals.
* A trial that produces no `gens.jsonl` is recorded `failed` with its
  `train.log` path — never silently dropped.
