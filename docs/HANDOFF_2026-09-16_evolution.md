# Greenfield ES Rover Brain — Status & Plan

Originally a handoff written 2026-09-16 ~21:40 MDT. **Rewritten 2026-09-17
~00:45 MDT** after a review session that found two bugs voiding runs 1–9,
confirmed a working ES, sped the Spark up ~20×, and added procedural
buildings with a curriculum. Pre-crash history is condensed in the last
sections for reference only.

Repo: `~/projects/ros2-rover` (GitHub `bensonjohnson/ros2-rover`, branch `main`).
Sweep machine: DGX Spark `benson@172.0.0.201`, repo at `~/projects/ros2-rover`,
`source /home/benson/venv/bin/activate` (torch 2.11+cu130). **The Spark checkout
is NOT a git clone — ship code with rsync, never `git pull` there.**
Companion skill with the running log of verdicts: `ros2-rover-revival`.

## TL;DR (2026-09-17 00:45)

1. **Runs 1–9 were void** — two bugs, both fixed and covered by tests:
   - *Reproduction:* children were built from the parent **index**, not the
     parent genome (`p1.unsqueeze(1).expand(-1, N) * mix`). Nothing was
     inherited; "ES" was elitism over random samples. Fixed in `62f412d`
     (`evolve.reproduce`, `python3 -m evo.test_reproduce`).
   - *Policy readout (since the first evo commit):* `einsum("pgh,pgHK->pgK")`
     summed the hidden units, so **both wheel commands were functions of one
     scalar**. Also explains run 8 (BC trained with the correct `h @ Wo`,
     replayed through the broken one). Fixed in `0f824b6`
     (`python3 -m evo.test_policy`).
2. **Working ES confirmed on two seeds** (runs 9b Spark seed 8, 9b_r V620
   seed 9 — both PASS all pre-registered criteria). First runs where the
   population climbs steadily and room crossings rise.
3. **~20× faster on both GPUs**: Spark `--fused --compile --graph` = **13 s/gen**
   (was 261 s); the 32 GB V620 matches it eager with `--fp16 --fused --compile`
   = **12.7 s/gen** (was 164 s) — compile replaces the graph capture ROCm can't do.
4. **Procedural buildings + curriculum** (`d82b843`): 6 building families with
   explicit rooms and door thresholds, resampled 1024-house pools per level,
   auto-advancing levels 0→3. Run 11 family (curriculum vs direct, 2 seeds,
   `w_coll 1.0`) is **running now**.

## Live now (Spark + V620)

All four run-11 jobs use the same config (9b genome: h128, obs v1, action lr,
fixed ES; pop 128 × 16 houses × 4 rotations, 5400 ticks, `--w-dist 0.005
--w-cov 0.3 --w-coll 1.0`, 40 gens, pool 1024, fresh houses every gen). Seed 8
runs on the Spark (`--fused --compile --graph`, `evo/run11_variant.sh`), seed 9
on the V620 (eager `--fp16 --fused --compile`, `evo/run11_variant620.sh`, both
in parallel on one GPU):

| # | Machine | Out dir | Seed | Mode | State (01:15) |
|---|---|---|---|---|---|
| 1 | Spark | `evo_runs/greenfield11` | 8 | curriculum: level 0 → 3, advance at elite cross rate ≥ 0.5 for 2 gens | **done** — L1→L2 at gen 12, L2→L3 at gen 24; not yet scored |
| 2 | Spark | `evo_runs/greenfield11c` | 8 | direct: level 3 from gen 0 | running |
| 3 | V620 | `/root/evo620-src/evo_runs/greenfield11_s9` | 9 | curriculum | running (started 01:11) |
| 4 | V620 | `/root/evo620-src/evo_runs/greenfield11c_s9` | 9 | direct | running (started 01:11) |

Completion marker per run: `RUN-COMPLETE <name>` in its `run.log`.
Per-generation readouts: `gens.jsonl`; report gens (every 5) with holdouts:
`evolution.jsonl`. Seeds 8 and 9 run on different engines (graph vs eager);
9b/9b_r already showed graph and eager runs agree in direction.

**Pre-registered criteria** (stated before launch, in `evo/run11.sh` and
`evo/run11_variant.sh`):
- *Curriculum run PASS* if at gen 39: level ≥ 2 reached; `hob_elite_cross_rate`
  (level-3 building holdout, never selected on) ≥ 1.5× its gen-0 value; and ≥
  the direct run's.
- *Curriculum verdict*: mean over seeds 8 and 9 of `hob_elite_cross_rate` at
  gen 39 — curriculum ≥ 1.2× direct → curriculum helps; ≤ direct → spreading
  from gen 0 is as good; each seed pair must agree in direction.
- *Transfer guard*: legacy deep-eval `best_genome` ≥ 0.37 (9b: 0.3948).
- *Collision guard* (added with w_coll 1.0): gen-39 `hob_champ_coll` ≤ 20 and
  elite train collisions ≤ 3 in all four runs.

**Why w_coll went 0.25 → 1.0:** the first run-11 attempt (kept as
`evo_runs/greenfield11_wc025_aborted`, killed at gen 11) grew collision
brute-forcers on the holdouts (building-holdout champion 624 collisions at
gen 10; legacy-holdout champion fitness −0.38 at gen 5). Collisions count per
tick in contact (~15/s pushing a wall): at 0.25 a one-second shove cost 0.04
fitness, at 1.0 it costs 0.15 ≈ one extra room in a six-room house. The
deep-eval (`baselines`) keeps its default fitness so its numbers stay
comparable with runs 5–10. That aborted attempt also showed the curriculum
advancing fast (L0→L1 at gen 3, L1→L2 at gen 5) and the building-holdout
elite cross rate rising 0.086 → 0.25 by gen 5 — the 0.5 advance threshold may
be too easy.

## Results since the review (all verdicts pre-registered)

| Run | What | Verdict | Key numbers |
|---|---|---|---|
| 9 / 9r | ES fix only, broken readout | killed at gen 1 | superseded by 9b once the readout bug surfaced |
| **9b** | ES + readout fix, 6a replica, Spark seed 8, 20 gens | **PASS** | fit_med +0.0171 (need +0.010); elite cells 1.47× (need 1.3×); mean child_win 0.196 (need 0.10); elite rooms 1.03→1.12 |
| **9b_r** | same, V620 eager seed 9 | **PASS** | fit_med +0.040; elite cells 1.49×; child_win 0.194; elite rooms 1.04→1.18 |
| **10** | genome v2 bundle: `--obs v2 --action vw --hidden 64`, vs 9b_r | **PASS** | elite cells 0.144 vs 0.117 (1.23×, need 1.2×); fit_med 0.431 > 0.416; but elite rooms 1.12 < 1.18. Bundle not split yet |
| 8b | BC at h128 (Hermes) | killed by user | superseded — BC was confounded by the readout bug |

**Deep-evals** (32 legacy holdout houses, 3600 ticks, default fitness):

| Genome | Fitness | Rooms | Coll |
|---|---|---|---|
| best scripted baseline (go_forward) | 0.352–0.357 | 1.00 | 0 |
| 9b best_genome | **0.3948** (+12%) | 1.12 | 0 |
| 9b population best | 0.4264 | 1.06 | 21.5 (pop mean) |
| 9b_r best_genome | 0.3690 | 1.09 | 0.3 |
| 10 best_genome | 0.3878 | 1.06 | 0 |

**Generalization gap (9b/9b_r):** rooms transfer (holdout ≈ train), but
holdout fitness fell to 0.16–0.25 vs 0.44 on train because some genomes
collide heavily in unseen layouts — brittleness, not memorization. This is
what the buildings/resampling work and w_coll 1.0 address.

## Speed (measured on the Spark)

Per-tick profile, eager, B = 8192 (`python3 -m evo.prof_tick`):

| Engine | ms/tick | Top cost |
|---|---|---|
| fp16 | 48.5 | raycast 89% |
| `--fused` | 6.85 | gate.process_scan 45% |
| `--fused --compile` | **2.83** | raycast 45% |

Production generation time (train only, from 3-gen smokes, rc 0, no Xid, no
capture failures): fp16+graph **261 s** → `--fused --graph` **34 s** →
`--fused --compile --graph` **13 s**; startup (capture + gen-0 holdout) 1062 s →
63 s.

V620 (32 GB, ROCm 7.1, torch 2.12.1, Triton 3.7.1), eager only: fused kernel
parity 99.86%, scan 1.16 ms/tick; profile fp16 31.1 → fused 5.17 → fused+compile
3.18 ms/tick; production smoke **12.7 s/gen**, startup 17 s (was 164 s/gen eager
fp16). MIOpen is irrelevant (no conv/BN/RNN ops). Batch scaling with
fused+compile: 16384 envs 1.58× the time of 8192, 32768 envs 2.8× — doubling
houses per genome is cheap.

Short-game proxy (`python3 -m evo.tick_rankcorr` on 9b's population): 2700
ticks ranks like 5400 (Spearman fitness 0.93, top-32 overlap 84%); 1800 is
marginal (0.79, 66%). Half-length early generations would roughly halve run
time again — not adopted yet.

## What changed in the code (all committed + pushed)

| Commit | Change |
|---|---|
| `62f412d` | `reproduce()` inherits the genome; mutation sigma0 0.03 default, `--sigma-max 0.15`, tau 1/√N, sigma from the copied parent (`--legacy-mutation` keeps the old step rule); honest readouts: per-gen `gens.jsonl` (elite = top 25%, `child_win`), holdout of the **train** champion/elite (`ho_champ_*`, `ho_elite_*`) |
| `76e98d4` | gate/preprocess per-beam constants cached (bit-identical); `evo/prof_tick.py` |
| `0f824b6` | readout fix (bmm); genome modes `--obs v2` (gate front/rear-blocked flags replace two pure-noise gyro channels; inputs centred), `--action vw`, `--compile`; npz carry `obs_mode`/`action_mode` |
| `3589243` | `--algo oes` (OpenAI-ES: mirrored pairs, rank shaping, Adam); `evo/tick_rankcorr.py`; perf queue |
| `d82b843` | **buildings** (below) |
| later | run-11 family scripts, Spark-only queue, w_coll 1.0 |

### Buildings (`pnn_sim/buildings.py`, `evo/worlds.py`)

- **Families:** `bsp` (3–9 rooms by recursive split), `hall` (corridor spine,
  rooms both sides), `lshape` (L / U / notched footprints), `open` (large rooms,
  1.5–2.5 m archways), `maze` (braided, rooms = zones of maze cells), `house`
  (legacy slab houses converted).
- **Ground truth per building:** 0.25 m room label grid; disjoint room
  rectangles; **door thresholds** = the open wall segment between two rooms
  (crossing it = passing the doorway); 0.05 m reachability raster flood-filled
  from the start (room and coverage ceilings count only what the rover can
  reach). Natural doors 0.75/1.0 m; furniture ≥ 0.6 m from thresholds; ≤ 128
  segments.
- **Curriculum levels:** L0 2–3 rooms, wide archways, start 0.8–2 m facing a
  door → L1 ≤ 4 rooms, natural doors, facing a door → L2 ≤ 6 rooms incl. halls,
  random start → L3 full mix incl. mazes.
- **Arena building mode** (`Arena(worlds=..., caps=...)`): rectangle room
  counter and door crossing / doors-used counters (elementwise only — no
  gather kernels), 16 m coverage lattice, `load_worlds()` swaps houses by
  `copy_` into fixed buffers so the captured graph replays on new geometry.
  Legacy mode verified bit-identical to the old arena.
- **evolve flags:** `--world buildings --level L [--curriculum --advance-at
  --advance-patience] --pool N --resample-every K`; building holdout = level 3,
  seed 778000, reported as `hob_*`; per-gen `level`, `elite_cross_rate`,
  `elite_door_crossings`, `elite_doors_used_frac`.
- **Pools:** generated in parallel, cached at `evo_runs/world_cache/`,
  deterministic across machines. `baselines --world buildings` deep-evals on
  the building holdout.
- **Tests:** `python3 -m evo.test_buildings --device cuda` (B1 room code =
  `room_id` replay, B2 crossings = numpy reference, B3 `load_worlds` = fresh
  arena, B4 generator invariants).

## Evaluation protocol (current)

- **Legacy holdout** seed 777000 (make_house, natural doors) — kept for
  comparability with runs 5–10.
- **Building holdout** seed 778000, level 3, 16 houses — the transfer metric
  for building-trained runs. Never used for selection.
- Readouts: elite (top 25% by train fitness) metrics from `gens.jsonl`; the
  old population-mean `train_*` and argmax-on-holdout `holdout_*` keys are kept
  only for comparability — don't draw conclusions from them.
- **Champion claims need the deep-eval gate** (`evo.baselines --genome
  --population`, legacy and `--world buildings`). Collision brute-forcers keep
  appearing on holdouts; always read the coll column.
- Every run states its pass/fail criteria in the script header before launch;
  verdicts go in the skill.

## Next actions, in priority order

1. **Read the run-11 family** when `QUEUE11-COMPLETE` appears: apply the
   pre-registered criteria (curriculum vs direct over both seeds, transfer and
   collision guards) and record the verdict in the skill.
2. **If collisions still fail the guard:** raise `w_coll` further or cap per-game
   collision credit; consider making the gate stop collisions rather than
   only penalizing them.
3. **If the curriculum advanced too fast** (levels in 2–3 gens): raise
   `--advance-at` (e.g. 0.7) or `--advance-patience`.
4. **Split the genome-v2 bundle** (run 10): obs v2 / action vw / h64 one at a
   time on buildings — cheap now at 13 s/gen.
5. **Half-length early generations** (2700 ticks) as a speed option, chained
   via `--seed-from`.
6. **Memory genome** only if a working ES with a working readout plateaus on
   buildings — the earlier case for it came from bugged runs.
7. `bc_seed.py`'s scripted expert still has a noise-like pivot direction
   (`sign(sin(37·front))`) and a one-sided "front" sector (bins 0–7); fix
   before any BC work.

## Operational gotchas

- **Launch pattern**: `ssh spark '... (setsid nohup bash evo/runX.sh > evo_runs/runX.out 2>&1 </dev/null &)'`
  — then VERIFY in a separate call: `pgrep -f "[e]vo.evolve"` (bracketed —
  unbracketed patterns self-match the ssh command line), `tail run.log`,
  `nvidia-smi`, `journalctl -k | grep -i xid`.
- **Don't edit a bash script while it's running** (bash reads scripts
  incrementally) — write a new script and queue it instead.
- **>1 live CUDA graph per process = illegal memory access** (torch 2.11/GB10).
  One merged arena, one graph; side evals eager; never recapture in-process.
  Swapping geometry with `copy_` (buildings `load_worlds`) is fine.
- CUDA graph replay is not bit-stable for continuous metrics (dist, cells);
  coarse metrics (rooms, collisions) are.
- **Hermes shares this checkout and the Spark** — check `git status` and
  running processes before launching or committing.
- V620 (root@172.0.0.19, 32 GB): run from `/root/evo620-src` with
  `source /root/evo620/bin/activate` and `--fp16 --fused --compile` — **never
  `--graph`** (ROCm graph capture segfaults). Temporarily drained from k8s by
  the user; microk8s containerd/cilium and a Ceph OSD still run on it. Build
  pools once before launching parallel runs (concurrent builds race on the
  cache file).

## File map (evo/)

`arena.py` (rollout, graph, coverage, fused, compile, building mode),
`evolve.py` (ES driver: GA/`--algo oes`, genome modes, buildings/curriculum),
`policy.py` (genome packing, obs/action modes), `worlds.py` (building pools),
`baselines.py` (deep-eval gate), `fused_scan.py` (Triton raycast),
`prof_tick.py` (stage profiler), `tick_rankcorr.py` (short-game proxy check),
tests: `test_reproduce.py`, `test_policy.py`, `test_buildings.py`,
`test_cells.py`; run scripts: `run9b.sh`, `run11_variant.sh`, `queue11.sh`
(plus historical `run6a.sh`…`run10_620.sh`); `bc_seed.py`,
`explore_probe.py` (pre-fix diagnostics). World generator:
`pnn_sim/buildings.py`.

## History before the review (void as evidence — kept for reference)

Runs 1–8 (all under both bugs) "falsified": reward weights and dist-term
dominance (1–3), doorway width with 2 m archways (4), cell-coverage novelty
`w_cov 0.3` (5a/5b), hidden 128 (6a), novelty price `w_cov 0.6` (7a), and BC
seeding (8: all 8 keepers mse 0.004–0.010 yet replay coverage 0.03–0.06). Each
of these tested random search with a one-scalar policy, so none is a verdict
on the task. Two lessons from the BC rounds still hold: a timer/RNG-driven
expert is unimitable from observations (use a Markovian expert), and DAgger
retrains can sink closed-loop coverage (keep best by replay). Useful older
facts that remain true: `explore_probe` showed scripted random-walk
personalities cover 0.11–0.14 but never cross doors; the fused raycast kernel
measured 31.26 → 1.52 ms per tick at B = 8192 with 99.83% beam parity.

## Standing rules from the user

Verify before claiming (processes, metrics, external state). Terse updates.
Git-pull/sync before touching projects; commit work as you go; push GitHub
origin. Pre-registered hypotheses get stated BEFORE the run, verdicts recorded
in the skill so they're never relitigated. Budget-conscious: prefer cheap
decisive diagnostics (probe, smoke) over expensive full runs.
