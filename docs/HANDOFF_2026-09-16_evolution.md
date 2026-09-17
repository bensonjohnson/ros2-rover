# Greenfield ES Rover Brain — Status & Plan (handoff, 2026-09-16 ~21:40 MDT)

Repo: `~/projects/ros2-rover` (GitHub `bensonjohnson/ros2-rover`, branch `main`).
Sweep machine: DGX Spark `benson@172.0.0.201`, repo at `~/projects/ros2-rover`,
`source /home/benson/venv/bin/activate` (torch 2.11+cu130). **The Spark checkout
is NOT a git clone — ship code with rsync, never `git pull` there.**
Companion skill with the same content in condensed form: `ros2-rover-revival`.

## The goal

Evolve a small recurrent-MLP "pet rover" brain (obs = 72 lidar bins + 8 proprio
+ last action → 2 wheel commands) that **explores a simulated house: crosses
doorways into new rooms and covers floor space**, in `pnn_sim.batched` physics
with the real safety gate in the loop. Evolutionary strategy (ES, population
128, per-gene mutation, self-adapted sigma), no gradients for evolution itself.
The real-rover deployment analogue: place-memory-driven curiosity pet.

## Evaluation protocol (do not change — all runs comparable)

- Holdout seed **777000**, natural doors (0.7–1.0 m), never used for selection.
- `train_rooms` (ground-truth room coverage) is the honest readout;
  `train_cells` (distinct 0.8 m cells, saturating, no room_id needed) is the
  novelty signal; holdout fitness/rooms are the transfer metric.
- Production config: pop 128 × 16 houses × 4 rotations = B=8192 envs,
  ticks=5400 (6 sim-min), `--w-dist 0.005 --w-coll 0.25 --fp16 --graph`.
- Champion claims require the deep-eval gate
  (`evo/baselines --genome --population` with default fitness): the
  train-fitness argmax under `w_cov` is a Goodhart trap (run 5a: "0.5009 best
  ever" genome deep-eval'd to 0.129 with 88.8 collisions = collision brute-forcer;
  honest population best was 0.4213 vs best scripted baseline 0.3558, +19%).

## What has been falsified (runs 1–7a, all ~40 gens, same protocol)

| # | Hypothesis tested | Result |
|---|---|---|
| 1–3 | reward weights / dist term dominance | dist out-earns rooms → fast wall-hugger plateau; rooms stuck 1.0 |
| 4 | doorways too narrow (basin width) | 2 m archways: `train_rooms` 1.06 flat. Falsified |
| 5a/5b | reward SHAPE: cell-coverage novelty `w_cov 0.3` | population frozen: cells 0.073, rooms 1.01, median pinned 0.3824 for all 40 gens. Elite 0.5009 was argmax noise + coll-cheating |
| 6a | CAPACITY: hidden 128 (+novelty) | cells 0.063–0.066, rooms 1.01 flat gens 5–39; sigma bloated 0.25→0.61 (ES on a plateau); champion deep-eval 0.390–0.415 ≈ h64 champion. **Width is not the wall** (and retroactively rules out 512) |
| 7a | reward PRICE: `w_cov 0.6` (novelty outbids cruising) | cells locked 0.062–0.065, rooms 1.03 through gen 30 (killed early). Falsified |

**The decisive diagnostic — `evo/explore_probe.py`** (128 randomized scripted
personalities, per-env bias/commitment-timers, same arena + same coverage metric):
- random-walk personalities cover **0.110–0.144 — 2× the evolved population's 0.066**.
  Evolution covers LESS ground than chance → exploration IS representable from
  scan72; the ES failure is **searchability**, not the reward or the body.
- NO scripted personality crossed a doorway (rooms 1.00–1.05 all modes) →
  room-crossing likely needs goal state held across the approach (memory),
  not just steering.

## Live: run 8 = BC-seeded ES (`evo/run8.sh`, chained pipeline)

Idea: the scripted personalities ARE correct explorers (probe + candidate round
found **cov 0.108–0.170** on the real train houses), so distill the best few into
MLP genomes with behavior cloning (+DAgger) and put them in the founding
population — hill-climb FROM an explorer instead of inventing one by mutation.

`evo/bc_seed.py` pipeline: 64 scripted candidates in train arena → rank by
coverage → keep top 8 → record expert (obs, cmd) streams over 32×2 houses →
distill each into the recurrent-MLP genome (Adam, segment-BPTT) → **2 DAgger
rounds** (student rolls houses, expert relabels the states the student actually
visits) → **keep-best-by-replay across stages** → gate: keep only members whose
MLP replay coverage ≥ 0.09 (cruise band 0.063–0.073, expert 0.11–0.17).
Then `evo.evolve --bc-from bc_thetas.npz` (kept members occupy founding slots,
rest random; sigma floor 0.15) → `evo/baselines` deep-eval guard at the end.

**Two hard lessons from failed BC rounds (already fixed in the code):**
1. The timer/RNG-driven expert is **unimitable**: internal clock states aren't in
   the observation stream → best MLP can only average the commits → standstill.
   (Keeper with mse 0.0022 yet replay cov 0.052.) Fix: expert is now **Markovian**
   — `cmd = f(scan72; constant personality)`, commitment is environmental
   (wall proximity keeps the pivot engaged). Markovian experts explore even
   better: cov 0.137–0.170.
2. Open-loop BC suffers **covariate shift**; DAgger fixes labels but each
   retrain can sink closed-loop coverage (cliffy) → keep-best-by-replay.

**State at time of writing:** run 8 relaunched 21:29 with keep-best distiller;
keepers 0–3 mid-run showing best-roll-cov 0.026–0.057 — *the keep-best gate may
still reject everyone.* If `RUN8-BCFAIL rc=1` (or rc=1 from the empty-bc guard
in evolve), the pre-registered conclusion is:

> A feedforward-backboned recurrent MLP (h64/h128) cannot hold even a
> Markovian scanner personality's exploration closed-loop → the genome needs
> explicit structure. **Next build = memory genome**: CAMEMBE-style memory
> vector + read/write heads in `evo/policy.py` (~200 lines), same flat-pack +
> ES recipe, and directly the right shape for the real rover's place-memory
> brain (see PCN lineage in this same repo).

If ≥4 keepers pass instead: ES runs 40 gens (≈3.5–4 h at current speed), then
deep-eval. Read `evo_runs/greenfield8/run.log` / `evolution.jsonl` for
`train_cells` movement (passes 0.073 = first population-level shift in 8 runs).

## Speed: the raycast is now a fused Triton kernel (your RT-core idea, right shape)

Bottleneck analysis (measured, runs 5–7 era): rollout was **memory-traffic-bound**
— the fp16 raycast materialized ~6 intermediates of [B=8192, 360 beams, ~30 segs]
per tick (~250 MB per tensor round-trips through DRAM); CUDA graphs only bought
1.06×; tensor cores idle (policy matmuls ~6% of time; TF32 toggles no-ops).
OptiX/RT cores exist on the GB10 but are the wrong tool for 2D segment fields.
`evo/fused_scan.py`: one Triton program per (env, 64 beams), segments streamed
into registers, one store. **Measured on Spark: 31.26 ms → 1.52 ms per tick
(20.5× scan speedup) at B=8192**; parity 99.83% of beams within 0.05 m of the
fp16 chain (grazing-ray fp16 diffs), deterministic, graph-capture-safe
(static out buffer — zero alloc in capture, the run-5 Xid-43 fault class).
Wired as `evo.evolve --fused` (implies --fp16; applies to train+holdout arenas).
3-gen production smoke passed with `--fused --graph` (`evo_runs/smoke_fused`,
reduced size). **Production-size fused+graph run not yet done** — expect
~2–3× generation-time cut (rollout remainder: gate ~, physics, policy, and
the fp32 collision check which we deliberately keep exact). Verify with the
standard checks before believing it (no Xid 43 in `journalctl -k`, log header,
GPU util — see pitfalls below).

## Operational gotchas (all earned the hard way — see skill too)

- **Launch pattern**: `ssh spark '... (setsid nohup bash evo/runX.sh > evo_runs/runX.out 2>&1 </dev/null &); sleep 20; tail log'` — then VERIFY separately: `pgrep -f "[e]vo.evolve"` (bracketed! unbracketed pgrep self-matches the ssh cmdline → false ALREADY-RUNNING), `tail run.log`, `nvidia-smi`. Never gate a launch on pgrep inside the same ssh command.
- **>1 live CUDA graph per process = illegal memory access** (torch 2.11/GB10). One merged arena, one graph; side evals eager; never recapture in-process (chain phases via `--seed-from`).
- Run 7a's mysterious 2× slowdown = my script bug: run7a.sh omitted `--games 16 --ticks 5400 --fp16 --graph --w-dist 0.005` (eager fp32 at CLI defaults). The GB10 is fine. `run6a.sh` is the correct template for the CLI flags.
- CUDA graph replay is not bit-stable for continuous metrics (chaotic raycast/matmul jitter over 5400 ticks); coarse metrics (rooms/collisions) are. Tests tolerance-check continuous ones.
- V620 AMD box (root@172.0.0.19): eager-only (HIP graph capture segfaults), roughly parity with Spark eager (130–169k vs 152k env-ticks/s). Parked (`/root/evo620`, `/root/evo620-src`) — viable as a parallel trainer, never the graph fast path.

## File map (evo/)

`arena.py` (rollout/graph/coverage/fused), `evolve.py` (ES driver, `--bc-from`,
`--fused`), `policy.py` (genome packing — the memory-genome change goes here),
`bc_seed.py` (distiller), `explore_probe.py` (scripted-personality diagnostic),
`fused_scan.py` (Triton kernel + P/B gates), `baselines.py` (deep-eval gate +
scripted baselines), `test_cells.py` (coverage accumulator suite — run before
trusting arena changes), `run6a.sh`/`run8.sh` (run templates), `bench620.py`.

## Suggested next actions for a fresh agent, in priority order

1. Check `evo_runs/greenfield8/run.log`: if `RUN8-BCFAIL` → implement the
   memory genome in `evo/policy.py` (add memory vector m∈R^8 with learned
   read/write heads into the flat-packed genome; keep ES recipe untouched;
   re-run the probe against it — a Markovian expert should be trivially
   distillable INTO the memory genome if BC keeps failing on MLP replay).
   If BC passed and ES is mid-flight: read `train_cells` per gen at report
   marks (0.073 = the ceiling to break), deep-eval the champion before any
   claims.
2. Production-size `--fused --graph` validation run (pop 128×16×4, ticks 5400,
   3 gens): check `journalctl -k` clean, capture completes, measure gen time
   (expect ≤ ~150 s/gen vs 288–340 s baseline). If it faults (Xid 43), fall
   back `--fp16 --graph` and keep the kernel for eager-only evals.
3. Only after the population moves: raise games/rotations for statistical
   power on the holdout (currently single-argmax reports are noisy).
4. Keep V620 in reserve for parallel seeds (different `--seed`/train-seed,
   same protocol) once per-run cost is the binding constraint again.

## Standing rules from the user

Verify before claiming (processes, metrics, external state). Terse updates.
Git-pull/sync before touching projects; commit work as you go; push GitHub
origin. Pre-registered hypotheses get stated BEFORE the run, verdicts recorded
in the skill so they're never relitigated. Budget-conscious: prefer cheap
decisive diagnostics (probe, smoke) over expensive full runs.
