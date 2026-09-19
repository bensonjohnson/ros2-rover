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

## TL;DR (2026-09-18 10:15)

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
   auto-advancing levels 0→3. **Run 11 verdict:** buildings training raises the
   unseen-building cross rate 5–7×; the curriculum adds nothing over training
   on the full mix; `w_coll 1.0` removed the collision brute-forcers.
5. **Run 12 verdict:** obs v2 / action vw / hidden 64 are all neutral — the
   base genome stands, and a validation-set champion (`val_best_genome.npz`)
   replaced the noisy train-argmax pick.
6. **Runs 13–14:** longer training took the champion to ~4 of 6 rooms on unseen
   buildings (plateau test split at 240 gens).
7. **Real rover:** the champion runs on hardware through the real safety
   monitor (`evo/deploy/`); it explores mostly in **reverse**, as in the sim.
8. **Reverse habit:** a reverse penalty alone (run 15) and a symmetric safety
   gate (run 16) both failed to produce a good forward explorer. The reverse
   habit is present from gen 0 and is not caused by the gate asymmetry.
9. **Sim-to-real hazard found:** the sim's 0.8 left trim is a physical
   slowdown, but on the real rover it is a driver-side correction. The best
   champion (run 14) takes 620 collisions with balanced tracks. Trim
   randomization (run 17) makes champions collision-robust (≤ 10 at every
   trim) at some cost in rooms.
10. **Run 18 verdict:** the doorway gate does NOT unlock forward exploring —
    cross rate 0.524 at gen 119 (H18a/b FAIL). Every forward arm (15f, 16sr,
    18d) plateaus 0.30–0.52 while reverse arms reach 0.86–0.89. The forward
    rules are not the doorway bottleneck; **do not change the real monitor**.
11. **Deploy hunt (run 19 queue, live):** `evo/deploy_pick.py` re-ranks a
    WHOLE final population across the trim envelope (5 trim settings × 32
    unseen buildings, worst case wins). On run 17's population it found idx
    56: worst-trim cross 0.688 / worst coll 9.7 vs the validation champion's
    0.375 worst-trim cross — a 1.8× robustness gain from genomes that already
    existed. Queue: 19e = extend run 17 (trim-rand, still climbing) +120
    gens; 19t = fresh seed-9 twin; both end with deep-evals + trim_eval +
    deploy_pick. Deploy candidate = best worst-trim genome from either.

## Live runs on the real rover

**Live runs 5–7 (09-18 ~18:46-18:52 MDT — note the rover's clock runs
several hours fast; times here are MDT, field session, 60 s each):**

| run | genome | scale | fwd/rev/pivot ticks | dist-wtd | still | ESTOPs | front min |
|---|---|---|---|---|---|---|---|
| 5 | champ15f_idx87 | 0.6 | 95 / 0 / 5 | 100% fwd | ~4% | 11 | 0.28 m |
| 6 | champ20m_idx115 | 0.6 | 0 / 43 / 61 | 100% rev | **56%** | 8 | 0.08 m |
| 7 | champ15f_idx87 | **0.8** | 90 / 0 / 10 | 100% fwd | **~4%** | **3** | 0.30 m |

- **FINDING: pivot-heavy genomes degrade on real hardware.** idx115
  (sim: best reverse, 0.833 cross) spent 56% of the 60 s physically still —
  its in-place pivots barely rotate on the real skid-steer (weak torque /
  friction vs the sim's instant yaw), and wall-facing pivots latch the
  monitor. Its sim score does not transfer. Sweeping-arc genomes (idx87)
  transfer cleanly.
- **Action scale 0.8 > 0.6 for idx87:** decisive arcs escape cul-de-sacs;
  fewer gate latches (3 vs 11) and more distance. Future forward runs at
  0.8.
- All three runs: zero observed contact, all ESTOPs self-recovered, safety
  chain (monitor + driver clamp) behaved identically to the sim's gate
  semantics.
- Deploy state: **champ15f_idx87 @ scale 0.8 = current best forward pilot.**
  idx115/idx95 (reverse, high-pivot) NOT field-recommended despite sim
  scores. Next decisive test needs a physical setup: rover ~1.5 m facing a
  doorway, 90 s run — count door crossings (rooms >= 2 events).

**Live run 4 (09-19 ~00:34) — `champ15f_idx87`, scale 0.6, 30 s — the
forward run.** Telemetry: 98% of ticks forward-commanded, 100% of
distance-weighted forward, 0% reverse (vs idx95's live run 3: 0% forward,
100% reverse). Mean cmd [-0.36,+0.95] = forward with a strong left-arc bias
(gyroscope yaw +1.0 to +1.3 rad/s — pirouetting forward). Lidar live, front
reads 0.4-1.3 m, one ESTOP front-block at t=28 s (monitor clamped forward,
exactly as designed), resumed next tick. Genome: mined from 15f population
(deploy_pick multi-draw): worst-trim cross 0.510, 0-3 collisions at every
trim x pose draw, rev 0.01, buildings deep-eval fit 0.402 rooms 1.62 coll
0.0. **Field tradeoff to remember: forward explorer = ~0.50 cross vs the
reverse explorer's 0.81.** No room-crossing observed in this short 30 s
run; next field test should place a doorway in view and run longer.

**Live run 3 (09-19 ~00:10) — `champ19e_idx95`:** behaved exactly as
evolved: 91% reverse ticks, 44% pivots, gate held it safe (user: "spun and
drove backwards"). Confirms sim->real behavior transfer; reverse is the
fitness-selected gait (front-watched gate + legal-blind reverse), not a bug.

## Live now: nothing (Spark idle since 09-18 18:33)

**Run 20m VERDICT (done 09-18 18:32): M-PARTIAL.** Plateau test B-A =
-0.034 (falling, warm pool already near basin top). Deploy pick (guard
23/128): idx115 = worst-trim cross 0.833 / coll 11.5 — beats both parent
picks (19e 0.823/11.9, 19t 0.667/5.7) on the cross axis with no collision
regression => M-PARTIAL holds; best reverse genome to date. But the merged
pool contained no forward genome (both parents were rev ~0.98 lineages), so
co-inheritance of *forward* exploring could not occur — forward quality
only comes from the w_rev lineage (15f). idx115 supersedes idx95 as the
best reverse deploy candidate; forward remains champ15f_idx87.

**Run 21 VERDICT (done 09-18 20:08): FAIL.** Best forward genome (rev
<= 0.15) reached only 0.510 worst-trim cross (idx103) — pre-registered
PASS needed >= 0.60, even the FAIL-line 0.56 wasn't reached. The ~0.5
forward ceiling is therefore STRUCTURAL for the reactive scan->wheels
policy class (matches explore_probe's scripted finding: no reactive
personality ever crossed a door reliably). Halving w_rev 0.3->0.15 did
buy honesty: 120/128 genomes pass the collision guard, winner worst-trim
coll 0.6 / 0.0-0.6 across every trim x pose draw, rev 0.02 — the cleanest
forward genome built so far (`champ21f_idx103`, committed). Plateau B-A
+0.049 = climbing but capped. DECISION per pre-registration: no more
reward knobs; forward-semantic improvement requires a structural change
(rear sensing channel / memory genome / blocked-direction memory), and
champ15f_idx87 @ scale 0.8 stays the forward pilot meanwhile. Field-test
idx103 vs idx87 at next rover session (expect equal cross rate, fewer
ESTOPs).

**Run 21 (launching):** `evo/run21_forward.sh` — forward-quality attempt:
15f recipe (w_rev) + trim-rand 0.75 + **halved penalty w_rev 0.15**, fresh
seed 8, 120 gens. Pre-registered: PASS if deploy_pick finds a genome with
worst-trim cross >= 0.60 AND rev <= 0.15 AND worst coll <= 15 (current
forward ceiling 0.510); FAIL if best forward genome stays < 0.56 cross
=> forward ceiling ~0.5 is structural for this policy class, accept
idx87 or reconsider rear sensing.

**Run 20m — merged-pool experiment** (`evo/run20_merged.sh`,
`greenfield20m_s10`, launched 09-18 16:57, ~2.2 h, watcher
proc_2935455f75c8): ES warm-started from `evo_runs/merged19_pool.npz` =
64 guard-passing genomes from 19e + 64 from 19t (`evo/merge_pools.py`).
Tests whether strong crossing (19e) and collision honesty (19t) co-inherit.
PRE-REGISTERED: M-PASS if deploy_pick at g119 finds worst-trim cross >=
0.85 with coll <= 12 (beats both parents 0.823/11.9 and 0.667/5.7);
M-PARTIAL = strict improvement on one axis, no regression on the other;
M-FAIL = deploy the existing verified finalists (below) and stop extending.

**Verified deploy finalists** (hardened multi-draw deploy_pick + trim_eval
re-checked at independent pose seeds 1/3/777 + buildings deep-eval at 5400
ticks; committed to `deploy/genomes/`):

| genome | worst-trim cross | worst-trim coll | rooms@nominal | deep-eval (32 unseen L3) |
|---|---|---|---|---|
| `champ19e_idx95.npz` | 0.823 | 11.9 | 3.97 | fit 0.606, 2.53 rooms, cross 0.812, coll 0.7 |
| `champ19t_idx16.npz` | 0.667 | 5.7 | 2.19 | fit 0.491, 1.97 rooms, cross 0.688, coll 0.9 |

Both: legacy gate (the real monitor's params), obs v1/action lr = directly
loadable by `evo/deploy/evo_runner.py`, reverse-driving ~0.98 (hardware-
proven style from live run 2). **First live test = champ19e_idx95** (strong
worst-case still well inside the collision envelope); fallback = idx16 if
the field shows any collision-braving behavior. Note the fitness formula
prices collisions at only coll/100 — champions can be crashers while the
population still hides honest genomes; deploy_pick exists because of this.

## Run 19 verdicts (see below) — kept for provenance

**Deploy picks (hardened, live):** multi-draw `deploy_pick` (3 independent
reset-pose draws per trim; guard = worst across draws) over both run-19
populations -> `evo_runs/deploy_pick_19{e,t}.log`, marker `picks.done`.
Rationale: run 19's single-draw pick (19e idx109) passed at worst-coll 11.8
but brute-forces 62-95 collisions on other pose draws — reset poses come
from the global RNG and one draw underestimates collision risk.

**Run 19 verdicts (both rc=0, done 09-18 14:00 / 16:25):**

`greenfield19e_s8` (extend run 17 s8 +120 gens, trim-rand 0.75):
- Plateau test B-A = +0.027 -> **AMBIGUOUS** (0.02-0.05 band; seed twin
  below is the pre-registered tiebreak).
- g119 hob elite cross **0.803**, val_best_fit 0.694, train rooms 3.2/game —
  strongest population so far.
- Deploy guard (val champion): worst/nominal rooms 0.66 PASS, max coll 11.4
  PASS. **Collision guards FAIL**: hob champ coll 52.8 (> 20), train elite
  coll 4.08 (> 3) — the champion is a brute-forcer; population mining is the
  way (and the single-draw pick was fooled, see above).
- trim_eval val champion: rooms 3.62-3.81 at 0.8-0.9 left trim (best
  absolute numbers of any lineage); drops to 2.38 @ 1.0/0.8.

`greenfield19t_s9` (fresh trim-rand twin):
- Pre-registered seed agreement: g119 hob elite cross 0.679 vs run 17's
  0.703 -> |diff| 0.024, **PASS**.
- Its own plateau window B-A = +0.19 -> **CLIMBING** — trim-rand MLP not
  converged; one more extension is justified.
- Collision guards PASS (hob champ 1.8, train elite 2.49). But the VAL
  champion brute-forces 105 coll at balanced 1.0/1.0 trim -> validation-set
  picking still rewards crashers; worst-trim population mining mandatory.
- Deploy pick (single-draw, pre-hardening): idx19 = worst cross 0.688,
  worst coll 4.0, trim_eval worst/nominal rooms 0.97, max coll 5.5 —
  **the most trim-honest genome measured so far**. Buildings deep-eval:
  fit 0.497, 2.00/6 rooms, cross 0.688, coll 1.6, rev 0.98.

Deploy status: 19t/deploy_genome.npz is already rover-grade by the numbers;
final pick waits on the hardened multi-draw ranking of both populations.

## Run 19 (superseded block — kept for provenance)

**Run 19 queue** (`evo/queue19.sh`, launched 2026-09-18 ~13:15, ~3 h total):

1. `greenfield19e_s8` (`evo/run19_extend.sh`) — extend run 17 s8 by 120 gens
   (warm `--seed-from` + `--keep-seed-sigma`, trim-rand 0.75, everything else
   identical). Run 17 ended cross 0.703 still climbing, so this is the
   strongest candidate lineage. Pre-registered: plateau test over local gens
   45–59 vs 105–119 (PLATEAU < 0.02 → memory genome; CLIMBING ≥ 0.05 →
   extend/accept); deploy guard trim_eval worst/nominal rooms ≥ 0.60 and
   max coll ≤ 20; collision guards as runs 11–18.
2. `greenfield19t_s9` (`evo/run19_twin.sh`) — fresh seed-9 trim-rand twin of
   run 17. Pre-registered: hob_elite_cross(g119) within 0.15 of 0.703
   (seed agreement); same deploy guard.

Both chains end with legacy+buildings deep-evals, `trim_eval`, and
`deploy_pick` (writes `deploy_candidates.json` + `deploy_genome.npz`,
hardware-compatible npz for `evo/deploy/evo_runner.py`). Selection is
worst-trim door-cross rate with a hard per-trim collision guard — the real
rover's track factors are unknown, so worst case, not nominal, decides.

## Run 18 — doorway forward gate (complete 2026-09-18 11:32, VERDICT: FAIL)

(`evo/run18_variant.sh`, `evo_runs/greenfield18d_s8`, Spark, launched
2026-09-18 ~09:55, rc=0): run 15f's config (fresh seed 8, `--w-rev 0.3`,
legacy trims) with one change, **`--gate doorway`** — a sim-only forward-gate
prototype (`arena.GATE_PRESETS`):

| Rule | Real rover today | Doorway prototype |
|---|---|---|
| Corridor half-width | 0.12 m (narrower than the 0.14 m robot radius) | 0.14 m (= robot radius) |
| Front stop | 0.15 m from bumper | 0.12 m (0.18 m from centre, 4 cm clear of the body) |
| Release hysteresis | 0.10 m | 0.05 m |
| Hold after a block | 0.3 s | 0.15 s |
| Side-equalization | inside 0.15 m | inside 0.10 m |

**Pre-registered** (in the script): H18a — forward exploring competitive:
building-holdout elite cross rate ≥ 0.770 **and** reverse fraction ≤ 0.30 at
gen 119; H18b — a clear improvement: cross ≥ 0.633 (1.3× run 15f's 0.487)
with reverse ≤ 0.30. Collision guard as before. Deep-evals run on the **legacy**
gate (the real rover's) and `evo.gate_rescore` compares both gates.

**RESULT — H18a FAIL, H18b FAIL.** g119 `hob_elite_cross_rate` 0.524 (need
0.770 / 0.633); reverse 0.014 ✓ — the penalty did make it drive forward, but
forward exploring still plateaus at run-15f level (0.487 → 0.524 = +8%, noise
at this seed spread). Cross rate climbed 0.057 → ~0.50 by gen 60 then went
flat (hob champion stuck at exactly 0.500 from gen 60 on; rooms/game 1.65).
Collision guard PASS (elite train coll 1.38, hob champ coll 0.0). Legacy-gate
deep-eval of the validation champion: fit 0.417 rooms 1.16 (vs scripted 0.36);
buildings deep-eval rooms 1.56/6, cross 0.469, coll 0.0. Gate rescore: legacy
vs doorway nearly identical (fit 0.410/0.413, cross 0.500/0.500) — the champion
is not gate-limited either way.

**Conclusion:** the forward safety rules are NOT the doorway bottleneck.
Across three attempts the forward habit caps at cross ~0.30–0.52 (15f legacy
+ w_rev 0.487, 16sr symmetric + w_rev 0.298, 18d doorway + w_rev 0.524) while
reverse arms reach 0.86–0.89 with equal ease. The forward/reverse gap is a
property of the policy class + sensing geometry (front-lidar arc prediction
against a forward-moving body), not of gate parameterization. **Do not change
the real `lidar_safety_monitor`.** If forward exploring is wanted, the lever
is in the genome/observation (e.g. rear sensing or a memory of blocked
directions), not the gate.

Re-scoring existing champions under the doorway gate changed almost nothing
(each is adapted to the gate it evolved under), so only evolution under it can
answer whether the forward rules are the doorway bottleneck. The real rover's
monitor is unchanged — any change there is the user's decision.

## Runs 16–17 — verdicts (2026-09-18)

**Run 16 — symmetric safety gate** (`--gate symmetric`, sim only: every forward
rule mirrored for reverse — rear corridor with reverse-arc prediction, same
0.15 m stop, hold/hysteresis latch, rear-flank 90–150° equalization). Fresh
seed 8, 120 gens. **All three pre-registered checks FAIL:**

| Arm | Reverse fraction (g119) | Holdout cross rate | Needed |
|---|---|---|---|
| 16s: symmetric, no penalty | 0.985 (H16a: < 0.50) | 0.488 (H16b: ≥ 0.770) | ✗ ✗ |
| 16sr: symmetric + `--w-rev 0.3` | 0.001 ✓ | 0.298 (H16c: ≥ 0.770) | ✗ |
| run 13 s8 (legacy gate, same start population) | reverse | 0.856 | reference |

- The reverse habit is **not** caused by the gate asymmetry: it is present from
  gen 0 (the best random genomes already reverse 57% of the time) and survives a
  symmetric gate.
- Mirroring the forward rules to the rear makes doorways hard in **both**
  directions (0.49 vs 0.86). **Recommendation: do not put the symmetric rules on
  the real monitor.**

**Gate diagnostics** (`evo/mirror_test.py`, `evo/gate_ablate.py`,
`evo/gate_rescore.py`): with the track trims swapped along with the frame, the
run-14 reverse champion driven *forward* keeps 3.56 of its 4.31 rooms, so
forward driving is not inherently hard. The first (uncorrected) mirror test's
862 collisions came from the trim flip. Relaxing any single forward rule (hold,
hysteresis, side-equalization) changes nothing for an evolved forward
champion; removing the arc prediction hurts.

**Run 17 — track-trim randomization** (`--trim-rand 0.75`: each generation,
each house column draws independent effective track factors ~ U(0.75, 1.0);
holdouts stay at the nominal 0.8/1.0). Legacy gate, fresh seed 8, 120 gens.

| Tracks (L/R) | Run-14 champion (no randomization) | Run-17 champion |
|---|---|---|
| 0.8 / 1.0 (sim nominal) | 4.28 rooms, 4 collisions | 3.16 rooms, 9 collisions |
| **1.0 / 1.0 (balanced)** | 1.78 rooms, **620 collisions** | 2.38 rooms, **9 collisions** |
| 1.0 / 0.8 | 1.38 rooms, 275 collisions | 1.72 rooms, 1 collision |
| worst / nominal rooms | 0.32 | 0.54 |

- **H17a FAIL** on rooms (worst/nominal 0.54, needed ≥ 0.80) but the collision
  part passes: no trim setting exceeds 10.4 collisions.
- **H17b FAIL:** g119 holdout cross rate 0.703 (needed ≥ 0.770) — still
  climbing (+0.12 over the last 20 gens), so it likely needs more generations.
- Champion on 32 unseen buildings: 2.56 rooms, cross 0.75, 2.4 collisions,
  reverse 0.97.
- **Why it matters:** the sim applies the 0.8 left trim as a physical 20%
  slowdown (a "straight" command curves 34° in 3 s), while the real rover's
  0.8 is a motor-driver correction for a faster left track. In live run 2 the
  rover curved the opposite way to the sim's prediction. **The run-14 champion
  is not safe to trust on balanced tracks; prefer trim-randomized champions for
  hardware.** `python3 -m evo.trim_eval --genome …` checks any genome.

## Runs 13–15 and the real rover (2026-09-17/18)

- **Run 13** (base genome, 120 gens, 32 houses/genome): **CLIMBING** on both seeds
  (late-window gain +0.053 s8 / +0.221 s9); cross rate 0.10 → 0.86 / 0.81.
- **Run 14** (+120 gens warm-start): plateau test **split** (s8 +0.0195 PLATEAU,
  s9 +0.0234 AMBIGUOUS) — cross rate saturates near 0.88–0.89. **Best genome:
  `greenfield14_s9/val_best_genome.npz`** — 32 unseen level-3 buildings: fitness
  0.767, 3.81/6 rooms, cross 0.875, 2.8 collisions; legacy 0.604 (scripted
  0.26 / ~1.1 rooms).
- **Real rover** (`evo/deploy/`, `6219f90`): numpy runner (4e-7 parity with the
  sim policy) + hardware-only launch (motor driver, STL-19P lidar, **BNO085** IMU —
  the LSM9DS1 is not fitted — and `lidar_safety_monitor` with parameters
  identical to the sim gate). Dry run by default; `--drive` only after the user
  confirms, HARDSTOP (:8090) open. Lessons: `/emergency_stop` = front block only
  (never zero on it — run 1 got stuck); the **right wheel encoder reads 0 while
  moving** → wheel proprio from the sim motor model. **Live run 2** (scale 0.6,
  20 s): ~1.5 m, 75% reversing / 23% pivots — the sim policy's style — no front
  blocks, gate intervened on 17% of ticks.
- **Run 15** (`--w-rev 0.3` reverse penalty): **both arms FAIL** — fresh drives
  forward but reaches only cross 0.487 (1.6 rooms); warm-from-14 keeps reversing
  (0.98) and pays the penalty.
- **Gate diagnostics** (`evo/mirror_test.py`, `evo/gate_ablate.py`): with track
  trims swapped correctly, the reverse champion driven **forward** keeps 3.56 of
  4.31 rooms — the legacy gate's front-heavy asymmetry is worth only ~15–20% to
  reversing. No single forward rule matters alone; arc prediction helps. The
  first mirror test's 862 collisions were a trim artifact, which also shows the
  champions are **extremely trim-sensitive** — a sim-to-real risk.
- Real gate geometry: side sectors cover only the front flanks (30–90°); the rear
  cone starts at 150° with a 0.30 m stop; the rear flanks (90–150°) are unwatched.

## Run 12 family — verdicts (7 runs, done 2026-09-17 11:36)

Bundle split on buildings (direct level 3, 40 gens, seeds 8/9, validation-set
champion). Primary metric `hob_elite_cross_rate` at gen 39; controls: seed 8
`greenfield11c` 0.410, seed 9 `greenfield12_ctl_s9` 0.510 (mean 0.460).
Summary: `evo_runs/run12_family_summary.json`.

| Change | Seed 8 | Seed 9 | Mean | vs control | Verdict |
|---|---|---|---|---|---|
| `--obs v2` | 0.453 | 0.436 | 0.445 | 0.97× | NEUTRAL |
| `--action vw` | 0.496 | 0.490 | 0.493 | 1.07× | NEUTRAL (needed 1.1× **and** both seeds over control) |
| `--hidden 64` | 0.484 | 0.438 | 0.461 | 1.00× | NEUTRAL |

- **The base genome stands** (obs v1, action lr, h128); run 10's bundle PASS was
  noise.
- **`--obs v2` has a downside:** its champion barely moves (7–8 m vs ~15 m) and
  took 12.3 collisions in the building deep-eval; every other champion 0–1.
- **Weak power:** the two controls differ by 0.10 while the effects are
  0.03–0.08 — hence run 13's 32 houses per genome.
- **Validation champion works:** the legacy transfer guard now passes **6 of 7**
  runs (run 11: 1 of 4).
- **Best genome to date** — `greenfield12_obs_s9` validation champion, on 32
  unseen level-3 buildings: fitness 0.3827, rooms 1.59, cross rate **0.500**,
  **0 collisions**; legacy 0.416. Best scripted: 0.272 buildings / 0.355 legacy.
- Spark (graph) and V620 (eager) agree: seed-9 control 0.510 vs 0.523.

## Run 11 family — verdicts (all complete; scored 2026-09-17 09:15)

Config: 9b genome, buildings, `w_coll 1.0`, 40 gens. Seed 8 on the Spark
(graph), seed 9 on the V620 (eager). Summary JSON:
`evo_runs/run11_family_summary.json` on the Spark.

| Criterion | 11 (s8, curriculum) | 11c (s8, direct) | 11_s9 (curriculum) | 11c_s9 (direct) |
|---|---|---|---|---|
| Level reached ≥ 2 | ✓ L3 (gens 3/12/24) | n/a | ✗ stuck at L1 from gen 3 | n/a |
| Building-holdout elite cross rate g0 → g39 | 0.082 → 0.412 | 0.090 → 0.410 | 0.076 → 0.510 | 0.090 → 0.523 |
| Curriculum ≥ direct (same seed) | ✓ (tie) | — | ✗ | — |
| Collision guard (hob champ ≤ 20, elite train ≤ 3) | ✓ 2.2 / 2.4 | ✓ 0.9 / 0.1 | ✓ 0.2 / 0.0 | ✓ 0.0 / 0.0 |
| Legacy deep-eval `best_genome` ≥ 0.37 | ✗ 0.140 (90.8 coll) | ✗ 0.357 | ✗ 0.325 | ✓ 0.375 |

- **Buildings training works:** elite cross rate on unseen level-3 buildings
  rose 5–7× in every run.
- **Curriculum: no benefit.** Mean over seeds 0.461 curriculum vs 0.467 direct
  (0.99×; "helps" needed 1.2×). Run 11 passes its own criteria (tie with 11c);
  11_s9 fails (its cross rate hovered around the 0.5 advance gate).
- **Collision guard passes in all four** — `w_coll 1.0` fixed the holdout
  brute-forcers.
- **Transfer guard fails 3/4 because of the champion pick**, not the
  population: `best_genome` was the train-fitness argmax on one generation's
  random 64 houses. Population best on legacy houses: 0.39–0.40 in all four.
  Fixed in `5014920`: champion = best on a fixed validation set
  (`val_best_genome.npz`, seed 779000).
- **Building deep-eval (32 unseen level-3 houses):** `best_genome` crosses
  rooms in 31–59% of games vs ≤ 16% for the best scripted controller. Run 11's
  genome: 2.09 rooms, cross 0.594, 1.5 collisions, fitness 0.459 vs 0.272.

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
| `5014920` | validation-set champion (`val_best_genome.npz`, seed 779000) |
| `6219f90` | `evo/deploy/`: numpy rover runner + hardware-only launch |
| `a8c9941` | `--w-rev` reverse penalty; forward/reverse travel metrics (`rev` columns) |
| `51229de` | `--gate symmetric`; `evo/mirror_test.py`, `evo/gate_ablate.py` |
| `f3dfee3` | `--trim-rand` track-speed randomization (`BatchedEnv.step` optional per-env trims) |
| `1d0af62` | `evo/trim_eval.py` trim-robustness check |
| `a7ff369` | `--gate doorway` prototype (`GATE_PRESETS`, `GateConfig.side_stop_distance`), `evo/gate_rescore.py` |

All sim options default off and were verified bit-identical to the previous
code at their defaults.

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
- **Before any hardware run:** `evo.trim_eval` on the genome (collisions at
  balanced tracks) and `evo.gate_rescore` on the legacy gate (the real monitor).

## Next actions, in priority order

1. **Extend run 17** (trim-randomized, still climbing) — the best candidate for
   hardware because it is collision-robust across track asymmetries.
2. **Combine** what works: trim randomization + forward penalty (the doorway
   gate is dead — run 18 FAIL), then a longer run and two seeds.
3. **Measure the real rover's track asymmetry** (drive a straight command on
   the floor, compare the gyro to the sim's prediction) and narrow the trim
   randomization range around it.
4. More live runs only with a trim-robust champion, longer and at full scale.
5. Cross rate saturates near 0.9: prefer rooms-visited as the primary metric
   for future plateau tests.
6. `bc_seed.py`'s scripted expert still has a noise-like pivot direction; fix
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
`test_cells.py`; diagnostics: `trim_eval.py`, `gate_rescore.py`,
`mirror_test.py`, `gate_ablate.py`; deploy: `deploy/evo_runner.py`,
`deploy/evo_hw.launch.py`; run scripts: `run9b.sh`, `run11_variant.sh`, `queue11.sh`
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
