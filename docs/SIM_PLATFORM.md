# SIM_PLATFORM — the rover sim as a clean building block for model & strategy ideas

**Draft for review — 2026-10-04.** No code moves until signed off.
Status: design; supersedes nothing (the `ros2-rover-revival` skill and `REVIVAL_PLAN.md`
remain the operational context).

## 0. Mission

Four directives from this session:

1. **Model plug-in API is the centerpiece.** Testing a new model idea must cost a small
   adapter file plus a documented contract — not a new harness lineage.
2. **Best possible sim for playing with more evo strategies.** The evaluation engine
   becomes reusable; evolutionary strategies become pluggable.
3. **Hardware-true geometry from the ROS files.** Rover shape/size from
   `src/tractor_bringup/urdf/tractor.urdf.xacro` + driver/monitor configs — not an
   idealized collision circle.
4. **Lidar + RGB camera as model inputs.** A camera obs channel, first-class in the
   obs spec, so multimodal models can be tested.

Constraints: **structure first** (fidelity upgrades after, validated one at a time);
**build alongside** (freeze legacy scripts, migrate one customer at a time with
equivalence tests); **ES genomes first-class in v1**.

## 1. Where the sim is today

### Inventory

| Piece | Lives in | Used by |
|---|---|---|
| numpy single-env sim (`World`/`make_house`, `SimRover`, `SimSafetyGate`) | `pnn_sim/{world,rover,safety_gate}.py` | nomad replay, deploy sanity, parity refs |
| batched torch sim (`BatchedEnv`, `BatchedGate`, `batched_preprocess`) | `pnn_sim/batched/env.py` | Arena, PCN lineage |
| building generator (families, levels, rooms, doors, reachability) | `pnn_sim/buildings.py` | evo pools, PCN |
| fast paths (`Fp32/Fp16/FusedEnv`, `NoSyncGate`, gate presets) | `evo/arena.py` | ES runs |
| fused Triton raycast | `evo/fused_scan.py` | ES runs |
| rollout engine (`Arena`: P×G layout, CUDA graphs, metrics, rooms/doors/cells, trim-rand, `load_worlds`) | `evo/arena.py` | ES runs |
| genome + memory (`PopulationNet`, packing, meta) | `evo/policy.py` | ES runs, deploy |
| ES loop (GA + OES, curriculum, warm start, BC seed) | `evo/evolve.py` | training |
| world pools/seeds (`build_pool`, `HOLDOUT_SEED`, `VAL_SEED`) | `evo/worlds.py` | ES runs, scorers |
| scoring/acceptance tools | `evo/{baselines,deploy_pick,trim_eval,orbit_audit,tick_rankcorr,merge_pools,...}.py` | humans |
| deployment twin | `evo/deploy/evo_runner.py` | rover |
| PCN lineage | `pnn_sim/batched/{model,trainer,place}.py`, `headless_runner.py`, `gate_tests.py`, `tools/` | PCN |

### Friction (why a new model idea costs a new harness)

1. **Three integration patterns, no shared interface.** ES: closure
   `policy_step(obs, prev_act)` with obs assembly hardcoded inside
   `Arena._rollout_body`. PCN: own trainer + own obs pipeline + own eval. NoMaD:
   bespoke single-env loop, own renderer, own metrics, own cadence (10 Hz vs the
   arena's 15 Hz).
2. **The CUDA-graph contract is tribal knowledge** (comments): static buffers, zero
   allocation in capture, no CPU sync, one live graph per process, reset hooks,
   default-RNG routing. Every integration re-derives it — or breaks in ways that
   cost runs (Xid 43, allocator asserts, generator registration).
3. **Scoring is fragmented** — five scripts + the arena metrics dict + PCN tables +
   nomad's `Episode`; no single comparable report across model families.
4. **World sources are fragmented** — `make_house` vs building pools vs ad-hoc
   loops; the val/holdout discipline exists only inside `evo`.
5. **The sim core is split across packages** — physics in `pnn_sim`, fast paths and
   metric accumulators in `evo/arena.py`. "The sim" cannot be imported as one thing.

## 2. Target architecture

```
rover_sim/
  core/            # one source of truth; no training code
    world.py       # World, make_house                       (moved from pnn_sim/world.py)
    buildings.py   # Building families + levels + rooms       (moved from pnn_sim/buildings.py)
    rover.py       # RoverConfig, SimRover                    (moved from pnn_sim/rover.py)
    gate.py        # GateConfig, SimSafetyGate                (moved from pnn_sim/safety_gate.py)
    batched.py     # BatchedEnv, BatchedGate, batched_preprocess  (moved from pnn_sim/batched/env.py)
    fast.py        # Fp32/Fp16/Fused envs, NoSyncGate, presets   (moved from evo/arena.py)
    rayscan.py     # fused Triton scan                        (moved from evo/fused_scan.py)
    geom.py        # BodyConfig, footprint collision, sensor mounts   (NEW — stage 4)
    render.py      # egocentric camera renderer                (NEW — stage 4)
  runner/
    engine.py      # RolloutEngine — Arena generalized (capture, metrics, buffers)
    obs.py         # ObsSpec + assembler
    policy.py      # Policy protocol + graph-safe helpers
  policies/
    es_genome.py   # PopulationNet + packing + meta           (moved from evo/policy.py)
  strategies/
    registry.py    # name -> strategy
    ga.py          # truncation GA + immigrants               (from evolve.py reproduce())
    oes.py         # OpenAI-ES (mirrored + rank + Adam)       (from evolve.py OES)
  adapters/
    es.py          # native ES adapter (thin)
    # later: pcn.py, nomad.py, scripted.py
  suites.py        # named world suites + pools/seeds          (from evo/worlds.py)
  evaluate.py      # evaluate(population, suite) — the shared primitive
  scoring.py       # fitness(), rev_frac(), metric semantics   (moved from evo/arena.py)
  bench.py         # report + baselines + acceptance gates      (stage 5)
  smoke.py         # 1-command sanity loop for new ideas
  tests/           # contract, equivalence, geometry, renderer
```

Principles:

- **One source of truth.** Physics, gate, worlds, obs, metrics live in `rover_sim`
  only. `evo/` becomes training scripts + scorers; `pnn_sim/` keeps the PCN brain
  lineage.
- **Frozen legacy via shims.** Old modules (`pnn_sim/world.py`, `pnn_sim/batched/env.py`,
  `evo/arena.py`, `evo/policy.py`, `evo/worlds.py`, `evo/fused_scan.py`, …) become
  explicit re-export shims, so every existing script and the PCN lineage keep
  importing unchanged. Shims die only after all consumers migrate.
- **Zero behavior change until stage 4.** Stages 1–3 are relocation + one interface.
  Physics, gate logic, obs layout (v1/v2), metrics semantics, seeds, fitness weights
  do not change. Equivalence bar: coarse metrics bit-identical, continuous metrics
  within the known replay-jitter tolerance.
- **The contract is enforced by tests**, not comments: graph-safety checker, capture
  smoke, equivalence suite, geometry validation.

## 3. The model plug-in contract (centerpiece)

### Policy protocol

```python
class Policy(Protocol):
    obs_spec: ObsSpec                  # which channels this policy consumes
    graph_safe: bool = False           # eligible for CUDA-graph capture
    def bind(self, P: int, G: int): ...      # allocate views/state for P×G envs
    def reset(self): ...                     # between games (installable as reset hook)
    def step(self, obs, prev_act): ...       # -> raw cmd [B, 2]
```

Runner responsibilities (so adapters don't re-implement them): obs assembly per
spec; device/graph mode selection + eager fallback; prev_act buffer; reset
orchestration; metric accumulation; optional recording.

### ObsSpec

| channel | shape | status |
|---|---|---|
| `scan_bins` | [B, 72] | existing (`batched_preprocess`) |
| `scan_raw` | [B, n_beams] | existing (optional) |
| `proprio` | [B, 8] `v1` \| `v2` | existing |
| `prev_act` | [B, 2] | existing |
| `camera` | [B, H, W, C] uint8 | NEW (stage 4) |
| `place` | — | future (PCN) |

obs delivered as a **dict of channels**; a `flat()` helper concatenates for
MLP-style models — byte-compatible with today's 82-dim layout (scan72 + proprio8 +
prev_act2). Channel set is versioned; adding a channel never changes an existing
policy's layout.

### Graph-safety checklist (the rules that cost runs to learn)

1. Parameters flow through a static buffer (`copy_` outside capture).
2. No fresh allocations inside the captured region — preallocate; use `out=`/in-place ops.
3. No host↔device sync in `step()` — no `.item()`, `int(tensor)`, `.cpu()`, no
   data-dependent Python control flow.
4. Randomness only via the default CUDA generator (or none).
5. Recurrent state updated in place (`copy_`/`zero_`), never rebound.
6. State zeroed between games (`reset()`; install as reset hook for hidden/memory state).
7. One live CUDA graph per process; never recapture in-process; `drop_graphs()` before teardown.
8. Fixed shapes for the engine's lifetime (`dynamic=False`).

Contract tests (stage 2): `check_graph_safe(policy)` — capture in a sandbox, replay
twice, assert equality; allocation counter during capture; capture failure always
falls back to eager with one log line (existing behavior, preserved).

### Determinism & pairing

- One shared noise stream per game (`manual_seed(noise_seed)` at reset) — every
  individual sees identical noise (paired comparison across the population).
- Coarse metrics (rooms, collisions, door counts) are replay-stable; continuous
  metrics (dist, cells) jitter across replays by a small known amount — tests
  tolerance-check them, never bit-check. (Documented lesson; keep it true.)

### Adapters

- **Native ES** (`adapters/es.py`, ~30 lines): wraps `ESGenomePolicy` (the moved
  `PopulationNet`) — views bound to a static theta buffer, hidden/memory zeroing as
  reset hooks, `step` = bmm + `copy_`. This is today's `GraphRunner` minus the
  bespoke parts.
- **External sketch (NoMaD, later):** ~60 lines — renderer → model → selector →
  cmd, single-env numpy mode. Proves the interface serves non-ES families; not in
  v1 scope.

## 4. Hardware truth — geometry & sensors from the ROS files

### As-built (sources: `urdf/tractor.urdf.xacro`, `hiwonder_motor_driver.py`,
`lidar_safety_monitor.py`, `evo/deploy/evo_hw.launch.py`, `config/realsense_config.yaml`)

| item | value | source |
|---|---|---|
| chassis | 0.2667 × 0.102 × 0.058 m | URDF base_link |
| tracks (×2) | 0.2667 × 0.042 × 0.090 m, centers ±0.072 m | URDF |
| **plan-view footprint** | **0.2667 × 0.186 m** (outer track edges ±0.093) | derived |
| wheel separation | driver 0.154 / URDF 0.144 | driver p42 / URDF |
| wheel radius | driver 0.025 / URDF 0.0485 | driver p44 / URDF |
| encoders / rates | 1980 PPR; cmd throttle 0.1 s (10 Hz); cmd_vel timeout 1.0 s | driver |
| lidar (LD19) | mount x=+0.07635 m (0.057 behind bumper), z=0.3945 m; CCW, beam 0 fwd | URDF laser_joint |
| lidar beams | field bags 2026-09-10: 482 beams @ ~10 Hz; sim today: 360 @ 12 m | bag_place_replay notes |
| camera (D435i) | mount x=+0.12085, z=0.3615 m; front flush; hFOV 1.396 rad (~80°); color 640×480; clip 6 m | URDF camera_joint; realsense_config |
| IMU | BNO085 fitted (URDF link is the older LSM9DS1 — not fitted) | field notes |
| gate | front 0.15/rear 0.30 (evo deploy) vs monitor defaults 0.20/0.40; offset 0.06; half-width 0.12; sides 30–90°; rear ≥150° | monitor + evo_hw.launch |

### Sim ↔ real deltas to close (stage 4 — all opt-in modes, legacy defaults kept)

| delta | today | hardware-true |
|---|---|---|
| collision shape | circle r=0.14 | rect 0.2667 × 0.186 (perimeter-sampled, batched) |
| lidar origin | robot center | lidar mount (x=+0.07635) |
| lidar beams | 360 | 482 (changes binning; opt-in) |
| camera | none | renderer channel |
| control/motor rates | single 15 Hz | multi-rate (fidelity backlog) |

### Geometry design (`core/geom.py`)

- `BodyConfig(shape="circle"|"rect", radius | (length, width), collision_samples)`.
- Collision: perimeter point sampling (corners + edge samples) against wall
  segments, reusing the existing batched point-clearance kernel; sample count
  configurable; validated against exact segment–rectangle distance on random poses
  (the test fixes the sample count and states the max error).
- Sensor mounts as config (lidar/camera pose); raycasts and renders originate from
  the true mount.
- Gate presets named: `sim_evo` (0.15/0.30 — current, default) and
  `monitor_default` (0.20/0.40); single config source shared by sim + deploy launch.

### Camera channel (`core/render.py`)

- Egocentric renderer: one ray fan per env (reuse scan machinery), walls as
  range-shaded bands (NoMaD-style), configurable H×W (default 96×96), grayscale or
  RGB, optional noise/dropout; batched torch + numpy twin.
- Cost: ~96 rays/env vs the 360-beam scan — small relative to the scan; render at
  policy rate, not every sim tick.
- Model-side: camera input is a search-space question for ES (conv stem vs
  downsampled luma vs separate vision genome). The platform provides the channel
  and benches it — architecture experiments are the point.

## 5. Evo strategies on the platform

- **Shared primitive:** `evaluate(population [P,N], policy_factory, suite, ticks) ->
  (fitness[P], metrics)` — what `evo.evaluate`/`GraphRunner` do today, generalized.
  Any strategy = `ask()`/`tell()` around it.
- **Registry:** `strategies/` with `ga` (truncation GA + immigrants) and `oes`
  (OpenAI-ES, mirrored + rank + Adam) ported byte-for-byte; future candidates:
  sep-CMA-ES, PGPE, MAP-Elites-style archives. New strategy = one file + registry
  entry.
- **Smoke tool:** `python3 -m rover_sim.smoke` — small-pop, few-gen, subset-worlds
  run printing the standard table; the fast loop for "does this idea behave" before
  a real run.
- **Comparability discipline (kept):** pre-registered criteria per run; same
  suites/metrics across strategies; val vs holdout separation enforced by the suite
  registry.

## 6. Migration stages (each verified before the next)

Implementation workflow: one scoped subagent per stage; the parent verifies at each
gate before dispatching the next.

**Stage 1 — core extraction (relocation only).**
Move the modules in §2 (fast classes + helpers out of `evo/arena.py`, sim modules
out of `pnn_sim`); add explicit re-export shims; run the full existing test battery
+ a fixed-seed 3-gen smoke; require identical numbers. No interface changes.

**Stage 2 — runner + obs + policy protocol.**
Generalize `Arena` into `RolloutEngine` (keep `Arena` as alias); obs assembly via
`ObsSpec`; `Policy` protocol; contract tests (graph-safety, capture, equivalence vs
legacy Arena on identical configs).

**Stage 3 — ES adapter + strategies.**
Move `PopulationNet` → `policies/es_genome.py`; add `adapters/es.py`; port GA/OES
into `strategies/`; rewire `evolve.py` CLI (flags unchanged); equivalence smoke:
3-gen run matches legacy within tolerance.

**Stage 4 — hardware-true geometry + camera.**
`geom.py` (rect footprint mode, mounts), `render.py` + camera obs channel; all
opt-in, legacy defaults unchanged; validation tests (geometry error bound; renderer
parity; cost bench at B=8192).

**Stage 5 — suites + bench.**
Registry (legacy/buildings/val/holdout); `bench.py` report + scripted baselines +
acceptance gates (worst-trim multi-draw pick, trim envelope, pivot cap) as
functions; migrate scoring tools one at a time.

Each stage: goal → moves → tests → smoke → rollback (git revert; shims make
rollback trivial until stage 5).

## 7. Verification methodology

- **Tier 1 (unit):** existing suites — `evo.test_buildings`, `evo.test_policy`,
  `evo.test_reproduce`, `pnn_sim.batched.verify` (CPU); `evo.test_cells --device
  cuda` (Spark). New: contract tests, geometry tests, renderer parity.
- **Tier 2 (equivalence):** same seed/config old-vs-new; coarse metrics exact,
  continuous within replay-jitter tolerance; capture smoke (capture_fail=0) on
  Spark.
- **Tier 3 (smoke runs):** fixed-seed 3-gen smoke before/after each stage; compare
  `gens.jsonl` (`fit_med`, `elite_cells`, cross rates).
- Spark: `ssh benson@172.0.0.201`, `source /home/benson/venv/bin/activate`, rsync
  code (exclude build/install/log).

## 8. Fidelity backlog (after structure; one change at a time, each validated)

1. **Multi-rate clocks** (lidar 10 Hz / control 15–30 / motor throttle 10 Hz) — the
   historical clock-mismatch failure; policy acts on stale obs in sim the same way
   it would on hardware.
2. **Skid-steer pivot model** (loaded track stalls; live-run lesson) — opt-in
   dynamics mode.
3. **Wheel proprio variants** (right-encoder-dead / model-based) for deploy parity.
4. **Beam count 482 + raycast from lidar mount** (hardware-true sensing).
5. **Trim semantics** (driver-side correction vs physical slowdown).
6. **Renderer upgrades** (textures/depth realism) — only if vision models show promise.

## 9. Risks & open questions

- **Graph pitfalls in new code paths** — mitigated by contract tests + capture smoke
  per stage; never recapture in-process; one live graph.
- **Camera cost at B=8192** — measure in stage 4; render at policy rate; low-res default.
- **Genome size with camera input** — model-side decision; platform provides the channel.
- **Config conflicts to resolve:** wheel_sep 0.144 (URDF) vs 0.154 (driver); gate
  0.15/0.30 (evo deploy) vs 0.20/0.40 (monitor defaults). Recommend: keep the
  sim+deploy pair consistent, expose both as presets, document the choice.
- **Package name** `rover_sim` — free; alternatives welcome.
- **Doc status:** this doc is the review artifact; code moves start only after sign-off.
