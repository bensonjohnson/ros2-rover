# Headless sim training for the PC active-inference brain

Trains the `pc_active_inference` brain (the `start_pc_brain_rover.sh awake`
stack) in a pure-Python 2D simulator — **no ROS, no Gazebo, no build step**.
The brain modules are imported unmodified from `src/tractor_bringup`, so a
checkpoint trained here is byte-compatible with the rover's
`~/.ros/pnn_brain.pt`. (Not related to `sim/`, the old Webots/SAC package.)

## What is simulated

- Procedural houses: outer shell, internal walls with doorways, furniture
  boxes (wall segments + exact ray-segment lidar, no grid artifacts).
- The rover's embodiment: cmd ±1 → 0.2 m/s per track, driver deadband,
  left-track trim 0.8, first-order motor lag, multiplicative track slip,
  collision stall.
- The full sensor diet: 360-beam lidar with noise/dropouts → the same
  72-bin preprocessing, wheel rad/s, gyro (yaw real, roll/pitch noise),
  accelerometer (gravity + body accel).
- The **safety gate**, ported from `lidar_safety_monitor.py` (arc-shifted
  front corridor, phantom suppression, hysteresis, hold time, rear/side
  logic) — the brain's interoceptive "hold" channel and its collision
  avoidance are learned from the gate's firings, so this is load-bearing.
- World switches reuse the lift-detection reset path (place memory cleared,
  slow-layer context dropped, weights kept).

Deliberately absent: teleop, dashboard, lift detection (no hands).

## Running (DGX Spark or any machine with python3 + numpy + torch)

```bash
# from the repo root
python3 -m pnn_sim.train_sim --ticks 650000 --workers 16
```

- `--ticks 650000` ≈ 12 sim-hours at the 15 Hz control rate.
- `--workers N`: N independent lifetimes (own world stream, seed, output
  under `sim_out/worker_NN/`). One per core; they don't share weights —
  this is for hyperparameter sweeps / picking the best-settled brain.
- `--switch-world-every 27000`: new house every ~30 sim-minutes (default).
- `--load ~/.ros/pnn_brain.pt`: continue from the rover's brain.
- `--device cuda`: runs the brain on GPU. Measured **slower** than CPU for
  the per-tick ≤82-dim tensors (kernel-launch overhead dominates); for
  single-stream, CPU + workers is the throughput path. To use a GPU
  properly, see batched mode below.
- `--freeze`: evaluate without learning.

## Batched mode — one brain, B worlds (the GPU path)

```bash
python3 -m pnn_sim.train_batched --envs 256 --ticks 100000   # --device cuda default
```

`pnn_sim/batched/` reimplements the world model, EFE actor, and slow layer
with a batch dimension: **shared weights, per-env recurrent state and
behavior, local PC updates averaged over the batch each tick**. Same
stationary point as the reference rules, a B-sample gradient estimate
instead of 1 — and tensors wide enough that the GPU finally pays.

- The whole pipeline is device-resident: raycast, scan binning, safety
  gate, place memory, and brain all run as batched tensor math on the GPU
  (plain CUDA/HIP arithmetic — RT cores are not involved or needed).
- Measured (AMD V620, ROCm): ~22,500 env-ticks/s at `--envs 8192`
  (~1,500× realtime into ONE brain), ~11,700 at 256. Throughput keeps
  scaling with batch size; big-B/short-tick runs (e.g. `--envs 8192
  --ticks 1200`) are a legitimate regime — few, very low-variance updates.
  Watch the eval's `rooms`/`dist_m` columns there: 1,200 ticks is only 80
  sim-seconds of *temporal* depth per env, thin for the slow layer and the
  novelty-appetite loop even when prediction error looks great.
- `--lr-scale`: the batch-averaged gradient is ~√B less noisy; 2–4 is
  worth trying for faster settling.
- No sequence replay in this mode (B fresh streams replace it); experience
  jsonl is logged for `--log-envs` streams (default 1) so sleep mode still
  works.
- `pnn_sim/batched/verify.py` asserts numerical equivalence against the
  reference modules (infer, learn, actor, preprocessing, dynamics, gate) —
  run it after touching either implementation.
- Checkpoints are the same schema; `--load ~/.ros/pnn_brain.pt` to start
  from the rover's brain, and the output loads on the rover unchanged.

Caveat: batch-averaged learning is a (principled) departure from the
single-stream online rule the hyperparameters were tuned on — treat the
first long runs as experiments, and compare against a single-stream
control before trusting a brain from here on the rover.

## Picking the transfer brain by data

Training drops numbered snapshots under `<out-dir>/snapshots/` every
`--snapshot-every` batch ticks (default 10k), and **automatically** ends
the run with a frozen eval of all snapshots in unseen houses, copying the
winner to `<out-dir>/best_pnn_brain.pt` (+ slow layer) with the table in
`eval_results.json` — that's the file to scp to the rover. Opt out with
`--no-eval-after`, or re-run/extend the sweep by hand:

```bash
python3 -m pnn_sim.eval_checkpoints sim_out_batched/snapshots/*.pt
```

Every checkpoint sees the same unseen worlds and noise streams, so the
columns are directly comparable: `err_early` (how fast a frozen brain
makes sense of a novel house), `err_late` (settled prediction quality),
`stops/h` / `coll/h` (gate pressure and gate failures), `rooms` / `dist_m`
(exploration actually happening). Stop training when these flatten between
snapshots; transfer the suggested pick (or your own read of the columns).

Note: frozen mode advances the recurrent state explicitly (learn() is what
normally advances it) — this fix also applies to `learn:=false` eval on
the rover runner.

Only `numpy` and `torch` are imported (plus the brain modules); on the
Spark, just `rsync` this repo (or only `pnn_sim/` + `src/tractor_bringup/`)
and run.

## Transfer to the rover

```bash
scp sim_out/pnn_brain.pt sim_out/pnn_brain_slow.pt rover:~/.ros/
```

Keep `action_scale` consistent (saved in the checkpoint; the rover warns on
mismatch). Expect a re-adaptation period on real tracks — sim slip/lag are
approximations and the brain learns online anyway.

The sim also writes `sim_out/pnn_experience.jsonl` in the rover's format,
so the sleep consolidator works on sim experience unchanged:

```bash
python3 src/tractor_bringup/tractor_bringup/active_inference/sleep_consolidator.py \
  --model_path sim_out/pnn_brain.pt --experience_log_path sim_out/pnn_experience.jsonl
```

## Throughput

Single worker, single torch thread: ~100–150 ticks/s (~7–10× realtime)
measured on a desktop CPU core, replay included. Per-worker speed is flat
as workers scale (independent processes).
