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
  the per-tick ≤82-dim tensors (kernel-launch overhead dominates); CPU +
  workers is the throughput path. The flag exists for experiments.
- `--freeze`: evaluate without learning.

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
