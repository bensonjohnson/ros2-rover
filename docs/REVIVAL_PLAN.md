# ros2-rover Revival Plan — Live Sample-Efficient Training

Assessment date: 2026-09-10. Written after a full review of the repo (~993 commits,
Nov 2025 – Jul 2026) to restart the project with a fresh, single-track process.

## 1. What's in the repo (the algorithm graveyard)

Nine distinct training approaches, in chronological order:

| Approach | Last real commit | Class | Verdict |
|---|---|---|---|
| `v620_pgpe_trainer.py` | 2025-11-09 | Evolutionary (policy-vector) | Abandoned. Needs hundreds of full-episode rollouts per update — worst possible fit for a live robot. |
| `v620_map_elites_trainer.py` | 2025-12-04 | Evolutionary (quality-diversity) | Abandoned. `compute_fitness()` (L1353) is hand-tuned tank-behavior shaping, no learning signal. |
| `v700_es_sac_trainer.py` | 2025-12-19 | Population SAC (OpenAI-ES style) | Abandoned. |
| `v620_sac_trainer.py` | 2026-03-15 | Off-policy SAC + UnifiedBEV, NATS JetStream transport, DroQ options | Abandoned. Had "create phased learning" as final state. |
| `v620_ppo_trainer.py`, `ppo_bev_trainer*.py` | 2026-03-27 | On-policy PPO | Abandoned. Last commit — *"fix spin-in-place local optimum: continuous penalty, idling cost, entropy annealing"* — is the epitaph: on-policy PPO on a real rover with no task definition collapses into degenerate cruising. |
| `v620_dreamer_trainer.py` | 2026-05-10 | Model-based (RSSM), rewards: coverage/frontier/collision/episodic | Abandoned mid-build (ROCM_MEMORY_FIX.md documents GPU memory fights). |
| **`v620_rlpd_trainer.py` + `rlpd_networks.py` + `rlpd_replay.py` + `rlpd_remote_runner.py`** | 2026-05-14 (ZMQ loop hardened through 2026-06-28) | **RLPD / HIL-SERL** | **The good one. See §3.** |
| `pnn_sim/` predictive-coding brain (batched GPU trainer, place memory, frontier explorer) | 2026-06-14 | Active-inference PCN, trained only in the 2D raycast sim | Interesting research substrate, never closed the sim→real loop. |
| `explorer_trainer.py` + `src/tractor_explorer/` "Deep Explorer Network" | 2026-06-28 | Novelty/coverage-driven (NGU-style `intrinsic_rewards.py`), 15 Hz control, ZMQ 5557/5558 | Freshest; completed the full ZMQ train→RKNN→hot-reload loop but never validated driving quality. |

**The pattern is the diagnosis.** Roughly every 4–8 weeks a new algorithm family was
started from scratch, taken as far as "it runs," hit its first plateau, and was replaced.
Each family duplicated the whole stack: its own collector (`remote_training_collector.py`,
`ppo_remote_runner.py`, `rlpd_remote_runner.py`, `explorer_runner.py`,
`map_elites_episode_runner.py`, `pc_active_inference_runner.py`), its own transport
(NATS for SAC, ZMQ 5555/5556 for PPO/RLPD, ZMQ 5557/5558 for explorer), its own reward,
its own ONNX→RKNN export. No single data path was ever hardened long enough to become
trustworthy. The failure mode was orchestration, not algorithms.

## 2. Platform facts (verified in code)

- **Rover:** RK3588 (NPU) + ROS2 Jazzy. Hiwonder I2C skid-steer driver
  (`src/tractor_control/tractor_control/hiwonder_motor_driver.py`): `encoder_ppr: 1980`,
  `wheel_separation: 0.154`, `wheel_radius: 0.025`, odometry at 100 Hz,
  **`motor_command_rate_limit_secs: 0.1` (cmd throttle to 10 Hz)** and
  **`cmd_vel_timeout_secs: 1.0` watchdog auto-stop**. Good hardware-level safety bones.
- **Sensors:** LD19 lidar ~10 Hz (100 ms/rev), RealSense D435i at 424×240@15 Hz
  (`realsense_config.yaml`, "minimum supported framerate"), BNO085 IMU
  (`imu_type: bno085` default). EKF (`robot_localization.yaml`) deliberately fuses
  rf2o **position only** — yaw not fused because 10 Hz scans smear at the skid-steer's
  ~2.5 rad/s yaw rate. Heading therefore rides on IMU; position on wheel odom + rf2o.
  Skid-steer slip makes wheel odom optimistic; this is the obs-quality bottleneck.
- **Safety:** `lidar_safety_monitor.py` is genuinely well built — front stop 0.20 m /
  rear 0.40 m with hysteresis, side/rear sector blocking,
  `stale_timeout: 0.2 s` (lidar dropout ⇒ stop), estop publishing, min-block duration.
  Keep this node permanently in the loop for all training modes.
- **HIL:** Xbox RB deadman (HIL-SERL style) in `rlpd_remote_runner.py`: holding RB
  overrides policy, steps get `is_intervention[t]=True` and `rewards[t,4]=-1.0`;
  `teleop_mode` collects pure demos. Demo buffer (`rlpd_replay.py DemoReplay`) is
  append-only and persisted atomically to `demos.npz` across restarts. Correct design.
- **Trainer:** `v620_rlpd_trainer.py` defaults are genuine RLPD: **10 critics**,
  batch 256, **UTD=10**, encoder_lr 1e-4 / actor&critic 3e-4, **demo_ratio 0.5**,
  online FIFO + demo buffers, chunked ingestion (T−1 transitions/chunk; note it
  explicitly does *not* stitch cross-chunk next-states — drops 1 transition/chunk and
  leaves boundary transitions unverifiable).
- **Server:** AMD V620 under ROCm. `buffer_test.log`/`startup_full.log` show repeated
  MIOpen "SearchImpl ... be patient" autotune stalls; `CONTINUOUS_MODE.md` exists
  because staged train bursts caused GPU hangs. The ROCm stack has been a recurring
  failure source in its own right. (DGX Spark at 172.0.0.201 is an available CUDA
  alternative; rover network is 192.168.1.x — verify routing before switching.)

## 3. Why nothing worked — fresh diagnosis

1. **No task.** No trainer has goal-reaching or a curriculum. `progress = max(0, vx)`
   on an open floor has one global optimum: drive forward forever (and the PPO commit
   history shows exactly that degenerate basin, patched with hand-rolled penalties
   instead of a task). Reward shaping cannot substitute for a terminal condition.
2. **Clock stack is incoherent.** Policy at 30 Hz (`inference_rate_hz: 30`), camera at
   15 Hz, lidar at ~10 Hz, motor driver throttled to 10 Hz. So 2 of every 3 policy steps
   act on stale observations *and* get discarded by the driver anyway — while the replay
   buffer records them as if they were real (state, action) transitions. The learned MDP
   is not the MDP the robot is in.
3. **Every family reimplemented observation normalization** (`normalize_proprio` et al.
   per runner) — training/inference skew bugs were inevitable and invisible.
4. **Cross-chunk transition stitching skipped** (see §2) — at chunk_len 64 that's ~1.5 %
   poisoned samples plus broken `is_first` semantics at chunk joins.
5. **Sim doesn't match the rover** (depth 848×100 vs 424×240 in `TractorRover.proto`,
   no skid-steer slip model) and was never in the training loop anyway — all real
   experience was coming from an unshaped, un-tasked robot.
6. **NPU hot-reload adds race conditions early**: ONNX→RKNN quantization changes
   behavior (there is a whole `CALIBRATION_AND_QUANTIZATION.md` + `verify_architecture.py`
   because of it), ZMQ PUB 5558 swap can land mid-episode. Fine later; poison during
   the fragile early phase.

## 4. The one thing to keep

The **RLPD / HIL-SERL lineage** (`v620_rlpd_trainer.py`, `rlpd_*`, `rlpd_remote_runner.py`,
`start_rlpd_*.sh`) is exactly the published recipe that has repeatedly produced real
robot policies in **~1–3 hours of wall-clock interaction** (Luo et al. RLPD; Jannott et
al. HIL-SERL). It is also the most-recently-hardened code in the repo. Sample efficiency
here comes from three multiplier effects, all of which the code already half-supports:

- **Off-policy replay** — reuse every real step thousands of times (UTD=10).
- **Human demos as data** — teleop episodes in the persistent demo buffer, replayed at
  ratio 0.5. This is *the* cheat code; HIL-SERL's ablations show demos beat any reward
  design.
- **Frozen pretrained vision** — HIL-SERL's key ablation finding. The SAC trainer already
  had "freezing pre-trained BEV encoder"; RLPD should adopt it (R3M/TCD, or start with a
  lidar-only MLP and add vision after).

Archive everything else (`git tag archive/pre-revival main; git checkout -b revival`) —
PGPE, map-elites, ES-SAC, NATS-SAC, PPO-BEV, Dreamer, pnn_sim, deep-explorer. Keep them
readable; stop feeding them energy.

## 5. What it will take, in order

**Stage 0 — Freeze & fix the loop (no robot time)**
- Single observation builder module shared by runner and trainer; delete per-runner
  duplicates. Unit tests: shapes, dtypes, normalization round-trip.
- Fix the clock stack: `inference_rate_hz` = 10 (match motor throttle + lidar), or raise
  the driver's `motor_command_rate_limit_secs` to ~0.033 — one coherent rate everywhere,
  stamped with sensor timestamps in the chunk.
- Stitch cross-chunk transitions (hold the last chunk frame; the `DemoReplay` stash-last
  pattern is already there — reuse it for online chunks).
- Buffer audit script: sample N random entries, decode (s, a, r, s′), verify s′ really is the
  frame ~100 ms after s (replay-consistency test; catches the stale-obs class of bug).
- Move training server to CUDA (DGX Spark, 172.0.0.201). `start_rlpd_server.sh` already
  auto-detects CUDA/ROCm/Blackwell — "V620" in the names is a legacy label — so this is
  copy-repo-to-Spark + run the script there; ZMQ transport unchanged. Do it from day one
  to retire the MIOpen autotune stalls entirely.

**Stage 1 — Task + demos (teleop days)**
- Define the task with a terminal state: goal-reaching to 2–3 visible waypoints
  (yard_map.pgm + ArUco or lidar-visible posts), then doorways, then a sequence.
  Keep the 5 RLPD reward channels as small shaping only; collision/intervention dominate.
- Run `start_rlpd_teleop_collection.sh` until the demo buffer holds ~30–50 varied
  episodes: straight approach, tight turns, reversing out of corners, obstacle avoidance.
  Budget: a few evenings. This is the highest-leverage hour-per-policy you'll spend.

**Stage 2 — First training runs (supervised autonomy)**
- Start on the existing **v3 obs pipeline** (depth 96×72 uint8 + lidar resampled to 360
  beams + 6·K proprio, frame_stack K=4; `rlpd_remote_runner.py` L5-13, depth clipped to
  [0.2, 6.0] m) with the small trainable depth/lidar CNN (`rlpd_networks.py` L221+,
  ~20× fewer encoder params than the ResNet path). CPU inference on the RK3588, policy
  published over ROS. **No RKNN, no hot-reload yet.**
- Run with lidar_safety_monitor + cmd_vel watchdog mandatory; RB deadman always armed;
  human-speed speed curriculum (start `_curriculum_max_speed` ≈ 0.10 m/s).
- Gate before any unsupervised run: Q-curves monotonic in TensorBoard, replay-consistency
  passed, demo replay eval shows demo-imitation ≥ 80 %.

**Stage 3 — Scale up**
- Add frozen pretrained depth/lidar BEV encoder (SERL-style R3M/TCD) only after the MLP
  actor learns the task — encoder changes invalidate buffer value estimates, so freeze it
  once and for all.
- Only when a CPU policy holds a task: re-enable ONNX→RKNN (with
  `verify_architecture.py` numerical parity as a hard gate) and the ZMQ 5558 hot-reload.
- Log every run; keep `demos.npz` in git-lfs or off-repo backup — it's the crown jewel.

**Success metric that ends the project:** from a cold buffer with Stage-1 demos, policy
reaches 3 waypoints over ≥10 min of unsupervised running within **≤2 hours of cumulated
robot time** (HIL-SERL-class sample efficiency). If RLPD can't do that with the fixed
loop, *then* it's a legitimate moment to revisit a second family (e.g. the BEV explorer
as the pre-training phase for the RLPD buffer) — as data generation, not as a new algo.

## 6. Housekeeping
- `build/`, `install/`, `log/`, `*.sync-conflict-*`, `buffer_test.log`, `test_run.log`,
  `last_commit.txt`, `status.txt`, screenshots: gitignore + purge.
- The `.sync-conflict` files show the repo syncs via Syncthing — pick git-only sync for
  this repo to stop the drift.
