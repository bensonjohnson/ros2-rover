# NPU Depth Exploration Improvement Plan (OrangePi5+ RK3588)

Format: Checklist you can tick ([ ] -> [x]) as you progress. Keep commits small per subsection.

NOTE: Phase 0 baseline capture intentionally skipped per decision (poor initial latency, proceed directly to improvements).

## Phase 0: Baseline Capture (Skipped)
[ ] (skipped) Record current model latency
[ ] (skipped) Reward stats log
[ ] (skipped) Session bag
[ ] (skipped) Archive models

## Phase 1: Data & Preprocessing
[x] Implement depth normalization (clip 0..4m, scale to [0,1])
[x] Reduce input resolution to 160x288 (benchmark latency)
[ ] Add optional central vertical crop (parameterized)
[ ] Add temporal stacking (last 3 frames) OR lightweight ConvGRU (infrastructure added, not enabled)
[x] Add proprio features: last_action, wheel_diff, min/mean depth, near_collision flag
[ ] Gate verbose prints behind debug parameter (prints removed; add debug toggle later)

## Phase 2: Action & Network Output Hygiene
[ ] Add tanh on action head; scale: linear = tanh(a0)*max_speed; angular = tanh(a1)*max_yaw
[ ] Separate confidence/value head (remove current sigmoid(reward) regression)
[ ] Move model definition to dedicated file (`models/depth_policy.py`)
[ ] Add weight initialization (orthogonal/Xavier)
[ ] Add model version number in state dict

## Phase 3: Training Paradigm Shift
[ ] Introduce episode concept (termination: collision, timeout, low battery)
[ ] Store experiences with done flag
[ ] Implement Advantage Actor-Critic (A2C lightweight) OR PPO-mini
[ ] Value head: train with (r + γ * (1-done)*V(s')) target
[ ] Normalize returns (running mean/std)
[ ] Clip or scale per-step rewards to sane range (e.g. [-10,10])
[ ] Optional: entropy bonus to encourage exploration

## Phase 4: Reward System Simplification
[ ] Externalize weights to YAML (`config/reward_config.yaml`)
[ ] Core components only: forward_progress, novelty, collision, proximity, smoothness, spin_penalty
[ ] Implement novelty via grid hashing (0.5m) + visitation count decay
[ ] Replace min-distance with percentile (5th) for stable proximity
[ ] Add reward debug publisher (`/reward_breakdown` JSON string)
[ ] Remove overlapping bonuses (wall/frontier unless re-added later)

## Phase 5: Safety & Feedback Integration
[ ] Feed near_collision & emergency_stop as inputs to policy
[ ] Distinguish soft_brake vs emergency_stop events
[ ] Penalize emergency_stop more than soft_brake
[ ] Add slip detector (commanded vs encoder discrepancy)
[ ] Penalize slip; publish `/slip_events`

## Phase 6: Replay & Sampling Enhancements
[ ] Switch to prioritized replay (TD-error based) if off-policy adopted
[ ] Cap buffer by time (e.g. 2h) and remove oldest episodes
[ ] Periodic dataset snapshot for analysis (every N episodes)

## Phase 7: Quantization & RKNN Pipeline
[ ] Auto-generate calibration dataset (200 random normalized depth tensors) -> `models/calib/`
[ ] Export ONNX with dynamic batch=1, channels=stack_frames
[ ] Validate RKNN inference numeric parity (MAE < 0.02 on actions)
[ ] Measure NPU latency vs CPU (log to file)
[ ] Add fallback logic if NPU init fails (graceful degrade)

## Phase 8: Performance & Profiling
[ ] Add simple FPS / inference time moving average publisher `/npu_perf`
[ ] Profile CPU usage before/after changes (top / psutil)
[ ] Prune channels (L1 norm) if latency too high
[ ] Optionally apply knowledge distillation to smaller net

## Phase 9: Persistence & Robustness
[ ] Save model on graceful shutdown (signal trap) and on low battery event
[ ] Keep rotating last K checkpoints + `latest`
[ ] Add checksum validation on load; fallback if corrupted
[ ] Add safe-mode flag to disable training while keeping inference

## Phase 10: Evaluation Mode
[ ] Launch arg `train:=false` for inference-only
[ ] Episode evaluator node logs: distance_traveled, collisions, average_speed
[ ] Generate comparison report pre/post refactor

## Phase 11: Monitoring & Tooling
[ ] Integrate TensorBoard (scalars: loss, value_loss, entropy, reward mean, components)
[ ] Add structured JSON logging (daily file) for post-mortem
[ ] Foxglove panel for reward components & action distribution

## Phase 12: Advanced Exploration (Optional)
[ ] Reintroduce adaptable wall/frontier heuristic as separate intrinsic reward channel
[ ] Curiosity via prediction error (forward model) or RND (random network distillation)
[ ] Multi-head: discrete angular bins + continuous speed (hybrid policy) test branch

## Phase 13: Documentation & Ops
[ ] Update README / ARCHITECTURE doc for new pipeline
[ ] Add diagram of data flow (depth→preproc→policy→action→env→reward)
[ ] Provide hyperparameter table with defaults & tuning notes
[ ] Add recovery guide (what to do if model diverges)

## Phase 14: Validation Metrics (Success Criteria)
[ ] Mean forward_progress per minute ↑ vs baseline (target +30%)
[ ] Collision rate per meter ↓ (target -50%)
[ ] Emergency stops vs soft brakes ratio ↓
[ ] Novel grid cells per 5 min ↑
[ ] Inference latency (NPU) < 15ms (66 FPS budget) or chosen target
[ ] Stable reward mean (no unbounded drift) over 30 min session

## Implementation Notes
- All configuration driven from `start_npu_exploration_depth.sh` (no new launch args) per preference.
- Stacked frame infra present; will enable after on-device latency measurement.

## Risk Mitigation
[ ] Watchdog: if action NaNs appear, auto-switch to safe stop
[ ] Cap absolute angular velocity commands before publish
[ ] Validate depth frame timestamp freshness (< 0.5s)

## Immediate Low-Effort Wins
[x] Remove verbose prints in inference loop
[x] Normalize depth & clamp distances
[ ] Add save on shutdown
[ ] Add tanh squashing

(End of plan)
