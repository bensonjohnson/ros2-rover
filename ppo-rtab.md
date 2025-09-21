# PPO + RTAB-Map Integration Plan

## Phase 0 – Prep & System Baseline
- [ ] Verify RealSense RGB-D topics (`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/depth/color/points_registered`) are stable outdoors.
- [ ] Confirm IMU (`/lsm9ds1_imu_publisher/imu/data`) and wheel encoder topics are time-synced; log rosbag for reference.
- [ ] Audit existing launch files/scripts for RTAB-Map usage; document current parameters and missing sensor inputs.
- [ ] Capture a 5–10 min dataset in the yard for off-robot replay/testing.
- [ ] Snapshot current rover CPU/GPU/NPU utilization during BEV runs for before/after comparison.

## Phase 1 – Sensor Fusion & RTAB-Map Configuration
- [ ] Add/verify EKF (`robot_localization`) node fusing wheel encoders + IMU + optional vision odom to publish `odom -> base_link`.
- [ ] Feed fused odom + IMU into RTAB-Map (`/rtabmap/rtabmap`) via launch remaps; enable `RGBD/AngularUpdate`, `RGBD/LinearUpdate`.
- [ ] Tune ground parameters for grass (`Grid/NormalsSegmentation`, `Grid/MaxGroundAngle`, noise filters); validate floor detection logs.
- [ ] Ensure RTAB-Map publishes required outputs (`/rtabmap/grid_map`, `/rtabmap/local_grid_obstacles`, `/rtabmap/frontiers`).
- [ ] Validate RTAB-Map TF tree (`map -> odom -> base_link`) against EKF output on hills; adjust IMU orientation filters if needed.
- [ ] Document final RTAB parameter set in `config/rtabmap_outdoor.yaml` and add to repo.

## Phase 2 – Observation Builder (RGB-D + Occupancy + Proprio)
- [x] Create new ROS2 node (`tractor_bringup/rtab_observation_node.py`) to:
  - Subscribe to downsampled depth/RGB frames and `/rtabmap/grid_map`.
  - Crop/resample a local occupancy patch around `base_link`.
  - Stack depth + occupancy + frontier masks + proprio scalars into an `ExplorationObservation` message.
  - Publish to `/exploration/observation` (float tensor + metadata) at ~10 Hz.
- [x] Update safety monitor to compute min-forward distance from RTAB data (occupancy or `/local_grid_obstacles`) instead of BEV.
- [x] Replace BEV references in runtime guardian (`npu_exploration_bev.py` → `npu_exploration_rtab.py`) with new observation inputs; keep IMU + wheel features.
- [x] Retire BEV processor launch stanza or guard it behind a launch argument.
- [x] Add Frontier channel or mask derived from `/rtabmap/frontiers` to guide exploration reward shaping.
- [ ] Define observation message schema (ROS2 custom msg vs. `Float32MultiArray`) and update documentation.
- [x] Log sample observations to disk (numpy/rosbag) for trainer and RKNN calibration use.

## Phase 3 – PPO Trainer Rework
- [x] Modify training data pipeline to record RGB-D/occupancy observations and updated proprio vectors.
- [x] Adapt `ppo_manager_node.py` (or new `ppo_manager_rtab.py`) to mirror runtime observation preprocessing, reward shaping, and rollout storage.
- [x] Redesign policy network architecture (CNN + proprio MLP) for the new observation shape.
- [ ] Update optimization hyperparameters, logging, and checkpoint schema.
- [x] Build simulation/offline training harness that replays rosbag data through the new observation builder.
- [x] Rework reward function to emphasize coverage gain, frontier reduction, smooth motion, and slope safety penalties.
- [ ] Stand up experiment tracking (Weights & Biases or CSV logs) for PPO iterations.

## Phase 4 – RKNN Export Path
- [ ] Update RKNN conversion scripts to handle new model input tensor.
- [ ] Validate RKNN inference with recorded observations; measure latency on the rover NPU.
- [ ] Implement hot-reload service compatibility (`/reload_rknn`) for the new runtime node.
- [ ] Prepare representative calibration set for RKNN quantization (RGB-D, occupancy, proprio ranges).
- [ ] Add unit tests for export pipeline to guard against shape mismatches.

## Phase 5 – Pathing & Run Modes
- [ ] Define map-complete criterion (coverage % or frontier exhaustion) to trigger pathing mode entry.
- [ ] Implement coverage planner using RTAB-Map occupancy (boustrophedon or grid-based sweep) and produce waypoint list.
- [ ] Create run-mode controller that follows waypoints, controls mower relay, and reuses safety guardrails.
- [ ] Add mode manager node/service to switch between Mapping / Pathing / Run.
- [ ] Integrate mower relay control topic/service with safety interlocks (e-stop, blade inhibit while moving backwards).
- [ ] Allow manual override/teleop mode for recovery and testing.

## Phase 6 – Testing & Validation
- [ ] Unit/integration tests for observation node (shape, latency, NaN handling).
- [ ] Simulation or bag replay tests for PPO runtime + trainer with RTAB-Map data.
- [ ] Outdoor dry-runs: mapping-only, then mapping+pathing, finally supervised mowing.
- [ ] Document performance metrics (coverage %, map quality, energy usage) and iterate.
- [ ] Record regression baselines and add automated checks to CI (lint + minimal unit tests) before deployment to rover.

## Phase 7 – Documentation & Deployment
- [ ] Update README/bringup docs with new launch instructions and mode descriptions.
- [ ] Create operator checklist for Mapping/Pathing/Run flows, including safety checks.
- [ ] Package launch + config changes into rover deployment script; verify `colcon build` + install on device.
- [ ] Schedule periodic map maintenance workflow (e.g., seasonal re-mapping, boundary updates).

## Notes & Considerations
- Maintain BEV code behind feature flag until RTAB path is stable for fallback comparisons.
- Use rosbag2 recordings to benchmark CPU/GPU load before/after removing BEV processing.
- Ensure guardian can fall back to raw depth min-distance if RTAB-Map stalls.
- Plan RKNN quantization with representative RGB-D/occupancy datasets to avoid accuracy loss.
- Perform final builds, tests, and deployments on the rover hardware; avoid local binary generation.
