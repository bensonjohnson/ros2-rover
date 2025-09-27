# RTAB Optimization To-Do

## Launch & Operator Controls
- [x] Add CLI/interactive toggles for `max_speed`, `safety_distance`, and PPO training cadence in `start_exploration.sh` and `start_npu_exploration_ppo.sh`.
- [x] Surface launch arguments for RTAB observation resolution and export them through `npu_exploration_ppo.launch.py`.

## Observation Pipeline
- [x] Profile `rtab_observation_node.py` and cache NumPy buffers to minimize allocations; consider `float16` outputs for export paths.
- [ ] Gate frontier/occupancy recompute to every N frames to cut CPU usage; make interval configurable.

## Safety Monitor
- [ ] Share occupancy/min-distance buffers between observation node and `simple_safety_monitor_rtab.py`.
- [ ] Implement adaptive safety-distance scaling tied to commanded speed.

## Runtime Inference
- [ ] Lazily initialize RKNN/PyTorch contexts in `npu_exploration_rtab.py` and add TorchAO INT8 fallback.
- [ ] Trim proprio vector to essentials and enable `torch.compile` for CPU inference path.

## PPO Training Stack
- [ ] Convert rollout storage to `float16` and add dynamic minibatch/epoch adjustments in `ppo_manager_rtab.py` / `ppo_trainer_rtab.py`.
- [ ] Gate RKNN exports on low-activity windows and thin experience replay when idle.

## RKNN Export
- [ ] Build RTAB-based calibration dataset and integrate hybrid INT8 quantization before export.
- [ ] Automate `/reload_rknn` smoke test post-export, logging latency deltas.

## Testing & Telemetry
- [ ] Add observation tensor shape/unit tests and integrate into CI scripts.
- [ ] Extend startup diagnostics to log NPU/CPU utilization and topic rates over time.
