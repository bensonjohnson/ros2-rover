"""Pure-PNN spatial brain — the deployable half.

`pc_map` (predictive-coding occupancy map) and `policy` (delta-rule map-policy
+ pursuit control) live HERE, in the ROS package, because they run on the
rover. `pnn_sim.spatial` imports them unmodified so the sim trains exactly the
stack that deploys — same rule as `tractor_bringup.active_inference`. Training
harness (distillation, validation, batched smoke tests) stays in pnn_sim.
"""
