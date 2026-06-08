"""Active-inference 'brain' for the rover.

A from-scratch predictive-coding world model that learns purely online with
local (non-backprop) update rules, and acts to maximize expected information
gain (pure epistemic value). Lidar in, per-track drive out.

Modules:
    scan_preprocess : LaserScan -> fixed-length normalized vector
    pc_world_model  : temporal predictive-coding generative model (the cortex)
    efe_actor       : action selection by expected free energy (epistemic only)
"""
