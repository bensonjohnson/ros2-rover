#!/usr/bin/env bash
# Spark-only: run-12 bundle split (criteria in evo/run12_variant.sh).
set -u
cd ~/projects/ros2-rover
bash evo/run12_variant.sh greenfield12_ctl_s9 9
for S in 8 9; do
    bash evo/run12_variant.sh greenfield12_obs_s$S $S --obs v2
    bash evo/run12_variant.sh greenfield12_act_s$S $S --action vw
    bash evo/run12_variant.sh greenfield12_h64_s$S $S --hidden 64
done
echo "QUEUE12-COMPLETE $(date)" | tee -a evo_runs/queue12.log
