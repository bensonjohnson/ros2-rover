#!/usr/bin/env bash
# Spark-only: run 14 = extend run 13 seeds 8 and 9 (criteria in run14_variant.sh).
set -u
cd ~/projects/ros2-rover
bash evo/run14_variant.sh greenfield14_s8 8 evo_runs/greenfield13_s8
bash evo/run14_variant.sh greenfield14_s9 9 evo_runs/greenfield13_s9
echo "QUEUE14-COMPLETE $(date)" | tee -a evo_runs/queue14.log
