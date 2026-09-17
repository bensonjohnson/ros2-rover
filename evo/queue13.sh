#!/usr/bin/env bash
# Spark-only: run 13 seeds 8 and 9 (criteria in evo/run13_variant.sh).
set -u
cd ~/projects/ros2-rover
bash evo/run13_variant.sh greenfield13_s8 8
bash evo/run13_variant.sh greenfield13_s9 9
echo "QUEUE13-COMPLETE $(date)" | tee -a evo_runs/queue13.log
