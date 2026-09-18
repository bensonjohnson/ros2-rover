#!/usr/bin/env bash
# Spark-only: run 15 forward-driving preference (criteria in run15_variant.sh).
set -u
cd ~/projects/ros2-rover
bash evo/run15_variant.sh greenfield15f_s8 8
bash evo/run15_variant.sh greenfield15w_s9 9 evo_runs/greenfield14_s9
echo "QUEUE15-COMPLETE $(date)" | tee -a evo_runs/queue15.log
