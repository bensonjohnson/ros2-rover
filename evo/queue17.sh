#!/usr/bin/env bash
# Spark-only: run 17 trim randomization, chained after run 16.
set -u
cd ~/projects/ros2-rover
until grep -q "QUEUE16-COMPLETE" evo_runs/queue16.log 2>/dev/null; do sleep 60; done
bash evo/run17_variant.sh greenfield17t_s8 8
echo "QUEUE17-COMPLETE $(date)" | tee -a evo_runs/queue17.log
