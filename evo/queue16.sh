#!/usr/bin/env bash
# Spark-only: run 16 symmetric gate (criteria in run16_variant.sh).
set -u
cd ~/projects/ros2-rover
bash evo/run16_variant.sh greenfield16s_s8 8
bash evo/run16_variant.sh greenfield16sr_s8 8 --w-rev 0.3
echo "QUEUE16-COMPLETE $(date)" | tee -a evo_runs/queue16.log
