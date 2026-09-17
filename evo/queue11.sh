#!/usr/bin/env bash
# Spark-only queue after run 11 (seed 8, curriculum): the 11c control and
# seed-9 replicates of both modes. V620 retired for evo (Spark fused+compile
# graph = ~13 s/gen vs V620 eager ~164 s/gen).
set -u
cd ~/projects/ros2-rover
until grep -q "RUN11-COMPLETE" evo_runs/greenfield11/run.log 2>/dev/null; do sleep 30; done
bash evo/run11_variant.sh greenfield11c 8 direct
bash evo/run11_variant.sh greenfield11_s9 9 curriculum
bash evo/run11_variant.sh greenfield11c_s9 9 direct
echo "QUEUE11-COMPLETE $(date)" | tee -a evo_runs/queue11.log
