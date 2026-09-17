#!/usr/bin/env bash
# Spark-only run-11 family, all with w_coll 1.0 (see run11_variant.sh):
# curriculum vs direct level 3, seeds 8 and 9.
set -u
cd ~/projects/ros2-rover
bash evo/run11_variant.sh greenfield11 8 curriculum
bash evo/run11_variant.sh greenfield11c 8 direct
bash evo/run11_variant.sh greenfield11_s9 9 curriculum
bash evo/run11_variant.sh greenfield11c_s9 9 direct
echo "QUEUE11-COMPLETE $(date)" | tee -a evo_runs/queue11.log
