#!/usr/bin/env bash
# Spark-only queue 19 (trim-robust deploy hunt):
#   1. greenfield19e_s8 = extend run 17 s8 by 120 gens (strongest candidate)
#   2. greenfield19t_s9 = fresh seed-9 trim-rand twin (second population for
#      the worst-trim deploy pick + seed agreement)
# Each ends with deep-evals + trim_eval + deploy_pick. ~3 h total on GB10.
set -u
cd ~/projects/ros2-rover
if pgrep -f "[e]vo.evolve" >/dev/null; then
    echo "QUEUE19 ABORT: an evo.evolve is already running" | tee -a evo_runs/queue19.log
    exit 1
fi
bash evo/run19_extend.sh greenfield19e_s8 8 evo_runs/greenfield17t_s8
bash evo/run19_twin.sh greenfield19t_s9 9
echo "QUEUE19-COMPLETE $(date)" | tee -a evo_runs/queue19.log
