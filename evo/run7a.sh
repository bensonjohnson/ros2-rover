#!/usr/bin/env bash
# Run 7a: novelty-dominant reward test (representation-vs-search gate part 1).
# Probe data (evo/explore_probe.py): best scripted personality cov 0.144 vs
# evolved population 0.066 — random walk OUT-covers evolution, i.e. w_cov=0.3
# under-prices exploration vs the dist/collision comfort of wall cruising.
# w_cov 0.6 makes the 0.078-coverage band worth ~0.047 — must outbid cruising.
# Readout: train_cells climbs off 0.066 = reward mispricing confirmed.
#          converges to ~0.14 then flat, rooms 1.0 = memory genome needed.
set -uo pipefail
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
mkdir -p evo_runs/greenfield7a
echo "=== 7a (hidden 128 + w_cov 0.6) start $(date) ===" > evo_runs/greenfield7a/run.log
python -u -m evo.evolve --out evo_runs/greenfield7a --pop 128 --gens 40 \
    --train-seed 64000 --holdout-seed 777000 --w-cov 0.6 --hidden 128 \
    >> evo_runs/greenfield7a/run.log 2>&1
echo "=== 7a rc=$? done $(date) ===" >> evo_runs/greenfield7a/run.log
echo "RUN7A-COMPLETE rc=$?" >> evo_runs/greenfield7a/run.log
