#!/usr/bin/env bash
# Run 4: doorway-curriculum evolution on the Spark, GB10 fast path.
# Two chained processes (one live CUDA graph per process is a hard limit
# on this stack — see gmbisect notes in the ros2-rover skill):
#   phase 1: wide 2.0 m archways -> crossing behavior basin is findable
#   phase 2: warm-started from phase-1 population, real-ish 0.94 m doors
# Holdout houses keep natural (0.7-1.0) door widths in BOTH phases: that
# is the transfer metric — phase-1 train rooms vs phase-1 holdout rooms.
set -uo pipefail
cd ~/projects/ros2-rover
source /home/benson/venv/bin/activate

OUT=evo_runs/greenfield4
mkdir -p $OUT
COMMON="--pop 128 --games 16 --hidden 64 --ticks 5400 --w-dist 0.005
        --device cuda --fp16 --graph --train-rotations 4 --report-every 5"

echo "=== phase 1 (door 2.0 m) start $(date) ===" | tee -a $OUT/run.log
python3 -m evo.evolve $COMMON --door-w 2.0 --gens 20 \
    --train-seed 61000 --seed 4 \
    --out-dir $OUT/phase1 >> $OUT/run.log 2>&1
RC1=$?
echo "=== phase 1 rc=$RC1 done $(date) ===" | tee -a $OUT/run.log
[ $RC1 -ne 0 ] && exit $RC1

echo "=== phase 2 (door 0.94 m, warm-start) start $(date) ===" | tee -a $OUT/run.log
python3 -m evo.evolve $COMMON --door-w 0.94 --gens 20 \
    --train-seed 61000 --seed 5 \
    --seed-from $OUT/phase1/final_population.npz \
    --out-dir $OUT/phase2 >> $OUT/run.log 2>&1
RC2=$?
echo "=== phase 2 rc=$RC2 done $(date) ===" | tee -a $OUT/run.log
echo "RUN4-COMPLETE rc=$RC2" | tee -a $OUT/run.log
