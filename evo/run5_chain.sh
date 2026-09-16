#!/usr/bin/env bash
# Run 5: cell-coverage NOVELTY on the Spark, GB10 fast path.
# Verdict sought by runs 1-4: reward weights, search budget and doorway
# width are all cleanly excluded as explanations for the rooms~1.0 wall-
# hugger plateau. The missing half is an intrinsic "go where your own
# perception says you haven't been" signal: w_cov pays per distinct 0.8 m
# cell visited (saturating; self-computable, no room_id -> deployable as
# place memory on the real rover). Rooms stays the ground-truth metric.
#
# Two processes, SEQUENTIAL (one live CUDA graph per process is a hard
# limit on this stack). Same fast path as run 3/4 (fp16+graph, 8192 envs,
# w_dist 0.005); fresh train-seed 62000 so run-4 houses never recur;
# holdout stays seed 777000 natural doors -> comparable to runs 1-4.
set -uo pipefail
cd ~/projects/ros2-rover
source /home/benson/venv/bin/activate

COMMON="--pop 128 --games 16 --hidden 64 --ticks 5400 --w-dist 0.005
        --w-cov 0.3 --device cuda --fp16 --graph --train-rotations 4
        --report-every 5"

# 5a FIRST (the clean control): can novelty BOOTSTRAP room exploration
# from a fresh population at all? (40 gens ~4.4 h)
OUT=evo_runs/greenfield5a
rm -rf $OUT; mkdir -p $OUT
echo "=== 5a (fresh pop + w_cov 0.3) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.evolve $COMMON --gens 40 \
    --train-seed 62000 --seed 6 \
    --out-dir $OUT >> $OUT/run.log 2>&1
RC1=$?
echo "=== 5a rc=$RC1 done $(date) ===" | tee -a $OUT/run.log

# 5b SECOND (continuation test): does novelty reward push the run-4
# champion population (trained w_cov=0) past its plateau? (20 gens ~2 h)
OUT=evo_runs/greenfield5b
rm -rf $OUT; mkdir -p $OUT
echo "=== 5b (warm-start from run4 phase2 pop + w_cov 0.3) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.evolve $COMMON --gens 20 \
    --train-seed 62000 --seed 7 \
    --seed-from evo_runs/greenfield4/phase2/final_population.npz \
    --out-dir $OUT >> $OUT/run.log 2>&1
RC2=$?
echo "=== 5b rc=$RC2 done $(date) ===" | tee -a $OUT/run.log
echo "RUN5-COMPLETE rc=5a:$RC1 5b:$RC2" | tee -a evo_runs/greenfield5a/run.log
