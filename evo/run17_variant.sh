#!/usr/bin/env bash
# Run 17 (Spark): TRACK-TRIM DOMAIN RANDOMIZATION. usage: NAME SEED [extra]
# Same config as run 13 s8 (base genome, LEGACY gate = the real rover's
# current safety monitor, buildings direct L3, pool 1024 resampled per gen,
# 32 houses/genome, 120 gens, w_dist 0.005 w_cov 0.3 w_coll 1.0, validation
# champion) + --trim-rand 0.75: each generation, each house column gets
# independent effective track factors ~ U(0.75, 1.0) for left and right.
#
# Why (2026-09-18): the sim's 0.8 left trim is a physical slowdown (a
# 'straight' command curves 34 deg in 3 s) but on the real rover 0.8 is a
# driver-side correction; live run 2 turned the opposite way to the sim's
# prediction while reversing. evo.trim_eval on the run-14 champion: 4.28
# rooms at 0.8/1.0 but 1.78 rooms + 620 collisions balanced (1.0/1.0), worst
# /nominal rooms 0.32 — it steers WITH the sim's built-in curve.
#
# PRE-REGISTERED (stated before launch), fresh seed 8, vs run 13 s8:
#   H17a robust: evo.trim_eval of val_best_genome worst/nominal rooms >= 0.80
#        (run-14 champion 0.32) AND no trim setting with > 20 collisions.
#   H17b cheap: g119 hob_elite_cross_rate (holdout at nominal 0.8/1.0) >=
#        0.770 (0.9 x run 13 s8's 0.856).
#   Collision guard as runs 11-16.
set -u
NAME=$1; SEED=$2; shift 2; EXTRA="$*"
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (trim-rand 0.75, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 \
    --report-every 5 --gens 120 --train-seed 63000 --holdout-seed 777000 \
    --seed $SEED --sigma0 0.05 --sigma-max 0.15 --pool 1024 \
    --resample-every 1 --world buildings --level 3 --trim-rand 0.75 $EXTRA"
python3 -u -m evo.evolve --out-dir $OUT $COMMON $FLAGS >> $OUT/run.log 2>&1
RC=$?
if [ $RC -ne 0 ] && [ "$(grep -c fit_med $OUT/gens.jsonl 2>/dev/null || echo 0)" -lt 2 ]; then
    echo "=== $NAME engine '$FLAGS' failed early (rc=$RC) -> eager fp16 $(date) ===" | tee -a $OUT/run.log
    mv $OUT/gens.jsonl $OUT/gens_failed.jsonl 2>/dev/null
    rm -f $OUT/evolution.jsonl
    python3 -u -m evo.evolve --out-dir $OUT $COMMON --fp16 >> $OUT/run.log 2>&1
    RC=$?
fi
for W in legacy buildings; do
    echo "--- deep-eval world=$W champion=val_best_genome" >> $OUT/run.log
    python3 -u -m evo.baselines --games 32 --device cuda --fp16 --world $W \
        --genome $OUT/val_best_genome.npz --population $OUT/final_population.npz \
        >> $OUT/run.log 2>&1
done
echo "--- trim robustness" >> $OUT/run.log
python3 -u -m evo.trim_eval --genome $OUT/val_best_genome.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
