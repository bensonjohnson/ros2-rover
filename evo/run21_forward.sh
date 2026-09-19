#!/usr/bin/env bash
# Run 21 (Spark): FORWARD-QUALITY attempt. usage: NAME SEED [extra]
# 15f recipe (w_rev forward enforcement) + trim-rand 0.75 (the hardware
# robustness lesson from runs 17/19) + HALVED penalty w_rev 0.15 (15f used
# 0.3 and its ceiling was cross 0.510 at rev 0.01 — the penalty may have
# over-taxed the door-escape reverses; 16sr suggests the gate, not the
# penalty, is the forward cost, so probe the middle ground).
# Fresh seed 8, base genome h128 obs v1 action lr, LEGACY gate (the real
# monitor), buildings direct L3, pool 1024 resampled per gen, 32x4 houses,
# 120 gens, w_dist 0.005 w_cov 0.3 w_coll 1.0.
#
# PRE-REGISTERED (stated before launch), deploy_pick multi-draw at g119:
#   PASS: a genome with worst-trim cross >= 0.60 AND nominal rev <= 0.15
#     AND worst-trim coll <= 15  (current forward ceiling: 15f idx87 at
#     0.510 / rev 0.01).
#   FAIL: best forward genome (rev <= 0.15) stays < 0.56 cross -> the ~0.5
#     forward ceiling is structural for this policy class + front-lidar
#     geometry; accept champ15f_idx87 for forward duty, reconsider rear
#     sensing rather than another reward knob.
#   (0.56-0.60 = PARTIAL: extend 40 gens before deciding.)
set -u
NAME=$1; SEED=$2; shift 2; EXTRA="$*"
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (trim-rand 0.75 + w_rev 0.15, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --w-rev 0.15 --device cuda --train-rotations 4 \
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
echo "--- deploy pick (multi-draw)" >> $OUT/run.log
python3 -u -m evo.deploy_pick --population $OUT/final_population.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
