#!/usr/bin/env bash
# Run 15 (Spark): FORWARD-DRIVING preference. usage: NAME SEED [FROM_DIR]
# Same config as runs 13/14 (base genome h128 obs v1 action lr, buildings
# direct L3, pool 1024 resampled per gen, 32 houses/genome, 120 gens, ticks
# 5400, w_dist 0.005 w_cov 0.3 w_coll 1.0, validation champion) plus
# --w-rev 0.3: fitness -= 0.3 x fraction of distance driven in reverse.
# FROM_DIR given -> warm-start that run's final population (keep sigma).
#
# Why: the run-14 champion explores BACKWARDS (sim mean command -0.72; on the
# real rover 75% reversing, 23% pivots, 0% forward — run 2, 2026-09-17).
# Nothing in the fitness priced direction; the real rover's bumper geometry
# and camera face forward. 0.3 x a 0.75 reverse fraction = -0.225, about a
# third of a champion's fitness, so forward driving is strongly preferred
# while a short reverse to escape a front block stays cheap.
#
# PRE-REGISTERED (stated before launch). Arms: 15f = fresh, seed 8 (control:
# greenfield13_s8, same config without w_rev); 15w = warm from
# greenfield14_s9, seed 9 (control: greenfield14_s9 champion).
#   DIRECTION: val_best_genome building deep-eval rev <= 0.30 AND g119
#     hob_elite_rev_frac <= 0.30.
#   COST 15f: g119 hob_elite_cross_rate >= 0.770 (0.9 x run 13 s8's 0.856).
#   COST 15w: val_best_genome building deep-eval rooms >= 3.24 and default
#     fitness >= 0.652 (0.85 x run-14 s9 champion's 3.81 / 0.767).
#   Arm PASS = DIRECTION and COST. Collision guard as runs 11-14.
# Deep-evals keep the DEFAULT fitness (no w_rev) so they compare with runs
# 11-14; the new rev column reports the reverse fraction.
set -u
NAME=$1; SEED=$2; FROM=${3:-}
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (w_rev 0.3, seed $SEED, from '${FROM:-fresh}') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
WARM=""
[ -n "$FROM" ] && WARM="--seed-from $FROM/final_population.npz --keep-seed-sigma"
echo "[engine] $FLAGS  [warm] ${WARM:-none}" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --w-rev 0.3 --device cuda --train-rotations 4 \
    --report-every 5 --gens 120 --train-seed 63000 --holdout-seed 777000 \
    --seed $SEED --sigma0 0.05 --sigma-max 0.15 --pool 1024 \
    --resample-every 1 --world buildings --level 3 $WARM"
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
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
