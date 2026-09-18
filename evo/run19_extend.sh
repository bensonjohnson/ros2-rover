#!/usr/bin/env bash
# Run 19e (Spark): EXTEND run 17 (trim-rand 0.75) by 120 gens. usage: NAME SEED FROM_DIR
# Warm-starts run 17's final population (--seed-from 128 rows, --keep-seed-sigma)
# with every other setting identical to run17_variant.sh (base genome h128
# obs v1 action lr, LEGACY gate = the real rover's monitor, buildings direct
# L3, pool 1024 resampled per gen, 32 houses x 4 rotations per genome,
# 120 gens, w_dist 0.005 w_cov 0.3 w_coll 1.0, validation champion,
# --trim-rand 0.75). Local gen g here = run-17 global gen 120 + g.
#
# Why: run 17 s8 ended gen 119 with hob_elite_cross 0.703 STILL CLIMBING
# (+0.12 over the last 20 gens), fit_med 0.477 rising, trim_eval worst/
# nominal rooms 0.54 with <= 10.4 collisions at every trim — the best
# hardware candidate lineage. No w_rev (runs 15f/16sr/18d: forward penalties
# cost more cross than they buy).
#
# PRE-REGISTERED (stated before launch):
#   Plateau test (same form as runs 13/14): A = mean hob_elite_cross_rate
#   over LOCAL gens 45-59, B = over LOCAL gens 105-119.
#     PLATEAU  if B - A < 0.02  -> trim-rand MLP converged; memory genome
#     CLIMBING if B - A >= 0.05 -> extend again or accept and deploy-test
#     AMBIGUOUS in between      -> rely on the seed-9 twin (19t) agreement
#   Deploy guard (the point of the run): trim_eval of val_best_genome at
#   g119 worst/nominal rooms >= 0.60 AND max collisions across trims <= 20
#   (run 17 end state: 0.54 / 10.4).
#   Collision guards as runs 11-18: g119 hob_champ_coll <= 20; train elite
#   coll <= 3 (run 17 ended 3.69 — judge the trend, not the last row).
#   deploy_pick then re-ranks the WHOLE final population across the trim
#   envelope; the deploy candidate is whatever survives worst-case scoring.
set -u
NAME=$1; SEED=$2; FROM=$3
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (extend $FROM +120 gens, trim-rand 0.75, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS  [warm-start] $FROM/final_population.npz" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 --report-every 5 \
    --gens 120 --train-seed 63000 --holdout-seed 777000 --seed $SEED \
    --sigma0 0.05 --sigma-max 0.15 --pool 1024 --resample-every 1 \
    --world buildings --level 3 --trim-rand 0.75 \
    --seed-from $FROM/final_population.npz --keep-seed-sigma"
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
echo "--- deploy pick (whole final population, trim envelope)" >> $OUT/run.log
python3 -u -m evo.deploy_pick --population $OUT/final_population.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
