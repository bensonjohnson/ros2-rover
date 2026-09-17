#!/usr/bin/env bash
# Run 14 (Spark): EXTEND run 13 by 120 more gens. usage: NAME SEED FROM_DIR
# Warm-starts the whole run-13 final population (--seed-from ... 128 rows,
# --keep-seed-sigma so the self-adapted step sizes carry over ~0.047) and
# keeps every other setting identical to run 13 (base genome h128 obs v1
# action lr, buildings direct L3, pool 1024 resampled per gen, 32 houses per
# genome, ticks 5400, w_dist 0.005 w_cov 0.3 w_coll 1.0, validation champion).
# Local gen g here = global gen 120 + g.
#
# Run-13 verdict that sets this up: CLIMBING on both seeds (building-holdout
# elite cross rate late-window gain +0.053 s8 / +0.221 s9, threshold 0.05), so
# the pre-registered response is to extend rather than change the genome.
# Run-13 end state: hob elite cross 0.856 (s8) / 0.805 (s9); validation
# champions on 32 unseen buildings 0.587 / 0.612 fitness, 2.78 / 3.06 rooms,
# cross 0.812 / 0.844 vs best scripted 0.270 / 1.19 rooms / 0.19.
#
# PRE-REGISTERED (stated before launch), same plateau test on a later window:
# A = mean hob_elite_cross_rate over LOCAL gens 45-59, B = over LOCAL 105-119.
#   PLATEAU  if B - A < 0.02  -> the base recurrent MLP has converged; the
#            memory genome is then the justified next build.
#   CLIMBING if B - A >= 0.05 -> extend again (or accept and deploy-test).
#   AMBIGUOUS in between -> add seeds before any structural decision.
# Both seeds must agree. Guards unchanged: local-g119 hob_champ_coll <= 20 and
# train elite coll <= 3 (run 13: s8 2.42 PASS, s9 7.17 FAIL — if s9 stays over,
# the fix is capping per-game collision credit, NOT another w_coll raise,
# which would also suppress doorway attempts).
set -u
NAME=$1; SEED=$2; FROM=$3
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (extend $FROM by 120 gens, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS  [warm-start] $FROM/final_population.npz" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 --report-every 5 \
    --gens 120 --train-seed 63000 --holdout-seed 777000 --seed $SEED \
    --sigma0 0.05 --sigma-max 0.15 --pool 1024 --resample-every 1 \
    --world buildings --level 3 --seed-from $FROM/final_population.npz \
    --keep-seed-sigma"
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
