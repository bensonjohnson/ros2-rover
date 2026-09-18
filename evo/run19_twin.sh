#!/usr/bin/env bash
# Run 19t (Spark): FRESH seed-9 twin of run 17 (trim-rand 0.75). usage: NAME SEED [extra]
# Identical to run17_variant.sh (base genome h128 obs v1 action lr, LEGACY
# gate, buildings direct L3, pool 1024 resampled per gen, 32 houses x 4
# rotations, 120 gens, w_dist 0.005 w_cov 0.3 w_coll 1.0, --trim-rand 0.75)
# + a deploy_pick of the final population at the end.
#
# Why: run 17 (seed 8) is the only lineage whose champions are trim-robust
# (<= 10.4 collisions at every trim setting) — the hardware requirement.
# Seed spread between controls has been ~0.10 cross in this family, so one
# seed is not evidence: 19t gives the second trim-rand population for the
# worst-trim deploy pick AND the seed-agreement check for the 19e extension
# trajectory. No w_rev (runs 15f/16sr/18d: forward penalties cost more
# cross than they buy).
#
# PRE-REGISTERED (stated before launch), vs run 17 s8 at local g119:
#   T-agree: hob_elite_cross_rate(g119) within 0.15 of run 17's 0.703.
#   Deploy guard: trim_eval val_best_genome worst/nominal rooms >= 0.60 and
#   max coll across trims <= 20.
set -u
NAME=$1; SEED=$2; shift 2; EXTRA="$*"
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (trim-rand 0.75 fresh, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
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
echo "--- deploy pick (whole final population, trim envelope)" >> $OUT/run.log
python3 -u -m evo.deploy_pick --population $OUT/final_population.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
