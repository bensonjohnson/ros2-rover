#!/usr/bin/env bash
# Run 22 (Spark): EXTEND run 21 (w_rev 0.15 + trim-rand) by 120 gens.
# usage: NAME SEED FROM_DIR
# Warm-starts run 21's final population (--seed-from, --keep-seed-sigma)
# with every setting identical to run21_forward.sh (base genome h128, LEGACY
# gate, buildings direct L3, pool 1024 resampled, 32x4 houses, 120 gens,
# w_dist 0.005 w_cov 0.3 w_coll 1.0 w_rev 0.15, --trim-rand 0.75).
# Local gen g here = global 120 + g.
#
# Why: every cross-rate leap in this project came from EXTENSION of a
# still-climbing lineage (13->14 reverse 0.86; 17->19e 0.703->0.803), and
# run 21's plateau window read B-A +0.049 = still rising at gen 119. The
# pre-registered FAIL said stop turning reward knobs; extension turns none.
# This is the last forward-ceiling probe before the memory-genome build.
#
# PRE-REGISTERED (stated before launch), fresh seed 9, deploy_pick
# multi-draw at local g119 on genomes with nominal rev <= 0.15:
#   PASS: worst-trim cross >= 0.60 (run 21 ceiling 0.510). If PASS, ship it
#     (verify parity + trim_eval + FIELD pivot-fraction <= 40%: pivot-heavy
#     genomes stall on the real skid-steer, live run 6).
#   FAIL: forward elite cross at g239 < 0.56 -> ~0.5 forward ceiling is
#     confirmed at 240 gens of w_rev search; forward gains then require the
#     memory genome (CAMEMBE read/write heads), not more search.
#   In between = PARTIAL: one more 60-gen extension maximum, then decide.
#   Guards: plateau test A=g45-59 B=g105-119 on hob_elite_cross; deploy
#   winner must sweep (pivot-fraction reported via live telemetry later).
set -u
NAME=$1; SEED=$2; FROM=$3
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (extend $FROM +120 gens, w_rev 0.15 + trim-rand, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS  [warm-start] $FROM/final_population.npz" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --w-rev 0.15 --device cuda --train-rotations 4 \
    --report-every 5 --gens 120 --train-seed 63000 --holdout-seed 777000 \
    --seed $SEED --sigma0 0.05 --sigma-max 0.15 --pool 1024 \
    --resample-every 1 --world buildings --level 3 --trim-rand 0.75 \
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
echo "--- deploy pick (multi-draw)" >> $OUT/run.log
python3 -u -m evo.deploy_pick --population $OUT/final_population.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
