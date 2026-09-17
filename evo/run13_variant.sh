#!/usr/bin/env bash
# Run 13 (Spark): LENGTH + POWER on the base config. usage: NAME SEED
# Base = run 11c/12 control config (9b genome h128, obs v1, action lr, fixed
# ES, buildings direct level 3, pool 1024 resampled every gen, w_coll 1.0,
# validation-set champion) with two changes:
#   --gens 120   (run 12 was still climbing at gen 39 in every run)
#   --games 32   (B=16384: 2x houses per genome for ~1.6x the tick cost —
#                 seed noise dominated run 12, where the two controls differed
#                 by 0.10 while the genome effects were 0.03-0.08)
#
# Run-12 verdict that sets this up: --obs v2 / --action vw / --hidden 64 are
# all NEUTRAL (means 0.97x / 1.07x / 1.00x of control), so the base genome
# stands and the open question is whether the base config plateaus.
#
# PRE-REGISTERED (stated before launch). Primary series:
# hob_elite_cross_rate (level-3 building holdout, never selected on) at report
# gens. Let A = mean over gens 75-89, B = mean over gens 105-119.
#   PLATEAU  if B - A < 0.02  -> the base recurrent MLP has converged; the
#            memory genome branch is then justified by evidence (first honest
#            case for it, since runs 1-9 were void).
#   CLIMBING if B - A >= 0.05 -> extend the run rather than change the genome.
#   AMBIGUOUS in between -> add seeds before any structural decision.
# Secondary (report, not verdicts): val_best_genome building deep-eval cross
# rate (run-12 best 0.500) and legacy fitness >= 0.37; collision guard as
# run 11 (g119 hob_champ_coll <= 20, elite train coll <= 3).
# Both seeds must agree in direction for a verdict.
set -u
NAME=$1; SEED=$2
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (buildings L3, base genome, 120 gens, games 32, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 --report-every 5 \
    --gens 120 --train-seed 63000 --holdout-seed 777000 --seed $SEED \
    --sigma0 0.05 --sigma-max 0.15 --pool 1024 --resample-every 1 \
    --world buildings --level 3"
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
