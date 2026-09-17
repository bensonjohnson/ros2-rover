#!/usr/bin/env bash
# Run 12 family (Spark only): split the run-10 genome bundle on BUILDINGS.
#   run12_variant.sh NAME SEED [extra evolve flags...]
# Base = run 11c config (9b genome h128 obs v1 action lr, fixed ES, direct
# level 3 buildings, pool 1024 resampled every gen, w_coll 1.0, 40 gens) plus
# the validation-set champion (val_best_genome.npz, seed 779000). Each variant
# changes ONE thing: --obs v2 | --action vw | --hidden 64.
#
# Run-11 verdicts that set this up: curriculum gave no benefit (mean hob elite
# cross rate 0.461 curriculum vs 0.467 direct over seeds 8/9) -> direct L3;
# collision guard passed in all four runs at w_coll 1.0; train-argmax
# best_genome was an unreliable pick (run 11 legacy deep-eval 0.14, 90.8
# collisions) -> champion now chosen on a separate validation set.
#
# PRE-REGISTERED (stated before launch). Primary metric: hob_elite_cross_rate
# at gen 39 (level-3 building holdout, never selected on). Controls: seed 8 =
# greenfield11c (0.410, Spark graph), seed 9 = greenfield12_ctl_s9 (this
# queue, Spark rerun of 11c_s9 so both seeds share the engine).
#   A change HELPS if its mean over seeds 8/9 >= 1.1x the control mean AND
#     each seed >= its control; HURTS if mean <= 0.9x control; else NEUTRAL.
#   Secondary (report, not a verdict): building deep-eval cross rate and
#     collisions of val_best_genome; legacy deep-eval val_best_genome >= 0.37.
#   Collision guard as run 11: g39 hob_champ_coll <= 20, elite train coll <= 3.
set -u
NAME=$1; SEED=$2; shift 2; EXTRA="$*"
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (buildings L3 direct, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
HID=128
case " $EXTRA " in *" --hidden "*) HID="";; esac
COMMON="--pop 128 --games 16 ${HID:+--hidden $HID} --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 --report-every 5 \
    --gens 40 --train-seed 63000 --holdout-seed 777000 --seed $SEED \
    --sigma0 0.05 --sigma-max 0.15 --pool 1024 --resample-every 1 \
    --world buildings --level 3 $EXTRA"
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
