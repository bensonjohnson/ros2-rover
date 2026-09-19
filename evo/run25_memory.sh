#!/usr/bin/env bash
# Run 25 (Spark): MEMORY GENOME — the pre-registered post-reactive build.
# usage: NAME SEED [extra]
# Engine line copied verbatim from run23_transit.sh (proven flags) with
# --mem 8 added. Recipe = run 23's (w_net 0.15 + w_spin 0.3 + trim-rand
# 0.75 + w_rev 0.15), hidden 128, L3 buildings, fresh seed 13.
#
# PRE-REGISTERED VERDICT (holdout L3, deploy-pick winner):
#   PASS    = worst-trim cross >= 0.60 AND rooms/game >= 2.0
#             AND spin <= 30% AND range >= 2.5 m  -> runner v2+mem port,
#             field-verify (doorway test), ship candidate
#   PARTIAL = rooms >= 2.0 but cross < 0.60 -> memory reaches rooms;
#             cross rate is then a gate/safety problem, not architecture
#   FAIL    = rooms < 2.0 after 120 gens -> memory never opened; next
#             step is the hand-designed waypoint scratchpad, NOT slots++
set -u
NAME=$1; SEED=$2; EXTRA=${3:-}
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (memory genome: run-23 recipe + --mem 8, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --mem 8 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --w-rev 0.15 --w-net 0.15 --w-spin 0.3 \
    --device cuda --train-rotations 4 --report-every 5 --gens 120 \
    --train-seed 63000 --holdout-seed 777000 --seed $SEED \
    --sigma0 0.05 --sigma-max 0.15 --pool 1024 --resample-every 1 \
    --world buildings --level 3 --trim-rand 0.75 $EXTRA"
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
echo "--- orbit audit (population)" >> $OUT/run.log
python3 -u -m evo.orbit_audit --fp16 --population $OUT/final_population.npz >> $OUT/run.log 2>&1
echo "--- trim robustness" >> $OUT/run.log
python3 -u -m evo.trim_eval --genome $OUT/val_best_genome.npz >> $OUT/run.log 2>&1
echo "--- deploy pick (multi-draw)" >> $OUT/run.log
python3 -u -m evo.deploy_pick --population $OUT/final_population.npz >> $OUT/run.log 2>&1
echo "--- orbit audit (deploy winner)" >> $OUT/run.log
python3 -u -m evo.orbit_audit --fp16 --genome $OUT/deploy_genome.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
