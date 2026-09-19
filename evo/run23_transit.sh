#!/usr/bin/env bash
# Run 23 (Spark): TRANSLATION fitness. usage: NAME SEED [extra]
# run 21 recipe (w_rev 0.15 + trim-rand 0.75, fresh seed) PLUS the orbit
# guard born from live runs 5-8:
#   --w-net 0.15   pays furthest straight-line distance from start / 10 m
#                  (dist_m pays ARC length; zero-turn orbits earned 23 m
#                  of arc with 3 m of net travel and drove the real rover
#                  in circles)
#   --w-spin 0.3   penalizes pivot-tick fraction: the real skid-steer
#                  stalls the loaded track on opposite-sign commands
#                  (drive_diag: pivL/pivR -> gyro 0.00, right track dead)
#
# PRE-REGISTERED (stated before launch), deploy_pick multi-draw winner at
# g119, audited with evo.orbit_audit on the same holdout:
#   PASS: worst-trim cross >= 0.60 AND rev <= 0.15 AND coll <= 15
#         AND range >= 2.5 m AND spin_frac <= 0.30. Then field-verify
#         (parity + dry run + 60 s @ 0.8: expect real translation).
#   FAIL: winner spin_frac > 0.35 OR range < 2.0 m OR best forward cross
#         < 0.56 -> translation/doorway gains need the memory genome
#         (CAMEMBE read/write heads), not more reward shaping.
#   Between = PARTIAL: extend 60 gens once, same bar.
set -u
NAME=$1; SEED=$2; EXTRA=${3:-}
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (translation fitness: w_net 0.15 + w_spin 0.3 + trim-rand 0.75 + w_rev 0.15, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
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
