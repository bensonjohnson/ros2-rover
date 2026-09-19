#!/usr/bin/env bash
# Run 24 (Spark): BLOCKED-FLAG INPUTS. usage: NAME SEED [extra]
# Run 21's recipe (w_rev 0.15 + trim-rand 0.75, fresh seed, NO w_net/w_spin)
# with --obs v2: proprio channels 2/3 (pure gyro noise in v1) replaced by
# the gate's latched front-blocked / rear-blocked flags. The doorway
# analyses (runs 15-18) showed the reactive policy's blind spot is not
# knowing forward is latched until it feels the override through prev_act;
# v2 makes it explicit. Never combined with the modern recipe (v2 last ran
# in the run-12 era: pre-buildings, pre-trim-rand, pre-w_rev).
#
# ARCHITECTURE knob, not reward shaping (run 23 exhausted the latter) —
# the blocked-direction input was one of the three fixes named there.
#
# PRE-REGISTERED (stated before launch), deploy_pick multi-draw winner at
# g119, audited (evo.orbit_audit) + trim_eval x2 seeds:
#   PASS: worst-trim cross >= 0.60 AND rev <= 0.15 AND coll <= 15 AND
#         range >= 2.5 m AND spin_frac <= 0.30 AND rooms/game >= 2.0.
#         -> field-verify on the rover (parity + dry + 90 s @ 0.8; runner
#            needs a v2 obs_mode branch + monitor publishing latch flags).
#   FAIL: best worst-trim cross < 0.56 OR rooms < 1.8 -> reactive class is
#        DONE (three consecutive structural FAILs: 21 ceiling, 23 shaping,
#        24 inputs). Next build: memory genome (explicit read/write slots,
#        CAMEMBE-style), design doc in docs/.
#   In between = PARTIAL: extend once (+60 gens), same bar.
set -u
NAME=$1; SEED=$2; EXTRA=${3:-}
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (v2 obs: blocked-flag channels; w_rev 0.15 + trim-rand 0.75, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --w-rev 0.15 --obs v2 \
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
