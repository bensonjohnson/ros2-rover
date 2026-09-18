#!/usr/bin/env bash
# Run 18 (Spark): DOORWAY FORWARD GATE (sim prototype). usage: NAME SEED [extra]
# = run 15f (base genome, buildings direct L3, pool 1024 resampled per gen,
# 32 houses/genome, 120 gens, w_dist 0.005 w_cov 0.3 w_coll 1.0, w_rev 0.3,
# validation champion, NO trim randomization) with ONE change: --gate doorway
# (arena.GATE_PRESETS): corridor half-width 0.14 (= robot radius; the real
# monitor's 0.12 is narrower than the robot), stop 0.12 m from the bumper
# (0.18 m from centre), hysteresis 0.05 m, hold 0.15 s, side equalisation
# inside 0.10 m. The real rover's monitor is unchanged (user's decision).
#
# Background: forward exploring keeps coming out expensive — run 15f (legacy
# gate + w_rev 0.3) cross 0.487; run 16sr (symmetric gate + w_rev) 0.298 —
# while reverse explorers reach 0.86-0.89. Static re-scoring of evolved
# champions under the doorway gate changes little (they are adapted to their
# gate); only evolution under it can tell whether the forward rules are the
# doorway bottleneck.
#
# PRE-REGISTERED (stated before launch), fresh seed 8, vs run 15f:
#   H18a doorway gate makes forward exploring competitive: g119
#        hob_elite_cross_rate >= 0.770 AND hob_elite_rev_frac <= 0.30
#   H18b it at least helps clearly: g119 hob_elite_cross_rate >= 0.633
#        (1.3 x run 15f's 0.487) with rev <= 0.30
#   Collision guard as runs 11-17. The deep-evals run on the LEGACY gate (the
#   real rover's), showing how the champion would fare on today's monitor.
set -u
NAME=$1; SEED=$2; shift 2; EXTRA="$*"
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (doorway gate + w_rev 0.3, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --w-rev 0.3 --device cuda --train-rotations 4 \
    --report-every 5 --gens 120 --train-seed 63000 --holdout-seed 777000 \
    --seed $SEED --sigma0 0.05 --sigma-max 0.15 --pool 1024 \
    --resample-every 1 --world buildings --level 3 --gate doorway $EXTRA"
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
    echo "--- deep-eval world=$W champion=val_best_genome (LEGACY gate)" >> $OUT/run.log
    python3 -u -m evo.baselines --games 32 --device cuda --fp16 --world $W \
        --genome $OUT/val_best_genome.npz --population $OUT/final_population.npz \
        >> $OUT/run.log 2>&1
done
echo "--- gate rescore (legacy vs doorway)" >> $OUT/run.log
python3 -u -m evo.gate_rescore $OUT/val_best_genome.npz >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
