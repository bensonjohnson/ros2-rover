#!/usr/bin/env bash
# Run 16 (Spark): SYMMETRIC SAFETY GATE in the sim. usage: NAME SEED [extra]
# Same config as run 13/15 (base genome, buildings direct L3, pool 1024
# resampled per gen, 32 houses/genome, 120 gens, w_dist 0.005 w_cov 0.3
# w_coll 1.0, validation champion) + --gate symmetric: every forward gate
# rule mirrored for reverse (rear corridor with arc, 0.15 m stop, hold +
# hysteresis latch, rear-flank 90-150 deg equalization). The real rover keeps
# its current gate until the user decides.
#
# Background (2026-09-18): runs 11-15 evolved REVERSE explorers. The corrected
# mirror test (track trims swapped with the frame) shows forward driving is
# nearly as good for the same policy (run-14 champion 4.31 rooms reverse vs
# 3.56 mirrored forward, cross 0.875 both), so the legacy gate's front-heavy
# asymmetry is worth ~15-20% to reversing; run 15f (w_rev 0.3, legacy gate)
# only found a 1.6-room forward basin. Ablation: no single forward rule
# (hold / hysteresis / side-eq) matters alone; the arc prediction helps.
#
# PRE-REGISTERED (stated before launch), seed 8, both arms fresh:
#   16s  (symmetric, no w_rev) vs run 13 s8 (legacy gate, same config):
#     H16a gate drives the reverse habit: g119 hob_elite_rev_frac < 0.50
#     H16b symmetric gate costs little: g119 hob_elite_cross_rate >= 0.770
#          (0.9 x run 13 s8's 0.856)
#   16sr (symmetric + w_rev 0.3) vs run 15f (legacy + w_rev 0.3, 0.487):
#     H16c forward exploring is cheap once the gate is symmetric: g119
#          hob_elite_cross_rate >= 0.770 AND hob_elite_rev_frac <= 0.30
#   Collision guard as runs 11-15. Deep-evals use default fitness; the
#   legacy-gate deep-eval shows how each champion fares on today's rover gate.
set -u
NAME=$1; SEED=$2; shift 2; EXTRA="$*"
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (symmetric gate, seed $SEED, extra '$EXTRA') start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 \
    --report-every 5 --gens 120 --train-seed 63000 --holdout-seed 777000 \
    --seed $SEED --sigma0 0.05 --sigma-max 0.15 --pool 1024 \
    --resample-every 1 --world buildings --level 3 --gate symmetric $EXTRA"
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
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
