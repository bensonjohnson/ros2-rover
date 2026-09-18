#!/usr/bin/env bash
# Run 20m (Spark): MERGED-POOL experiment. usage: NAME SEED POOL
# Warm-starts ES from the deploy_pick-verified genomes of BOTH run-19 arms:
# 64 from greenfield19e_s8 (strong: worst-trim cross up to 0.82, but its
# champions brute-force collisions) + 64 from greenfield19t_s9 (honest:
# worst-trim coll <= 6, weaker cross). Everything else identical to
# run19_extend.sh (base genome h128, LEGACY gate, buildings direct L3, pool
# 1024 resampled per gen, 32x4 houses, 120 gens, w_dist 0.005 w_cov 0.3
# w_coll 1.0, --trim-rand 0.75). Fresh sigma handling: --seed-from the
# merged npz carries each donor genome's sigma (keep-seed-sigma).
#
# Hypothesis under test: honest exploration and strong room-crossing live in
# DIFFERENT genomes of the same algorithm family and therefore co-inherit
# under ES mutation. If true, the whole-population worst-trim numbers should
# rise toward 19e's level WITHOUT its collision pathology. If false (both
# metrics revert to one arm's level), the arms are separate attractor
# basins and we simply deploy 19e's verified idx95 or 19t's idx16.
#
# PRE-REGISTERED (stated before launch), fresh seed 10:
#   M-PASS: deploy_pick (multi-draw) at g119 shows a genome with worst-trim
#     cross >= 0.85 AND worst-trim coll <= 12 — better worst-case than BOTH
#     parents (19e: 0.823/11.9, 19t: 0.667/5.7).
#   M-PARTIAL: worst-trim cross >= 0.823 with coll <= 11.9, or cross >=
#     0.667 with coll < 5.7 (strict improvement on one axis, no regression
#     on the other).
#   M-FAIL: best genome inside the parents' envelope -> deploy the existing
#     verified finalists instead; do not extend further.
#   Guards as usual: plateau windows A=g45-59 B=g105-119 on hob_elite_cross;
#   g119 hob_champ_coll <= 20 (population-level truth comes from deploy_pick,
#   not the champion).
set -u
NAME=$1; SEED=$2; POOL=$3
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (merged pool $POOL, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
fi
echo "[engine] $FLAGS  [warm-start] $POOL" | tee -a $OUT/run.log
COMMON="--pop 128 --games 32 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 --report-every 5 \
    --gens 120 --train-seed 63000 --holdout-seed 777000 --seed $SEED \
    --sigma0 0.05 --sigma-max 0.15 --pool 1024 --resample-every 1 \
    --world buildings --level 3 --trim-rand 0.75 \
    --seed-from $POOL --keep-seed-sigma"
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
