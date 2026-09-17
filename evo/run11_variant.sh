#!/usr/bin/env bash
# Run 11 family on the Spark: usage run11_variant.sh NAME SEED MODE
#   MODE curriculum = buildings from level 0, auto-advance (as evo/run11.sh)
#   MODE direct     = buildings at level 3 from gen 0, no curriculum (11c)
# Same config, engine selection and pre-registered criteria as evo/run11.sh;
# the curriculum verdict now uses BOTH seeds (8, 9) per mode: mean
# hob_elite_cross_rate at gen 39, and each seed pair must agree in direction.
#
# w_coll 1.0 (was 0.25 in runs 1-11a): the first run 11 attempt (seed 8,
# killed at gen 11, evo_runs/greenfield11_wc025_aborted) grew collision
# brute-forcers on the holdouts (building-holdout champion 624 collisions at
# gen 10, legacy-holdout champion fitness -0.38 at gen 5). Collisions count
# per tick in contact (~15 per second pushing a wall): at 0.25 a one-second
# shove cost 0.04 fitness, at 1.0 it costs 0.15 ~ one extra room in a
# six-room house. Deep-eval (baselines) keeps its default fitness so its
# numbers stay comparable with runs 5-10; collisions are reported alongside.
# Added pre-registered guard: gen-39 hob_champ_coll <= 20 and elite train
# coll <= 3 in all four runs, else collision shaping is still insufficient.
set -u
NAME=$1; SEED=$2; MODE=$3
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (buildings $MODE, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
elif grep -q "\[smoke fused\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --graph"
fi
if [ "$MODE" = curriculum ]; then
    WORLD="--world buildings --level 0 --curriculum --advance-at 0.5 --advance-patience 2"
else
    WORLD="--world buildings --level 3"
fi
echo "[engine] $FLAGS  [world] $WORLD" | tee -a $OUT/run.log
COMMON="--pop 128 --games 16 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --w-coll 1.0 --device cuda --train-rotations 4 --report-every 5 --gens 40 \
    --train-seed 63000 --holdout-seed 777000 --seed $SEED --sigma0 0.05 \
    --sigma-max 0.15 --pool 1024 --resample-every 1 $WORLD"
python3 -u -m evo.evolve --out-dir $OUT $COMMON $FLAGS >> $OUT/run.log 2>&1
RC=$?
if [ $RC -ne 0 ] && [ "$(grep -c fit_med $OUT/gens.jsonl 2>/dev/null || echo 0)" -lt 2 ]; then
    echo "=== $NAME engine '$FLAGS' failed early (rc=$RC) -> eager fp16 $(date) ===" | tee -a $OUT/run.log
    mv $OUT/gens.jsonl $OUT/gens_failed.jsonl 2>/dev/null
    rm -f $OUT/evolution.jsonl
    python3 -u -m evo.evolve --out-dir $OUT $COMMON --fp16 >> $OUT/run.log 2>&1
    RC=$?
fi
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
python3 -u -m evo.baselines --games 32 --device cuda --fp16 --world buildings \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== $NAME rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN-COMPLETE $NAME rc=$RC" | tee -a $OUT/run.log
