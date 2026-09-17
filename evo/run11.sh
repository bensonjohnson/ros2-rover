#!/usr/bin/env bash
# Run 11 (Spark): BUILDINGS + CURRICULUM. 9b genome config (h128, obs v1,
# action lr, fixed ES) — the only change is the world: resampled building
# pools (1024 per level, fresh 64-house draw every gen) starting at level 0
# (2-3 rooms, 1.5-2.5 m archways, start 0.8-2.0 m from a door facing it),
# auto-advancing when the elite crosses into a 2nd room in >= 50% of games
# for 2 consecutive gens, up to level 3 (full mix incl. halls + mazes).
# Control: run11c_620.sh (V620) = same config, level 3 from gen 0, no
# curriculum. Chained after the Spark perf queue; uses the fastest engine
# that passed its production smoke (fused+compile > fused > fp16), falls
# back to eager fp16 if the building-mode graph dies before gen 1.
#
# PRE-REGISTERED (stated before launch):
#   H11 curriculum: PASS if ALL at gen 39: (a) level >= 2 reached;
#     (b) hob_elite_cross_rate (level-3 building holdout, never selected on)
#     >= 1.5x its gen-0 value; (c) hob_elite_cross_rate >= run 11c's.
#     FAIL if stuck at level 0 all 40 gens (elite cross rate < 0.5).
#   Curriculum verdict (with 11c): 11 >= 1.2x 11c on hob_elite_cross_rate at
#     gen 39 -> curriculum helps; 11 <= 11c -> spreading from gen 0 is as
#     good; between -> inconclusive (single seed each).
#   Transfer guard: legacy deep-eval best_genome >= 0.37 (9b: 0.3948) —
#     buildings training must not cost the old house distribution.
set -u
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/greenfield11
until grep -q "PERFQ-COMPLETE" evo_runs/perf_queue.log 2>/dev/null; do sleep 60; done
rm -rf $OUT; mkdir -p $OUT
echo "=== 11 (buildings + curriculum, 9b genome) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.test_buildings --device cuda >> $OUT/run.log 2>&1 \
    || { echo "RUN11-TESTFAIL" | tee -a $OUT/run.log; exit 1; }
FLAGS="--fp16 --graph"
if grep -q "\[smoke fused_compile\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --compile --graph"
elif grep -q "\[smoke fused\] rc=0 xid=0 capture_fail=0" evo_runs/perf_queue.log; then
    FLAGS="--fused --graph"
fi
echo "[engine] $FLAGS" | tee -a $OUT/run.log
COMMON="--pop 128 --games 16 --hidden 128 --ticks 5400 --w-dist 0.005 \
    --w-cov 0.3 --device cuda --train-rotations 4 --report-every 5 --gens 40 \
    --train-seed 63000 --holdout-seed 777000 --seed 8 --sigma0 0.05 \
    --sigma-max 0.15 --world buildings --level 0 --curriculum \
    --advance-at 0.5 --advance-patience 2 --pool 1024 --resample-every 1"
python3 -u -m evo.evolve --out-dir $OUT $COMMON $FLAGS >> $OUT/run.log 2>&1
RC=$?
if [ $RC -ne 0 ] && [ "$(grep -c fit_med $OUT/gens.jsonl 2>/dev/null || echo 0)" -lt 2 ]; then
    echo "=== 11 engine '$FLAGS' failed early (rc=$RC) -> eager fp16 $(date) ===" | tee -a $OUT/run.log
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
echo "=== 11 rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN11-COMPLETE rc=$RC" | tee -a $OUT/run.log
