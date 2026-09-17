#!/usr/bin/env bash
# Run 11c (V620, eager fp16): CONTROL for run 11 — same 9b genome config and
# seed 8, same building pools machinery, but level 3 (full mix: bsp, halls,
# L/U footprints, open plans, mazes, legacy houses) from gen 0 and NO
# curriculum. Answers "spread across all houses from the start" vs run 11's
# "easy houses first". Criteria are in evo/run11.sh. Chained after run 10.
set -u
source /root/evo620/bin/activate
cd /root/evo620-src
OUT=evo_runs/greenfield11c
until grep -q "RUN10-COMPLETE" evo_runs/greenfield10/run.log 2>/dev/null; do sleep 60; done
rm -rf $OUT; mkdir -p $OUT
echo "=== 11c (buildings L3, no curriculum; control) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.test_buildings --device cuda >> $OUT/run.log 2>&1 \
    || { echo "RUN11C-TESTFAIL" | tee -a $OUT/run.log; exit 1; }
python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 128 \
    --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 \
    --train-rotations 4 --report-every 5 --gens 40 \
    --train-seed 63000 --holdout-seed 777000 --seed 8 \
    --sigma0 0.05 --sigma-max 0.15 --world buildings --level 3 \
    --pool 1024 --resample-every 1 >> $OUT/run.log 2>&1
RC=$?
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
python3 -u -m evo.baselines --games 32 --device cuda --fp16 --world buildings \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== 11c rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN11C-COMPLETE rc=$RC" | tee -a $OUT/run.log
