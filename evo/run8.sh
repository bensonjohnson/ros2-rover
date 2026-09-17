#!/usr/bin/env bash
# Run 8: BC-seeded ES (behavior-cloned explorer seeding).
# Runs 1-7a verdict: reward weights (1-4), basin width (4), reward SHAPE at
# h64 (5), CAPACITY h128 (6a), reward PRICE w_cov 0.6 (7a) all falsified —
# population never left the cruise attractor (train_cells 0.063-0.073,
# rooms ~1.01). explore_probe: scripted personalities cover 0.11-0.144 =
# 2x the evolved pop -> exploration IS representable from scan72; what is
# missing is a mutation path INTO that basin. Run 8 = put BC-distilled
# scripted explorers in the founding population, hill-climb from there.
# hidden 64 (6a proved width is not the wall; smaller genome = easier ES).
# w_cov 0.3 = run 5/6a champion-ceiling price (BC members' 0.11-0.14
# coverage now has elite competitors IN the population from gen 0).
# Fresh train seed 65000 (bc data + train match); holdout 777000 unchanged.
set -u
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/greenfield8
BC=evo_runs/bc8/bc_thetas.npz
rm -rf $OUT evo_runs/bc8; mkdir -p $OUT
echo "=== 8 (BC-seeded ES, h64, w_cov 0.3) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.bc_seed --device cuda --train-seed 65000 --hidden 64 \
    --out $BC >> $OUT/run.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "=== 8 BC-FAIL rc=$RC (min-cov gate rejected too many keepers) $(date) ===" | tee -a $OUT/run.log
    echo "RUN8-BCFAIL rc=$RC" | tee -a $OUT/run.log
    exit $RC
fi
python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 64 \
    --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 --graph \
    --train-rotations 4 --report-every 5 --gens 40 \
    --train-seed 65000 --holdout-seed 777000 --seed 8 \
    --bc-from $BC >> $OUT/run.log 2>&1
RC=$?
# Goodhart guard: deep-eval champion + population on the honest default
# fitness (runs 5a/6a taught us train-fitness argmax != deployable).
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== 8 rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN8-COMPLETE rc=$RC" | tee -a $OUT/run.log
