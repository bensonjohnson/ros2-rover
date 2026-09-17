#!/usr/bin/env bash
# Run 8b: the pre-registered loophole of the run-8 BCFAIL verdict —
# can a WIDER MLP hold the expert closed-loop where h64 could not?
# Run 8 (h64): 8/8 keepers near-perfect open-loop (mse<=0.01) but ALL
# collapsed to cov 0.026-0.062 = the cruise basin ES also froze in
# (runs 1-7a). If h128 also fails the min-cov 0.09 gate, the MLP
# genome is conclusively unable to hold explorers at any width ES can
# search -> memory genome (evo/policy.py) is the confirmed next build.
# NOTE: 6a falsified width for the ES SEARCH; that does not automatically
# falsify width for the distillation REPRESENATATION — this run closes it.
set -u
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/greenfield8b
BC=evo_runs/bc8b/bc_thetas.npz
rm -rf $OUT evo_runs/bc8b; mkdir -p $OUT
echo "=== 8b (BC h128 loophole) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.bc_seed --device cuda --train-seed 65000 --hidden 128 \
    --out $BC >> $OUT/run.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "=== 8b BC-FAIL rc=$RC — MLP CANNOT hold experts at h64 OR h128 -> memory genome $(date) ===" | tee -a $OUT/run.log
    echo "RUN8B-BCFAIL rc=$RC" | tee -a $OUT/run.log
    exit $RC
fi
python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 128 \
    --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 --graph \
    --train-rotations 4 --report-every 5 --gens 40 \
    --train-seed 65000 --holdout-seed 777000 --seed 8 \
    --bc-from $BC >> $OUT/run.log 2>&1
RC=$?
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== 8b rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN8B-COMPLETE rc=$RC" | tee -a $OUT/run.log
