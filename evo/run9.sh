#!/usr/bin/env bash
# Run 9: FIXED ES — exact run-6a replica (h128, w_cov 0.3, train seed 63000,
# seed 8, same houses/flags) except the reproduction operator.
#
# Discovery (2026-09-16 review): runs 1-8 built children from the parent
# INDEX, not the parent genome (`p1.unsqueeze(1).expand(-1, N) * mix`), so
# every child was a constant vector + noise. Nothing was ever inherited:
# the "ES" was elitism over gen-0 samples + 10% random immigrants. Every
# falsification in runs 1-7a (reward shape/price, width, doors) was a test
# of RANDOM SEARCH, not of evolution. Also fixed: step size (sigma0 0.25 ->
# 0.05, clamp 1.0 -> 0.15, tau 0.4 -> 1/sqrt(N), sigma from the copied
# parent instead of max-of-two — the measured behaviour cliff sits between
# sigma 0.1 and 0.25), and readouts (elite + train-champion-on-holdout;
# per-gen gens.jsonl with child_win).
#
# PRE-REGISTERED (stated before launch), vs 6a on identical houses:
#   6a reference: fit_med pinned 0.3735-0.3744 gens 5-39; fit_best
#   0.395-0.4253; pop train_cells 0.063-0.066.
#   PASS (the operator was the wall) if by gen 19 ALL of:
#     (a) fit_med >= 0.385 (6a never left 0.3744)
#     (b) elite_cells at gen 19 >= 1.3x elite_cells at gen 1
#     (c) mean child_win over gens 1-19 >= 0.10 (children compete)
#   FAIL if fit_med <= 0.376 at gen 19 and elite_cells flat (+/-10%):
#     then a working ES on scan72 recurrent-MLP still plateaus -> the
#     memory-genome branch is supported by evidence, not by a bugged search.
#   Anything between = PARTIAL: extend to 40 gens before any verdict.
# Deep-eval gate at the end (Goodhart guard) before any champion claim.
set -u
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
OUT=evo_runs/greenfield9
rm -rf $OUT; mkdir -p $OUT
echo "=== 9 (fixed ES, 6a replica h128 w_cov 0.3) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 128 \
    --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 --graph \
    --train-rotations 4 --report-every 5 --gens 20 \
    --train-seed 63000 --holdout-seed 777000 --seed 8 \
    --sigma0 0.05 --sigma-max 0.15 >> $OUT/run.log 2>&1
RC=$?
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== 9 rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN9-COMPLETE rc=$RC" | tee -a $OUT/run.log
