#!/usr/bin/env bash
# Run 9b_r (V620 eager replicate of run9b.sh, seed 9): FIXED ES + FIXED POLICY READOUT — 6a replica (h128, w_cov 0.3,
# train seed 63000, seed 8, same houses/flags), obs v1 / action lr.
#
# Supersedes run 9 (killed at gen 1): a second bug predating run 1 —
# PopulationNet used einsum("pgh,pgHK->pgK", h, Wo); the mismatched h/H
# letters sum each side, so BOTH wheel commands were tanh(sum(h) *
# colsum(Wo) + bo), functions of ONE scalar (evo.test_policy P2). Speed and
# steering were never independently controllable in runs 1-9, and bc_seed
# (trained with the correct h @ Wo) replayed its students through the
# broken readout — the run-8 "mse fine, replay collapses" signature.
# Plus the run-9 ES fix (children inherit the genome, not the index;
# sigma0 0.05, sigma <= 0.15, tau 1/sqrt(N)).
#
# PRE-REGISTERED (stated before launch). Gen-0 fitness is not comparable to
# 6a (readout change), so criteria are within-run:
#   PASS (search + readout were the wall) if by gen 19 ALL of:
#     (a) fit_med(g19) - fit_med(g1) >= +0.010   (6a: +0.0004 over 34 gens)
#     (b) elite_cells(g19) >= 1.3x elite_cells(g1)
#     (c) mean child_win over gens 1-19 >= 0.10
#   FAIL if (a) < +0.003 AND elite_cells within +/-10% of g1 -> a working
#     ES with a working readout still plateaus on scan72 -> memory-genome
#     branch supported by evidence.
#   Anything between = PARTIAL -> extend to 40 gens before a verdict.
#   Replicate run9b_r (V620, seed 9) must agree in direction.
# Deep-eval gate at the end (Goodhart guard) before any champion claim.
set -u
source /root/evo620/bin/activate
cd /root/evo620-src
OUT=evo_runs/greenfield9b_r
rm -rf $OUT; mkdir -p $OUT
echo "=== 9b_r (V620 replicate, seed 9, eager) (fixed ES + readout, 6a replica h128 w_cov 0.3) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 128 \
    --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 \
    --train-rotations 4 --report-every 5 --gens 20 \
    --train-seed 63000 --holdout-seed 777000 --seed 9 \
    --sigma0 0.05 --sigma-max 0.15 >> $OUT/run.log 2>&1
RC=$?
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== 9b_r (V620 replicate, seed 9, eager) rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN9BR-COMPLETE rc=$RC" | tee -a $OUT/run.log
