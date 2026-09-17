#!/usr/bin/env bash
# Run 10 (V620, eager): genome v2 — obs v2 (gate flags replace the two
# pure-noise gyro channels, inputs centred), action vw (forward/turn outputs),
# hidden 64 (width verdicts from runs 1-8 were void; fewer genes = faster
# truncation-GA search). Chained after run 9b_r on the same box so the
# comparison is same machine, same eager path, same seed 9, same houses.
#
# PRE-REGISTERED (stated before launch), vs 9b_r at the same generation:
#   PASS (genome v2 helps) if at gen 19: elite_cells >= 1.2x 9b_r's AND
#     fit_med > 9b_r's fit_med (fitness is defined on env metrics only, so
#     it is comparable across genome modes)
#   FAIL if elite_cells <= 9b_r's: v2 bundle does not help at h64 — split
#     the bundle (h128 v2/vw) before discarding any single change.
#   Otherwise PARTIAL. Deep-eval gate before any champion claim.
set -u
source /root/evo620/bin/activate
cd /root/evo620-src
until grep -q "RUN9BR-COMPLETE" evo_runs/greenfield9b_r/run.log 2>/dev/null; do sleep 60; done
OUT=evo_runs/greenfield10
rm -rf $OUT; mkdir -p $OUT
echo "=== 10 (genome v2: obs v2, action vw, h64; V620 seed 9) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 64 \
    --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 \
    --train-rotations 4 --report-every 5 --gens 20 \
    --train-seed 63000 --holdout-seed 777000 --seed 9 \
    --sigma0 0.05 --sigma-max 0.15 --obs v2 --action vw >> $OUT/run.log 2>&1
RC=$?
python3 -u -m evo.baselines --games 32 --device cuda --fp16 \
    --genome $OUT/best_genome.npz --population $OUT/final_population.npz \
    >> $OUT/run.log 2>&1
echo "=== 10 rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN10-COMPLETE rc=$RC" | tee -a $OUT/run.log
