#!/usr/bin/env bash
# V620 twin of evo/run11_variant.sh (same config, criteria, w_coll 1.0):
#   run11_variant620.sh NAME SEED MODE     MODE = curriculum | direct
# Engine: eager --fp16 --fused --compile (ROCm graph capture segfaults; the
# compile path measured 12.7 s/gen here vs Spark graph 13.2 s/gen). Falls
# back to plain eager fp16 if the fused/compile engine dies before gen 1.
set -u
NAME=$1; SEED=$2; MODE=$3
source /root/evo620/bin/activate
cd /root/evo620-src
OUT=evo_runs/$NAME
rm -rf $OUT; mkdir -p $OUT
echo "=== $NAME (V620 buildings $MODE, seed $SEED) start $(date) ===" | tee -a $OUT/run.log
FLAGS="--fp16 --fused --compile"
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
