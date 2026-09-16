#!/usr/bin/env bash
# Run 6a: capacity escalation test (hidden 128) + novelty.
# Runs 1-5 excluded reward weights, search budget, doorway width, and
# (5a/5b) reward *shape at hidden 64*: the cruiser/wall-hugger attractor
# survives all of them with median fitness pinned at 0.3824. Pre-registered
# next lever: expressivity. hidden 128 = 27k params (3.2x), policy still
# NOT the dominant rollout cost (raycast is 94% at h=64; h=128 adds ~15%
# total, h=512 would add ~200% while wrecking ES searchability at pop 128
# in 300k+ dimensions). Fresh train seed 63000; holdout 777000 unchanged.
set -uo pipefail
cd ~/projects/ros2-rover
source /home/benson/venv/bin/activate

OUT=evo_runs/greenfield6a
rm -rf $OUT; mkdir -p $OUT
echo "=== 6a (hidden 128 + w_cov 0.3) start $(date) ===" | tee -a $OUT/run.log
python3 -u -m evo.evolve --pop 128 --games 16 --hidden 128 --ticks 5400 \
    --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 --graph \
    --train-rotations 4 --report-every 5 --gens 40 \
    --train-seed 63000 --seed 8 \
    --out-dir $OUT >> $OUT/run.log 2>&1
RC=$?
echo "=== 6a rc=$RC done $(date) ===" | tee -a $OUT/run.log
echo "RUN6A-COMPLETE rc=$RC" | tee -a $OUT/run.log
