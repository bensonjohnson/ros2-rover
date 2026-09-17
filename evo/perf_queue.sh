#!/usr/bin/env bash
# Spark perf validation queue — waits for run 9b to finish (never shares the
# GPU with a pre-registered run), then:
#   1. prof_tick: per-stage split for fp16 / fused / fused+compile
#   2. production-size 3-gen smoke --fused --graph      (9b config)
#   3. production-size 3-gen smoke --fused --compile --graph
#   4. tick_rankcorr on the 9b final population
# Each smoke: Xid count from journalctl -k, per-gen elapsed from gens.jsonl
# (compare against evo_runs/greenfield9b/gens.jsonl for the same gens).
set -u
source /home/benson/venv/bin/activate
cd ~/projects/ros2-rover
Q=evo_runs/perf_queue.log
until grep -q "RUN9B-COMPLETE" evo_runs/greenfield9b/run.log 2>/dev/null; do sleep 60; done
echo "=== perf queue start $(date) ===" | tee -a $Q
xid() { journalctl -k --since "$1" 2>/dev/null | grep -ci "xid"; }

T0=$(date "+%Y-%m-%d %H:%M:%S")
python3 -u -m evo.prof_tick --device cuda --hidden 128 >> $Q 2>&1
echo "[prof_tick] rc=$? xid=$(xid "$T0")" | tee -a $Q

for V in "fused:--fused" "fused_compile:--fused --compile"; do
    NAME=${V%%:*}; FLAGS=${V#*:}
    OUT=evo_runs/smoke9b_$NAME; rm -rf $OUT
    T0=$(date "+%Y-%m-%d %H:%M:%S")
    python3 -u -m evo.evolve --out-dir $OUT --pop 128 --games 16 --hidden 128 \
        --ticks 5400 --w-dist 0.005 --w-cov 0.3 --device cuda --fp16 --graph \
        --train-rotations 4 --report-every 100 --gens 3 \
        --train-seed 63000 --holdout-seed 777000 --seed 8 \
        --sigma0 0.05 --sigma-max 0.15 $FLAGS > $OUT.log 2>&1
    RC=$?
    echo "[smoke $NAME] rc=$RC xid=$(xid "$T0") capture_fail=$(grep -c 'graph capture failed' $OUT.log)" | tee -a $Q
    cat $OUT/gens.jsonl >> $Q 2>/dev/null
done

python3 -u -m evo.tick_rankcorr --device cuda \
    --population evo_runs/greenfield9b/final_population.npz >> $Q 2>&1
echo "=== perf queue done $(date) ===" | tee -a $Q
echo "PERFQ-COMPLETE" | tee -a $Q
