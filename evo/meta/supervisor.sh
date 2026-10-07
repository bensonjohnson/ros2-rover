#!/usr/bin/env bash
# Keep-alive wrapper for an unattended meta search (evo/meta/run.py).
#
#   nohup evo/meta/supervisor.sh --hours 48 --device cuda --parallel 1 &
#
# Why a wrapper at all: the manager is resumable, so the correct response to a
# crash (OOM, CUDA Xid, a reboot, an ssh drop killing the process group) is to
# start it again — it will skip every finished trial and continue where it
# stopped. A clean exit (rc 0) means the search finished or the budget was
# spent, so the wrapper stops instead of looping forever.
#
# Launch detached: `nohup ... &` or `setsid`. Do NOT launch it under a plain
# `ssh host 'command'` — when the ssh channel closes the process group dies
# with it (this is the same failure that once killed a rover colcon build).
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${META_ROOT:-$REPO/evo_runs/meta}"
PY="${META_PY:-python3}"
LOG="$ROOT/supervisor.log"
mkdir -p "$ROOT"
cd "$REPO"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[supervisor] attempt $attempt start $(date -Is) :: $PY -m evo.meta.run --root $ROOT $*" >> "$LOG"
  "$PY" -m evo.meta.run --root "$ROOT" "$@" >> "$LOG" 2>&1
  rc=$?
  echo "[supervisor] attempt $attempt exit rc=$rc $(date -Is)" >> "$LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] clean exit — search complete or budget spent; stopping" >> "$LOG"
    break
  fi
  sleep 30
done
