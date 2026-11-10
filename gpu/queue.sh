#!/usr/bin/env bash
# The whole sequence after the pilot, in the protocol's order of priority: E1, E2, E3, one process at a time.
# Usage, from the folder that contains gpu/:  BENCH_SEEDS=0,1 bash gpu/queue.sh
set -uo pipefail
. "$(dirname "$0")/env.sh"
. "$RND/venv/bin/activate"
OUT="$RND/out"
bash gpu/run.sh bench 1 "${BENCH_SEEDS:-0,1,2}" "${BENCH_TASKS:-cat-dog,deer-horse,automobile-truck,airplane-ship,bird-frog}"
python gpu/manifest.py "$OUT" bench
bash gpu/run.sh rephase 1
python gpu/manifest.py "$OUT" rephase
bash gpu/run.sh people
python gpu/manifest.py "$OUT" people
echo "QUEUE END $(date -u +%T)"
