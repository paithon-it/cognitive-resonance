#!/usr/bin/env bash
# E16, confirmation: the primary judge now, the other two when the card has room (after the trainings of E15 and E14).
set -uo pipefail
. "$(dirname "$0")/env.sh"
export HF_HUB_DISABLE_TELEMETRY=1
PY="$RND/venv/bin/python"; OUT="$RND/out/judge"; LOG="$RND/out/logs"
echo "JUDGE START $(date -u +%FT%TZ)" >> "$LOG/queue.log"
"$PY" -I gpu/judge.py "$OUT" Qwen/Qwen2.5-1.5B-Instruct test 6000 > "$LOG/judge-test-qwen15.log" 2>&1
echo "JUDGE primary END $(date -u +%FT%TZ) rc=$?" >> "$LOG/queue.log"
while pgrep -f "gpu/crossing.py" > /dev/null; do sleep 20; done
"$PY" -I gpu/judge.py "$OUT" microsoft/Phi-3.5-mini-instruct test 6000 > "$LOG/judge-test-phi.log" 2>&1
echo "JUDGE phi END $(date -u +%FT%TZ) rc=$?" >> "$LOG/queue.log"
while pgrep -f "gpu/shortcut.py" > /dev/null; do sleep 20; done
"$PY" -I gpu/judge.py "$OUT" Qwen/Qwen2.5-7B-Instruct test 6000 > "$LOG/judge-test-qwen7.log" 2>&1
echo "JUDGE qwen7 END $(date -u +%FT%TZ) rc=$?" >> "$LOG/queue.log"
echo "JUDGE ALL END $(date -u +%FT%TZ)" >> "$LOG/queue.log"
