#!/usr/bin/env bash
# Runs one stage on the GPU machine, with several processes sharing the GPU (the networks are small).
# Usage, from the folder that contains gpu/:
#   bash gpu/run.sh pilot   [WORKERS]   # timing, frame probe, epochs x learning-rate grid at mu = 0, choice
#   bash gpu/run.sh bench   [WORKERS] [SEEDS] [TASKS]    # E1
#   bash gpu/run.sh rephase [WORKERS] [SEEDS] [TASKS]    # E2
#   bash gpu/run.sh people                               # E3: two base machines, then the fine-tuning
# Results go to $RND/out/<stage>/; the configuration chosen by the pilot is $RND/out/pilot/config.json.
set -uo pipefail
. "$(dirname "$0")/env.sh"
. "$RND/venv/bin/activate"
STAGE="$1"; W="${2:-1}"
OUT="$RND/out"; CFG="$OUT/pilot/config.json"
mkdir -p "$OUT/$STAGE/units" "$OUT/logs"
start=$(date +%s)
pids=()
case "$STAGE" in
  pilot)
    python gpu/pilot.py timing "$OUT/pilot" > "$OUT/logs/pilot-timing.log" 2>&1
    python gpu/pilot.py probe "$OUT/pilot" > "$OUT/logs/pilot-probe.log" 2>&1
    for w in $(seq 0 $((W - 1))); do
      python gpu/pilot.py grid "$OUT/pilot" "$w" "$W" > "$OUT/logs/pilot-grid-$w.log" 2>&1 & pids+=($!)
    done
    wait "${pids[@]}"
    python gpu/pilot.py choose "$OUT/pilot" > "$OUT/logs/pilot-choose.log" 2>&1 ;;
  bench|rephase)
    for w in $(seq 0 $((W - 1))); do
      python "gpu/$STAGE.py" "$OUT/$STAGE" "$CFG" "$w" "$W" ${3:+"$3"} ${4:+"$4"} > "$OUT/logs/$STAGE-$w.log" 2>&1 & pids+=($!)
    done
    wait "${pids[@]}" ;;
  people)
    # one process at a time: on this GPU concurrent processes lower the total throughput
    for s in 0 1; do
      [ -f "$OUT/people/base_s$s.npz" ] || python gpu/people.py base "$OUT/people" "$CFG" "$s" > "$OUT/logs/people-base-$s.log" 2>&1
    done
    for s in 0 1; do
      python gpu/people.py finetune "$OUT/people" "$CFG" "$s" > "$OUT/logs/people-ft-$s.log" 2>&1
    done ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
end=$(date +%s)
mkdir -p "$OUT/$STAGE/units"
printf '{"stage": "%s", "workers": %s, "start_unix": %s, "end_unix": %s, "wall_seconds": %s}\n' \
  "$STAGE" "$W" "$start" "$end" $((end - start)) > "$OUT/$STAGE/units/_stage.json"
touch "$OUT/$STAGE.done"
echo "DONE $STAGE $((end - start)) s"
