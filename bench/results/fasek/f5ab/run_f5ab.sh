#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/f5ab"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== f5ab start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 rep=$2
  shift 2
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 8k --single-request --gate-tokens --min-free-gb 16 --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
# o = submission-order (default); c = completion-order
run_arm o 1
run_arm c 1 OMLX_EXPERT_STREAMING_RUN_WINDOW=completion
run_arm o 2
run_arm c 2 OMLX_EXPERT_STREAMING_RUN_WINDOW=completion
echo "=== f5ab done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
