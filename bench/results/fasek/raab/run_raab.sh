#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/raab"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== raab start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 rep=$2
  shift 2
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --min-free-gb 16 --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
# on = RA default (F_RDADVISE on, STASH off); off = RA=0
run_arm on 1 OMLX_EXPERT_STREAMING_PROFILE=1
run_arm off 1 OMLX_EXPERT_STREAMING_PROFILE=1 OMLX_EXPERT_STREAMING_RA=0
run_arm on 2 OMLX_EXPERT_STREAMING_PROFILE=1
run_arm off 2 OMLX_EXPERT_STREAMING_PROFILE=1 OMLX_EXPERT_STREAMING_RA=0
run_arm on 3 OMLX_EXPERT_STREAMING_PROFILE=1
run_arm off 3 OMLX_EXPERT_STREAMING_PROFILE=1 OMLX_EXPERT_STREAMING_RA=0
echo "=== raab done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
