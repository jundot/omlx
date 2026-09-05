#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/rem"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== remedida start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 rep=$2
  shift 2
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}

run_arm a 1
run_arm a 2
run_arm a 3
run_arm b 1 OMLX_EXPERT_STREAMING_CTX_ROLLING=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
run_arm b 2 OMLX_EXPERT_STREAMING_CTX_ROLLING=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
run_arm b 3 OMLX_EXPERT_STREAMING_CTX_ROLLING=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
run_arm d 1 OMLX_EXPERT_STREAMING_CTX_AHEAD=1 OMLX_EXPERT_STREAMING_RUN_QD=1
run_arm d 2 OMLX_EXPERT_STREAMING_CTX_AHEAD=1 OMLX_EXPERT_STREAMING_RUN_QD=1
run_arm d 3 OMLX_EXPERT_STREAMING_CTX_AHEAD=1 OMLX_EXPERT_STREAMING_RUN_QD=1
echo "=== remedida done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
