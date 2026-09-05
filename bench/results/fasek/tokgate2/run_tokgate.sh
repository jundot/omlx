#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/tokgate2"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== tokgate2 start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1
  shift
  local dir="$OUT/$arm"
  mkdir -p "$dir"
  echo "--- $arm start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
run_arm t1
run_arm t2 OMLX_EXPERT_STREAMING_CTX_ROLLING=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
run_arm t3 OMLX_EXPERT_STREAMING_CTX_AHEAD=1 OMLX_EXPERT_STREAMING_RUN_QD=1
echo "=== tokgate2 done $(date +%Y-%m-%dT%H:%M:%S)" >> "$LOG"
echo ALLDONE
