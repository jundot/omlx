#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/attrib"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== attrib start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 rep=$2
  shift 2
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}

# pipeline default (AHEAD=3, promote on, merge gap 2)
run_arm x1 1
# N0: rolling without prefetch (AHEAD=0)
run_arm x2 1 OMLX_EXPERT_STREAMING_CTX_AHEAD=0
# P0: per-expert promote (single-promotion off)
run_arm x3 1 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
# G0: run-gap merge off
run_arm x4 1 OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
# legacy union
run_arm x5 1 OMLX_EXPERT_STREAMING_CTX_ROLLING=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
run_arm x1 2
run_arm x2 2 OMLX_EXPERT_STREAMING_CTX_AHEAD=0
run_arm x3 2 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
run_arm x4 2 OMLX_EXPERT_STREAMING_RUN_MERGE_GAP=0
run_arm x5 2 OMLX_EXPERT_STREAMING_CTX_ROLLING=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE=0 OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0
# p1: pipeline default + per-stage profile
run_arm p1 1 OMLX_EXPERT_STREAMING_PROFILE=1
echo "=== attrib done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
