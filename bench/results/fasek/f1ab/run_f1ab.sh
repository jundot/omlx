#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/f1ab"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== f1ab retry start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 plen=$2 rep=$3
  shift 3
  local dir="$OUT/$arm$rep"
  mkdir -p "$dir"
  echo "--- $arm rep $rep ($plen) retry start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len "$plen" --single-request --gate-tokens --min-free-gb 16 --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm rep $rep done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}
run_arm h 2k 1
run_arm r 2k 1 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
run_arm h 2k 2
run_arm r 2k 2 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
run_arm h 2k 3
run_arm r 2k 3 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
run_arm h 8k 1
run_arm r 8k 1 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
run_arm h 8k 2
run_arm r 8k 2 OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS=0
echo "=== f1ab retry done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
