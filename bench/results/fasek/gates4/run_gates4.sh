#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/gates4"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== gates4 start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"

run_arm() {
  local arm=$1 plen=$2
  shift 2
  local dir="$OUT/$arm"
  mkdir -p "$dir"
  echo "--- $arm start $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)] env=[$*]" >> "$LOG"
  env $* "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len "$plen" --single-request --gate-tokens --out-dir "$dir" --out "$dir/result.json" >> "$LOG" 2>&1
  echo "--- $arm done rc=$? $(date +%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
}

run_arm s1 2k OMLX_EXPERT_STREAMING_STASH=1
run_arm k4a 8k
run_arm k4b 8k OMLX_EXPERT_STREAMING_PREFILL_QD=24
echo "=== gates4 done $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
echo ALLDONE
