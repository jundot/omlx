#!/bin/bash
set -u
ROOT="/Volumes/SSD 4TB/DEV/omlx"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/bench/results/fasek/tokgate3"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== tokgate3 start $(date +%Y-%m-%dT%H:%M:%S) load=[$(sysctl -n vm.loadavg)]" >> "$LOG"
env "$PY" -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --out-dir "$OUT/t1" --out "$OUT/t1/result.json" >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
echo "=== tokgate3 done $(date +%Y-%m-%dT%H:%M:%S)" >> "$LOG"
echo ALLDONE
