#!/bin/bash
set -e
D=bench/results/ppl_topk
mkdir -p $D
for m in none 0.9 0.85; do
  echo "=== ppl topk=$m $(date +%H:%M:%S) ==="
  args="--model qwen-jang --corpus bench/corpus/pg1342.txt --max-windows 24 --out $D/4s_topk_$m.json"
  if [ "$m" != "none" ]; then args="$args --topk $m"; fi
  .venv/bin/python bench/ppl_expert_streaming.py $args 2>&1 | grep -E 'ppl|nll|PPL' || true
done
echo PPL_DONE
