#!/bin/bash
set -e
D=bench/results/ppl_topk
mkdir -p $D
run_ppl() {
  local model=$1 topk=$2 out=$3
  echo "=== $out topk=$topk $(date +%H:%M:%S) ==="
  if [ "$topk" = "none" ]; then
    .venv/bin/python bench/ppl_expert_streaming.py --streaming --model "$model" --cold-tier none --budget 0 --max-windows 24 --corpus bench/corpus/pg1342.txt --out $D/$out.json 2>&1 | tail -n 3
  else
    .venv/bin/python bench/ppl_expert_streaming.py --streaming --model "$model" --cold-tier none --budget 0 --topk $topk --max-windows 24 --corpus bench/corpus/pg1342.txt --out $D/$out.json 2>&1 | tail -n 3
  fi
}
for i in 1 2 3; do
  run_ppl qwen-jang none 4s_topk_none_$i
  run_ppl qwen-jang 0.9 4s_topk_090_$i
  run_ppl qwen-jang 0.85 4s_topk_085_$i
done
echo 4S_PPL_DONE
for i in 1 2 3; do
  run_ppl qwen-jang4m none 4m_topk_none_$i
  run_ppl qwen-jang4m 0.85 4m_topk_085_$i
done
echo 4M_PPL_DONE
