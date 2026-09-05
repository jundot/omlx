#!/bin/bash
set -e
D=bench/results/fasel/m_pins
P_DEC=bench/results/fasel/l2/profiles/decode.json
COMMON='--model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --min-free-gb 16 --knob pins_enabled --knob pin_budget_gib'
run_arm() {
  local name=$1; shift
  local envs=() cli=()
  for a in "$@"; do
    case "$a" in
      *=*) envs+=("$a") ;;
      *) cli+=("$a") ;;
    esac
  done
  echo "=== M-pins completion $name $(date +%H:%M:%S) ==="
  env OMLX_EXPERT_STREAMING_PROFILE=1 "${envs[@]}" .venv/bin/python -m bench.bench_expert_streaming $COMMON "${cli[@]}" --out-dir $D --out $D/$name.json
}
# Balanced completion: c3 (pins) then a4 (control) — the interleaving so
# far is a1,c1,a2,c2,a3, so c3..a4 keeps >=3 interleaved pairs.
run_arm m_c3 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC --pins --pin-gib 0.5 --pin-regime decode
run_arm m_a4
echo M_PINS_COMPLETE