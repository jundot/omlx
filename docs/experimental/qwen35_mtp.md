# Qwen3.5/3.6 native MTP modes

Qwen-family checkpoints with native MTP heads use a fixed draft depth of three.
A fixed verify width keeps the default fast path deterministic and was faster
than the adaptive controller across code and prose in local testing. Other MTP
architectures retain their adaptive policy.

## Default fast mode

The default path verifies every draft with the target model while using
verify-width quantized kernels. Multi-row quantized accumulation can differ from
serial one-row arithmetic, so this mode prioritizes throughput rather than
serial-token bit parity.

Post-load and verify optimizations include:

- physical dense gate/up projection fusion;
- a register-neutral Q4 gate/up + SwiGLU kernel;
- Qwen GDN fused prework, two-SIMD-group recurrence, and rejection-only state
  replay;
- MTP-head Q/K/V projection fusion; and
- cached routing geometry for fixed-width quantized projections.

## Serial-exact mode

Set `OMLX_QWEN35_EXACT_VERIFY=1` before model load to route Qwen target verifies
through cross-row Q4/Q5 kernels that preserve serial M=1 accumulation order:

```bash
OMLX_QWEN35_EXACT_VERIFY=1 omlx serve
```

Exact routing supports verify widths 2–4, group size 64, normal and tiny GDN
output widths, and a dedicated large-vocabulary launch geometry. Unsupported
shapes fall back to independent one-row projections. The opt-in is deliberately
Qwen-specific; it does not alter DeepSeek, Gemma, Inkling, or other MTP paths.

The exact path is slower than the default fast mode but was independently
validated against `mtp_enabled=false` output hashes on code and prose prompts.
