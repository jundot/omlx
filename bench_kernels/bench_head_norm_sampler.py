#!/usr/bin/env python
"""Component-level GPU microbenchmark for LLM decode on Apple Silicon (M2 Ultra).

Targets everything OUTSIDE the 40 transformer decoder layers of
Ornith-1.0-35B-4bit (Qwen3.5-MoE hybrid, linear+full attention, 256 experts):
  - lm_head (4-bit quantized output projection, hidden -> vocab logits)
  - embedding_lookup (token id -> hidden, 4-bit quantized embedding table)
  - final_norm (the model's trailing RMSNorm)
  - layer_norm (ONE representative per-layer RMSNorm; model has 2 x 40 = 80 of these)
  - sampler (omlx's eager temp/top_p/top_k sampler, from omlx.utils.sampling)
  - dispatch_overhead (per-GPU-dispatch floor + per-call sync cost calibration)

Run with the omlx venv interpreter:
  /Volumes/Untitled/GitHub/omlx/.venv/bin/python bench_kernels/bench_head_norm_sampler.py

Methodology (must match the task spec exactly):
  - Real mlx-lm module classes/args: nn.Embedding, nn.Linear, nn.RMSNorm are the
    literal classes mlx_lm.models.qwen3_5.{Qwen3_5TextModel,TextModel,DecoderLayer}
    instantiate for embed_tokens / lm_head / {input,post_attention}_layernorm. We
    build ONLY those submodules (not the full 40-layer / 256-expert stack — that
    would need ~70GB transient bf16 memory to construct on a 64GB machine and its
    timing is explicitly out of scope for this component list) using shapes/eps
    parsed from the REAL TextModelArgs.from_dict() on the model's actual config.json
    (nested under text_config). Quantization (4-bit affine group_size=64) is applied
    to lm_head/embedding exactly as mlx_lm's TextModel.quant_predicate would: MoE
    router/shared_expert_gate paths get 8-bit, everything else (including lm_head
    and embed_tokens, per this model's config.json) gets the default 4-bit/64.
  - bf16 for all non-quantized params (norms).
  - Decode shapes: hidden state [B, 1, 2048] for B in {1, 2, 4, 8}.
  - Warmup 20 iters with mx.eval.
  - Mode A (amortized): K=50 forward calls chained WITHOUT intermediate eval,
    input varied each iteration so MLX can't collapse identical subgraphs, ONE
    mx.eval on the final outputs at the end; report wall/K ms.
  - Mode B (synced): K=50 calls, mx.eval after every call; report wall/K ms.
    (B - A) approximates per-call sync/dispatch overhead.
  - time.perf_counter for all wall-clock measurement.
"""

import json
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_5 import TextModelArgs
from omlx.utils.sampling import make_sampler

CONFIG_PATH = "/Volumes/Untitled/Models/Athena/mlx/Ornith-1.0-35B-4bit/config.json"
BATCH_SIZES = (1, 2, 4, 8)
WARMUP = 20
K_MODEL = 50
K_DISPATCH = 200
DISPATCH_SHAPE = (64, 64)

# ---------------------------------------------------------------------------
# Config loading — real TextModelArgs, real config.json (text_config nesting)
# ---------------------------------------------------------------------------


def load_text_args() -> TextModelArgs:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    text_cfg = cfg["text_config"]
    # TextModelArgs.from_dict (via BaseModelArgs) only consumes fields that are
    # declared dataclass fields; extras are dropped, so passing the raw nested
    # dict is exactly what mlx_lm's own model loader does.
    args = TextModelArgs.from_dict(text_cfg)
    return args, cfg


def quant_settings(cfg: dict) -> dict:
    q = cfg["quantization"]
    return {"group_size": q["group_size"], "bits": q["bits"], "mode": q["mode"]}


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def time_mode_a(make_input, forward, k=K_MODEL):
    """Chained, no intermediate eval — one mx.eval at the end. Returns ms/call."""
    x = make_input(0)
    outputs = []
    t0 = time.perf_counter()
    for i in range(k):
        x = make_input(i)
        y = forward(x)
        outputs.append(y)
    mx.eval(outputs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / k


def time_mode_b(make_input, forward, k=K_MODEL):
    """Synced — mx.eval after every call. Returns ms/call."""
    t0 = time.perf_counter()
    for i in range(k):
        x = make_input(i)
        y = forward(x)
        mx.eval(y)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / k


def warmup(make_input, forward, n=WARMUP):
    for i in range(n):
        mx.eval(forward(make_input(i)))


def bench_component(make_input, forward, k=K_MODEL):
    warmup(make_input, forward)
    ms_a = time_mode_a(make_input, forward, k)
    ms_b = time_mode_b(make_input, forward, k)
    return ms_a, ms_b


# ---------------------------------------------------------------------------
# Build components from the real mlx-lm classes/args
# ---------------------------------------------------------------------------


def build_components(args: TextModelArgs, qcfg: dict):
    hidden = args.hidden_size
    vocab = args.vocab_size
    eps = args.rms_norm_eps

    mx.random.seed(0)

    # embed_tokens: identical construction to Qwen3_5TextModel.__init__
    embed = nn.Embedding(vocab, hidden)
    embed.set_dtype(mx.bfloat16)
    embed_q = nn.QuantizedEmbedding.from_embedding(
        embed, group_size=qcfg["group_size"], bits=qcfg["bits"], mode=qcfg["mode"]
    )

    # lm_head: identical construction to TextModel.__init__ (tie_word_embeddings=False)
    lm_head = nn.Linear(hidden, vocab, bias=False)
    lm_head.set_dtype(mx.bfloat16)
    lm_head_q = nn.QuantizedLinear.from_linear(
        lm_head, group_size=qcfg["group_size"], bits=qcfg["bits"], mode=qcfg["mode"]
    )

    # final_norm: identical construction to Qwen3_5TextModel.norm
    final_norm = nn.RMSNorm(hidden, eps=eps)
    final_norm.set_dtype(mx.bfloat16)

    # layer_norm: identical construction to DecoderLayer.input_layernorm /
    # .post_attention_layernorm (both are plain nn.RMSNorm(hidden, eps); there
    # are 2 per layer x 40 layers = 80 in the real model, all the same shape/cost)
    layer_norm = nn.RMSNorm(hidden, eps=eps)
    layer_norm.set_dtype(mx.bfloat16)

    mx.eval(embed_q.parameters(), lm_head_q.parameters(), final_norm.parameters(), layer_norm.parameters())

    return embed_q, lm_head_q, final_norm, layer_norm


# ---------------------------------------------------------------------------
# Per-component benchmarks
# ---------------------------------------------------------------------------


def bench_lm_head(lm_head_q, hidden, batch):
    x0 = mx.random.normal((batch, 1, hidden)).astype(mx.bfloat16)

    def make_input(i):
        return x0 + (i * 1e-4)

    def forward(x):
        return lm_head_q(x)

    return bench_component(make_input, forward)


def bench_embedding(embed_q, vocab, batch):
    def make_input(i):
        # vary ids each iter (mod vocab) so MLX can't collapse identical subgraphs
        return mx.remainder(mx.full((batch, 1), i, dtype=mx.uint32), vocab)

    def forward(ids):
        return embed_q(ids)

    return bench_component(make_input, forward)


def bench_norm(norm_mod, hidden, batch):
    """RMSNorm output shape == input shape, so per spec we feed the output back
    in as the next call's input (norm(norm(norm(...x...)))) for Mode A chaining
    — this both satisfies the "feed output back if shape-compatible" rule and
    naturally varies the data each iteration (no manual epsilon needed)."""
    x0 = mx.random.normal((batch, 1, hidden)).astype(mx.bfloat16)

    warmup(lambda i: x0, norm_mod)

    # Mode A: chain norm(norm(norm(...x...))) for K calls, one eval at the end
    x = x0
    outs = []
    t0 = time.perf_counter()
    for i in range(K_MODEL):
        x = norm_mod(x)
        outs.append(x)
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_a = (t1 - t0) * 1000.0 / K_MODEL

    # Mode B: synced each call
    x = x0
    t0 = time.perf_counter()
    for i in range(K_MODEL):
        x = norm_mod(x)
        mx.eval(x)
    t1 = time.perf_counter()
    ms_b = (t1 - t0) * 1000.0 / K_MODEL

    return ms_a, ms_b


def bench_sampler(sampler, vocab, batch):
    logits0 = mx.random.normal((batch, vocab)).astype(mx.bfloat16)

    def make_input(i):
        return logits0 + (i * 1e-6)

    def forward(logits):
        return sampler(logits)

    return bench_component(make_input, forward)


def bench_dispatch_overhead():
    x0 = mx.random.normal(DISPATCH_SHAPE).astype(mx.float32)

    def forward(x):
        return x * 1.0001

    warmup(lambda i: x0, forward, n=WARMUP)

    # Mode A: chained, one eval at end
    x = x0
    t0 = time.perf_counter()
    for i in range(K_DISPATCH):
        x = x * 1.0001
    mx.eval(x)
    t1 = time.perf_counter()
    ms_a = (t1 - t0) * 1000.0 / K_DISPATCH

    # Mode B: synced every call
    x = x0
    t0 = time.perf_counter()
    for i in range(K_DISPATCH):
        x = x * 1.0001
        mx.eval(x)
    t1 = time.perf_counter()
    ms_b = (t1 - t0) * 1000.0 / K_DISPATCH

    return ms_a, ms_b


# ---------------------------------------------------------------------------
# Rough estimate of dispatches/decode-step from reading qwen3_5.py / qwen3_next.py
# ---------------------------------------------------------------------------

DISPATCH_ESTIMATE_NOTES = """
Rough op-count estimate per DecoderLayer.__call__ (qwen3_5.py), by reading code
(NOT measured — MLX doesn't expose a per-call dispatch counter used here):
  - input_layernorm (RMSNorm, fused mx.fast.rms_norm)              ~1 dispatch
  - linear_attn (GatedDeltaNet, 30/40 layers) OR self_attn (10/40):
      GatedDeltaNet: 4 in_proj matmuls + concat/conv1d/silu + split/reshape
                     + 2 rms_norm (q,k) + gated_delta_update (custom, treated
                     as ~3 fused kernels) + RMSNormGated (~2) + out_proj  ~20
      Attention (full, every 4th layer): q/k/v proj (3) + rope (2) +
                     sdpa (1 fused) + o_proj (1)                          ~10-12
  - post_attention_layernorm (RMSNorm)                                    ~1
  - SparseMoeBlock (num_experts=256, top-8): router matmul+softmax+topk
    (~5) + gathered switch_mlp expert matmuls (gate/up/down, grouped)
    (~8) + shared_expert (3 matmuls + gate, ~4) + combine (~3)          ~20-25
  Per-layer total: ~40-55 dispatches x 40 layers  =>  ~1,700-2,200 dispatches
  Plus embedding(1) + final_norm(1) + lm_head(1) + sampler(~5-8 for
  top_p/top_k/categorical) => decode step total order-of-magnitude ~1,800-2,300
  GPU dispatches. Labeled ESTIMATE ONLY, not a measured count.
"""


def main():
    args, cfg = load_text_args()
    qcfg = quant_settings(cfg)
    hidden = args.hidden_size
    vocab = args.vocab_size
    n_layers = args.num_hidden_layers

    print(f"hidden_size={hidden} vocab_size={vocab} num_hidden_layers={n_layers} "
          f"rms_norm_eps={args.rms_norm_eps} quant={qcfg}")

    embed_q, lm_head_q, final_norm, layer_norm = build_components(args, qcfg)

    gen_cfg = cfg.get("generation_config", {})
    sampler = make_sampler(
        temp=gen_cfg.get("temperature", 1.0),
        top_p=gen_cfg.get("top_p", 0.95),
        top_k=gen_cfg.get("top_k", 20),
    )
    print(f"sampler: temp={sampler.temp} top_p={sampler.top_p} top_k={sampler.top_k}")

    results = []

    for batch in BATCH_SIZES:
        ms_a, ms_b = bench_lm_head(lm_head_q, hidden, batch)
        results.append(("lm_head", batch, ms_a, ms_b))
        print(f"lm_head        B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms")

        ms_a, ms_b = bench_embedding(embed_q, vocab, batch)
        results.append(("embedding_lookup", batch, ms_a, ms_b))
        print(f"embedding_lookup B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms")

        ms_a, ms_b = bench_norm(final_norm, hidden, batch)
        results.append(("final_norm", batch, ms_a, ms_b))
        print(f"final_norm     B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms")

        ms_a, ms_b = bench_norm(layer_norm, hidden, batch)
        results.append(("layer_norm", batch, ms_a, ms_b))
        print(f"layer_norm     B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms")

        ms_a, ms_b = bench_sampler(sampler, vocab, batch)
        results.append(("sampler", batch, ms_a, ms_b))
        print(f"sampler        B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms")

    ms_a, ms_b = bench_dispatch_overhead()
    results.append(("dispatch_overhead", 1, ms_a, ms_b))
    print(f"dispatch_overhead (K={K_DISPATCH}, shape={DISPATCH_SHAPE})  "
          f"amortized={ms_a:8.5f} ms  synced={ms_b:8.5f} ms  "
          f"(sync_delta~{ms_b - ms_a:.5f} ms)")

    print(DISPATCH_ESTIMATE_NOTES)

    print("\n--- JSON ---")
    print(json.dumps(
        {
            "config": {
                "hidden_size": hidden,
                "vocab_size": vocab,
                "num_hidden_layers": n_layers,
                "rms_norm_eps": args.rms_norm_eps,
                "quantization": qcfg,
            },
            "results": [
                {"component": c, "batch": b, "ms_amortized": a, "ms_synced": s}
                for c, b, a, s in results
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
