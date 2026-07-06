#!/usr/bin/env python
"""
Component-level GPU microbenchmark for Ornith-1.0-35B-4bit (Qwen3.5-MoE hybrid) decode.

Benchmarks two decode-critical sub-modules built from the REAL mlx-lm model
classes (mlx_lm.models.qwen3_5 / qwen3_next), instantiated from the real
ModelArgs read out of the deployed model's config.json, with randomly
initialized weights (bf16 for non-quantized params, real 4-bit affine
quantization for the MoE expert weights matching the checkpoint's own
quantization_config):

  1. moe_block  - Qwen3NextSparseMoeBlock (SwitchGLU, 256 experts, top-8,
                  moe_intermediate_size 512, + dense shared-expert) run
                  QUANTIZED (4-bit experts / shared-expert, 8-bit
                  router-gate + shared-expert-gate, matching the
                  checkpoint's quant_predicate exactly).
  2. moe_router - just the gate Linear + softmax + top-k select/renorm
                  slice of (1), reported separately as a diagnostic
                  (NOTE: this cost is *already included inside* moe_block's
                  number -- see methodology_notes, do not double count).
  3. full_attn_block - Qwen3NextAttention (GQA, 16 Q heads / 2 KV heads,
                  head_dim 256) run bf16 (not quantized -- see
                  methodology_notes) with a real KVCache prefilled to 2048
                  tokens via one real "causal" prefill forward, then timed
                  over T=1 decode steps (cache grows 2048 -> 2048+K across
                  the timed loop, exactly like real decode).

Run with the omlx venv interpreter:
  /Volumes/Untitled/GitHub/omlx/.venv/bin/python \
      /Volumes/Untitled/GitHub/omlx/bench_kernels/moe_attn_component_bench.py

Writes nothing to disk; prints a JSON report to stdout.
"""

import json
import time

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import TextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

CONFIG_PATH = (
    "/Volumes/Untitled/Models/Athena/mlx/Ornith-1.0-35B-4bit/config.json"
)

HIDDEN = 2048
CTX_LEN = 2048  # realistic decode context to preload the KV cache to
BATCHES = [1, 2, 4, 8]
K = 50  # timed forward calls per mode
WARMUP = 20

# real-engine reference totals (ms/token) given in the task, for the sanity
# check against physically-absurd component shares.
REAL_ENGINE_MS = {1: 11.3, 2: 15.6, 4: 20.7, 8: 32.0}


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def load_args():
    cfg = json.load(open(CONFIG_PATH))
    text_cfg = cfg["text_config"]
    args = TextModelArgs.from_dict(text_cfg)
    quant_cfg = cfg.get("quantization_config", cfg.get("quantization", {}))
    return args, cfg, quant_cfg


def moe_quant_predicate(path, module):
    """Mirrors mlx_lm.models.qwen3_5.TextModel.quant_predicate exactly:
    the router gate and the shared-expert gate are 8-bit, everything else
    quantizable gets the default (4-bit / group_size 64 / affine)."""
    if not hasattr(module, "to_quantized"):
        return False
    leaf = path.split(".")[-1]
    if leaf in ("gate", "shared_expert_gate"):
        return {"group_size": 64, "bits": 8, "mode": "affine"}
    return True


def build_moe_block(args, group_size, bits):
    block = Qwen3NextSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    nn.quantize(
        block,
        group_size=group_size,
        bits=bits,
        mode="affine",
        class_predicate=moe_quant_predicate,
    )
    mx.eval(block.parameters())
    return block


class RouterOnly(nn.Module):
    """The gate/top-k slice of Qwen3NextSparseMoeBlock.__call__, isolated."""

    def __init__(self, args):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

    def __call__(self, x):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return scores


def build_router(args, group_size, bits):
    router = RouterOnly(args)
    router.set_dtype(mx.bfloat16)
    # the real checkpoint quantizes mlp.gate to 8-bit (see quant_predicate);
    # match that here for fidelity even though not explicitly required.
    nn.quantize(
        router,
        group_size=group_size,
        bits=8,
        mode="affine",
    )
    mx.eval(router.parameters())
    return router


def build_full_attn(args):
    attn = Qwen3NextAttention(args)
    attn.set_dtype(mx.bfloat16)
    mx.eval(attn.parameters())
    return attn


# --------------------------------------------------------------------------
# Timing harness
# --------------------------------------------------------------------------


def time_stateless(forward_fn, B, K=K, warmup=WARMUP):
    tiny = mx.array(1e-4, dtype=mx.bfloat16)

    def fresh_x():
        x = mx.random.normal((B, 1, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)
        return x

    # warmup (Mode B style: eval every iter)
    x = fresh_x()
    for _ in range(warmup):
        x = x + tiny
        out = forward_fn(x)
        mx.eval(out)

    # Mode A: chained, single eval at the end
    x = fresh_x()
    outs = []
    t0 = time.perf_counter()
    for _ in range(K):
        x = x + tiny
        out = forward_fn(x)
        outs.append(out)
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_amortized = (t1 - t0) * 1000.0 / K

    # Mode B: synced every call
    x = fresh_x()
    t0 = time.perf_counter()
    for _ in range(K):
        x = x + tiny
        out = forward_fn(x)
        mx.eval(out)
    t1 = time.perf_counter()
    ms_synced = (t1 - t0) * 1000.0 / K

    return ms_amortized, ms_synced


def prefill_cache(attn, B, ctx_len=CTX_LEN):
    cache = KVCache()
    xp = mx.random.normal((B, ctx_len, HIDDEN)).astype(mx.bfloat16)
    mx.eval(xp)
    mask = create_attention_mask(xp, cache)
    out = attn(xp, mask=mask, cache=cache)
    mx.eval(out, cache.keys, cache.values)
    return cache


def time_full_attn(attn, B, K=K, warmup=WARMUP):
    tiny = mx.array(1e-4, dtype=mx.bfloat16)

    # warmup (fresh cache, own prefill, evaluated every decode step)
    cache = prefill_cache(attn, B)
    x = mx.random.normal((B, 1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)
    for _ in range(warmup):
        x = x + tiny
        out = attn(x, mask=None, cache=cache)
        mx.eval(out)

    # Mode A: chained decode steps (cache genuinely grows 2048 -> 2048+K,
    # exactly as real decode does), single eval at the end
    cache = prefill_cache(attn, B)
    x = mx.random.normal((B, 1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)
    outs = []
    t0 = time.perf_counter()
    for _ in range(K):
        x = x + tiny
        out = attn(x, mask=None, cache=cache)
        outs.append(out)
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_amortized = (t1 - t0) * 1000.0 / K

    # Mode B: synced every decode step
    cache = prefill_cache(attn, B)
    x = mx.random.normal((B, 1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)
    t0 = time.perf_counter()
    for _ in range(K):
        x = x + tiny
        out = attn(x, mask=None, cache=cache)
        mx.eval(out)
    t1 = time.perf_counter()
    ms_synced = (t1 - t0) * 1000.0 / K

    return ms_amortized, ms_synced


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    mx.random.seed(0)
    args, cfg, quant_cfg = load_args()
    group_size = quant_cfg.get("group_size", 64)
    bits = quant_cfg.get("bits", 4)

    full_attn_count = sum(
        1
        for i in range(args.num_hidden_layers)
        if (i + 1) % args.full_attention_interval == 0
    )
    # Verified from code: mlx_lm.models.qwen3_5.DecoderLayer.__init__ does
    #   if args.num_experts > 0: self.mlp = SparseMoeBlock(args)
    #   else: self.mlp = MLP(...)
    # with NO decoder_sparse_step / mlp_only_layers gating in this arch's
    # DecoderLayer (that gating exists only in the plain qwen3_next.py
    # DecoderLayer, unused by qwen3_5). Since num_experts=256 > 0 for every
    # layer, ALL 40 layers get the sparse MoE FFN -- none are dense-only.
    moe_ffn_count = args.num_hidden_layers
    linear_attn_count = args.num_hidden_layers - full_attn_count

    layer_counts = {
        "moe_ffn": moe_ffn_count,
        "full_attention": full_attn_count,
        "linear_attention": linear_attn_count,
        "moe_router": moe_ffn_count,  # 1 router per MoE layer; NOT additive, see notes
        "total_layers": args.num_hidden_layers,
    }

    print(f"# layer_counts = {layer_counts}", flush=True)
    print(
        f"# quant: group_size={group_size} bits={bits} "
        f"(router-gate/shared_expert_gate overridden to 8-bit per checkpoint)",
        flush=True,
    )

    print("# building moe_block (quantized)...", flush=True)
    moe_block = build_moe_block(args, group_size, bits)
    print("# building moe_router (8-bit, diagnostic)...", flush=True)
    router = build_router(args, group_size, bits)
    print("# building full_attn_block (bf16, unquantized)...", flush=True)
    attn = build_full_attn(args)

    timings = []
    for B in BATCHES:
        print(f"# batch {B}: moe_block", flush=True)
        a, s = time_stateless(moe_block, B)
        timings.append({"component": "moe_block", "batch": B, "ms_amortized": a, "ms_synced": s})
        print(f"#   moe_block   B={B}  amortized={a:.4f}ms  synced={s:.4f}ms", flush=True)

        print(f"# batch {B}: moe_router", flush=True)
        a, s = time_stateless(router, B)
        timings.append({"component": "moe_router", "batch": B, "ms_amortized": a, "ms_synced": s})
        print(f"#   moe_router  B={B}  amortized={a:.4f}ms  synced={s:.4f}ms", flush=True)

        print(f"# batch {B}: full_attn_block", flush=True)
        a, s = time_full_attn(attn, B)
        timings.append({"component": "full_attn_block", "batch": B, "ms_amortized": a, "ms_synced": s})
        print(f"#   full_attn   B={B}  amortized={a:.4f}ms  synced={s:.4f}ms", flush=True)

    # extrapolate to a full decode step at batch 1 (Mode A / amortized numbers,
    # since Mode A approximates the marginal per-block cost inside one big
    # chained per-token graph -- which is exactly how the real engine's
    # 40-layer forward + single eval per token amortizes dispatch overhead).
    b1 = {t["component"]: t for t in timings if t["batch"] == 1}
    extrapolated = {
        "moe_ffn": b1["moe_block"]["ms_amortized"] * moe_ffn_count,
        "full_attention": b1["full_attn_block"]["ms_amortized"] * full_attn_count,
    }
    extrapolated["moe_ffn_plus_full_attention_total"] = (
        extrapolated["moe_ffn"] + extrapolated["full_attention"]
    )
    extrapolated["real_engine_b1_ms"] = REAL_ENGINE_MS[1]
    extrapolated["share_of_real_engine_b1"] = (
        extrapolated["moe_ffn_plus_full_attention_total"] / REAL_ENGINE_MS[1]
    )

    print("\n# ---- extrapolation @ batch=1 ----", flush=True)
    print(json.dumps(extrapolated, indent=2), flush=True)

    report = {
        "layer_counts": layer_counts,
        "timings": timings,
        "extrapolated_ms_per_step": extrapolated,
    }
    print("\n# ---- FULL JSON REPORT ----")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
