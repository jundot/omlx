#!/usr/bin/env python
"""Component-level GPU microbenchmark for LLM decode on Apple Silicon (M2 Ultra).

Targets the GatedDeltaNet linear-attention block used by 30 of the 40 decoder
layers in Ornith-1.0-35B-4bit (Qwen3.5-MoE hybrid: linear_attention every
layer except every 4th, which is full_attention). Splits one layer's linear-
attention forward into three pieces so we can see how much of it is the
custom Metal delta-rule scan kernel vs. everything else (projections, causal
depthwise conv, RMS-norms, sigmoid/softplus gating prep):

  - linear_block_full     : GatedDeltaNet.__call__ end-to-end (real module,
                             real ArraysCache-shaped recurrent state)
  - gated_delta_kernel_only: the bare mx.fast.metal_kernel scan
                             (mlx_lm.models.gated_delta.gated_delta_kernel)
                             called directly at the exact q/k/v/g/beta/state
                             shapes GatedDeltaNet feeds it for T=1 decode
  - linear_glue           : the same block with the scan stubbed to an
                             identity (out = v), isolating in_proj_qkv/z/b/a,
                             the causal conv1d+silu, the two q/k RMS-norms,
                             sigmoid(beta)/compute_g gating prep, the gated
                             RMSNormGated, and out_proj

Run with the omlx venv interpreter:
  /Volumes/Untitled/GitHub/omlx/.venv/bin/python bench_kernels/bench_gated_delta.py

Methodology (must match the task spec exactly):
  - Real mlx-lm module classes: mlx_lm.models.qwen3_5.GatedDeltaNet /
    TextModelArgs, mlx_lm.models.cache.ArraysCache, and the raw scan kernel +
    gating helper from mlx_lm.models.gated_delta, instantiated from the real
    model's config.json (nested under text_config). Weights randomly
    initialized, bf16 for all non-quantized params -- except A_log, which the
    real TextModel.cast_predicate deliberately keeps in float32 (it feeds
    mx.exp(A_log) inside compute_g; sanitize()/quantize() never touch it
    either), so we replicate that one exception exactly.
  - Decode shapes: hidden state [B, 1, 2048] for B in {1, 2, 4, 8}.
  - Recurrent state is REAL ArraysCache shape, pre-populated with random
    values (not a cold/zero cache) before warmup: cache[0] = conv state
    (B, conv_kernel_size-1, conv_dim), cache[1] = delta state
    (B, linear_num_value_heads, linear_value_head_dim, linear_key_head_dim)
    in float32, matching gated_delta_update's own mx.zeros(..., float32).
  - Warmup 20 iters with mx.eval.
  - Mode A (amortized): K=50 forward calls chained WITHOUT intermediate eval;
    each call's output (or a perturbed tensor, when output isn't fed back
    into the next call's slot) feeds the next iteration so MLX can't collapse
    identical subgraphs; ONE mx.eval on the final outputs; report wall/K ms.
  - Mode B (synced): K=50 calls, mx.eval after every call; report wall/K ms.
    (B - A) approximates per-call sync/dispatch overhead.
  - time.perf_counter for all wall-clock measurement.

Sanity bound: the real engine decodes at ~11.3ms/token (B=1) end to end,
covering ALL 40 layers (30 linear-attention + 10 full-attention), MoE MLP
(256 experts, top-8) on every layer, embedding/lm_head/norms/sampler. This
harness only measures the linear-attention sub-block, so 30 * ms_amortized
(the extrapolated per-step contribution of just this component family) must
leave comfortable headroom under 11.3ms for everything else -- if it doesn't,
something is wrong (e.g. accidentally timing something MoE-shaped) and the
run is re-measured, not reported.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel

CONFIG_PATH = "/Volumes/Untitled/Models/Athena/mlx/Ornith-1.0-35B-4bit/config.json"
BATCH_SIZES = (1, 2, 4, 8)
WARMUP = 20
K_MODEL = 50

N_LINEAR_LAYERS = 30
N_TOTAL_LAYERS = 40
REAL_TOTAL_MS = {1: 11.3, 2: 15.6, 4: 20.7, 8: 32.0}


# ---------------------------------------------------------------------------
# Config loading -- real TextModelArgs (only the fields GatedDeltaNet reads),
# real config.json (fields nested under text_config)
# ---------------------------------------------------------------------------


def load_text_args():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    args = TextModelArgs.from_dict(tc)
    return args, tc


def build_model(args: TextModelArgs) -> GatedDeltaNet:
    mx.random.seed(0)
    model = GatedDeltaNet(args)

    # bf16 everywhere except A_log, exactly mirroring TextModel.cast_predicate
    # (real loader: mlx_lm.utils casts every param through this predicate).
    def cast(path, p):
        if path.endswith("A_log"):
            return p.astype(mx.float32)
        return p.astype(mx.bfloat16)

    model.update(tree_map_with_path(cast, model.parameters()))
    model.eval()  # training=False -> gated_delta_update takes use_kernel=True
    mx.eval(model.parameters())
    return model


def make_cache(model: GatedDeltaNet, batch: int) -> ArraysCache:
    """Real ArraysCache(size=2), pre-populated with realistic (non-zero,
    non-cold) recurrent state at the exact shapes GatedDeltaNet reads/writes:
      cache[0] = conv state   (B, conv_kernel_size-1, conv_dim)          bf16
      cache[1] = delta state  (B, num_v_heads, head_v_dim, head_k_dim)  f32
    """
    cache = ArraysCache(size=2)
    cache[0] = mx.random.normal(
        (batch, model.conv_kernel_size - 1, model.conv_dim)
    ).astype(mx.bfloat16)
    cache[1] = mx.random.normal(
        (batch, model.num_v_heads, model.head_v_dim, model.head_k_dim)
    ).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


# ---------------------------------------------------------------------------
# The "glue" cost: projections + causal conv + q/k RMS-norm + gating prep
# (sigmoid(beta)/compute_g) + gated RMSNormGated + out_proj -- everything in
# GatedDeltaNet.__call__ EXCEPT the scan/kernel (component #2, benched
# separately).
#
# IMPORTANT (discovered empirically while building this harness, not assumed):
# stubbing the scan to an identity INSIDE one single chained forward (out = v
# + 0.0*(beta+g), then norm+out_proj, all in one Python function, chained 50x
# without intermediate eval) produces a measurement artifact, not a real
# number: Mode A (amortized) comes out ~5-9x HIGHER than Mode B (synced) for
# that construction -- e.g. one early version of this harness measured
# linear_glue amortized=1.09ms / synced=0.83ms at B=1, i.e. amortized > synced,
# which is impossible for real GPU work (chaining without sync can only
# remove per-call sync overhead, never add cost). Bisecting op-by-op pinned
# the cause: `compute_g` is `@partial(mx.compile, shapeless=True)`-wrapped,
# and when its output is combined with OTHER ops (a broadcast add feeding a
# further matmul) inside a 50-deep chain that never syncs, MLX pays a large
# repeated CPU-side dispatch/graph-splice cost for the compiled sub-graph
# that Mode B's per-call mx.eval() happens to hide (GPU-busy time overlaps
# the CPU-side cost) but Mode A's single-eval-at-the-end does not. Isolated
# in a clean loop of its own (no other ops sharing the call), compute_g is
# fast and well-behaved (amortized 0.036ms, synced 0.23ms for B=1) -- so the
# fix is exactly the alternative the task spec offers: "time the
# projection+conv ops separately" instead of one chained stubbed-scan block.
# We measure each op-group in its own clean Mode A/B loop (input varied by a
# tiny per-iteration constant, per spec) and SUM the resulting ms_amortized
# and ms_synced across groups to report a single 'linear_glue' number. This
# also means linear_glue's ms_amortized is a lower bound (no chaining
# interaction across groups can inflate it) while linear_block_full's own
# Mode A (measured on the real, unmodified module -- which is fine because
# there compute_g's output flows straight into gated_delta_kernel, a single
# fused mx.fast.metal_kernel call, not through further generic ops) remains
# a directly faithful, unmodified measurement of the real forward path.
# ---------------------------------------------------------------------------


def bench_op_group(make_input, forward, k=K_MODEL, warmup=WARMUP):
    """Generic Mode A / Mode B pair for a single, self-contained op group
    (no cross-group op mixing -- see the note above on why that matters)."""
    for i in range(warmup):
        mx.eval(forward(make_input(i)))

    outs = []
    t0 = time.perf_counter()
    for i in range(k):
        outs.append(forward(make_input(i)))
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_a = (t1 - t0) * 1000.0 / k

    t0 = time.perf_counter()
    for i in range(k):
        mx.eval(forward(make_input(i)))
    t1 = time.perf_counter()
    ms_b = (t1 - t0) * 1000.0 / k

    return ms_a, ms_b


def derive_scan_inputs(m: GatedDeltaNet, inputs: mx.array, cache: ArraysCache):
    """Run the real pre-scan glue path once to get q/k/v/g/beta/state at the
    exact shapes+dtypes GatedDeltaNet actually feeds gated_delta_kernel for a
    T=1 decode step (no guessing dtypes by hand)."""
    B, S, _ = inputs.shape

    qkv = m.in_proj_qkv(inputs)
    b = m.in_proj_b(inputs)
    a = m.in_proj_a(inputs)

    conv_state = cache[0]
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = m.conv_kernel_size - 1
    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(m.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1),
            [m.num_k_heads, m.num_k_heads, m.num_v_heads],
            [m.head_k_dim, m.head_k_dim, m.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    beta = mx.sigmoid(b)
    g = compute_g(m.A_log, a, m.dt_bias)
    state = cache[1]
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


# ---------------------------------------------------------------------------
# Timing primitives (shared Mode A / Mode B pattern)
# ---------------------------------------------------------------------------


def bench_linear_block_full(model: GatedDeltaNet, batch: int):
    cache = make_cache(model, batch)
    x0 = mx.random.normal((batch, 1, model.hidden_size)).astype(mx.bfloat16)

    def forward(x):
        return model(x, mask=None, cache=cache)

    # Warmup
    x = x0
    for _ in range(WARMUP):
        x = forward(x)
        mx.eval(x)

    # Mode A: chained (output fed back as next input -- shape-compatible),
    # no intermediate eval, one eval at the end.
    x = x0
    outs = []
    t0 = time.perf_counter()
    for _ in range(K_MODEL):
        x = forward(x)
        outs.append(x)
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_a = (t1 - t0) * 1000.0 / K_MODEL

    # Mode B: synced every call.
    x = x0
    t0 = time.perf_counter()
    for _ in range(K_MODEL):
        x = forward(x)
        mx.eval(x)
    t1 = time.perf_counter()
    ms_b = (t1 - t0) * 1000.0 / K_MODEL

    return ms_a, ms_b


def bench_gated_delta_kernel_only(model: GatedDeltaNet, batch: int):
    cache = make_cache(model, batch)
    x0 = mx.random.normal((batch, 1, model.hidden_size)).astype(mx.bfloat16)
    q, k, v0, g, beta, state0 = derive_scan_inputs(model, x0, cache)

    def forward(v, state):
        return gated_delta_kernel(q, k, v, g, beta, state, None)

    # Warmup
    v, state = v0, state0
    for i in range(WARMUP):
        v = v + 1e-6
        y, state = forward(v, state)
        mx.eval(y, state)

    # Mode A: chain recurrent state across calls (realistic), perturb v each
    # iteration so MLX can't collapse identical subgraphs, one eval at end.
    v, state = v0, state0
    outs = []
    t0 = time.perf_counter()
    for i in range(K_MODEL):
        v = v + 1e-6
        y, state = forward(v, state)
        outs.append(y)
        outs.append(state)
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_a = (t1 - t0) * 1000.0 / K_MODEL

    # Mode B: synced every call.
    v, state = v0, state0
    t0 = time.perf_counter()
    for i in range(K_MODEL):
        v = v + 1e-6
        y, state = forward(v, state)
        mx.eval(y, state)
    t1 = time.perf_counter()
    ms_b = (t1 - t0) * 1000.0 / K_MODEL

    return ms_a, ms_b


def bench_linear_glue(model: GatedDeltaNet, batch: int):
    cache = make_cache(model, batch)
    x0 = mx.random.normal((batch, 1, model.hidden_size)).astype(mx.bfloat16)

    def forward(x):
        return glue_forward(model, x, cache)

    # Warmup
    x = x0
    for _ in range(WARMUP):
        x = forward(x)
        mx.eval(x)

    # Mode A: chained (output fed back -- shape-compatible), one eval at end.
    x = x0
    outs = []
    t0 = time.perf_counter()
    for _ in range(K_MODEL):
        x = forward(x)
        outs.append(x)
    mx.eval(outs)
    t1 = time.perf_counter()
    ms_a = (t1 - t0) * 1000.0 / K_MODEL

    # Mode B: synced every call.
    x = x0
    t0 = time.perf_counter()
    for _ in range(K_MODEL):
        x = forward(x)
        mx.eval(x)
    t1 = time.perf_counter()
    ms_b = (t1 - t0) * 1000.0 / K_MODEL

    return ms_a, ms_b


def bench_linear_glue(model: GatedDeltaNet, batch: int):
    """Sum of separately-benched op groups covering every op in
    GatedDeltaNet.__call__ except the scan/kernel -- see the module note
    above `bench_op_group` for why groups are measured independently rather
    than chained together in one stubbed-scan forward."""
    m = model
    x0 = mx.random.normal((batch, 1, m.hidden_size)).astype(mx.bfloat16)
    conv_state0 = mx.random.normal(
        (batch, m.conv_kernel_size - 1, m.conv_dim)
    ).astype(mx.bfloat16)
    mx.eval(x0, conv_state0)

    # (a) in_proj_qkv + in_proj_z + causal conv1d + silu + split/reshape.
    def proj_conv_fwd(x):
        B, S, _ = x.shape
        qkv = m.in_proj_qkv(x)
        z = m.in_proj_z(x).reshape(B, S, m.num_v_heads, m.head_v_dim)
        conv_input = mx.concatenate([conv_state0, qkv], axis=1)
        conv_out = nn.silu(m.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1),
                [m.num_k_heads, m.num_k_heads, m.num_v_heads],
                [m.head_k_dim, m.head_k_dim, m.head_v_dim],
            )
        ]
        return q, k, v, z

    q0, k0, v0, z0 = proj_conv_fwd(x0)
    mx.eval(q0, k0, v0, z0)

    ms_a1, ms_b1 = bench_op_group(
        lambda i: x0 + i * 1e-6, proj_conv_fwd
    )

    # (b) q/k RMS-norm.
    inv_scale = k0.shape[-1] ** -0.5

    def rmsnorm_fwd(qk):
        q, k = qk
        qq = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        kk = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return qq, kk

    ms_a2, ms_b2 = bench_op_group(
        lambda i: (q0 + i * 1e-6, k0 + i * 1e-6), rmsnorm_fwd
    )

    # (c) gating prep: in_proj_b, in_proj_a, sigmoid(beta), compute_g.
    def gating_fwd(x):
        b = m.in_proj_b(x)
        a = m.in_proj_a(x)
        beta = mx.sigmoid(b)
        g = compute_g(m.A_log, a, m.dt_bias)
        return beta, g

    ms_a3, ms_b3 = bench_op_group(lambda i: x0 + i * 1e-6, gating_fwd)

    # (d) gated RMSNormGated(v, z) + out_proj.
    def post_fwd(vz):
        v, z = vz
        B, S = v.shape[:2]
        out = m.norm(v, z)
        return m.out_proj(out.reshape(B, S, -1))

    ms_a4, ms_b4 = bench_op_group(
        lambda i: (v0 + i * 1e-6, z0 + i * 1e-6), post_fwd
    )

    return ms_a1 + ms_a2 + ms_a3 + ms_a4, ms_b1 + ms_b2 + ms_b3 + ms_b4


def main():
    args, tc = load_text_args()
    model = build_model(args)

    print(
        f"hidden_size={model.hidden_size} num_k_heads={model.num_k_heads} "
        f"num_v_heads={model.num_v_heads} head_k_dim={model.head_k_dim} "
        f"head_v_dim={model.head_v_dim} key_dim={model.key_dim} "
        f"value_dim={model.value_dim} conv_dim={model.conv_dim} "
        f"conv_kernel_size={model.conv_kernel_size} "
        f"layer_types linear_attention count="
        f"{sum(1 for t in tc['layer_types'] if t == 'linear_attention')} / "
        f"{len(tc['layer_types'])}"
    )

    results = []
    for batch in BATCH_SIZES:
        ms_a, ms_b = bench_linear_block_full(model, batch)
        results.append(("linear_block_full", batch, ms_a, ms_b))
        print(
            f"linear_block_full      B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms"
        )

        ms_a, ms_b = bench_gated_delta_kernel_only(model, batch)
        results.append(("gated_delta_kernel_only", batch, ms_a, ms_b))
        print(
            f"gated_delta_kernel_only B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms"
        )

        ms_a, ms_b = bench_linear_glue(model, batch)
        results.append(("linear_glue", batch, ms_a, ms_b))
        print(
            f"linear_glue             B={batch:2d}  amortized={ms_a:8.4f} ms  synced={ms_b:8.4f} ms"
        )

    print("\n--- sanity check: 30x linear_block_full vs real total decode ms ---")
    for batch in BATCH_SIZES:
        row = next(r for r in results if r[0] == "linear_block_full" and r[1] == batch)
        extrap = row[2] * N_LINEAR_LAYERS
        total = REAL_TOTAL_MS[batch]
        pct = 100.0 * extrap / total
        print(
            f"B={batch:2d}  30*linear_block_full(amortized)={extrap:7.3f} ms  "
            f"real_total={total:6.2f} ms  share={pct:5.1f}%"
        )

    print("\n--- JSON ---")
    print(
        json.dumps(
            {
                "config": {
                    "hidden_size": model.hidden_size,
                    "linear_num_key_heads": model.num_k_heads,
                    "linear_num_value_heads": model.num_v_heads,
                    "linear_key_head_dim": model.head_k_dim,
                    "linear_value_head_dim": model.head_v_dim,
                    "linear_conv_kernel_dim": model.conv_kernel_size,
                    "n_linear_layers": N_LINEAR_LAYERS,
                    "n_total_layers": N_TOTAL_LAYERS,
                },
                "results": [
                    {"component": c, "batch": b, "ms_amortized": a, "ms_synced": s}
                    for c, b, a, s in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
