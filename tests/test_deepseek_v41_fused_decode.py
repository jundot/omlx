"""Fused single-token decode (decode_fused) against the unfused DeepSeek V4.1 block.

A five-layer model at the fused kernels' shapes (dim 1024, eight 512-wide heads, MXFP8
attention and shared-expert projections, MXFP4 routed experts, an MXFP8-exact bf16 wo_a, the
checkpoint's parameter dtypes) runs every decode layer with the fused path on and off from the
same input. The fused kernels only change accumulation orders the DeepSeek reference leaves
free, so each layer's output must agree to rounding noise. build(fmt="affine") gives the
projections the affine layouts of oQ2-oQ3 exports instead.
"""

import zlib
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v41 import decode_fused
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Block, LanguageModel
from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan, OffloadedExpert
from omlx.patches.deepseek_v41.quantization import QuantizedProjection
from tests.test_deepseek_v41 import run_two_ranks

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="fused decode kernels are Metal-only"
)


def config():
    return ModelConfig(
        vocab_size=64,
        dim=1024,
        moe_inter_dim=256,
        n_layers=5,
        n_heads=8,
        q_lora_rank=256,
        head_dim=512,
        rope_head_dim=64,
        o_groups=2,
        o_lora_rank=256,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=3,
        n_routed_experts=8,
        n_activated_experts=2,
        window_size=4,
        dtype="bf16",
        expert_dtype=None,
        n_mtp_layers=0,
        max_seq_len=64,
        max_batch_size=1,
        candidate_source_layer=3,
        candidate_block_size=2,
        candidate_topk_blocks=2,
        index_source_layers=(1, 3, 4),
    )


_FP32 = ("hc_", "attn_sink", "gate.bias", "compressor.")


# Every width oQ2-oQ3 exports use: affine attention projections of mixed widths, 2-bit w1 / w3
# and 3-bit w2 routed experts (oQ2.5e), 8-bit shared experts.
_AFFINE_BITS = {"wq_a": 8, "wkv": 4, "wq_b": 4, "wo_b": 6, "w1": 2, "w3": 2, "w2": 3, "shared_experts": 8}


def build(seed=4121, fmt="mxfp", **overrides):
    model = LanguageModel(replace(config(), **overrides))
    weights = []
    for name, parameter in tree_flatten(model.parameters()):
        rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
        data = rng.normal(0, 0.03, parameter.shape).astype(np.float32)
        if "norm.weight" in name:
            data += 1
        if "weights_proj.weight" in name:
            data = np.abs(data)
        dtype = mx.float32 if any(key in name for key in _FP32) else mx.bfloat16
        weights.append((name, mx.array(data).astype(dtype)))
    model.load_weights(weights, strict=True)

    def quantized(weight, bits, mode):
        if fmt == "affine":
            bits = _AFFINE_BITS[role]
            packed, scales, biases = mx.quantize(weight.astype(mx.bfloat16), group_size=64, bits=bits)
            return QuantizedProjection(packed, scales, bits, "affine", biases=biases, group_size=64)
        packed, scales = mx.quantize(weight.astype(mx.bfloat16), group_size=32, bits=bits, mode=mode)[:2]
        return QuantizedProjection(packed, scales, bits, mode)

    for layer in model.layers:
        att, ffn = layer.attn, layer.ffn
        for name in ("wq_a", "wkv", "wq_b", "wo_b"):
            role = name
            setattr(att, name, quantized(getattr(att, name).weight, 8, "mxfp8"))
        # convert.py dequantizes the FP8 checkpoint's wo_a to bf16: exactly MXFP8 values
        packed, scales = mx.quantize(att.wo_a.weight, group_size=32, bits=8, mode="mxfp8")[:2]
        att.wo_a.weight = mx.dequantize(packed, scales, None, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)
        for name in ("w1", "w2", "w3"):
            role = name
            setattr(ffn.experts, name, quantized(getattr(ffn.experts, name).weight, 4, "mxfp4"))
            role = "shared_experts"
            setattr(ffn.shared_experts, name, quantized(getattr(ffn.shared_experts, name).weight, 8, "mxfp8"))
    mx.eval(model.parameters())
    return model


def verify_exact(model):
    """Decode on the verify blocks' kernels, as configure_mtp(True) sets it up."""
    for layer in model.layers:
        object.__setattr__(layer, "_omlx_verify_exact", True)
    return model


@pytest.fixture(scope="module")
def shared_build():
    """Cache build() per arguments. Only for tests that read weights through kernels:
    no forward pass, no attribute writes, nothing replaced."""
    models = {}

    def get(fmt="mxfp", **overrides):
        key = (fmt, *sorted(overrides.items()))
        if key not in models:
            models[key] = build(fmt=fmt, **overrides)
        return models[key]

    return get


@pytest.mark.parametrize("fmt,tiles", [("mxfp", False), ("affine", False), ("affine", True)])
def test_fused_decode_matches_unfused_block(monkeypatch, fmt, tiles):
    model = verify_exact(build(fmt=fmt)) if tiles else build(fmt=fmt)
    c = model._config
    decode_fused._window_idx.cache_clear()
    assert all(decode_fused.supported(layer, c) for layer in model.layers)
    calls = {"fused": 0}
    real = decode_fused.decode_block

    def counted(*args, **kwargs):
        calls["fused"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block", counted)
    supported, fused_on = decode_fused.supported, [False]
    monkeypatch.setattr(decode_fused, "supported", lambda layer, c: fused_on[0] and supported(layer, c))
    ids = np.random.default_rng(3).integers(3, 60, 24).tolist()
    caches = []
    for _ in range(2):
        cache = model.make_cache()
        mx.eval(model(mx.array([ids[:17]]), cache=cache))
        caches.append(cache)
    # The fused pass feeds every layer the unfused layer's input, so rounding differences do
    # not compound through the layers (on this random model a one-ulp change per embedding row
    # moves the logits by 30-50%).
    unfused, worst = [], 0.0
    block_call = Block.__call__

    def layer_by_layer(self, h, pre, *args, **kwargs):
        nonlocal worst
        if not fused_on[0]:
            unfused.append((h, pre, block_call(self, h, pre, *args, **kwargs)))
            return unfused[-1][2]
        h, pre, (expected, _) = unfused.pop(0)
        out = block_call(self, h, pre, *args, **kwargs)
        ref = np.asarray(expected.astype(mx.float32))
        worst = max(worst, float(np.abs(np.asarray(out[0].astype(mx.float32)) - ref).max() / np.abs(ref).max()))
        return out

    monkeypatch.setattr(Block, "__call__", layer_by_layer)
    for token in ids[17:23]:
        for fused_on[0], cache in ((False, caches[0]), (True, caches[1])):
            mx.eval(model(mx.array([[token]]), cache=cache))
        assert not unfused
    assert calls["fused"] == 6 * c.n_layers
    # Full-window steps at six positions share one cached window index array.
    assert decode_fused._window_idx.cache_info().currsize == 1
    # Free-order rounding moves a layer's output by up to ~2.5% of its largest value (2-8 bit
    # affine maps); a wrong layout or kernel selection gives O(1) errors.
    assert worst < 0.05, worst


def test_fused_decode_rejects_unsupported_layouts():
    model = build()
    c = model._config
    layer = model.layers[1]
    assert decode_fused.supported(layer, c)
    att = model.layers[2].attn
    packed, scales = mx.quantize(
        mx.dequantize(att.wq_b.weight, att.wq_b.scales, None, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16),
        group_size=32, bits=4, mode="mxfp4",
    )
    att.wq_b = QuantizedProjection(packed, scales, 4, "mxfp4")
    assert not decode_fused.supported(model.layers[2], c)
    other = model.layers[3].attn
    scaled = (other.wo_a.weight.astype(mx.float32) * 1.01).astype(mx.bfloat16)   # off the FP8 grid
    assert not mx.array_equal(scaled, other.wo_a.weight).item()
    other.wo_a.weight = scaled
    assert not decode_fused.supported(model.layers[3], c)


def test_offloaded_experts_decode_on_the_unfused_path(tmp_path):
    model = build()
    c = model._config
    specs, tensors = {}, {}
    for i, layer in enumerate(model.layers):
        prefix = f"language_model.layers.{i}.ffn.experts"
        specs.update({f"{prefix}.{proj}": dict(bits=4, mode="mxfp4") for proj in ("w1", "w2", "w3")})
        tensors.update({f"{prefix}.{k}": v for k, v in tree_flatten(layer.ffn.experts.parameters())})
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    raw = {"omlx_deepseek_v41": {"version": 1, "quantized_modules": specs}}
    plan = ExpertOffloadPlan(tmp_path, raw, {k: "model.safetensors" for k in tensors}, c, 0.5)
    try:
        for i, layer in enumerate(model.layers):
            layer.ffn.experts = OffloadedExpert(layer.ffn.experts, plan, f"language_model.layers.{i}.ffn.experts")
        assert not decode_fused.supported(model.layers[0], c)
        # Requests then verify one by one, so the batch policy must not cap their depth.
        assert model.mtp_shared_verify_rows() == 0
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3, 4, 5]]), cache=cache))
        assert mx.isfinite(model(mx.array([[6]]), cache=cache)).all().item()
    finally:
        plan.close()


@pytest.mark.parametrize("fmt", ["mxfp", "affine"])
@pytest.mark.parametrize("batch", [3, 8, 17])   # odd: a tile lane holds half a vector pair
def test_batched_decode_matches_single_requests(monkeypatch, batch, fmt):
    """On the row kernels (below decode_moe.MMA_MIN_BATCH requests without the drafter), B
    single-token requests through decode_block_nb give each request the logits it gets
    decoding alone (per-request arithmetic is the single-request kernels')."""
    from omlx.patches.deepseek_v41 import decode_moe
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    monkeypatch.setattr(decode_moe, "use_mma", lambda nb: False)
    # Affine batches always run on matrix tiles; so do MXFP8 ones once the drafter verifies.
    model = verify_exact(build(fmt=fmt)) if fmt == "affine" else build(fmt=fmt)
    batches = []
    real = decode_fused.decode_block_nb

    def counted(*args, **kwargs):
        batches.append(args[1].shape[0])
        assert 1 <= batches[-1] <= decode_fused.MAX_BATCH
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block_nb", counted)
    real_attention = decode_fused.batched_attention
    attention = {"batched": 0}

    def counted_attention(*args, **kwargs):
        out = real_attention(*args, **kwargs)
        attention["batched"] += out is not None
        return out

    monkeypatch.setattr(decode_fused, "batched_attention", counted_attention)
    rng = np.random.default_rng(11)
    prompts = [rng.integers(3, 60, 13 + 2 * r).tolist() for r in range(batch)]   # different lengths
    singles, rows = [], []
    for p in prompts:
        for target in (singles, rows):
            cache = model.make_cache()
            mx.eval(model(mx.array([p[:-1]]), cache=cache))
            target.append(cache)
    batched = [DeepseekV41Cache.merge([rows[r][i] for r in range(batch)]) for i in range(len(rows[0]))]
    for item in batched:
        item.prepare(lengths=[4] * batch)
    tokens = [p[-1] for p in prompts]
    # The first step decodes an even position, the second an odd one: only the second
    # completes a ratio-2 compression group.
    for step in range(2):
        expected = [np.asarray(model(mx.array([[t]]), cache=c_)[0, -1].astype(mx.float32)) for t, c_ in zip(tokens, singles)]
        got = np.asarray(model(mx.array([[t] for t in tokens]), cache=batched)[:, -1].astype(mx.float32))
        for r in range(batch):
            np.testing.assert_allclose(got[r], expected[r], rtol=0, atol=1e-6 * np.abs(expected[r]).max())
            for i, (item, single) in enumerate(zip(batched, singles[r])):
                row = item.extract(r)
                assert row.size() == single.size()
                assert row._ds41_buffers is rows[r][i]._ds41_buffers
        assert all(item.lengths.tolist() == [3 - step] * batch for item in batched)
        tokens = [int(e.argmax()) for e in expected]
    chunks = [min(decode_fused.MAX_BATCH, batch - start)
              for start in range(0, batch, decode_fused.MAX_BATCH)]
    assert batches == [n for n in chunks for _ in model.layers] * 2
    assert attention["batched"] == len(batches)   # windows are full: request-batched attention


def _aq(rng, nb, k):
    from omlx.patches.deepseek_v41.quantization import quantize_activation

    x = mx.array(rng.normal(0, 1, (nb, k)).astype(np.float32)).astype(mx.bfloat16)
    return quantize_activation(x).astype(mx.float32)


@pytest.mark.parametrize("nb", [4, 5, 8])
def test_batched_fp8_mma_projection_within_fp32_summation_of_exact(
    monkeypatch, shared_build, nb
):
    """Batched projections run on simdgroup matrices from decode_moe.MMA_MIN_BATCH vectors.
    Every product is exact (fp8 weight x act_quant activation), so each output must equal
    the float64 dot product up to fp32 summation rounding (K * 2^-24 * sum|w x|) plus the
    bf16 rounding."""
    from omlx.patches.deepseek_v41 import decode_layer

    names = []
    real_kernel = decode_layer.kernel
    monkeypatch.setattr(decode_layer, "kernel", lambda name, *f: (names.append(name), real_kernel(name, *f))[1])
    att = shared_build().layers[0].attn
    k = att.wq_a.weight.shape[1] * 4
    x = _aq(np.random.default_rng(nb), nb, k)
    ya, yb = decode_fused.gemv_fp8_nb(x, att.wq_a, att.wkv)
    assert names == ["gemv_fp8_mma"]
    X = np.asarray(x).astype(np.float64)
    for proj, y in ((att.wq_a, ya), (att.wkv, yb)):
        W = mx.dequantize(proj.weight, proj.scales, None, group_size=32, bits=8, mode="mxfp8", dtype=mx.float32)
        W = np.asarray(W).astype(np.float64)
        exact = X @ W.T
        bound = k * 2.0**-24 * (np.abs(X) @ np.abs(W).T)
        got = np.asarray(y.astype(mx.float32)).astype(np.float64)
        top = np.maximum(np.abs(got), np.abs(exact))
        half_ulp = 2.0 ** (np.floor(np.log2(np.maximum(top, 2.0**-126))) - 8)
        assert np.all(np.abs(got - exact) <= bound + half_ulp)


@pytest.mark.parametrize("nb", [4, 8, 13])
def test_batched_mma_kernels_match_rows_kernels(monkeypatch, nb):
    """Each simdgroup-matrix kernel (wq_b+RoPE, wo_a+act_quant, MoE shared expert) agrees with
    its fp8_rows_nb counterpart up to summation rounding; MoE routing is untouched."""
    from omlx.patches.deepseek_v41 import decode_layer as DL, decode_moe

    model = build()
    c = model._config
    blk = model.layers[0]
    att = blk.attn
    rng = np.random.default_rng(31 + nb)
    nq, rows = c.n_heads * c.head_dim, c.o_groups * c.o_lora_rank
    qrq = _aq(rng, nb, c.q_lora_rank)
    cos = mx.array(rng.normal(0, 1, (nb, c.rope_head_dim // 2)).astype(np.float32))
    sin = mx.array(rng.normal(0, 1, (nb, c.rope_head_dim // 2)).astype(np.float32))
    ks = decode_fused._mma_split(nq // 8, c.q_lora_rank // 32)
    tmpl = [("N", nq), ("K", c.q_lora_rank), ("HD", c.head_dim), ("RD", c.rope_head_dim), ("NB", nb)]
    q_mma = DL.kernel("wqb_rope_mma", (decode_moe.MXFP8,))(
        inputs=[qrq, att.wq_b.weight, att.wq_b.scales, decode_moe._biases(att.wq_b), cos, sin], template=tmpl + [("KS", ks)],
        grid=(nq // 8 * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
        output_shapes=[(nb * nq,)], output_dtypes=[mx.bfloat16])[0]
    q_rows = DL.kernel("wqb_rope_nb", (decode_moe.MXFP8,))(
        inputs=[qrq, att.wq_b.weight, att.wq_b.scales, decode_moe._biases(att.wq_b), cos, sin], template=tmpl,
        grid=(nq // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(nb * nq,)], output_dtypes=[mx.bfloat16])[0]
    np.testing.assert_allclose(q_mma.astype(mx.float32), q_rows.astype(mx.float32), rtol=2**-6, atol=1e-6)
    oq = _aq(rng, nb, nq)
    wq, ws = decode_fused._woa_mxfp8(att)
    tmpl = [("R", c.o_lora_rank), ("K", nq // c.o_groups), ("G", c.o_groups), ("NB", nb)]
    y_mma = DL.kernel("woa_mma")(
        inputs=[oq, wq, ws], template=tmpl + [("KS", 2)], grid=(rows // 32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(nb * rows,)], output_dtypes=[mx.float32])[0]
    y_rows = DL.kernel("woa_nb")(
        inputs=[oq, wq, ws], template=tmpl, grid=(rows // 32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(nb * rows,)], output_dtypes=[mx.float32])[0]
    assert np.mean(np.asarray(y_mma) == np.asarray(y_rows)) > 0.99   # act_quant can flip a step at a rounding tie
    x = mx.array(rng.normal(0, 1, (nb, c.dim)).astype(np.float32)).astype(mx.bfloat16)
    out_mma, idx_mma, _ = decode_moe.fused_decode_moe_nb(blk.ffn, x, c)
    monkeypatch.setattr(decode_moe, "use_mma", lambda nb: False)
    out_rows, idx_rows, _ = decode_moe.fused_decode_moe_nb(blk.ffn, x, c)
    np.testing.assert_array_equal(idx_mma, idx_rows)
    np.testing.assert_allclose(out_mma.astype(mx.float32), out_rows.astype(mx.float32), rtol=2**-6, atol=1e-5)


@pytest.mark.parametrize("fmt", ["mxfp", "affine"])
def test_second_column_tile_matches_an_eight_vector_tile(shared_build, fmt):
    """Matrix-tile kernels run vectors 8..15 as a second column tile: every vector gets bit
    for bit what an 8-vector call gives it, so a request's rows do not depend on where they
    sit in a batch (batched decode and shared verify rely on it)."""
    from omlx.patches.deepseek_v41 import decode_moe

    model = shared_build(fmt=fmt)
    c = model._config
    blk = model.layers[0]
    att = blk.attn
    rng = np.random.default_rng(43)
    x = _aq(rng, 16, c.dim)
    halves = (slice(0, 8), slice(8, 16))
    wide = decode_fused.gemv_fp8_nb(x, att.wq_a, att.wkv, mma=True)
    for half in halves:
        part = decode_fused.gemv_fp8_nb(x[half], att.wq_a, att.wkv, mma=True)
        assert all(mx.array_equal(w[half], p).item() for w, p in zip(wide, part))
    qrq = _aq(rng, 16, c.q_lora_rank)
    cos = mx.array(rng.normal(0, 1, (16, c.rope_head_dim // 2)).astype(np.float32))
    sin = mx.array(rng.normal(0, 1, (16, c.rope_head_dim // 2)).astype(np.float32))
    wide = decode_fused.wqb_rope_nb(att, qrq, cos, sin, c, True)
    for half in halves:
        assert mx.array_equal(wide[half], decode_fused.wqb_rope_nb(att, qrq[half], cos[half], sin[half], c, True)).item()
    oq = _aq(rng, 16, c.n_heads * c.head_dim)
    wide = decode_fused.woa_nb(att, oq, c, True)
    for half in halves:
        assert mx.array_equal(wide[half], decode_fused.woa_nb(att, oq[half], c, True)).item()
    h = mx.array(rng.normal(0, 1, (16, c.dim)).astype(np.float32)).astype(mx.bfloat16)
    wide = decode_moe.fused_decode_moe_nb(blk.ffn, h, c, mma=True)[0]
    for half in halves:
        assert mx.array_equal(wide[half], decode_moe.fused_decode_moe_nb(blk.ffn, h[half], c, mma=True)[0]).item()


def test_small_batches_keep_rows_kernels(monkeypatch, shared_build):
    """Below MMA_MIN_BATCH the fp8_rows_nb kernels stay in use (faster there)."""
    from omlx.patches.deepseek_v41 import decode_layer, decode_moe

    names = []
    real_kernel = decode_layer.kernel
    monkeypatch.setattr(decode_layer, "kernel", lambda name, *f: (names.append(name), real_kernel(name, *f))[1])
    att = shared_build().layers[0].attn
    decode_fused.gemv_fp8_nb(_aq(np.random.default_rng(3), decode_moe.MMA_MIN_BATCH - 1, att.wq_a.weight.shape[1] * 4), att.wq_a, att.wkv)
    assert names == ["gemv_fp8_nb"]


def test_paired_weight_decoders_are_exact():
    """fp8_decode4 / fp4_decode8 give every E4M3 / E2M1 code its value times 2^-8 / 2^-14
    (E4M3FN NaN codes 0x7F / 0xFF read as 480, like the e4m3() helper; weights never hold them)."""
    from omlx.patches.deepseek_v41 import decode_moe

    source = r"""
        const uint i = thread_position_in_grid.x;
        float v[8];
        fp8_decode4(words8[i], v);
        for (int j = 0; j < 4; ++j) out8[i * 4 + j] = v[j] * 256.0f;
        if (i < 2) {
            fp4_decode8(words4[i], v);
            for (int j = 0; j < 8; ++j) out4[i * 8 + j] = v[j] * 16384.0f;
        }
    """
    kernel = mx.fast.metal_kernel(name="v41_test_weight_decoders", input_names=["words8", "words4"],
                                  output_names=["out8", "out4"], source=source, header=decode_moe._HEADER)
    codes = np.arange(256, dtype=np.uint8)
    words8 = codes.view(np.uint32)                                   # code 4i + j at bits 8j of word i
    words4 = np.array([sum(n << (4 * j) for j, n in enumerate(range(8 * w, 8 * w + 8))) for w in range(2)], np.uint32)
    out8, out4 = kernel(inputs=[mx.array(words8), mx.array(words4)], grid=(64, 1, 1), threadgroup=(64, 1, 1),
                        output_shapes=[(256,), (16,)], output_dtypes=[mx.float32, mx.float32])
    c = codes.astype(np.int64)
    e, m = (c >> 3) & 15, c & 7
    e4m3 = np.where(e == 0, m * 2.0**-9, 2.0 ** (e - 7.0) * (1 + m / 8)) * np.where(c & 128, -1, 1)
    e2m1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    np.testing.assert_array_equal(np.asarray(out8), e4m3.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(out4), e2m1.astype(np.float32))
    assert np.array_equal(np.signbit(np.asarray(out8)), codes >= 128)
    assert np.array_equal(np.signbit(np.asarray(out4)), np.arange(16) >= 8)


@pytest.mark.parametrize("fmt", ["mxfp", "affine"])
@pytest.mark.parametrize("nb", [3, 8, 16])
def test_batched_moe_matches_single_token_moe(monkeypatch, shared_build, nb, fmt):
    """Without matrix tiles, each token of the batched MoE gets the single-token kernels'
    output, routing indices and weights bit for bit, also when tokens share experts: the
    routed kernels then dispatch every token of a row block together (see _UP_NB)."""
    from omlx.patches.deepseek_v41 import decode_moe

    monkeypatch.setattr(decode_moe, "use_mma", lambda nb: False)
    model = shared_build(fmt=fmt, n_routed_experts=16, n_activated_experts=4)   # >2 slots: their sum order shows
    c = model._config
    rng = np.random.default_rng(23 + nb)
    x = rng.normal(0, 1, (nb, c.dim)).astype(np.float32)
    x[nb // 2:] = x[: nb - nb // 2] + rng.normal(0, 0.05, (nb - nb // 2, c.dim))   # neighbours share experts
    x = mx.array(x).astype(mx.bfloat16)
    shared = 0
    for layer in model.layers:
        out, idx, w = decode_moe.fused_decode_moe_nb(layer.ffn, x, c)
        for r in range(nb):
            single, single_idx, single_w = decode_moe.fused_decode_moe(layer.ffn, x[r].reshape(1, 1, -1), c)
            assert mx.array_equal(out[r], single.reshape(-1)).item()
            assert mx.array_equal(idx[r], single_idx).item() and mx.array_equal(w[r], single_w).item()
        shared += len(np.unique(np.asarray(idx))) < nb * c.n_activated_experts
    assert shared


@pytest.mark.parametrize("batch", [2, 8, 13])
def test_batched_engram_projects_each_request_like_a_single_request(batch):
    """Batched decode feeds Engram one row per request. Each row gets the one-row arithmetic
    (mxfp8_gemv_each, as when the request decodes alone); a B-row quantized_matmul rounds its
    fp32 sums differently, which made a request's tokens depend on its batch."""
    from test_deepseek_v41 import tiny

    from omlx.patches.deepseek_v41.engram import Engram

    c = tiny(
        dim=512,   # wkv output 2560 wide: there MLX's B-row kernel rounds differently
        engram_layer_ids=(1,),
        engram_num_embeddings=(97,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_compressed_vocab_size=64,
    )
    engram = Engram(c, 0)
    rng = np.random.default_rng(19)   # draws whose B-row projection differs in a few elements
    engram.embed.weight = mx.array(rng.normal(0, 1, engram.embed.weight.shape).astype(np.float32)).astype(mx.bfloat16)
    weight = mx.array(rng.normal(0, 0.02, engram.wkv.weight.shape).astype(np.float32)).astype(mx.bfloat16)
    packed, scales = mx.quantize(weight, group_size=32, bits=8, mode="mxfp8")[:2]
    engram.wkv = QuantizedProjection(packed, scales, 8, "mxfp8")
    ids = rng.integers(0, 97, (batch, 1, 3 * 8))
    ids[-1] = ids[0]   # requests can share an n-gram: their rows must stay one per request
    # Engram computes in fp32 and returns h's dtype: fp32 h keeps every projection bit visible.
    h = mx.array(rng.normal(0, 1, (batch, 1, c.hc_mult, c.dim)).astype(np.float32))
    batched = engram(h, ids)
    single = mx.concatenate([engram(h[r : r + 1], ids[r : r + 1]) for r in range(batch)])
    assert mx.array_equal(batched, single).item()


@pytest.mark.parametrize("fmt", ["mxfp", "affine"])
@pytest.mark.parametrize("prompt_length", [11, 2])   # full window (batched attention) / filling
@pytest.mark.parametrize("rows,accepted", [(2, 0), (3, 2), (6, 3)])
def test_verify_block_matches_token_by_token_decode(monkeypatch, rows, accepted, prompt_length, fmt):
    """An MTP verify block through the fused kernels gives every token bit for bit the
    logits of decoding it alone, and after a partial rollback the cache continues exactly
    like one that decoded only the accepted tokens."""
    model = verify_exact(build(fmt=fmt, vocab_size=4096))   # the vocab head kernel (per-row exact) needs >= 4096 rows
    calls = []
    real = decode_fused.decode_block_verify

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block_verify", counted)
    blocks = []
    real_block = decode_fused.compress_index_block
    monkeypatch.setattr(decode_fused, "compress_index_block", lambda *a: (blocks.append(1), real_block(*a))[1])
    rng = np.random.default_rng(29 + rows)
    prompt = rng.integers(3, 60, prompt_length).tolist()
    tokens = rng.integers(3, 60, rows + 1).tolist()   # confirmed token, drafts, next step
    sequential, verified, accepted_only = (model.make_cache() for _ in range(3))
    for cache in (sequential, verified, accepted_only):
        mx.eval(model(mx.array([prompt]), cache=cache))
    expected = [model(mx.array([[t]]), cache=sequential)[0, -1] for t in tokens[:rows]]
    got = model(mx.array([tokens[:rows]]), cache=verified, n_confirmed=1)
    for j in range(rows):
        assert mx.array_equal(got[0, j], expected[j]).item(), f"row {j}"
    assert len(calls) == model._config.n_layers
    # A full window with enough index keys runs the compressor and indexer once per block.
    assert len(blocks) == (4 if prompt_length == 11 else 0)
    assert model.mtp_partial_rollback(verified, accepted, rows - 1)
    for t in tokens[: accepted + 1]:
        mx.eval(model(mx.array([[t]]), cache=accepted_only))
    assert verified[0].size() == accepted_only[0].size()
    nxt = mx.array([[tokens[accepted + 1]]])
    assert mx.array_equal(model(nxt, cache=verified), model(nxt, cache=accepted_only)).item()


@pytest.mark.parametrize("fmt", ["mxfp", "affine"])
def test_requests_verified_together_match_their_batched_decode(fmt):
    """mtp_verify_rows gives each request bit for bit the logits that decoding the same
    requests token by token as a batch gives, and after rolling back to the same accepted
    count the caches continue exactly like that batch's."""
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    model = verify_exact(build(fmt=fmt, vocab_size=4096, dspark_target_layer_ids=(2, 3, 4)))
    rng = np.random.default_rng(37)
    # One request with a full window, one still filling it; different positions.
    prompts = [rng.integers(3, 60, 11).tolist(), rng.integers(3, 60, 3).tolist()]
    blocks = [rng.integers(3, 60, 3).tolist() for _ in prompts]

    def prefilled():
        caches = []
        for prompt in prompts:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt]), cache=cache))
            caches.append(cache)
        return caches

    def batch(rows):
        return [DeepseekV41Cache.merge([rc[i] for rc in rows]) for i in range(len(rows[0]))]

    def decode(cache, column):
        return model(mx.array([[block[column]] for block in blocks]), cache=cache)[:, -1]

    sequential, accepted_only = batch(prefilled()), batch(prefilled())
    expected = [decode(sequential, j) for j in range(3)]
    together = prefilled()
    assert model.mtp_verify_rows_ok(together, 3)
    logits, hidden, together = model.mtp_verify_rows(mx.array(blocks), together)
    assert hidden.shape == (2, 3, 3 * model._config.dim)
    for j, want in enumerate(expected):
        assert mx.array_equal(logits[:, j], want).item(), f"row {j}"
    for cache in together:
        assert model.mtp_partial_rollback(cache, 1, 2)
    for j in range(2):
        decode(accepted_only, j)
    together = batch(together)
    assert all(a.size() == b.size() for a, b in zip(together, accepted_only))
    assert mx.array_equal(decode(together, 2), decode(accepted_only, 2)).item()


@pytest.mark.parametrize("fmt", ["mxfp", "affine"])
def test_mtp_generation_matches_greedy_with_fused_kernels(monkeypatch, fmt):
    """DSpark speculation over the fused decode kernels reproduces greedy decoding token for
    token (verify blocks are token-exact, rejected drafts roll back exactly)."""
    from mlx_lm.generate import BatchGenerator

    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    model = build(
        fmt=fmt,
        vocab_size=4096,
        preserve_mtp=True,
        n_mtp_layers=1,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_markov_rank=32,
    )
    prompt = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    model.configure_mtp(True, 3)
    expected, cache = [], model.make_cache()
    # The generator prefills prompt[:-1] in steps of 4; other prefill chunks round differently.
    for s in range(0, len(prompt) - 1, 4):
        model(mx.array([prompt[s : min(s + 4, len(prompt) - 1)]]), cache=cache)
    ids = mx.array([prompt[-1:]])
    for _ in range(24):
        token = int(mx.argmax(model(ids, cache=cache)[0, -1]).item())
        expected.append(token)
        ids = mx.array([[token]])
    calls = []
    real = decode_fused.decode_block_verify

    def counted(*args, **kwargs):
        calls.append(args[1].shape[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block_verify", counted)
    generator = BatchGenerator(model, max_tokens=24, prefill_step_size=4, sampler=lambda logits: mx.argmax(logits, -1))
    result = []
    try:
        generator.insert([prompt])
        for _ in range(96):
            _, responses = generator.next()
            result.extend(response.token for response in responses)
            if len(result) >= 24:
                break
    finally:
        generator.close()
    assert calls and max(calls) > 1, "verify blocks must take the fused path"
    assert result[:24] == expected


@pytest.mark.parametrize("count", [2, 8])   # 8: the policy caps the depth below the drafts'
def test_mtp_generation_of_several_requests_verifies_them_together(monkeypatch, count):
    """Requests decoding with DSpark verify their blocks in one forward and each
    reproduces its own greedy decoding token for token."""
    from mlx_lm.generate import BatchGenerator

    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    # Affine projections run on matrix tiles alone and in a batch, so each request's greedy
    # tokens are its single-request ones (batched MXFP8 rows round like batched decode).
    model = build(
        fmt="affine",
        vocab_size=4096,
        preserve_mtp=True,
        n_mtp_layers=1,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_markov_rank=32,
    )
    prompts = [[3 + r + j for j in range(9)] for r in range(0, 9 * count, 9)]
    model.configure_mtp(True, 3)
    # A fixed depth keeps the batch policy from timing-based parking.
    model._omlx_mtp_depth_fixed = True
    expected = []
    for prompt in prompts:
        tokens, cache = [], model.make_cache()
        for s in range(0, len(prompt) - 1, 4):
            model(mx.array([prompt[s : min(s + 4, len(prompt) - 1)]]), cache=cache)
        ids = mx.array([prompt[-1:]])
        for _ in range(16):
            token = int(mx.argmax(model(ids, cache=cache)[0, -1]).item())
            tokens.append(token)
            ids = mx.array([[token]])
        expected.append(tokens)
    together = []
    real = type(model).mtp_verify_rows

    def counted(self, inputs, row_caches):
        together.append(inputs.shape)
        return real(self, inputs, row_caches)

    monkeypatch.setattr(type(model), "mtp_verify_rows", counted)
    generator = BatchGenerator(
        model, max_tokens=16, prefill_step_size=4, prefill_batch_size=count,
        sampler=lambda logits: mx.argmax(logits, -1),
    )
    result = {}
    try:
        uids = generator.insert(prompts)
        for _ in range(96):
            _, responses = generator.next()
            for response in responses:
                result.setdefault(response.uid, []).append(response.token)
            if all(len(result.get(uid, [])) >= 16 for uid in uids):
                break
    finally:
        generator.close()
    assert together and max(shape[0] for shape in together) == count, "the requests must verify in one forward"
    for uid, want in zip(uids, expected):
        assert result[uid][:16] == want


# Two pipeline ranks run batched Lightning MTP for two requests (rank 0 holds the head and
# the DSpark target layers, rank 1 replays its cycles).
_CLUSTER_MTP_BATCH_RANK = """
import json
import sys

import mlx.core as mx

group = mx.distributed.init(backend="ring", strict=True)
sys.path.insert(0, sys.argv[1])
import test_deepseek_v41_fused_decode as t
from mlx_lm.generate import BatchGenerator
from omlx.cluster.control_plane import RankControlPlane
from omlx.cluster.mtp_sync import ClusterMtpSync
from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan
from omlx.patches.mlx_lm_mtp import batch_generator

batch_generator.apply()
port, rank, limit = int(sys.argv[2]), group.rank(), 12
prompts = [[3 + r + j for j in range(9)] for r in range(0, 18, 9)]

def model():
    return t.build(fmt="affine", vocab_size=4096, preserve_mtp=True, n_mtp_layers=1, dspark_block_size=3,
                   dspark_noise_token_id=2, dspark_target_layer_ids=(3, 4), dspark_markov_rank=32)

full = model()
expected = []
for prompt in prompts:
    tokens, cache = [], full.make_cache()
    full(mx.array([prompt[:-1]]), cache=cache)
    ids = mx.array([prompt[-1:]])
    for _ in range(limit):
        token = int(mx.argmax(full(ids, cache=cache)[0, -1]).item())
        tokens.append(token)
        ids = mx.array([[token]])
    expected.append(tokens)

stage = model()
plan = (PipelineAssignment("a", 0, 3, 5, 1, 0, 0, 1), PipelineAssignment("b", 1, 0, 3, 1, 0, 0, 1))
with install_unequal_pipeline_plan(plan):
    stage.pipeline(group)
stage.configure_mtp(True, 2, verify_only=rank != 0)
together, accepted = [], []
real = type(stage).mtp_verify_rows
def counted(self, inputs, row_caches):
    together.append(int(inputs.shape[0]))
    return real(self, inputs, row_caches)
type(stage).mtp_verify_rows = counted
share_result = ClusterMtpSync.share_cycle_result
def recorded(self, *a):
    out = share_result(self, *a)
    accepted.append(out[0])
    return out
ClusterMtpSync.share_cycle_result = recorded

def oracle_drafts(jobs):
    # Rank 0 drafts each request's true next tokens, with a wrong last draft every third
    # position, so cycles accept and reject.
    for row, state, _, _, _ in jobs:
        i = uids.index(state.uid)
        done = len(row.tokens[0]) - len(prompts[i]) + len(state.queue)
        ids = (expected[i][done : done + state.depth] + [0] * state.depth)[: state.depth]
        if done % 3 == 0:
            ids[-1] = (ids[-1] + 1) % 4096
        state.drafts = mx.array(ids, dtype=mx.uint32)
        state.draft_lps, state.draft_accept_lps = [], []
if rank == 0:
    batch_generator._dspark_draft_jobs = oracle_drafts

with RankControlPlane(rank=rank, world_size=2, host="127.0.0.1", port=port, token="v41-mtp-batch") as plane:
    stage._omlx_cluster_mtp = ClusterMtpSync(rank=rank, control_plane=plane)
    gen = BatchGenerator(stage, max_tokens=limit, prefill_batch_size=len(prompts),
                         sampler=lambda logits: mx.argmax(logits, -1))
    result = {}
    try:
        uids = gen.insert(prompts)
        for _ in range(8 * limit):
            _, responses = gen.next()
            for response in responses:
                result.setdefault(response.uid, []).append(response.token)
            if all(len(result.get(uid, [])) >= limit for uid in uids):
                break
    finally:
        gen.close()
assert [result[uid][:limit] for uid in uids] == expected, (result, expected)
assert together and max(together) == len(prompts), together
assert max(accepted) > 0 and min(accepted) < max(accepted), accepted
print(json.dumps({"rank": rank, "ok": True, "together": len(together)}), flush=True)
"""


@pytest.mark.slow
def test_cluster_mtp_verifies_several_requests_together(tmp_path):
    """Pipeline ranks verify two requests' DSpark blocks in one forward, and each request
    reproduces its own greedy tokens."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    run_two_ranks(tmp_path, _CLUSTER_MTP_BATCH_RANK, Path(__file__).parent, port)

# One rank of the cluster decode test below.
_CLUSTER_DECODE_RANK = """
import json
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

group = mx.distributed.init(backend="ring", strict=True)
sys.path.insert(0, sys.argv[1])
import test_deepseek_v41_fused_decode as t
from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan
from omlx.patches.deepseek_v41 import decode_fused, decode_moe
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.language import Block
from omlx.patches.deepseek_v41.loading import tensor_slice
from omlx.patches.deepseek_v41.quantization import QuantizedProjection

rng = np.random.default_rng(11)
prompts = [rng.integers(3, 60, 13 + 2 * r).tolist() for r in range(2)]
x = mx.array(rng.normal(0, 1, (2, 1024)).astype(np.float32)).astype(mx.bfloat16)

def prefill(model):
    rows = []
    for p in prompts:
        cache = model.make_cache()
        mx.eval(model(mx.array([p[:-1]]), cache=cache))
        rows.append(cache)
    return rows

def merged(rows):
    return [DeepseekV41Cache.merge(list(pair)) for pair in zip(*rows)]

# Tensor parallel. Layer 0 sees the same embeddings and KV on every rank, so
# its summed attention output and a MoE fed the same input differ from one
# process only by the summation order. Later layers amplify that rounding on
# this random model.
sums, fused = [], {"decode_block": 0, "decode_block_nb": 0}
real_sum = Block.sum_ranks

def record(self, value):
    out = real_sum(self, value)
    if self.attn._layer == 0:
        sums.append(out.astype(mx.float32).reshape(-1))
    return out

Block.sum_ranks = record
for name in fused:
    def counted(*args, _real=getattr(decode_fused, name), _name=name):
        fused[_name] += 1
        return _real(*args)
    setattr(decode_fused, name, counted)

def tensor_run(model):
    sums.clear()
    singles, batched = prefill(model), merged(prefill(model))
    for token in range(3, 6):
        for cache in singles:
            mx.eval(model(mx.array([[token]]), cache=cache))
        mx.eval(model(mx.array([[token], [token + 1]]), cache=batched))
    block, c = model.layers[0], model._config
    one = decode_moe.fused_decode_moe(block.ffn, x[:1].reshape(1, 1, -1), c)[0]
    both = decode_moe.fused_decode_moe_nb(block.ffn, x, c)[0]
    # Each block sums its attention output, then its MoE output.
    return mx.concatenate(sums[0::2]), block.sum_ranks(one), block.sum_ranks(both)

full = t.build(n_heads=16, moe_inter_dim=512)
reference = tensor_run(full)
model = t.build(n_heads=8, o_groups=1, moe_inter_dim=256)
model.load_weights(
    [(k, tensor_slice(k, v, group.rank(), group.size())) for k, v in tree_flatten(full.parameters())],
    strict=True,
)
for block in model.layers:
    block._tensor_group = group
fused = dict.fromkeys(fused, 0)
got = tensor_run(model)
assert fused["decode_block"] and fused["decode_block_nb"], fused
for r, g in zip(reference, got):
    error = (mx.abs(g - r).max() / mx.abs(r.astype(mx.float32)).max()).item()
    assert error < 0.02, error

# Pipeline. Layer 4's MXFP4 wo_b is a layout the fused kernels do not cover,
# so only rank 0's stage [3, 5) lacks fused decode.
def mixed():
    model = t.build()
    att = model.layers[4].attn
    weight = mx.dequantize(
        att.wo_b.weight, att.wo_b.scales, None, group_size=32, bits=8, mode="mxfp8"
    )
    packed, scales = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
    att.wo_b = QuantizedProjection(packed, scales, 4, "mxfp4")
    return model

def pipeline_run(model):
    batched = merged(prefill(model))
    outs = []
    for token in range(3, 6):
        outs.append(model(mx.array([[token], [token + 1]]), cache=batched))
        mx.eval(outs[-1])
    return outs

reference = pipeline_run(mixed())
model = mixed()
plan = (
    PipelineAssignment("a", 0, 3, 5, 1, 0, 0, 1),
    PipelineAssignment("b", 1, 0, 3, 1, 0, 0, 1),
)
with install_unequal_pipeline_plan(plan):
    model.pipeline(group)
assert all(mx.array_equal(r, g).item() for r, g in zip(reference, pipeline_run(model)))
print(json.dumps({"rank": group.rank(), "ok": True}), flush=True)
"""


# Two real ranks on one GPU: a rank can hit the Metal timeout on a busy runner.
@pytest.mark.slow
def test_cluster_decode_matches_single_process(tmp_path):
    # Ranks sum their partial outputs, and pipeline stages agree on batched
    # decode even when only one stage's layers support it.
    run_two_ranks(tmp_path, _CLUSTER_DECODE_RANK, Path(__file__).parent)
