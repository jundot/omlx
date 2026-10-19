"""Fused single-token decode (decode_fused) against the unfused DeepSeek V4.1 block.

A five-layer model at the fused kernels' shapes (dim 1024, eight 512-wide heads, MXFP8
attention and shared-expert projections, MXFP4 routed experts, an MXFP8-exact bf16 wo_a, the
checkpoint's parameter dtypes) decodes the same tokens from identical caches with the fused
path on and off. The fused kernels only change accumulation orders the DeepSeek reference
leaves free, so the logits must agree to rounding noise.
"""

import zlib
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v41 import decode_fused
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.quantization import QuantizedProjection

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


def build(seed=4121, **overrides):
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
        packed, scales = mx.quantize(weight.astype(mx.bfloat16), group_size=32, bits=bits, mode=mode)[:2]
        return QuantizedProjection(packed, scales, bits, mode)

    for layer in model.layers:
        att, ffn = layer.attn, layer.ffn
        for name in ("wq_a", "wkv", "wq_b", "wo_b"):
            setattr(att, name, quantized(getattr(att, name).weight, 8, "mxfp8"))
        # convert.py dequantizes the FP8 checkpoint's wo_a to bf16: exactly MXFP8 values
        packed, scales = mx.quantize(att.wo_a.weight, group_size=32, bits=8, mode="mxfp8")[:2]
        att.wo_a.weight = mx.dequantize(packed, scales, None, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)
        for name in ("w1", "w2", "w3"):
            setattr(ffn.experts, name, quantized(getattr(ffn.experts, name).weight, 4, "mxfp4"))
            setattr(ffn.shared_experts, name, quantized(getattr(ffn.shared_experts, name).weight, 8, "mxfp8"))
    mx.eval(model.parameters())
    return model


def test_fused_decode_matches_unfused_block(monkeypatch):
    model = build()
    c = model._config
    assert all(decode_fused.supported(layer, c) for layer in model.layers)
    calls = {"fused": 0}
    real = decode_fused.decode_block

    def counted(*args, **kwargs):
        calls["fused"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block", counted)
    supported = decode_fused.supported
    ids = np.random.default_rng(3).integers(3, 60, 24).tolist()
    caches = []
    monkeypatch.setattr(decode_fused, "supported", lambda layer, c: False)
    for _ in range(2):
        cache = model.make_cache()
        mx.eval(model(mx.array([ids[:17]]), cache=cache))
        caches.append(cache)
    token, worst = ids[17], 0.0
    for step in range(6):
        monkeypatch.setattr(decode_fused, "supported", lambda layer, c: False)
        reference = model(mx.array([[token]]), cache=caches[0])[0, -1].astype(mx.float32)
        monkeypatch.setattr(decode_fused, "supported", supported)
        fused = model(mx.array([[token]]), cache=caches[1])[0, -1].astype(mx.float32)
        reference, fused = np.asarray(reference), np.asarray(fused)
        scale = np.abs(reference).max()
        worst = max(worst, float(np.abs(fused - reference).max() / scale))
        top2 = np.sort(reference)[-2:]
        if top2[1] - top2[0] > 0.02 * scale:
            assert fused.argmax() == reference.argmax()
        token = ids[18 + step]
    assert calls["fused"] == 6 * c.n_layers
    # Free-order rounding differences grow through the layers: on this random model the fused
    # path stays within 0-3% of the largest logit (seeds 11-13, 4121), while a one-ulp change
    # of a single element per embedding row already moves the logits by 30-50%. A wrong layout
    # or kernel selection gives O(1) garbage. Rounding-point conformance per stage is checked
    # on the released weights by the decode audit, not here.
    assert worst < 0.08, worst


def test_fused_decode_rejects_unsupported_layouts():
    model = build()
    c = model._config
    layer = model.layers[1]
    assert decode_fused.supported(layer, c)
    att = model.layers[2].attn
    packed, scales, biases = mx.quantize(
        mx.dequantize(att.wq_b.weight, att.wq_b.scales, None, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16),
        group_size=64, bits=4,
    )
    att.wq_b = QuantizedProjection(packed, scales, 4, "affine", biases=biases, group_size=64)
    assert not decode_fused.supported(model.layers[2], c)
    other = model.layers[3].attn
    scaled = (other.wo_a.weight.astype(mx.float32) * 1.01).astype(mx.bfloat16)   # off the FP8 grid
    assert not mx.array_equal(scaled, other.wo_a.weight).item()
    other.wo_a.weight = scaled
    assert not decode_fused.supported(model.layers[3], c)


@pytest.mark.parametrize("batch", [2, 3, 8])
def test_batched_decode_matches_single_requests(monkeypatch, batch):
    """On the row kernels (below decode_moe.MMA_MIN_BATCH requests, and in MTP verify blocks),
    B single-token requests through decode_block_nb give each request the logits it gets
    decoding alone (per-request arithmetic is the single-request kernels')."""
    from omlx.patches.deepseek_v41 import decode_moe
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    monkeypatch.setattr(decode_moe, "use_mma", lambda nb: False)
    model = build()
    calls = {"nb": 0}
    real = decode_fused.decode_block_nb

    def counted(*args, **kwargs):
        calls["nb"] += 1
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
    tokens = [p[-1] for p in prompts]
    for step in range(4):
        expected = [np.asarray(model(mx.array([[t]]), cache=c_)[0, -1].astype(mx.float32)) for t, c_ in zip(tokens, singles)]
        got = np.asarray(model(mx.array([[t] for t in tokens]), cache=batched)[:, -1].astype(mx.float32))
        for r in range(batch):
            np.testing.assert_allclose(got[r], expected[r], rtol=0, atol=1e-6 * np.abs(expected[r]).max())
        tokens = [int(e.argmax()) for e in expected]
    assert calls["nb"] == 4 * model._config.n_layers
    assert attention["batched"] == 4 * model._config.n_layers   # windows are full: request-batched attention


def _aq(rng, nb, k):
    from omlx.patches.deepseek_v41.quantization import quantize_activation

    x = mx.array(rng.normal(0, 1, (nb, k)).astype(np.float32)).astype(mx.bfloat16)
    return quantize_activation(x).astype(mx.float32)


@pytest.mark.parametrize("nb", [4, 5, 8])
def test_batched_fp8_mma_projection_within_fp32_summation_of_exact(monkeypatch, nb):
    """Batched projections run on simdgroup matrices from decode_moe.MMA_MIN_BATCH vectors.
    Every product is exact (fp8 weight x act_quant activation), so each output must equal
    the float64 dot product up to fp32 summation rounding (K * 2^-24 * sum|w x|) plus the
    bf16 rounding."""
    from omlx.patches.deepseek_v41 import decode_layer

    names = []
    real_kernel = decode_layer.kernel
    monkeypatch.setattr(decode_layer, "kernel", lambda name: (names.append(name), real_kernel(name))[1])
    att = build().layers[0].attn
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


@pytest.mark.parametrize("nb", [4, 8])
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
    q_mma = DL.kernel("wqb_rope_mma")(
        inputs=[qrq, att.wq_b.weight, att.wq_b.scales, cos, sin], template=tmpl + [("KS", ks)],
        grid=(nq // 8 * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
        output_shapes=[(nb * nq,)], output_dtypes=[mx.bfloat16])[0]
    q_rows = DL.kernel("wqb_rope_nb")(
        inputs=[qrq, att.wq_b.weight, att.wq_b.scales, cos, sin], template=tmpl,
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


def test_small_batches_keep_rows_kernels(monkeypatch):
    """Below MMA_MIN_BATCH the fp8_rows_nb kernels stay in use (faster there)."""
    from omlx.patches.deepseek_v41 import decode_layer, decode_moe

    names = []
    real_kernel = decode_layer.kernel
    monkeypatch.setattr(decode_layer, "kernel", lambda name: (names.append(name), real_kernel(name))[1])
    att = build().layers[0].attn
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


@pytest.mark.parametrize("nb", [2, 3, 8])
def test_batched_moe_matches_single_token_moe(monkeypatch, nb):
    """Without matrix tiles, each token of the batched MoE gets the single-token kernels'
    output, routing indices and weights bit for bit, also when tokens share experts: the
    routed kernels then dispatch every token of a row block together (see _UP_NB)."""
    from omlx.patches.deepseek_v41 import decode_moe

    monkeypatch.setattr(decode_moe, "use_mma", lambda nb: False)
    model = build(n_routed_experts=16, n_activated_experts=4)   # >2 slots: their sum order shows
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


@pytest.mark.parametrize("batch", [2, 8])
def test_batched_engram_projects_each_request_like_a_single_request(batch):
    """Batched decode feeds Engram one row per request. Each row is projected on its own
    (mxfp8_gemv, as when the request decodes alone); a B-row quantized_matmul rounds its fp32
    sums differently, which made a request's tokens depend on its batch."""
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
    # Engram computes in fp32 and returns h's dtype: fp32 h keeps every projection bit visible.
    h = mx.array(rng.normal(0, 1, (batch, 1, c.hc_mult, c.dim)).astype(np.float32))
    batched = engram(h, ids)
    single = mx.concatenate([engram(h[r : r + 1], ids[r : r + 1]) for r in range(batch)])
    assert mx.array_equal(batched, single).item()


@pytest.mark.parametrize("prompt_length", [11, 2])   # full window (batched attention) / filling
@pytest.mark.parametrize("rows,accepted", [(2, 0), (3, 2), (4, 1), (6, 3)])
def test_verify_block_matches_token_by_token_decode(monkeypatch, rows, accepted, prompt_length):
    """An MTP verify block through the fused kernels gives every token bit for bit the
    logits of decoding it alone, and after a partial rollback the cache continues exactly
    like one that decoded only the accepted tokens."""
    model = build(vocab_size=4096)   # the vocab head kernel (per-row exact) needs >= 4096 rows
    calls = []
    real = decode_fused.decode_block_verify

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block_verify", counted)
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
    assert model.mtp_partial_rollback(verified, accepted, rows - 1)
    for t in tokens[: accepted + 1]:
        mx.eval(model(mx.array([[t]]), cache=accepted_only))
    assert verified[0].size() == accepted_only[0].size()
    nxt = mx.array([[tokens[accepted + 1]]])
    assert mx.array_equal(model(nxt, cache=verified), model(nxt, cache=accepted_only)).item()


def test_mtp_generation_matches_greedy_with_fused_kernels(monkeypatch):
    """DSpark speculation over the fused decode kernels reproduces greedy decoding token for
    token (verify blocks are token-exact, rejected drafts roll back exactly)."""
    from mlx_lm.generate import BatchGenerator

    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    model = build(
        vocab_size=4096,
        preserve_mtp=True,
        n_mtp_layers=1,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_markov_rank=32,
    )
    prompt = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected, cache = [], model.make_cache()
    ids = mx.array([prompt])
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
    model.configure_mtp(True, 3)
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
