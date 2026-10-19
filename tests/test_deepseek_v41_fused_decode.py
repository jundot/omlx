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


@pytest.mark.parametrize("batch", [2, 3])
def test_batched_decode_matches_single_requests(monkeypatch, batch):
    """B single-token requests through decode_block_nb give each request the logits it gets
    decoding alone (per-request arithmetic is the single-request kernels')."""
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    model = build()
    calls = {"nb": 0}
    real = decode_fused.decode_block_nb

    def counted(*args, **kwargs):
        calls["nb"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_fused, "decode_block_nb", counted)
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


@pytest.mark.parametrize("nb", [2, 3, 8])
def test_batched_moe_matches_single_token_moe(monkeypatch, nb):
    """Without matrix tiles, each token of the batched MoE gets the single-token kernels'
    output, routing indices and weights bit for bit, also when tokens share experts: the
    routed kernels then dispatch every token of a row block together (see _UP_NB)."""
    from omlx.patches.deepseek_v41 import decode_moe

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
