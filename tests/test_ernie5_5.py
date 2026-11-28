# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import struct
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.generate import BatchGenerator
from mlx_vlm.models.cache import CacheList, KVCache, PoolingCache, RotatingKVCache
from mlx_vlm.models.deepseek_v4.hyper_connection import (
    _hc_expand_op,
    _hc_split_sinkhorn_ops,
)
from mlx_vlm.models.mla import MultiLinear
from mlx_vlm.models.switch_layers import SwitchLinear
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from omlx import oq
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.custom_kernels.nax import is_nax_available
from omlx.engine.vlm import VLMBatchedEngine, _apply_template_thinking_mode
from omlx.exceptions import InvalidRequestError
from omlx.models.vlm import VLMModelAdapter
from omlx.patches import fp4_expert_rows
from omlx.patches.deepseek_v4 import apply_pooling_cache_support
from omlx.patches.ernie5_5 import kernels, language, loading, sparse_prefill_available
from omlx.patches.ernie5_5 import oq as ernie_oq
from omlx.patches.ernie5_5.config import ModelConfig
from omlx.patches.ernie5_5.dspark import DSpark
from omlx.patches.ernie5_5.language import HCAAttention, LanguageModel, MLAAttention
from omlx.patches.ernie5_5.loading import _convert_shard, repack
from omlx.patches.ernie5_5.model import Model
from omlx.patches.ernie5_5.processing import Processor, TokenizedPrompt
from omlx.patches.mimo_v2.decode_fast import combine_rms
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, cache_rollback
from omlx.patches.mlx_lm_mtp import batch_generator as mtp_batch_generator
from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import _PRIME_CTX_ATTR
from omlx.scheduler import _patched_merge_caches
from omlx.utils.sampling import make_sampler

_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
_needs_sparse_kernels = pytest.mark.skipif(
    not sparse_prefill_available(), reason="native sparse kernels are not built"
)


def test_fp4_experts_repack_to_mxfp4_bit_exact():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, (4, 32), dtype=np.uint8)
    exponents = rng.integers(112, 123, (4, 2), dtype=np.uint8)
    weight, scales, ok = repack(mx.array(codes.view(np.int8)), mx.array(exponents))
    nibbles = np.stack([codes & 15, codes >> 4], axis=-1).reshape(4, 64)
    values = np.where(nibbles & 8, -1.0, 1.0) * _E2M1[nibbles & 7]
    expected = values * np.repeat(2.0 ** (exponents.astype(np.float32) - 127), 32, 1)
    actual = mx.dequantize(
        weight, scales, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32
    )
    np.testing.assert_array_equal(np.array(actual), expected)
    assert ok.item()


def test_fp8_blocks_repack_to_mxfp8_bit_exact():
    rng = np.random.default_rng(1)
    signs = rng.integers(0, 2, (192, 256), dtype=np.uint8) << 7
    codes = rng.integers(0, 0x7F, (192, 256), dtype=np.uint8) | signs
    block_scales = 2.0 ** rng.integers(-12, -3, (2, 2))
    weight, scales, ok = repack(
        mx.array(codes), mx.array(block_scales.astype(np.float32))
    )
    expected = np.array(mx.from_fp8(mx.array(codes), dtype=mx.float32)) * np.repeat(
        np.repeat(block_scales, 128, 0)[:192], 128, 1
    )
    actual = mx.dequantize(
        weight, scales, group_size=32, bits=8, mode="mxfp8", dtype=mx.float32
    )
    np.testing.assert_array_equal(np.array(actual), expected)
    assert ok.item()


def test_convert_shard_layout_and_scale_rejection():
    experts = "model.layers.1.block_sparse_moe.experts"
    query = "model.layers.0.self_attn.linear_q_down_proj.weight"
    raw = {
        f"{experts}.{e}.w1.{name}": value
        for e in range(2)
        for name, value in (
            ("weight", mx.zeros((4, 16), dtype=mx.int8)),
            ("weight_scale_inv", mx.full((4, 1), 127, dtype=mx.uint8)),
        )
    }
    raw.update(
        {
            query: mx.zeros((128, 128), dtype=mx.uint8),
            query + "_scale_inv": mx.ones((1, 1)),
            "model.layers.7.self_attn.kv_b_proj.weight": mx.full(
                (128, 128), 0x38, dtype=mx.uint8
            ),
            "model.layers.7.self_attn.kv_b_proj.weight_scale_inv": mx.full(
                (1, 1), 0.125
            ),
            "model.layers.1.self_attn.linear_o_group_proj": mx.zeros(
                (256, 128), dtype=mx.uint8
            ),
            "model.layers.1.self_attn.linear_o_group_proj_scale_inv": mx.ones((2, 1)),
            "model.mtp.0.enorm.weight": mx.ones((4,)),
            "model.norm.weight": mx.ones((4,), dtype=mx.bfloat16),
        }
    )
    pending = {}
    out = _convert_shard(raw, pending, n_experts=2)
    assert pending == {}
    assert {key: (value.shape, value.dtype) for key, value in out.items()} == {
        f"{experts}.w1.weight": ((2, 4, 4), mx.uint32),
        f"{experts}.w1.scales": ((2, 4, 1), mx.uint8),
        "model.layers.0.self_attn.linear_q_down_proj.weight": ((128, 32), mx.uint32),
        "model.layers.0.self_attn.linear_q_down_proj.scales": ((128, 4), mx.uint8),
        "model.layers.7.self_attn.kv_b_proj.weight": ((128, 128), mx.bfloat16),
        "model.layers.1.self_attn.linear_o_group_proj.weight": ((256, 32), mx.uint32),
        "model.layers.1.self_attn.linear_o_group_proj.scales": ((256, 4), mx.uint8),
        "model.norm.weight": ((4,), mx.bfloat16),
    }
    assert mx.all(out["model.layers.7.self_attn.kv_b_proj.weight"] == 0.125).item()
    raw[query + "_scale_inv"] = mx.full((1, 1), 0.3)
    with pytest.raises(ValueError, match="linear_q_down_proj"):
        _convert_shard(raw, {}, n_experts=2)


class _Renderer:
    def render_chat(self, messages, tools=None, **kwargs):
        self.kwargs = kwargs
        text = messages[-1]["content"]
        return SimpleNamespace(
            text="<|ernie:im_start|>" + text,
            segments=(("<|ernie:im_start|>", True), (text, False)),
        )

    def encode_segments(self, segments):
        return [7 if control else 1000 + len(text) for text, control in segments]

    def __call__(self, *args, **kwargs):
        raise AssertionError("a rendered prompt was tokenized again")


def test_chat_prompt_keeps_renderer_token_ids():
    renderer = _Renderer()
    engine = VLMBatchedEngine("ernie")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type="ernie5_5_moe")
    )
    engine._processor = Processor(renderer)
    text = "hi <|ernie:im_end|>"
    ids, *_ = engine._prepare_vision_inputs(
        [{"role": "user", "content": text}],
        images=[],
        chat_template_kwargs={"reasoning_effort": "LOW", "options": {}, "other": 1},
    )
    assert ids == [7, 1000 + len(text)]
    assert renderer.kwargs == {
        "add_generation_prompt": True,
        "continue_final_message": False,
        "enable_thinking": None,
        "options": {},
        "reasoning_effort": "low",
    }
    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": text}], images=[object()]
        )


def test_reasoning_effort_off_aliases_and_invalid():
    for effort, expected in (
        ("none", {"enable_thinking": False}),
        ("xhigh", {"reasoning_effort": "high"}),
        (0.5, {}),
    ):
        kwargs = {"reasoning_effort": effort}
        _apply_template_thinking_mode("ernie5_5_moe", kwargs)
        assert kwargs == expected
    with pytest.raises(InvalidRequestError, match="reasoning_effort"):
        _apply_template_thinking_mode("ernie5_5_moe", {"reasoning_effort": "loud"})


def test_tokenized_prompt_survives_pickle():
    clone = pickle.loads(pickle.dumps(TokenizedPrompt("hi", [7, 1002])))
    assert clone == "hi" and clone.token_ids == (7, 1002)


def test_cached_prefill_matches_full_forward_and_skips_logits():
    mx.random.seed(0)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        sliding_window=16,
        compress_ratios=[16, -2],
        hybrid_mla_num_attention_heads=4,
        hybrid_mla_q_lora_rank=32,
        hybrid_mla_kv_lora_rank=32,
        hybrid_mla_qk_nope_head_dim=16,
        hybrid_mla_qk_rope_head_dim=8,
        hybrid_mla_v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        moe_intermediate_size=32,
        moe_latent_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        vha_rank=4,
    )
    model = LanguageModel(config)
    router = model.layers[1].ffn.gate
    router.weight = mx.random.normal(router.weight.shape)
    router.weight_1 = mx.random.normal(router.weight.shape)
    ids = mx.random.randint(0, 64, (1, 70))
    full = model(ids).logits[:, -1]
    cache = model.make_cache()
    assert model(ids[:, :32], cache=cache, skip_logits=True).logits is None
    model(ids[:, 32:69], cache=cache)
    step = model(ids[:, 69:], cache=cache).logits[:, -1]
    assert mx.allclose(full, step, atol=1e-4).item()


def _rel(actual, expected):
    actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
    return (mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item()


def test_hyper_connection_matches_reference_and_rows_are_independent():
    mx.random.seed(4)
    config = ModelConfig()
    width = config.hidden_size
    hc, norm = language.HyperConnection(config), nn.RMSNorm(width)
    hc.fn = mx.random.normal(hc.fn.shape) * (4 * width) ** -0.5
    hc.base = mx.random.normal(hc.base.shape) * 0.3
    hc.scale = mx.array([0.7, 0.9, 1.3])
    norm.weight = (1 + 0.1 * mx.random.normal((width,))).astype(mx.bfloat16)
    x = mx.random.normal((1, 8, 4, width)).astype(mx.bfloat16)
    y = mx.random.normal((1, 8, width)).astype(mx.bfloat16)
    normed, mixes = hc.collapse(x, norm)
    out = hc.expand(y, x, mixes)

    eps = config.rms_norm_eps
    flat = x.astype(mx.float32).flatten(-2)
    ref_mixes = (flat @ hc.fn.T) / (
        mx.sqrt(mx.mean(flat * flat, -1, keepdims=True)) + eps
    )
    pre, post, comb = _hc_split_sinkhorn_ops(
        ref_mixes, hc.scale, hc.base, 4, config.hc_sinkhorn_iters, config.hc_eps
    )
    collapsed = (pre[..., None] * x.astype(mx.float32)).sum(-2).astype(x.dtype)
    assert _rel(mixes, ref_mixes) < 1e-5
    assert _rel(normed, mx.fast.rms_norm(collapsed, norm.weight, eps)) < 1e-3
    assert _rel(out, _hc_expand_op(y, x, post, comb)) < 1e-3
    for rows in (1, 2, 4):
        part_normed, part_mixes = hc.collapse(x[:, :rows], norm)
        part_out = hc.expand(y[:, :rows], x[:, :rows], part_mixes)
        assert mx.array_equal(part_mixes, mixes[:, :rows]).item()
        assert mx.array_equal(part_normed, normed[:, :rows]).item()
        assert mx.array_equal(part_out, out[:, :rows]).item()


def _latent_moe(mode):
    mx.random.seed(5)
    config = ModelConfig(
        hidden_size=256,
        moe_latent_size=512,
        moe_intermediate_size=512,
        n_routed_experts=32,
        num_experts_per_tok=4,
    )
    moe = language.LatentMoE(config)
    moe.update(
        tree_unflatten(
            [
                (k, mx.random.normal(v.shape) * v.shape[-1] ** -0.5)
                for k, v in tree_flatten(moe.parameters())
            ]
        )
    )
    moe.set_dtype(mx.bfloat16)
    moe.e_score_correction_bias = mx.random.normal((32,)) * 0.01
    moe.gate.routed_scaling_factor_param = 2 + mx.random.uniform(shape=(32,))
    bits = 8 if mode == "mxfp8" else 4
    nn.quantize(moe.switch_mlp, group_size=32, bits=bits, mode=mode)
    nn.quantize(
        moe,
        group_size=32,
        bits=8,
        mode="mxfp8",
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    return moe, mx.random.normal((1, 8, 256)).astype(mx.bfloat16)


def test_decode_moe_kernels_match_reference_and_rows_are_independent(monkeypatch):
    moe, x = _latent_moe("mxfp4")
    fast = moe(x)

    rows = x[0]
    gate = moe.gate
    ids, weights = kernels.route(
        rows,
        gate.weight,
        gate.weight_1,
        moe.e_score_correction_bias,
        gate.routed_scaling_factor_param,
        gate.top_k,
        True,
    )
    for i in range(8):
        ref_ids, ref_weights = gate(rows[i : i + 1], moe.e_score_correction_bias)
        assert mx.array_equal(ids[i : i + 1], ref_ids).item()
        assert mx.array_equal(weights[i : i + 1], ref_weights).item()
    experts = moe.switch_mlp
    latent = moe.fc1_latent_proj(rows)
    pairs = mx.expand_dims(latent, (-2, -3))
    stock = experts.down_proj(
        experts.activation(experts.up_proj(pairs, ids), experts.gate_proj(pairs, ids)),
        ids,
    ).reshape(*ids.shape, -1)
    act = fp4_expert_rows.gate_up(
        latent,
        ids,
        experts.gate_proj.weight,
        experts.gate_proj.scales,
        experts.up_proj.weight,
        experts.up_proj.scales,
        n_out=512,
        scale_group_size=32,
        activation=kernels.SITU,
        extra=(moe._situ,),
    )
    out = fp4_expert_rows.down(
        act,
        ids,
        experts.down_proj.weight,
        experts.down_proj.scales,
        scale_group_size=32,
    )
    assert mx.array_equal(out.reshape(stock.shape), stock).item()
    zeros = mx.zeros_like(latent)
    _, combined = combine_rms(
        zeros, stock, weights, moe.latent_norm.weight, moe.latent_norm.eps
    )
    for n in (1, 2, 4):
        part = fp4_expert_rows.down(
            fp4_expert_rows.gate_up(
                latent[:n],
                ids[:n],
                experts.gate_proj.weight,
                experts.gate_proj.scales,
                experts.up_proj.weight,
                experts.up_proj.scales,
                n_out=512,
                scale_group_size=32,
                activation=kernels.SITU,
                extra=(moe._situ,),
            ),
            ids[:n],
            experts.down_proj.weight,
            experts.down_proj.scales,
            scale_group_size=32,
        )
        assert mx.array_equal(part, out[: n * 4]).item()
        _, part_combined = combine_rms(
            zeros[:n],
            stock[:n],
            weights[:n],
            moe.latent_norm.weight,
            moe.latent_norm.eps,
        )
        assert mx.array_equal(part_combined, combined[:n]).item()
        part_ids, part_weights = kernels.route(
            rows[:n],
            gate.weight,
            gate.weight_1,
            moe.e_score_correction_bias,
            gate.routed_scaling_factor_param,
            gate.top_k,
            True,
        )
        assert mx.array_equal(part_ids, ids[:n]).item()
        assert mx.array_equal(part_weights, weights[:n]).item()

    monkeypatch.setattr(kernels, "route", lambda *args: None)
    assert _rel(fast, moe(x)) < 1e-2


@pytest.mark.skipif(not is_nax_available(), reason="needs an M5 (NAX) GPU")
def test_batched_prefill_offsets_keep_the_nax_hca_kernel(monkeypatch):
    mx.random.seed(6)
    attn = _real_shape_layer(HCAAttention(ModelConfig(), 0))
    q = mx.random.normal((1, 64, 32, 512)).astype(mx.bfloat16)
    kv = mx.random.normal((1, 1, 32, 512)).astype(mx.bfloat16)
    pooled = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    sinks = attn.attn_sink.astype(mx.bfloat16)
    assert attn._prefill_attention(q, kv, pooled, sinks, mx.array([128])) is None
    ran, kernel = [], language.sparse_topk_attention_nax

    def spy(*args):
        ran.append(kernel(*args))
        return ran[-1]

    monkeypatch.setattr(language, "sparse_topk_attention_nax", spy)
    assert attn._prefill_attention(q, kv, pooled, sinks, 128) is ran[0] is not None


def test_decode_moe_rows_on_mxfp8_experts_match_reference(monkeypatch):
    moe, x = _latent_moe("mxfp8")
    fast = moe(x)
    monkeypatch.setattr(kernels, "route", lambda *args: None)
    assert _rel(fast, moe(x)) < 1e-2


def _each_row_matches_alone(fn, x, axis, out_axis=None):
    full = fn(x)
    for n in (1, 2, 4):
        for start in range(0, x.shape[axis], n):
            rows = [slice(None)] * x.ndim
            rows[axis] = slice(start, start + n)
            out = [slice(None)] * full.ndim
            out[axis if out_axis is None else out_axis] = slice(start, start + n)
            assert mx.array_equal(fn(x[tuple(rows)]), full[tuple(out)]).item()


def test_decode_rows_get_their_one_row_bits_on_release_formats():
    mx.random.seed(8)
    dense = nn.QuantizedLinear.from_linear(
        nn.Linear(1024, 8192, bias=False), group_size=32, bits=8, mode="mxfp8"
    )
    x = mx.random.normal((8, 1024)).astype(mx.bfloat16)
    _each_row_matches_alone(lambda r: language._linear(dense, r), x, 0)
    bf16 = nn.Linear(1024, 8192, bias=False)
    bf16.set_dtype(mx.bfloat16)
    _each_row_matches_alone(lambda r: language._linear(bf16, r), x, 0)

    config = ModelConfig(
        hidden_size=256, q_lora_rank=256, num_attention_heads=8, o_groups=2
    )
    attn = _real_shape_layer(HCAAttention(config, 0))
    grouped = mx.random.normal((2, 8, 2048)).astype(mx.bfloat16)

    def project(g):
        return attn._project_groups(g, g.shape[1], 1)

    _each_row_matches_alone(project, grouped, 1, 0)
    nn.quantize(
        attn,
        group_size=32,
        bits=8,
        mode="mxfp8",
        class_predicate=lambda path, _: path in ("wo_a", "wq_b"),
    )
    _each_row_matches_alone(project, grouped, 1, 0)
    qr = mx.random.normal((1, 16, 256)).astype(mx.bfloat16)
    flat = attn.wq_b(qr).reshape(1, 16, 8, 512).transpose(0, 2, 1, 3)
    assert _rel(attn._queries(qr), flat) < 1e-3

    o = mx.random.normal((1, 8, 8, 512)).astype(mx.bfloat16)
    mix = (mx.eye(8) + 0.1 * mx.random.normal((8, 8))).astype(mx.bfloat16)
    mixed = language.kernels.vha(o, mix, 2)
    ref = (mix @ o.transpose(0, 2, 1, 3)).reshape(8, 2, -1).swapaxes(0, 1)
    assert _rel(mixed, ref) < 1e-2
    _each_row_matches_alone(lambda r: language.kernels.vha(r, mix, 2), o, 2, 1)
    batched = language.kernels.vha(mx.contiguous(o.swapaxes(0, 2)), mix, 2)
    assert mx.array_equal(batched, mixed).item()

    rows = mx.random.normal((8, 4096)).astype(mx.bfloat16).astype(mx.float32)
    weight = (mx.random.normal((512, 4096)) * 0.02).astype(mx.bfloat16)
    _each_row_matches_alone(lambda r: language.project_logits(r, weight), rows, 0)

    for mode in ("mxfp4", "mxfp8", "affine"):
        moe, hidden = _latent_moe(mode)
        _each_row_matches_alone(moe, hidden, 1)


def _real_shape_layer(module):
    params = tree_flatten(module.parameters())
    module.update(
        tree_unflatten(
            [(k, mx.random.normal(v.shape) * v.shape[-1] ** -0.5) for k, v in params]
        )
    )
    module.set_dtype(mx.bfloat16)
    module.attn_sink = mx.random.normal(module.attn_sink.shape) * 0.5
    return module


@_needs_sparse_kernels
def test_prefill_rows_keep_prefix_window_and_top_keys():
    mx.random.seed(1)
    config = ModelConfig()
    attn = _real_shape_layer(MLAAttention(config))
    x = mx.random.normal((1, 4608, config.hidden_size)).astype(mx.bfloat16)
    new_cache, ref_cache = KVCache(), KVCache()
    for start in range(0, 4608, 1536):
        chunk = x[:, start : start + 1536]
        qr = attn.q_a_layernorm(attn.q_a_proj(chunk))
        rows = attn.indexer.select(chunk, qr, new_cache)
        mask = language.create_attention_mask(chunk, ref_cache, return_array=True)
        ref, _ = attn.indexer(chunk, qr, mask, ref_cache)
        total = start + 1536
        rows = np.array(rows[0, 0])
        ref = np.array(mx.broadcast_to(ref, (1, 1, 1536, total))[0, 0])
        overlap = []
        for r, row in enumerate(rows):
            p = start + r
            picked = row[row < total]
            assert row.max() <= total and len(set(picked)) == len(picked)
            assert set(range(max(0, p - 127), p + 1)) <= set(picked)
            if p < config.index_topk + config.sliding_window:
                assert sorted(picked) == list(range(p + 1))
            else:
                expected = set(np.flatnonzero(ref[r]))
                overlap.append(len(expected & set(picked)) / len(expected))
        if overlap:
            assert np.mean(overlap) > 0.99 and min(overlap) > 0.95


@_needs_sparse_kernels
def test_sparse_prefill_matches_dense_attention_on_the_same_keys(monkeypatch):
    mx.random.seed(2)
    config = ModelConfig()
    x = mx.random.normal((1, 4096, config.hidden_size)).astype(mx.bfloat16)
    picked, windowed = [], []
    select, prefill = language.Indexer.select, HCAAttention._prefill_attention

    def select_spy(self, *args):
        picked.append(select(self, *args))
        return picked[-1]

    def prefill_spy(self, *args):
        windowed.append(prefill(self, *args))
        return windowed[-1]

    monkeypatch.setattr(language.Indexer, "select", select_spy)
    monkeypatch.setattr(HCAAttention, "_prefill_attention", prefill_spy)
    layers = (
        (
            _real_shape_layer(MLAAttention(config)),
            lambda: CacheList(KVCache(), KVCache()),
        ),
        (
            _real_shape_layer(HCAAttention(config, 0)),
            lambda: CacheList(RotatingKVCache(max_size=128), PoolingCache(128)),
        ),
    )
    outputs = []
    for layer, make_cache in layers:
        cache = make_cache()
        outputs.append([layer(x[:, i : i + 2048], cache) for i in (0, 2048)])
        mx.eval(outputs[-1])
    assert len(picked) == 2 and [out is not None for out in windowed] == [True] * 2
    monkeypatch.setattr(language, "sparse_prefill_available", lambda: False)
    monkeypatch.setattr(HCAAttention, "_prefill_attention", lambda *args: None)
    dense_call = language.Indexer.__call__

    def same_keys(self, chunk, qr, mask, cache=None):
        dense_call(self, chunk, qr, mask, cache)
        rows = picked.pop(0)[0, 0]
        hits = mx.zeros((rows.shape[0], cache.offset + 1), dtype=mx.bool_)
        hits = mx.put_along_axis(hits, rows.astype(mx.int32), mx.array(True), axis=-1)
        return hits[None, None, :, :-1], None

    monkeypatch.setattr(language.Indexer, "__call__", same_keys)
    for (layer, make_cache), sparse in zip(layers, outputs):
        cache = make_cache()
        for i, out in zip((0, 2048), sparse):
            ref = layer(x[:, i : i + 2048], cache).astype(mx.float32)
            err = mx.linalg.norm(out.astype(mx.float32) - ref) / mx.linalg.norm(ref)
            assert err.item() < 0.02


@_needs_sparse_kernels
def test_sparse_prefill_peak_memory_stays_below_one_score_tensor():
    mx.random.seed(3)
    config = ModelConfig()
    attn = _real_shape_layer(MLAAttention(config))
    x = mx.random.normal((1, 1024, config.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    cache = CacheList(KVCache(), KVCache())
    peaks = []
    for _ in range(8):
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        mx.eval(attn(x, cache))
        peaks.append(mx.get_peak_memory() - before)
    assert max(peaks) < 1 << 30


_TINY = dict(
    vocab_size=64,
    hidden_size=128,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    head_dim=64,
    q_lora_rank=64,
    qk_rope_head_dim=8,
    o_groups=2,
    o_lora_rank=32,
    sliding_window=16,
    compress_ratios=[16, 16, -2],
    hybrid_mla_num_attention_heads=4,
    hybrid_mla_q_lora_rank=64,
    hybrid_mla_kv_lora_rank=32,
    hybrid_mla_qk_nope_head_dim=16,
    hybrid_mla_qk_rope_head_dim=8,
    hybrid_mla_v_head_dim=16,
    index_n_heads=2,
    index_head_dim=16,
    index_topk=8,
    moe_intermediate_size=64,
    moe_latent_size=64,
    n_routed_experts=8,
    num_experts_per_tok=2,
    vha_rank=4,
    dspark_target_layer_ids=[0, 1, 2],
    dspark_noise_token_id=5,
    dspark_markov_rank=8,
)


def _release_values(rng):
    hidden, vocab, experts = 128, 64, 8
    values = {}

    def add(name, kind, shape, std=None, fill=None):
        if fill is not None:
            values[name] = (kind, np.full(shape, fill, np.float32))
        else:
            std = shape[-1] ** -0.5 if std is None else std
            values[name] = (kind, rng.normal(0, std, shape).astype(np.float32))

    def hyper(prefix):
        for part in ("alpha_pre", "alpha_post", "alpha_res"):
            add(f"{prefix}.{part}", "f32", (1,), fill=1.0)
        add(f"{prefix}.bias", "f32", (24,), std=0.1)
        add(f"{prefix}.mapping_proj.weight", "f32", (4 * hidden, 24), std=0.02)

    def output_projections(a):
        add(f"{a}.linear_o_group_proj", "fp8", (64, 128))
        add(f"{a}.o_proj.weight", "fp8", (hidden, 64))
        add(f"{a}.gate_proj.weight", "fp8", (64, 64))

    def draft_attention(a):
        add(f"{a}.vha_postmix_U", "bf16", (4, 4), std=0.1)
        add(f"{a}.vha_postmix_V", "bf16", (4, 4), std=0.1)
        add(f"{a}.linear_q_down_proj.weight", "fp8", (64, hidden))
        add(f"{a}.q_layernorm.weight", "bf16", (64,), fill=1.0)
        add(f"{a}.linear_q_up_proj.weight", "fp8", (256, 64))
        add(f"{a}.linear_kv_proj.weight", "fp8", (64, hidden))
        add(f"{a}.kv_layernorm.weight", "bf16", (64,), fill=1.0)
        add(f"{a}.core_attention.attn_sink", "f32", (4,), fill=0.0)
        output_projections(a)

    def moe(prefix):
        add(f"{prefix}.gate.weight", "bf16", (experts, hidden))
        add(f"{prefix}.gate.weight_1", "bf16", (experts, hidden))
        add(f"{prefix}.gate.routed_scaling_factor_param", "f32", (experts,), fill=1.0)
        add(f"{prefix}.e_score_correction_bias", "f32", (experts,), fill=0.0)
        add(f"{prefix}.fc1_latent_proj.weight", "fp8", (64, hidden))
        add(f"{prefix}.fc2_latent_proj.weight", "fp8", (hidden, 64))
        add(f"{prefix}.latent_norm.weight", "bf16", (64,), fill=1.0)
        for part, shape in (
            ("w1", (64, hidden)),
            ("w3", (64, hidden)),
            ("w2", (hidden, 64)),
        ):
            add(f"{prefix}.shared_experts.{part}.weight", "fp8", shape)
        for e in range(experts):
            for part in ("w1", "w2", "w3"):
                add(f"{prefix}.experts.{e}.{part}.weight", "expert", (64, 64), std=0.3)

    add("model.embed_tokens.weight", "bf16", (vocab, hidden), std=1.0)
    add("lm_head.weight", "bf16", (vocab, hidden))
    add("model.lm_head.multimax_ranges", "bf16", (4,), fill=0.0)
    add("model.lm_head.multimax_ts", "bf16", (4,), fill=0.0)
    add("model.norm.weight", "bf16", (hidden,), fill=1.0)
    add("model.mhc_contract.hc_head_base", "f32", (4,), std=0.1)
    add("model.mhc_contract.hc_head_fn", "f32", (4 * hidden, 4), std=0.02)
    add("model.mhc_contract.hc_head_scale", "f32", (1,), fill=1.0)
    for i, ratio in enumerate(_TINY["compress_ratios"]):
        p = f"model.layers.{i}"
        add(f"{p}.input_layernorm.weight", "bf16", (hidden,), fill=1.0)
        add(f"{p}.post_attention_layernorm.weight", "bf16", (hidden,), fill=1.0)
        hyper(f"{p}.self_attention_hyper_connection")
        hyper(f"{p}.mlp_hyper_connection")
        a = f"{p}.self_attn"
        add(f"{a}.vha_postmix_U", "bf16", (4, 4), std=0.1)
        add(f"{a}.vha_postmix_V", "bf16", (4, 4), std=0.1)
        if ratio == -2:
            add(f"{a}.q_a_proj.weight", "fp8", (64, hidden))
            add(f"{a}.q_a_layernorm.weight", "bf16", (64,), fill=1.0)
            add(f"{a}.q_b_proj.weight", "fp8", (96, 64))
            add(f"{a}.kv_a_proj_with_mqa.weight", "fp8", (40, hidden))
            add(f"{a}.kv_a_layernorm.weight", "bf16", (32,), fill=1.0)
            add(f"{a}.kv_b_proj.weight", "fp8", (128, 32))
            add(f"{a}.o_proj.weight", "fp8", (hidden, 64))
            add(f"{a}.gate_proj.weight", "fp8", (64, 64))
            add(f"{a}.core_attention.softmax_offset", "bf16", (4,), fill=0.0)
            x = f"{a}.core_attention.indexer"
            add(f"{x}.wq_b.weight", "fp8", (32, 64))
            add(f"{x}.wk.weight", "bf16", (16, hidden))
            add(f"{x}.k_norm.weight", "bf16", (16,), fill=1.0)
            add(f"{x}.k_norm.bias", "bf16", (16,), fill=0.0)
            add(f"{x}.weights_proj.weight", "bf16", (2, hidden))
        else:
            add(f"{a}.linear_q_down_proj.weight", "fp8", (64, hidden))
            add(f"{a}.q_layernorm.weight", "bf16", (64,), fill=1.0)
            add(f"{a}.linear_q_up_proj.weight", "fp8", (256, 64))
            add(f"{a}.linear_kv_proj.weight", "fp8", (64, hidden))
            add(f"{a}.kv_layernorm.weight", "bf16", (64,), fill=1.0)
            add(f"{a}.core_attention.attn_sink", "f32", (4,), fill=0.0)
            c = f"{a}.core_attention.compressor"
            add(f"{c}.ape", "f32", (ratio, 64), std=0.1)
            add(f"{c}.linear_wkv.weight", "bf16", (64, hidden))
            add(f"{c}.linear_wgate.weight", "bf16", (64, hidden))
            add(f"{c}.norm.weight", "bf16", (64,), fill=1.0)
            output_projections(a)
        if i == 0:
            add(f"{p}.mlp.gate_proj.weight", "fp8", (128, hidden))
            add(f"{p}.mlp.up_proj.weight", "fp8", (128, hidden))
            add(f"{p}.mlp.down_proj.weight", "fp8", (hidden, 128))
        else:
            moe(f"{p}.block_sparse_moe")
    for s in range(3):
        p = f"model.mtp.{s}"
        add(f"{p}.input_layernorm.weight", "bf16", (hidden,), fill=1.0)
        add(f"{p}.post_attention_layernorm.weight", "bf16", (hidden,), fill=1.0)
        hyper(f"{p}.self_attention_hyper_connection")
        hyper(f"{p}.mlp_hyper_connection")
        draft_attention(f"{p}.self_attn")
        moe(f"{p}.block_sparse_moe")
    add("model.mtp.feature_projection.weight", "fp8", (hidden, 3 * hidden))
    add("model.mtp.feature_norm.weight", "bf16", (hidden,), fill=1.0)
    add("model.mtp.norm.weight", "bf16", (hidden,), fill=1.0)
    add("model.mtp.mhc_contract.hc_head_base", "f32", (4,), std=0.1)
    add("model.mtp.mhc_contract.hc_head_fn", "f32", (4 * hidden, 4), std=0.02)
    add("model.mtp.mhc_contract.hc_head_scale", "f32", (1,), fill=1.0)
    add("model.mtp.markov_head.markov_w1.weight", "bf16", (8, vocab))
    add("model.mtp.markov_head.markov_w2.weight", "bf16", (vocab, 8))
    add("model.mtp.confidence_head.projection.weight", "bf16", (1, hidden + 8))
    return values


def _fp8_blocks(values):
    rows, cols = values.shape
    blocks = (-(-rows // 128), -(-cols // 128))
    padded = np.zeros((blocks[0] * 128, blocks[1] * 128), np.float32)
    padded[:rows, :cols] = np.abs(values)
    amax = padded.reshape(blocks[0], 128, blocks[1], 128).max(axis=(1, 3))
    scale = (2.0 ** np.ceil(np.log2(np.maximum(amax, 1e-30) / 448))).astype(np.float32)
    full = np.repeat(np.repeat(scale, 128, 0), 128, 1)[:rows, :cols]
    return np.array(mx.to_fp8(mx.array(values / full))), scale


def _write_release(path, values, fp4_experts):
    arrays = {}
    for name, (kind, value) in values.items():
        if kind == "expert" and fp4_experts:
            packed, scales = mx.quantize(
                mx.array(value), group_size=32, bits=4, mode="mxfp4"
            )
            arrays[name] = ("I8", np.array(packed).view(np.int8))
            arrays[name + "_scale_inv"] = ("F8_E8M0", np.array(scales))
        elif kind in ("fp8", "expert"):
            codes, scale = _fp8_blocks(value)
            arrays[name] = ("F8_E4M3", codes)
            arrays[name + "_scale_inv"] = ("F32", scale)
        elif kind == "bf16":
            arrays[name] = (
                "BF16",
                np.array(mx.array(value).astype(mx.bfloat16).view(mx.uint16)),
            )
        else:
            arrays[name] = ("F32", value)
    header, offset = {}, 0
    for name, (dtype, data) in arrays.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(data.shape),
            "data_offsets": [offset, offset + data.nbytes],
        }
        offset += data.nbytes
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.mkdir()
    with open(path / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded)
        for _, data in arrays.values():
            f.write(np.ascontiguousarray(data).tobytes())
    quantization = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    if fp4_experts:
        quantization["expert_dtype"] = "mxfp4"
    config = {
        **_TINY,
        "model_type": "ernie5_5_moe",
        "eos_token_id": 2,
        "dspark_block_size": 7,
        "quantization_config": quantization,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {k: "model-00001-of-00001.safetensors" for k in arrays}}
        )
    )
    vocabulary = ["<pad>", "<s>", "</s>"] + [f"t{i}" for i in range(3, 64)]
    backend = Tokenizer(
        models.WordLevel(dict(zip(vocabulary, range(64))), unk_token="<pad>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    ).save_pretrained(path)
    return path


def _logits(model, tokens):
    return [
        model.language_model(row[None]).logits[0].astype(mx.float32) for row in tokens
    ]


def _kl(reference, actual):
    total = 0.0
    for p, q in zip(reference, actual):
        log_p, log_q = p - mx.logsumexp(p, -1, keepdims=True), q - mx.logsumexp(
            q, -1, keepdims=True
        )
        total += (mx.exp(log_p) * (log_p - log_q)).sum(-1).mean().item()
    return total / len(reference)


def _output_tensors(path):
    tensors = {}
    for shard in path.glob("*.safetensors"):
        tensors.update(mx.load(str(shard)))
    return tensors


def test_oq_builds_keep_release_tensors_and_order_by_level(tmp_path, monkeypatch):
    values = _release_values(np.random.default_rng(0))
    fp4 = _write_release(tmp_path / "fp4", values, fp4_experts=True)
    fp8 = _write_release(tmp_path / "fp8", values, fp4_experts=False)
    tokens = mx.array(np.random.default_rng(1).integers(3, 64, (6, 48)))
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **k: tokens)
    with pytest.raises(ValueError, match="FP8 release"):
        oq.quantize_oq_streaming(str(fp4), str(tmp_path / "oq6-fp4"), 6)
    plain, _ = loading.load(fp4, trust_remote_code=True)
    assert plain.language_model.get_mtp_module() is None
    monkeypatch.setattr(loading, "is_mtp_active", lambda: True)
    taps = mx.random.normal((1, 5, 3 * 128), key=mx.random.key(4)).astype(mx.bfloat16)
    kl = {}
    for level, source in ((2.5, fp4), (3, fp4), (6, fp8)):
        output = tmp_path / f"oq{level:g}e"
        oq.quantize_oq_streaming(
            str(source),
            str(output),
            level,
            enhanced=True,
            preserve_mtp=True,
            imatrix_num_samples=len(tokens),
            imatrix_seq_length=tokens.shape[1],
        )
        reference, _ = loading.load(source, trust_remote_code=True)
        built, _ = loading.load(output, trust_remote_code=True)
        kl[level] = _kl(_logits(reference, tokens), _logits(built, tokens))
        drafts = [
            lm.dspark_forward(taps, mx.array([[7]]), lm.make_mtp_cache())[0]
            for lm in (reference.language_model, built.language_model)
        ]
        assert mx.array_equal(*drafts).item()
        tensors = _output_tensors(output)
        expected = dict(tree_flatten(reference.parameters()))
        expert_bytes = 0
        for key, value in tensors.items():
            if ".mtp." in key:
                continue
            if ".switch_mlp." in key:
                expert_bytes += value.nbytes
                continue
            assert mx.array_equal(value, expected[key]).item(), key
        for part in ("weight", "scales"):
            release = expected[f"language_model.dspark.layers.0.attn.wo_a.{part}"]
            built_part = tensors[
                f"language_model.model.mtp.0.self_attn.linear_o_group_proj.{part}"
            ]
            assert mx.array_equal(built_part.reshape(release.shape), release).item()
        target, cap = oq._OQ_BPW_TARGETS[level]
        assert target - 0.1 <= 8 * expert_bytes / (2 * 3 * 8 * 64 * 64) <= cap
        estimate = oq.estimate_bpw_and_size(str(source), level, preserve_mtp=True)
        written = sum(value.nbytes for value in tensors.values())
        assert estimate["output_size_bytes"] == pytest.approx(written, rel=1e-3)
        experts = "language_model.model.layers.1.ffn.switch_mlp.down_proj"
        report = json.loads((output / "oq_imatrix_report.json").read_text())
        assert report["refine"]["status"] == "done"
        assert experts in report["applied"]
        draft = "language_model.model.mtp.0.block_sparse_moe.experts.w1"
        source_bytes = mx.load(str(source / "model-00001-of-00001.safetensors"))
        first = source_bytes["model.mtp.0.block_sparse_moe.experts.0.w1.weight"]
        assert mx.array_equal(
            tensors[draft + ".weight"][0], first.view(mx.uint32)
        ).item()
    assert 0 < kl[6] < kl[3] < kl[2.5] < 0.5, kl


def test_layerwise_imatrix_matches_cached_full_forward(tmp_path):
    values = _release_values(np.random.default_rng(2))
    model, _ = loading.load(
        _write_release(tmp_path / "fp4", values, fp4_experts=True),
        trust_remote_code=True,
    )
    tokens = mx.array(np.random.default_rng(3).integers(3, 64, (3, 48)))
    walked = ernie_oq.collect_imatrix(model, tokens)
    language = model.language_model
    collector = oq.OQImatrixCollector()
    for i in (1, 2):
        prefix = f"language_model.model.layers.{i}.ffn.switch_mlp."
        collector.install(language.layers[i].ffn.switch_mlp, name_prefix=prefix)
    for row in tokens:
        mx.eval(language(row[None], cache=language.make_cache()).logits)
    assert walked.keys() == collector.entries.keys() and len(walked) == 6
    for name, entry in collector.entries.items():
        np.testing.assert_array_equal(walked[name].counts, entry.counts)
        np.testing.assert_array_equal(walked[name].in_sum2, entry.in_sum2)


_SMALL = dict(
    vocab_size=512,
    hidden_size=512,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    q_lora_rank=256,
    o_groups=2,
    o_lora_rank=256,
    compress_ratios=[128, 128, -2, 128],
    hybrid_mla_num_attention_heads=4,
    hybrid_mla_q_lora_rank=256,
    hybrid_mla_qk_nope_head_dim=64,
    hybrid_mla_v_head_dim=128,
    index_n_heads=4,
    index_topk=16,
    moe_intermediate_size=512,
    moe_latent_size=512,
    n_routed_experts=64,
    num_experts_per_tok=4,
    vha_rank=4,
    dspark_block_size=7,
    dspark_target_layer_ids=[1, 2, 3],
    dspark_noise_token_id=7,
)


def _small_model(**overrides):
    mx.random.seed(0)
    config = ModelConfig(**{**_SMALL, **overrides})
    model = Model(config)
    model.language_model.dspark = DSpark(config)
    float32 = ("_hc.", "hc_head", "mhc_contract", "attn_sink", "ape", "ffn.gate.")
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if key.endswith(("_hc.scale", "routed_scaling_factor_param")) or "norm" in key:
            value = mx.ones(value.shape)
        elif key.endswith("e_score_correction_bias") or "multimax" in key:
            value = mx.zeros(value.shape)
        elif value.ndim > 1:
            std = 0.02 if key.endswith("fn") else value.shape[-1] ** -0.5
            value = mx.random.normal(value.shape) * std
        else:
            value = mx.random.normal(value.shape) * 0.1
        if not any(name in key for name in float32):
            value = value.astype(mx.bfloat16)
        updates.append((key, value))
    model.update(tree_unflatten(updates))
    keep = ("compressor", "indexer.wk", "weights_proj", "lm_head", "markov", "embed")

    def spec(path, module):
        if any(name in path for name in keep):
            return False
        if isinstance(module, SwitchLinear):
            return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        return isinstance(module, (nn.Linear, MultiLinear)) and {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
        }

    nn.quantize(model, class_predicate=spec)
    lm = model.language_model
    lm.dspark.head = nn.QuantizedLinear.from_linear(
        lm.lm_head, group_size=32, bits=4, mode="mxfp4"
    )
    mx.eval(model.parameters())
    return model


def _cache_arrays(cache):
    return tree_flatten(
        [
            [
                CacheTypeRegistry.get_handler_for_object(c).serialize_state(c)
                for c in layer.caches
            ]
            for layer in cache
        ]
    )


def _restored(cache):
    handler = CacheTypeRegistry.get_handler_for_object(cache[0])
    return [
        handler.reconstruct_cache(
            handler.extract_state(layer), handler.serialize_meta_state(layer)
        )
        for layer in cache
    ]


@pytest.mark.parametrize("prompt_len", [40, 124, 250])
def test_verify_rows_and_rollback_match_one_token_decode(prompt_len):
    cache_rollback.apply()
    apply_pooling_cache_support()
    lm = _small_model().language_model
    prompt = mx.random.randint(0, 512, (1, prompt_len), key=mx.random.key(prompt_len))

    def prefill(variant):
        cache = lm.make_cache()
        for start in range(0, prompt_len, 100):
            lm(prompt[:, start : start + 100], cache=cache)
        if variant == "padded":
            longer = lm.make_cache()
            lm(mx.zeros((1, prompt_len + 300), dtype=mx.int32), cache=longer)
            merged = _patched_merge_caches([cache, longer])
            for layer in merged:
                layer.filter([0])
            return merged
        if variant.startswith("restored"):
            cache = _restored(cache)
        return _patched_merge_caches([cache]) if variant.endswith("merged") else cache

    for variant in ("single", "merged", "padded", "restored", "restored merged"):
        cache, tokens, logits = prefill(variant), [3], []
        for _ in range(19):
            logits.append(lm(mx.array([tokens[-1:]]), cache=cache).logits[0, -1])
            tokens.append(int(mx.argmax(logits[-1]).item()))
        seq = mx.array([tokens])
        for accepted in (0, 3, 7):
            cache, start = prefill(variant), 0
            for _ in range(2):
                language.set_dspark_verify_armed(True)
                cache_rollback.set_undo_armed(True)
                try:
                    rows = lm(seq[:, start : start + 8], cache=cache).logits[0]
                finally:
                    language.set_dspark_verify_armed(False)
                    cache_rollback.set_undo_armed(False)
                for r in range(8):
                    assert mx.array_equal(rows[r], logits[start + r]).item()
                if accepted < 7:
                    assert lm.mtp_partial_rollback(cache, accepted, 7)
                else:
                    mtp_batch_generator._clear_rollback(cache)
                    assert all(
                        getattr(c, "_ernie_verify_undo", None) is None
                        for layer in cache
                        for c in layer.caches
                    )
                start += accepted + 1
            step = lm(seq[:, start : start + 1], cache=cache).logits[0, -1]
            assert mx.array_equal(step, logits[start]).item()
            reference = prefill(variant)
            for i in range(start + 1):
                lm(seq[:, i : i + 1], cache=reference)
            for (_, got), (_, want) in zip(
                _cache_arrays(cache), _cache_arrays(reference), strict=True
            ):
                assert (got is None and want is None) or mx.array_equal(
                    got, want
                ).item()


@pytest.mark.parametrize(
    "overrides", [{}, {"hidden_size": 4096, "first_k_dense_replace": 4}]
)
def test_requests_verified_together_match_their_own_decode(overrides):
    cache_rollback.apply()
    apply_pooling_cache_support()
    lm = _small_model(**overrides).language_model
    prompts = [
        mx.random.randint(0, 512, (1, n), key=mx.random.key(n)) for n in (124, 250)
    ]

    def prefill(prompt):
        cache = lm.make_cache()
        for start in range(0, prompt.shape[1], 100):
            lm(prompt[:, start : start + 100], cache=cache)
        return cache

    blocks, decoded = [], []
    for prompt in prompts:
        cache, tokens, logits = prefill(prompt), [3], []
        for _ in range(4):
            logits.append(lm(mx.array([tokens[-1:]]), cache=cache).logits[0, -1])
            tokens.append(int(mx.argmax(logits[-1]).item()))
        blocks.append(tokens[:4])
        decoded.append(logits)
    rows, _, caches = lm.mtp_verify_rows(
        mx.array(blocks), [prefill(p) for p in prompts]
    )
    for prompt, block, logits, out, cache, accepted in zip(
        prompts, blocks, decoded, rows, caches, (1, 2)
    ):
        for r in range(4):
            assert mx.array_equal(out[r], logits[r]).item()
        assert lm.mtp_partial_rollback(cache, accepted, 3)
        reference = prefill(prompt)
        for token in block[: accepted + 1]:
            lm(mx.array([[token]]), cache=reference)
        for (_, got), (_, want) in zip(
            _cache_arrays(cache), _cache_arrays(reference), strict=True
        ):
            assert (got is None and want is None) or mx.array_equal(got, want).item()


@_needs_sparse_kernels
def test_mla_verify_rows_match_their_decode_at_real_shape():
    mx.random.seed(4)
    attn = _real_shape_layer(MLAAttention(ModelConfig()))
    lengths = (4300, 2100)
    x = [mx.random.normal((1, n + 4, 4096)).astype(mx.bfloat16) for n in lengths]

    def prefill(i):
        cache = CacheList(KVCache(), KVCache())
        for start in range(0, lengths[i], 1024):
            attn(x[i][:, start : min(start + 1024, lengths[i])], cache)
        return cache

    decoded = []
    for i, n in enumerate(lengths):
        cache = prefill(i)
        decoded.append(
            mx.concatenate(
                [attn(x[i][:, n + r : n + r + 1], cache) for r in range(4)], 1
            )
        )
    rows = attn(
        mx.concatenate([x[i][:, n:] for i, n in enumerate(lengths)]),
        [prefill(i) for i in range(len(lengths))],
    )
    for i in range(len(lengths)):
        assert mx.array_equal(rows[i], decoded[i][0]).item()


def test_mtp_generation_matches_greedy(monkeypatch):
    assert apply_mlx_lm_mtp_patch()
    apply_pooling_cache_support()
    model = _small_model()
    lm = model.language_model
    drafts, rollbacks = [], []
    forward, rollback = type(lm).dspark_forward, type(lm).mtp_partial_rollback

    def tracked_forward(self, *args, **kwargs):
        drafts.append(1)
        return forward(self, *args, **kwargs)

    def tracked_rollback(self, *args):
        rollbacks.append(args[1:])
        return rollback(self, *args)

    monkeypatch.setattr(type(lm), "dspark_forward", tracked_forward)
    monkeypatch.setattr(type(lm), "mtp_partial_rollback", tracked_rollback)
    prompt = mx.random.randint(0, 512, (250,), key=mx.random.key(1)).tolist()

    def generate(use_mtp):
        lm.configure_mtp(use_mtp, 7)
        generator = BatchGenerator(
            VLMModelAdapter(model),
            max_tokens=24,
            prefill_step_size=64,
            sampler=lambda lp: mx.argmax(lp, -1),
        )
        out = []
        try:
            generator.insert([prompt])
            while len(out) < 24:
                out.extend(r.token for r in generator.next()[1])
        finally:
            generator.close()
        return out

    assert generate(True) == generate(False)
    assert drafts and rollbacks


def test_parked_decode_rebuilds_the_never_parked_draft_context():
    assert apply_mlx_lm_mtp_patch()
    cache_rollback.apply()
    apply_pooling_cache_support()
    lm = _small_model().language_model
    lm.configure_mtp(True, 3)
    prompt = mx.random.randint(0, 512, (1, 140), key=mx.random.key(5))
    seq = mx.random.randint(0, 512, (1, 160), key=mx.random.key(6))
    anchor = mx.array([[11]])

    def prefill():
        cache = lm.make_cache()
        lm(prompt, cache=cache)
        return cache

    for parked in (20, 150):
        cache = prefill()
        for t in range(parked):
            lm(seq[:, t : t + 1], cache=cache)
        assert getattr(lm, _PRIME_CTX_ATTR).caches[0].offset == prompt.shape[1]
        out = lm(seq[:, parked : parked + 1], cache=cache, return_hidden=True)
        rings, _ = lm.mtp_take_primed(cache, None)
        block = lm.dspark_hidden_rows([out.hidden_states[0]], [anchor], [rings])

        cache = prefill()
        hidden = lm(seq[:, :1], cache=cache, return_hidden=True).hidden_states[0]
        reference, _ = lm.mtp_take_primed(cache, None)
        for start in range(1, parked + 1, 3):
            lm.dspark_append_context(hidden, reference)
            rows = seq[:, start : min(start + 3, parked + 1)]
            _, hidden, _ = lm.mtp_verify_rows(rows, [cache])
            mtp_batch_generator._clear_rollback(cache)
        expected = lm.dspark_hidden_rows([hidden], [anchor], [reference])
        assert mx.array_equal(block, expected).item()
        for got, want in zip(rings, reference, strict=True):
            assert got.offset == want.offset
            assert mx.array_equal(got.keys, want.keys).item()


@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_requests_draft_and_verify_together(monkeypatch, temp):
    assert apply_mlx_lm_mtp_patch()
    apply_pooling_cache_support()
    model = _small_model()
    lm = model.language_model
    lm.configure_mtp(True, 3)
    shared = []
    verify_rows = type(lm).mtp_verify_rows

    def tracked(self, inputs, row_caches):
        shared.append(inputs.shape[0])
        return verify_rows(self, inputs, row_caches)

    monkeypatch.setattr(type(lm), "mtp_verify_rows", tracked)
    monkeypatch.setattr(lm, "_omlx_mtp_depth_fixed", True, raising=False)
    generator = BatchGenerator(
        VLMModelAdapter(model),
        max_tokens=24,
        prefill_step_size=64,
        sampler=make_sampler(temp=temp, top_p=0.95, top_k=20),
    )
    out = {}
    try:
        for n in (250, 140):
            prompt = mx.random.randint(0, 512, (n,), key=mx.random.key(n)).tolist()
            (uid,) = generator.insert([prompt])
            out[uid] = []
            while not out[uid]:
                for r in generator.next()[1]:
                    out[r.uid].append(r.token)
        while any(len(tokens) < 24 for tokens in out.values()):
            for r in generator.next()[1]:
                out[r.uid].append(r.token)
    finally:
        generator.close()
    assert [len(tokens) for tokens in out.values()] == [24, 24]
    assert max(shared) == 2
