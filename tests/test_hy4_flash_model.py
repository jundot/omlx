# SPDX-License-Identifier: Apache-2.0
"""Hy4 Flash graph tests: synthetic values only, no public weight payloads."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten  # noqa: E402

from omlx.patches.hy_v4_flash.model import (  # noqa: E402
    HyperConnection,
    HyperHead,
    Model,
    ModelArgs,
    _ngram_inputs,
)

DSA = "deepseek_sparse_attention"
SWA = "sliding_attention"


def _args():
    return ModelArgs.from_dict(
        {
            "vocab_size": 48,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "mlp_layer_types": ["dense", "sparse"],
            "layer_types": [DSA, SWA],
            "intermediate_size": 24,
            "moe_intermediate_size": 12,
            "routed_expert_hidden_size": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "num_attention_heads": 2,
            "swa_num_attention_heads": 3,
            "sliding_window": 3,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "swa_qk_nope_head_dim": 4,
            "swa_qk_rope_head_dim": 4,
            "swa_v_head_dim": 4,
            "q_lora_rank": 8,
            "kv_lora_rank": 6,
            "rope_parameters": {
                t: {"rope_type": "default", "rope_theta": theta}
                for t, theta in [(DSA, 10000000), (SWA, 10000)]
            },
            "compressed_indexer_compress_ratio": 2,
            "compressed_indexer_group_topk": 2,
            "compressed_indexer_num_attention_heads": 2,
            "compressed_indexer_head_dim": 8,
            "hc_mult": 2,
            "hc_magnitude": 1.75,
            "engram_max_ngram_size": 3,
            "engram_layer_ids": [2],
            "engram_layer_head_vocab_sizes": {"2": [[11, 13], [17, 19]]},
            "engram_n_head_per_ngram": 2,
            "engram_n_embed_per_ngram": 8,
            "engram_pad_id": 3,
            "engram_lookup_table_size": 47,
        }
    )


class _TinyTables:
    def __init__(self):
        self.table = np.random.default_rng(553).normal(size=(60, 4)).astype(np.float32)
        self.reads = []

    def gather(self, block, ids):
        assert block == 1
        self.reads.append(ids.copy())
        return mx.array(self.table[ids]).astype(mx.bfloat16)


def _graph(public_dtypes):
    args = _args()
    model = Model(args)
    rng = np.random.default_rng(2026)
    weights = []
    # The public checkpoint keeps iHC, sinks, the router bias and the gated
    # pooler's reductions in FP32. Activations between decoder steps are BF16.
    fp32_markers = (
        "hc_attn_layer",
        "hc_mlp_layer",
        "model.hc_head",
        "intra_group_ape",
        "key_gate_projection",
        "key_value_projection",
        "compressed_indexer.key_norm",
        "e_score_correction_bias",
        "learnable_sink_param",
    )
    for name, leaf in sorted(tree_flatten(model.parameters())):
        if name.endswith("lookup_table"):
            # Include a non-identity map and an id outside the mapped vocabulary.
            arr = mx.arange(leaf.size, dtype=mx.int32) // 2
        else:
            x = rng.normal(0, 0.18, size=leaf.shape).astype(np.float32)
            if ("norm" in name and name.endswith("weight")) or name.endswith(
                "unified_pre_gamma"
            ):
                x += 1
            arr = mx.array(x)
            if public_dtypes and not any(tag in name for tag in fp32_markers):
                arr = arr.astype(mx.bfloat16)
        weights.append((name, arr))
    model.load_weights(weights, strict=True)
    tables = _TinyTables()
    model.set_engram_tables(tables)
    return model, tables


@pytest.mark.parametrize(
    "public_dtypes", [False, True], ids=["strict-fp32-cpu", "public-bf16-device"]
)
def test_full_logits_match_interleaved_cached_requests(public_dtypes):
    # A full-precision CPU control isolates cache correctness from M5 NAX's
    # different GEMM/GEMV rounding for prefill/decode. Also exercise the
    # actual mixed-dtype inference path on the default (Metal) device.
    with mx.stream(mx.default_device() if public_dtypes else mx.cpu):
        _check_cached_requests(public_dtypes)


def _check_cached_requests(public_dtypes):
    model, tables = _graph(public_dtypes)
    # Both exceed the tiny recall budget and sliding window. Request A's
    # mixed chunks also finish a pooling group begun by an earlier call.
    ids = [
        mx.array([[4, 47, 5, 7, 19, 8, 17, 2]]),
        mx.array([[8, 9, 4, 1, 38, 7, 21, 6]]),
    ]
    full = [np.array(model(x)) for x in ids]
    full_rows = list(tables.reads)
    tables.reads.clear()
    caches = [model.make_cache(), model.make_cache()]
    outputs, rows = [[], []], [[], []]
    for widths in zip((1, 3, 2, 2), (2, 1, 1, 4)):
        for i, width in enumerate(widths):
            cache = caches[i]
            start = cache.offset
            chunk = ids[i][:, start : start + width]
            outputs[i].append(np.array(model(chunk, cache=cache)))
            rows[i].append(tables.reads.pop())
            assert cache.offset == start + width
            assert cache.history.shape == (1, 2)
            assert all(kv.offset == cache.offset for kv in cache.attention)
    for i, expected in enumerate(full):
        assert expected.shape == (1, 8, 48)
        assert np.isfinite(expected).all() and expected.std() > 0.01
        np.testing.assert_array_equal(np.concatenate(rows[i], axis=1), full_rows[i])
        np.testing.assert_array_equal(
            np.array(caches[i].history), np.array(ids[i][:, -2:])
        )
        actual = np.concatenate(outputs[i], axis=1)
        assert actual.dtype == expected.dtype == np.float32
        if public_dtypes:
            # Different GEMM/GEMV reduction trees can cross BF16 rounding
            # boundaries. Compare relative L2 *per token*, not a batch-wide
            # average that could hide a bad decode step, and require the same
            # greedy decision. Row selection and history still must be exact.
            relative = np.linalg.norm(actual - expected, axis=-1) / np.linalg.norm(
                expected, axis=-1
            )
            assert np.max(relative) < 2 * 2**-7  # two BF16 epsilons
            np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(
            np.array(model(ids[i], logits_to_keep=1)), expected[:, -1:], atol=3e-6
        )


def test_engram_inputs_pad_then_cross_one_token_steps():
    history = None
    expected = [
        [[5, 3, 3], [8, 3, 3]],
        [[17, 5, 3], [7, 8, 3]],
        [[6, 17, 5], [1, 7, 8]],
    ]
    for want in expected:
        tokens = mx.array([[want[0][0]], [want[1][0]]])
        got, history = _ngram_inputs(tokens, history, order=3, pad_id=3)
        np.testing.assert_array_equal(np.asarray(got)[:, 0], want)


def test_ihc_and_head_match_independent_reference():
    args = _args()
    gate, head = HyperConnection(args), HyperHead(args)
    rng = np.random.default_rng(984)
    streams = mx.array(rng.normal(size=(1, 3, args.hc_mult, args.hidden_size))).astype(
        mx.bfloat16
    )
    x = np.asarray(streams.astype(mx.float32))
    flat = x.reshape(1, 3, -1)
    inv = 1 / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + args.rms_norm_eps)
    for module in (gate, head):
        module.load_weights(
            [
                (k, mx.array(rng.normal(size=v.shape).astype(np.float32)))
                for k, v in tree_flatten(module.parameters())
            ]
        )

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    p = gate.hc_pre
    projection = (flat @ np.array(p.hc_fn).T) * inv
    pre, post = np.split(projection, 2, axis=-1)
    pre_b, post_b = np.split(np.array(p.hc_base), 2)
    pre = sigmoid(pre * p.hc_scale[0].item() + pre_b) + args.hc_eps
    post = (
        args.hc_magnitude * sigmoid(post * p.hc_scale[1].item() + post_b) + args.hc_eps
    )
    got_post, got_pre = gate(streams)
    assert got_post.dtype == mx.float32 and got_pre.dtype == mx.bfloat16
    np.testing.assert_allclose(np.asarray(got_post), post, rtol=2e-6, atol=1e-6)

    def rounded(x):
        return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))

    np.testing.assert_array_equal(
        np.asarray(got_pre.astype(mx.float32)), rounded((pre[..., None] * x).sum(2))
    )
    update = rng.normal(size=(1, 3, args.hidden_size)).astype(np.float32)
    result = gate.combine(streams, got_post, mx.array(update))
    np.testing.assert_array_equal(
        np.array(result.astype(mx.float32)),
        rounded(x + post[..., None] * update[:, :, None]),
    )
    projection = (flat @ np.array(head.hc_head_fn).T) * inv
    pre = (
        sigmoid(projection * np.array(head.hc_head_scale) + np.array(head.hc_head_base))
        + args.hc_eps
    )
    np.testing.assert_array_equal(
        np.array(head(streams).astype(mx.float32)), rounded((pre[..., None] * x).sum(2))
    )


@pytest.mark.parametrize("packed", [False, True], ids=["bare-bf16", "native-oqh"])
def test_public_text_shapes_without_materializing_weights(monkeypatch, packed):
    # Shape-only: construct every normally random MLX weight as a scalar view,
    # not a random operation that could accidentally materialize a giant bank.
    # Routed banks already use scalar broadcasts; tables aren't MLX parameters.
    def scalar_view(shape, **kwargs):
        return mx.broadcast_to(mx.array(0.0), shape)

    monkeypatch.setattr(mx.random, "normal", scalar_view)
    monkeypatch.setattr(mx.random, "uniform", scalar_view)
    engram_ids = [2, 12, 22, 32]
    # We need only the number of heads per order: the moduli affect external
    # table rows, not a single shape of the core text graph.
    v = [[11, 13, 17, 19, 23, 29, 31, 37], [41, 43, 47, 53, 59, 61, 67, 71]]
    args = replace(
        _args(),
        hidden_size=3072,
        vocab_size=120832,
        num_hidden_layers=42,
        layer_types=[DSA if i < 2 or i % 4 == 1 else SWA for i in range(42)],
        mlp_layer_types=["dense"] + ["sparse"] * 41,
        intermediate_size=13824,
        moe_intermediate_size=1536,
        routed_expert_hidden_size=1536,
        n_routed_experts=512,
        n_shared_experts=1,
        num_experts_per_tok=16,
        num_attention_heads=32,
        swa_num_attention_heads=64,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        swa_qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        swa_qk_nope_head_dim=192,
        v_head_dim=256,
        swa_v_head_dim=256,
        sliding_window=512,
        compressed_indexer_compress_ratio=4,
        compressed_indexer_num_attention_heads=16,
        compressed_indexer_head_dim=128,
        compressed_indexer_group_topk=512,
        compressed_indexer_candidate_topk=2048,
        hc_mult=4,
        engram_layer_ids=engram_ids,
        engram_lookup_table_size=120818,
        engram_pad_id=120002,
        engram_n_head_per_ngram=8,
        engram_n_embed_per_ngram=768,
        engram_layer_head_vocab_sizes={str(i): v for i in engram_ids},
    )
    if packed:
        args.omlx_oqh = {
            "version": 1,
            "tensors": {
                f"model.layers.{b}.mlp.experts.{projection}": {
                    "bits": 8,
                    "block": 256,
                    "group_size": 128,
                }
                for b in range(1, 42)
                for projection in ("gate_up_proj", "down_proj")
            },
        }
    shapes = {k: t.shape for k, t in tree_flatten(Model(args).parameters())}
    assert shapes["model.layers.1.hc_attn_layer.hc_pre.hc_fn"] == (8, 12288)
    assert shapes["model.layers.41.mlp.experts.down_proj.weight"] == (
        512,
        1536,
        384 if packed else 1536,
    )
    assert shapes["model.layers.0.self_attn.q_b_proj.weight"] == (8192, 1024)
    assert shapes["model.layers.2.self_attn.q_b_proj.weight"] == (16384, 1024)
    assert shapes["model.hc_head.hc_head_fn"] == (4, 12288)
    assert shapes["lm_head.weight"] == (120832, 3072)
    # Digest generated from all 148 original tencent/Hy4-flash headers at
    # d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959. Select text tensors, exclude
    # four external .multi_head_embedding.embedding.weight tensors, append
    # .weight to both raw expert banks. No downloaded assets needed for CI.
    # For v1 oQh also expand the 82 rotated banks to their packed weight,
    # scales, biases and per-block signs.
    assert len(shapes) == (1678 if packed else 1432)
    lines = [f"{name}:{','.join(map(str, shape))}\n" for name, shape in shapes.items()]
    assert hashlib.sha256("".join(sorted(lines)).encode()).hexdigest() == (
        "e3b4a8e5e6fd9498971aca819343c26f7b430845b290a3c44947cca921ba2c46"
        if packed
        else "61639c87c9de2de9c78412734ba8c6625d992adb94a57d114af93b9bf7bf3148"
    )
