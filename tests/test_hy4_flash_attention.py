# SPDX-License-Identifier: Apache-2.0
"""Tiny, offline oracles for the released HY V4 Flash text attention."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.hy_v4_flash.attention import Attention

DSA = "deepseek_sparse_attention"
SWA = "sliding_attention"


def _args():
    return SimpleNamespace(
        layer_types=[DSA, SWA],
        hidden_size=12,
        q_lora_rank=7,
        kv_lora_rank=6,
        num_attention_heads=3,
        qk_nope_head_dim=5,
        qk_rope_head_dim=4,
        v_head_dim=7,
        swa_num_attention_heads=4,
        swa_qk_nope_head_dim=3,
        swa_qk_rope_head_dim=6,
        swa_v_head_dim=5,
        compressed_indexer_compress_ratio=3,
        compressed_indexer_group_topk=2,
        compressed_indexer_num_attention_heads=5,
        compressed_indexer_head_dim=8,
        compressed_indexer_candidate_topk=6,
        learnable_sink_init=0.0,
        rms_norm_eps=1e-5,
        attention_bias=False,
        sliding_window=4,
        max_position_embeddings=1024,
        rope_parameters={
            DSA: {
                "rope_type": "default",
                "rope_theta": 1e7,
                "mrope_section": [1, 1, 0],
            },
            SWA: {
                "rope_type": "default",
                "rope_theta": 1e4,
                "mrope_section": [1, 1, 1],
            },
        },
    )


def _fixture(block, length, args=None, bf16=False):
    args = args or _args()
    model = Attention(args, block)
    rng = np.random.default_rng(103)
    state = {}
    weights = []
    for name, template in sorted(tree_flatten(model.parameters())):
        if "norm.weight" in name:
            data = rng.uniform(0.5, 1.5, template.shape)
        elif name == "learnable_sink_param":
            data = np.linspace(-1, 2, template.size)
        else:
            data = rng.normal(0, 0.45 if name.endswith("ape") else 0.22, template.shape)
        dtype = mx.float32
        if bf16 and not any(
            k in name
            for k in (
                "learnable_sink_param",
                "intra_group_ape",
                "key_gate_projection",
                "key_value_projection",
                "compressed_indexer.key_norm",
            )
        ):
            dtype = mx.bfloat16
        w = mx.array(data.astype(np.float32)).astype(dtype)
        weights.append((name, w))
        state[name] = np.array(w.astype(mx.float32), dtype=np.float64)
    model.load_weights(weights, strict=True)
    x = mx.array(rng.normal(size=(2, length, args.hidden_size)).astype(np.float32))
    return model, x.astype(mx.bfloat16) if bf16 else x, state


def _reference(args, block, x, state, *, ignore_recall=False):
    """Independent expanded-KV/eager-mask form from Transformers c2ec8505.

    Unlike the MLX implementation this recomputes whole groups from scratch,
    scores *every* token, and scatters the per-query shortlist into a dense mask.
    """
    x = np.array(x.astype(mx.float32), dtype=np.float64)
    b, t, _ = x.shape
    sliding = args.layer_types[block] == SWA
    p = "swa_" if sliding else ""
    h = getattr(args, p + "num_attention_heads")
    rdim = getattr(args, p + "qk_rope_head_dim")
    ndim = getattr(args, p + "qk_nope_head_dim")
    vdim = getattr(args, p + "v_head_dim")
    theta = args.rope_parameters[args.layer_types[block]]["rope_theta"]
    slots = np.arange(t)

    def norm(a, name):
        return (
            a
            * state[name + ".weight"]
            / np.sqrt(np.mean(a**2, axis=-1, keepdims=True) + args.rms_norm_eps)
        )

    def linear(a, name):
        return a @ state[name + ".weight"].T + state.get(name + ".bias", 0.0)

    def rotate(a, positions):
        # Complex phase over adjacent real/imaginary pairs of the suffix.
        phase = np.exp(
            1j * positions[:, None] * theta ** (-np.arange(0, rdim, 2) / rdim)
        )
        z = a[..., -rdim::2] + 1j * a[..., -rdim + 1 :: 2]
        z = z * phase[None, :, None, :]
        return np.concatenate(
            (
                a[..., :-rdim],
                np.stack((z.real, z.imag), -1).reshape(*z.shape[:-1], rdim),
            ),
            -1,
        )

    def softmax(a, axis):
        z = np.exp(a - a.max(axis=axis, keepdims=True))
        return z / z.sum(axis=axis, keepdims=True)

    resid = norm(linear(x, "q_a_proj"), "q_a_layernorm")
    q = rotate(linear(resid, "q_b_proj").reshape(b, t, h, ndim + rdim), slots)
    combined = linear(x, "kv_a_proj_with_mqa")
    latent = norm(combined[..., :-rdim], "kv_a_layernorm")
    r = rotate(combined[..., -rdim:][:, :, None], slots)
    expanded = linear(latent, "kv_b_proj").reshape(b, t, h, ndim + vdim)
    k = np.concatenate((expanded[..., :ndim], np.repeat(r, h, axis=2)), axis=-1)
    v = expanded[..., ndim:]
    logits = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(ndim + rdim)
    allowed = np.broadcast_to(slots <= slots[:, None], (b, t, t)).copy()
    if sliding:
        allowed &= slots > slots[:, None] - args.sliding_window
    elif t > args.compressed_indexer_candidate_topk and not ignore_recall:
        p = "compressed_indexer."
        ratio = args.compressed_indexer_compress_ratio
        n, d = (
            args.compressed_indexer_num_attention_heads,
            args.compressed_indexer_head_dim,
        )
        iq = rotate(linear(resid, p + "query_projection").reshape(b, t, n, d), slots)
        raw = norm(linear(x, p + "token_key_projection"), p + "token_key_norm")
        groups = raw[:, : t // ratio * ratio].reshape(b, t // ratio, ratio, d)
        gates = linear(groups, p + "key_gate_projection") + state[p + "intra_group_ape"]
        keys = np.sum(linear(groups, p + "key_value_projection") * softmax(gates, 2), 2)
        keys = rotate(
            norm(keys, p + "key_norm")[:, :, None], np.arange(keys.shape[1]) * ratio
        )[:, :, 0]
        score = np.maximum(0, np.einsum("bqnd,bgd->bqng", iq, keys))
        weights = linear(x, p + "head_weight_projection") / np.sqrt(n * d)
        scores = np.einsum("bqn,bqng->bqg", weights, score)
        for i in range(b):
            for j in range(t):
                own = j // ratio
                previous = np.argsort(-scores[i, j, :own], kind="stable")[
                    : args.compressed_indexer_group_topk - 1
                ]
                allowed[i, j] &= np.isin(slots // ratio, [*previous, own])
    logits = np.where(allowed[:, None], logits, -np.inf)
    sink = np.broadcast_to(
        state["learnable_sink_param"][None, :, None, None], (b, h, t, 1)
    )
    probs = softmax(np.concatenate((logits, sink), axis=-1), -1)[..., :t]
    y = np.einsum("bhqk,bkhd->bqhd", probs, v).reshape(b, t, -1)
    y *= 1 / (1 + np.exp(-linear(x, "linear_gate")))
    return norm(linear(y, "o_proj"), "post_proj_norm")


@pytest.mark.parametrize(
    "block,length", [(0, 5), (0, 6), (0, 7), (0, 23), (1, 3), (1, 23), (1, 137)]
)
def test_attention_matches_expanded_reference(block, length):
    args = _args()
    attn, x, state = _fixture(block, length, args)
    expected = _reference(args, block, x, state)
    np.testing.assert_allclose(attn(x), expected, atol=2e-5, rtol=1e-5)
    if block == 0 and length > args.compressed_indexer_candidate_topk:
        dense = _reference(args, block, x, state, ignore_recall=True)
        assert np.max(np.abs(expected - dense)) > 0.1


@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize("chunks", [[1] * 17, [2, 1, 4, 1, 1, 1, 7], [7, 6, 4]])
def test_causal_chunk_cache_and_restore(block, chunks):
    args = _args()
    attn, x, state = _fixture(block, sum(chunks), args)
    cache = attn.make_cache()
    ys = []
    for n in chunks:
        start = cache.offset
        ys.append(attn(x[:, start : start + n], cache=cache))
        assert cache.offset == start + n
        # No token-dependent state may be hidden on the module or on the cache.
        # mlx-lm 0.32 includes metadata in state; 0.31 passes it separately.
        metadata = (cache.meta_state,) if hasattr(cache, "meta_state") else ()
        cache = type(cache).from_state(cache.state, *metadata)
    expected = _reference(args, block, x, state)
    actual = mx.concatenate(ys, axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(actual, attn(x), atol=2e-5, rtol=1e-5)
    # The cache retains only one shared MLA latent and rotary key per slot.
    assert cache.state[0].shape[1::2] == (1, args.kv_lora_rank)
    if block == 1:
        assert cache.size() == args.sliding_window


def test_trim_dsa_across_group_and_replace_continuation():
    args = _args()
    attn, x, state = _fixture(0, 17, args)
    cache = attn.make_cache()
    mx.eval(attn(x[:, :14], cache))
    assert cache.trim(6) == 6  # Keep two raw tokens of an unfinished group.
    changed = mx.concatenate((x[:, :8], -x[:, 8:]), axis=1)
    actual = attn(changed[:, 8:], cache)
    expected = _reference(args, 0, changed, state)[:, 8:]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("topk", [1, 2])
def test_public_mixed_dtypes_and_own_group_only(topk):
    args = _args()
    args.compressed_indexer_group_topk = topk
    args.compressed_indexer_candidate_topk = (
        topk * args.compressed_indexer_compress_ratio
    )
    attn, x, state = _fixture(0, 11, args, bf16=True)
    actual = attn(x)
    assert actual.dtype == mx.bfloat16
    expected = _reference(args, 0, x, state)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected, atol=0.04, rtol=0.02
    )


def test_non_default_rope_is_rejected():
    args = _args()
    args.rope_parameters[DSA] = dict(rope_type="yarn", rope_theta=16.0, factor=4.0)
    with pytest.raises(ValueError, match="Unsupported HY V4 Flash RoPE type"):
        Attention(args, 0)


def test_public_2049th_token_gathers_only_recalled_slots(monkeypatch):
    args = _args()
    args.compressed_indexer_group_topk = 512
    args.compressed_indexer_compress_ratio = 4
    args.compressed_indexer_candidate_topk = 2048
    attn, x, _ = _fixture(0, 2049, args)
    cache = attn.make_cache()
    # MLX is lazy; we only need the prefill's latent/group cache, not 2048 outputs.
    attn(x[:, :2048], cache)
    seen = []
    aggregate = Attention._aggregate

    def audit(self, q, latent, k_rot, allowed, sink):
        seen.append((latent.shape[-2], allowed))
        return aggregate(self, q, latent, k_rot, allowed, sink)

    monkeypatch.setattr(Attention, "_aggregate", audit)
    mx.eval(attn(x[:, 2048:], cache))
    assert len(seen) == 1
    gathered_width, allowed = seen[0]
    assert gathered_width == 2048  # Not all 2049 keys under a dense mask.
    assert mx.all(allowed.sum(axis=-1) == 511 * 4 + 1).item()


def test_no_dense_substitution_for_unsupported_indexer():
    args = _args()
    args.compressed_indexer_skip_fine_select = False
    with pytest.raises(ValueError, match="group-only"):
        Attention(args, 0)
    args = _args()
    args.layer_types[0] = "full_attention"
    with pytest.raises(ValueError, match="attention type"):
        Attention(args, 0)
