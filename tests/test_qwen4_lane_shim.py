# SPDX-License-Identifier: Apache-2.0
"""Serving-lifecycle test for the Qwen4ExpLaneShim (tiny model)."""

from __future__ import annotations

import mlx.core as mx

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models import qwen4_exp  # noqa: E402
from mlx_vlm.models.qwen4_exp.compiled_decode import Qwen4ExpLaneShim  # noqa: E402
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config  # noqa: E402


def _make_lm():
    config = _tiny_config()
    config.text_config.linear_key_head_dim = 32
    config.text_config.linear_value_head_dim = 32
    mx.random.seed(7)
    model = qwen4_exp.Model(config)
    model.eval()
    from mlx_vlm.models.qwen4_exp.language import fuse_resident_ple_embeddings

    if fuse_resident_ple_embeddings(model, minimum_physical_memory=0) == 0:
        import mlx.nn as nn

        for layer in model.language_model.model.layers:
            ple = getattr(layer, "ple", None)
            emb = getattr(
                getattr(ple, "ple_embedding", None), "ngram_embedding", None
            )
            if type(emb).__name__ == "ShardedEmbedding":
                emb.fused = nn.Embedding(emb.shard_offsets[-1], emb.dims)
    return model.language_model


def _build_caches(lm, B):
    caches = []
    for layer in lm.model.layers:
        if layer.is_linear:
            c = qwen4_exp.language.ArraysCache(size=4 if "ple" in layer else 2)
            c.left_padding = mx.array([0] * B)
            caches.append(c)
        else:
            caches.append(qwen4_exp.language.QSAKVCache().to_batch([0] * B))
    return caches


def _run(scheme):
    lm = _make_lm()
    length = 24
    mx.random.seed(4)
    tokens = mx.random.uniform(0, 40, (2, length)).astype(mx.uint32)
    tokens_b = mx.random.uniform(0, 40, (1, length + 6)).astype(mx.uint32)

    lm._position_ids = None
    caches = _build_caches(lm, 2)
    shim = Qwen4ExpLaneShim(lm)
    model_arg = shim if scheme == "shim" else lm
    out = model_arg(tokens, cache=caches).logits
    mx.eval(out)
    tok = mx.argmax(out[:, -1, :], axis=-1).astype(mx.uint32)
    out_tokens = [[], []]

    def decode(n):
        nonlocal tok
        for _ in range(n):
            logits = model_arg(tok[:, None], cache=caches)
            logits = logits.logits if hasattr(logits, "logits") else logits
            logits = logits[:, -1, :]
            mx.eval(logits)
            tok = mx.argmax(logits, axis=-1).astype(mx.uint32)
            out_tokens[0].append(int(tok[0].item()))
            out_tokens[1].append(int(tok[1].item()))

    def reshape(fn):
        if scheme == "shim":
            shim.before_reshape(caches)
        fn()
        if scheme == "shim":
            shim.after_reshape(caches)

    decode(6)

    def do_filter():
        for cache in caches:
            cache.filter([0])

    reshape(do_filter)

    def decode1(n):
        nonlocal tok
        for _ in range(n):
            logits = model_arg(tok[0:1, None], cache=caches)
            logits = logits.logits if hasattr(logits, "logits") else logits
            logits = logits[:, -1, :]
            mx.eval(logits)
            tok = mx.argmax(logits, axis=-1).astype(mx.uint32)
            out_tokens[0].append(int(tok[0].item()))

    decode1(4)

    def do_extend():
        lm._position_ids = None
        join_caches = _build_caches(lm, 1)
        lm(tokens_b, cache=join_caches).logits
        for a, b in zip(caches, join_caches):
            a.extend(b)

    reshape(do_extend)
    mx.random.seed(99)
    join_tok = mx.random.uniform(0, 40, (1,)).astype(mx.uint32)
    tok_row1 = join_tok

    def decode2(n):
        nonlocal tok, tok_row1
        tok_full = mx.concatenate([tok, tok_row1])
        for _ in range(n):
            logits = model_arg(tok_full[:, None], cache=caches)
            logits = logits.logits if hasattr(logits, "logits") else logits
            logits = logits[:, -1, :]
            mx.eval(logits)
            tok_full = mx.argmax(logits, axis=-1).astype(mx.uint32)
            out_tokens[0].append(int(tok_full[0].item()))
            out_tokens[1].append(int(tok_full[1].item()))
        tok = tok_full[0:1]
        tok_row1 = tok_full[1:2]

    decode2(5)

    def eager_multi():
        pair = mx.array(
            [
                [int(tok[0].item()), int(tok_row1[0].item())],
                [int(tok_row1[0].item()), int(tok[0].item())],
            ],
            dtype=mx.uint32,
        )
        out = lm(pair, cache=caches).logits
        mx.eval(out)

    reshape(eager_multi)
    decode2(3)
    return out_tokens


def test_lane_shim_full_serving_lifecycle():
    eager = _run("eager")
    shim = _run("shim")
    assert shim[0] == eager[0]
    same1 = sum(1 for a, b in zip(eager[1], shim[1]) if a == b)
    assert same1 >= len(eager[1]) - 3
