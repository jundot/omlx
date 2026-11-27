# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hy4 / Hy4-flash text models on tiny synthetic checkpoints."""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models.cache import (  # noqa: E402
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
)

from omlx import oq  # noqa: E402
from omlx.adapter.output_parser import correct_tool_parser_type  # noqa: E402
from omlx.api.thinking import prompt_opens_thinking  # noqa: E402
from omlx.custom_kernels import nax  # noqa: E402
from omlx.engine.vlm import (  # noqa: E402
    _apply_template_thinking_mode,
    _count_image_tokens_real,
)
from omlx.models.vlm import VLMModelAdapter  # noqa: E402
from omlx.patches.hy_v4 import (  # noqa: E402
    exact,
    indexing,
    language,  # noqa: E402
    layers,
    verify_dispatch,
)
from omlx.patches.hy_v4.config import VisionConfig  # noqa: E402
from omlx.patches.hy_v4.flash import CompressedIndexer, Engram  # noqa: E402
from omlx.patches.hy_v4.language import set_dspark_verify_armed  # noqa: E402
from omlx.patches.hy_v4.layers import Indexer  # noqa: E402
from omlx.patches.hy_v4.model import (  # noqa: E402
    Model,
    ModelConfig,
    dequantize_block_fp8,
)
from omlx.patches.hy_v4.processor import (  # noqa: E402
    HyV4ImageProcessor,
    HyV4Processor,
)
from omlx.patches.hy_v4.rope import rope_index  # noqa: E402
from omlx.patches.hy_v4.tool_parser import parse_tool_call  # noqa: E402
from omlx.patches.hy_v4.vision import VisionModel  # noqa: E402
from omlx.patches.mlx_lm_mtp import cache_rollback  # noqa: E402
from omlx.patches.qwen35_verify_qmm import apply_verify_qmm_patch  # noqa: E402

VOCAB = 320
SEQ = 40

_COMMON = dict(
    vocab_size=VOCAB,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    n_routed_experts=16,
    num_experts_per_tok=4,
    routed_scaling_factor=2.5,
    # A small top-k puts the sequences below on the sparse path. Many indexer
    # heads keep exact ReLU-zero ties, which order differently, unlikely.
    index_topk=16,
    index_n_heads=16,
    index_head_dim=64,
)


def _hy4_config():
    text = dict(
        _COMMON,
        model_type="hy_v4",
        indexer_types=["full", "full", "shared", "shared", "full", "shared"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "mrope_section": [4, 2, 2],
        },
    )
    return {"model_type": "hy_v4_vl", "text_config": text}


def _flash_config():
    primes = [101, 103, 107, 109, 113, 127, 131, 137]
    text = dict(
        _COMMON,
        model_type="hy_v4_flash",
        swa_num_attention_heads=8,
        routed_expert_hidden_size=64,
        layer_types=["deepseek_sparse_attention"] * 2
        + ["sliding_attention"] * 3
        + ["deepseek_sparse_attention"],
        rope_parameters={
            "deepseek_sparse_attention": {
                "rope_type": "default",
                "rope_theta": 1e7,
                "mrope_section": [4, 2, 2],
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 1e4,
                "mrope_section": [4, 2, 2],
            },
        },
        sliding_window=8,
        compressed_indexer_compress_ratio=4,
        engram_layer_ids=[2, 5],
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=32,
        engram_layer_head_vocab_sizes={
            "2": [primes[0:2], primes[2:4]],
            "5": [primes[4:6], primes[6:8]],
        },
        engram_pad_id=2,
        engram_lookup_table_size=300,
    )
    return {"model_type": "hy_v4_flash", "text_config": text}


def _model(config):
    mx.random.seed(0)
    model = Model(ModelConfig.from_dict(config))
    params = []
    for name, value in tree_flatten(model.parameters()):
        if name.endswith("lookup_table"):
            params.append((name, mx.arange(value.size, dtype=mx.int32) % 250))
        elif mx.issubdtype(value.dtype, mx.floating):
            scale = value.shape[-1] ** -0.5 if value.ndim > 1 else 0.3
            noise = mx.random.normal(value.shape) * scale
            base = 1.0 if ("norm" in name or "gamma" in name) else 0.0
            params.append((name, (base + noise).astype(mx.float32)))
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model


def _ids(seed, n=SEQ):
    return np.random.default_rng(seed).integers(10, VOCAB - 20, size=n)


@pytest.mark.parametrize("affine", [False, True])
def test_engram_ssd_matches_resident(tmp_path, monkeypatch, affine):
    from mlx_vlm import utils

    from omlx.patches.hy_v4 import apply_patch
    from omlx.patches.hy_v4.engram_ssd import load, plan_engram_ssd

    apply_patch()
    config = _flash_config()
    config["text_config"]["engram_n_embed_per_ngram"] = 64
    weights = dict(tree_flatten(_model(config).parameters()))
    recipe = {"bits": 4, "group_size": 32, "mode": "affine"}
    index = {}
    for i, (key, value) in enumerate(list(weights.items())):
        if not key.endswith(".multi_head_embedding.embedding.weight"):
            continue
        tensors = {key: value.astype(mx.bfloat16)}
        if affine:
            prefix = key.removesuffix(".weight")
            packed, scales, biases = mx.quantize(tensors[key], **recipe)
            tensors = {key: packed}
            tensors.update({f"{prefix}.scales": scales, f"{prefix}.biases": biases})
            config.setdefault("quantization", dict(recipe))[prefix] = dict(recipe)
        shard = f"table-{i}.safetensors"
        mx.save_safetensors(str(tmp_path / shard), tensors, metadata={"format": "mlx"})
        index.update(dict.fromkeys(tensors, shard))
        del weights[key]
    mx.save_safetensors(
        str(tmp_path / "backbone.safetensors"), weights, metadata={"format": "mlx"}
    )
    index.update(dict.fromkeys(weights, "backbone.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    monkeypatch.setattr(utils, "load_processor", lambda *a, **k: object())
    monkeypatch.setattr(utils, "load_image_processor", lambda *a, **k: None)

    resident, _ = utils.load(tmp_path)
    expected = _run(resident, _ids(3), [24, 9, 7])
    model, _ = load(tmp_path)
    try:
        params = dict(tree_flatten(model.parameters()))
        skipped = plan_engram_ssd(tmp_path).skipped.values()
        assert not any(k in params for keys in skipped for k in keys)
        np.testing.assert_array_equal(expected, _run(model, _ids(3), [24, 9, 7]))
    finally:
        model._omlx_engram_ssd.close()


def _run(model, ids, chunks):
    cache = model.make_cache()
    outs, start = [], 0
    for size in chunks:
        chunk = mx.array(ids[None, start : start + size].astype(np.int32))
        outs.append(model.language_model(chunk, cache=cache).logits)
        start += size
    return np.array(mx.concatenate(outs, axis=1))[0]


@pytest.fixture(params=["hy4", "flash"], scope="module")
def model(request):
    return _model(_hy4_config() if request.param == "hy4" else _flash_config())


def test_flash_engram_gate_matches_bf16_pytorch():
    config = _flash_config()
    config["text_config"].update(hidden_size=8, hc_mult=1)
    engram = Engram(ModelConfig.from_dict(config).text_config, layer_id=2)
    engram.set_dtype(mx.bfloat16)
    # Unit keys and values make each output channel equal to the gate.
    for proj in (engram.key_projs[0], engram.value_proj):
        proj.weight = mx.zeros_like(proj.weight)
        proj.bias = mx.ones_like(proj.bias)
    streams = mx.array(
        [[-2, -2, -1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 1, 1, 2]], dtype=mx.bfloat16
    )[None, :, None, :]
    ngrams = mx.zeros((1, 2, engram.max_order), dtype=mx.int32)
    out = engram(streams, ngrams, mx.ones((1, 2), dtype=mx.bool_))

    # Recorded from the Transformers gate with BF16 RMS-normalized queries.
    assert out.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        out[0, :, 0, 0].astype(mx.float32), [0.21484375, 0.74609375]
    )


def test_incremental_matches_full_forward(model):
    ids = _ids(1)
    full = _run(model, ids, [SEQ])
    # Prefill, ragged chunks (open compressed-indexer groups) and decode steps.
    for chunks in ([24] + [1] * 16, [7, 7, 7, 7, 7, 5], [1] * 20 + [20]):
        np.testing.assert_allclose(_run(model, ids, chunks), full, atol=2e-4)


def test_flash_engram_submits_ids_before_prefix_only_for_ssd(tmp_path, monkeypatch):
    from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding

    config = _flash_config()
    config["text_config"].update(hidden_size=8, hc_mult=1)
    engram = Engram(ModelConfig.from_dict(config).text_config, layer_id=2)
    engram.set_dtype(mx.bfloat16)
    weight = engram.multi_head_embedding.embedding.weight
    streams = mx.ones((1, 2, 1, 8), dtype=mx.bfloat16)
    ngrams = mx.zeros((1, 2, engram.max_order), dtype=mx.int32)
    keep = mx.ones((1, 2), dtype=mx.bool_)
    table_path = tmp_path / "engram.safetensors"
    mx.save_safetensors(str(table_path), {"weight": weight})
    submitted = []
    by_lookup = []
    async_eval = mx.async_eval
    lookup = Engram.lookup
    verifying = True

    def record(array):
        submitted.append(array)
        async_eval(array)

    def record_lookup(self, rows):
        by_lookup.append(len(submitted))
        return lookup(self, rows)

    monkeypatch.setattr(mx, "async_eval", record)
    monkeypatch.setattr(Engram, "lookup", record_lookup)
    monkeypatch.setattr(
        verify_dispatch, "short_gpu_verify", lambda rows: verifying and rows == 2
    )

    engram(streams, ngrams, keep)
    assert submitted == []
    reader = DiskEngramEmbedding(table_path, "weight", None)
    engram.multi_head_embedding.embedding = reader
    try:
        verifying = False
        expected = engram(streams, ngrams, keep)
        assert submitted == []
        verifying = True
        actual = engram(streams, ngrams, keep)
        assert len(submitted) == 2 and submitted[1] is streams
        assert by_lookup == [0, 0, 2]
        np.testing.assert_array_equal(submitted[0], engram.hash_rows(ngrams))
        np.testing.assert_array_equal(
            actual.astype(mx.float32), expected.astype(mx.float32)
        )
    finally:
        reader.close()


def _batch_cache(model, left):
    caches = []
    for c in model.make_cache():
        if isinstance(c, CacheList):
            caches.append(CacheList(*[BatchKVCache(left) for _ in c.caches]))
        else:
            caches.append(BatchRotatingKVCache(c.max_size, left))
    return caches


@pytest.mark.parametrize("short", [20, 8])
def test_left_padded_batch_rows_match_single_runs(model, short):
    # A short row (8 tokens) has fewer visible keys or closed groups than the
    # indexer keeps, so its selection also covers masked slots.
    a, b = _ids(2), _ids(3, n=short + 14)
    pa, pb = 26, short
    pad = pa - pb
    lm = model.language_model
    cache = _batch_cache(model, [0, pad])
    prompt = np.stack([a[:pa], np.concatenate([np.zeros(pad, int), b[:pb]])])
    out = lm(mx.array(prompt.astype(np.int32)), cache=cache).logits
    rows = [[np.array(out[0, -1])], [np.array(out[1, -1])]]
    for step in range(14):
        nxt = np.array([[a[pa + step]], [b[pb + step]]], dtype=np.int32)
        out = lm(mx.array(nxt), cache=cache).logits
        rows[0].append(np.array(out[0, -1]))
        rows[1].append(np.array(out[1, -1]))
    for seq, start, got in ((a, pa, rows[0]), (b, pb, rows[1])):
        single = _run(model, seq, [len(seq)])
        expected = single[start - 1 : start - 1 + len(got)]
        np.testing.assert_allclose(np.stack(got), expected, atol=2e-4)


def _with_dfly(config):
    return dict(
        config,
        dfly_num_hidden_layers=2,
        dfly_block_size=4,
        dfly_mask_token_id=VOCAB - 1,
        dfly_target_kv_layers=[1, 5],
        dfly_target_index_layer=[1, 5],
    )


def _draft_block(lm, hidden, row, anchor):
    block, _ = lm.dspark_forward(
        hidden[:, : row + 1], mx.array([int(anchor)]), draft_length=3
    )
    return np.array(block)


@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_dfly_cycles_roll_back_to_the_accepted_prefix(kind):
    # Verify blocks of 4 on an 8-slot sliding window rotate the ring, close
    # compressed-indexer groups and extend the Engram history; a rejected
    # tail must leave every cache as if only the accepted prefix had run.
    cache_rollback.apply()
    model = _model(_with_dfly(_hy4_config() if kind == "hy4" else _flash_config()))
    lm = model.language_model
    lm.configure_mtp(True, 3)
    seq = _ids(4, n=SEQ + 8)
    reference = _run(model, seq, [len(seq)])

    cache = model.make_cache()
    committed = 20
    out = lm(
        mx.array(seq[None, :committed].astype(np.int32)),
        cache=cache,
        return_hidden=True,
    )
    row = committed - 1
    for accepted in (3, 0, 1, 2, 3):
        anchor = seq[committed]
        drafted = _draft_block(lm, out.hidden_states[0], row, anchor)
        fresh = model.make_cache()
        plain = lm(
            mx.array(seq[None, :committed].astype(np.int32)),
            cache=fresh,
            return_hidden=True,
        )
        expected = _draft_block(lm, plain.hidden_states[0], committed - 1, anchor)
        np.testing.assert_allclose(drafted, expected, atol=2e-4)

        block = seq[committed : committed + 4].copy()
        block[accepted + 1 :] += 1
        cache_rollback.set_undo_armed(True)
        try:
            out = lm(
                mx.array(block[None].astype(np.int32)), cache=cache, return_hidden=True
            )
        finally:
            cache_rollback.set_undo_armed(False)
        np.testing.assert_allclose(
            np.array(out.logits[0, : accepted + 1]),
            reference[committed : committed + accepted + 1],
            atol=2e-4,
        )
        assert lm.mtp_partial_rollback(cache, accepted, 3)
        committed += accepted + 1
        row = accepted


def _quantized_model(kind, bits, dfly=False, calm=False, dtype=mx.bfloat16):
    config = _hy4_config() if kind == "hy4" else _flash_config()
    if calm:
        # No discrete routing or slot selection, and wide logit gaps, so batch
        # rounding (in fp32) does not flip a token; logic errors still do.
        config["text_config"].update(
            n_routed_experts=4, num_experts_per_tok=4, index_topk=64
        )
    model = _model(_with_dfly(config) if dfly else config)
    if calm:
        lm = model.language_model
        lm.lm_head.weight = lm.lm_head.weight * 8
    model.set_dtype(dtype)
    if bits:
        mxfp8 = bits == "mxfp8"
        nn.quantize(
            model.language_model,
            group_size=32 if mxfp8 else 64,
            bits=8 if mxfp8 else bits,
            mode="mxfp8" if mxfp8 else "affine",
            # oQ keeps the compressed indexer's fp32 key pooling in float.
            class_predicate=lambda path, m: hasattr(m, "to_quantized")
            and not isinstance(m, nn.Embedding)
            and "compressed_indexer.key_" not in path
            and m.weight.shape[-1] % 64 == 0,
        )
    mx.eval(model.parameters())
    return model


def _decode_logits(lm, cache, tokens):
    return [
        lm(mx.array([[int(t)]], dtype=mx.int32), cache=cache).logits for t in tokens
    ]


@pytest.mark.parametrize(
    ("kind", "bits", "batched"),
    [
        ("hy4", None, False),
        ("hy4", 4, True),
        ("hy4", 3, False),
        ("hy4", "mxfp8", True),
        ("flash", None, True),
        ("flash", 4, False),
        ("flash", 3, True),
        ("flash", "mxfp8", False),
    ],
)
def test_exact_verify_rows_match_decode(monkeypatch, kind, bits, batched):
    # Past the index top-k and the 8-slot sliding window, so sparse selection
    # and a rotated ring are both in play. mlx-lm's generator hands even one
    # request one-row batch caches.
    apply_verify_qmm_patch()
    cache_rollback.apply()
    monkeypatch.setattr(language, "EXACT_VERIFY", True)
    model = _quantized_model(kind, bits)
    lm = model.language_model
    ids = _ids(7, n=SEQ)
    prompt, block = 28, 6

    def primed():
        cache = _batch_cache(model, [0]) if batched else model.make_cache()
        lm(mx.array(ids[None, :prompt].astype(np.int32)), cache=cache)
        return cache

    decoded = _decode_logits(lm, primed(), ids[prompt : prompt + block])
    cache = primed()
    cache_rollback.set_undo_armed(True)
    set_dspark_verify_armed(True)
    try:
        rows = lm(
            mx.array(ids[None, prompt : prompt + block].astype(np.int32)), cache=cache
        ).logits
    finally:
        set_dspark_verify_armed(False)
        cache_rollback.set_undo_armed(False)
    assert mx.array_equal(rows, mx.concatenate(decoded, axis=1)).item()

    # Keep the bonus row and two drafts, then both caches decode alike.
    assert lm.mtp_partial_rollback(cache, accepted=2, num_drafts=block - 1)
    reference = primed()
    _decode_logits(lm, reference, ids[prompt : prompt + 3])
    tail = ids[prompt + 3 : prompt + 6] + 1
    for got, want in zip(
        _decode_logits(lm, cache, tail), _decode_logits(lm, reference, tail)
    ):
        assert mx.array_equal(got, want).item()


def test_exact_block_kernels_match_one_row_calls():
    # Release-sized layers: the weight-once MXFP8 kernel and the bf16-storage
    # fp32 head, whose tiny-model shapes take the per-row fallback instead.
    linear = nn.Linear(512, 1024, bias=False)
    linear.weight = mx.random.normal((1024, 512)) * 0.05
    mxfp8 = linear.to_quantized(group_size=32, bits=8, mode="mxfp8")
    head = nn.Linear(256, 4096, bias=False)
    head.weight = (mx.random.normal((4096, 256)) * 0.05).astype(mx.bfloat16)
    x = mx.random.normal((1, 7, 512)).astype(mx.bfloat16)
    h = mx.random.normal((1, 7, 256)).astype(mx.bfloat16)
    one_x = [mxfp8(x[:, i : i + 1]) for i in range(7)]
    one_h = [exact.fp32_head(head, h[:, i : i + 1]) for i in range(7)]
    one_d = [head(h[:, i : i + 1]) for i in range(7)]
    exact.set_armed(True)
    try:
        with exact.scope(mx.zeros((1, 7), dtype=mx.int32), [None]):
            rows_x = exact.linear(mxfp8, x)
            rows_h = exact.fp32_head(head, h)
            rows_d = exact.linear(head, h)
    finally:
        exact.set_armed(False)
    assert mx.array_equal(rows_x, mx.concatenate(one_x, axis=1)).item()
    assert mx.array_equal(rows_d, mx.concatenate(one_d, axis=1)).item()
    assert rows_h.dtype == mx.float32
    assert mx.array_equal(rows_h, mx.concatenate(one_h, axis=1)).item()


@pytest.mark.skipif(not indexing.available(), reason="needs sparse_index kernels")
def test_exact_block_index_selection_matches_one_row_decode():
    # Release geometry past the top-k, where a verify block scores its rows
    # in one decode-kernel call; each row must select what its decode step did.
    config = _hy4_config()
    config["text_config"].update(
        index_topk=2048, index_n_heads=32, index_head_dim=128, hidden_size=256
    )
    args = ModelConfig.from_dict(config).text_config
    indexer = Indexer(args)
    mx.random.seed(5)
    indexer.update(
        tree_unflatten(
            [
                (k, mx.random.normal(v.shape) * v.shape[-1] ** -0.5)
                for k, v in tree_flatten(indexer.parameters())
            ]
        )
    )
    indexer.set_dtype(mx.bfloat16)
    prompt, rows = 2100, 6
    x = mx.random.normal((1, prompt + rows, args.hidden_size)).astype(mx.bfloat16)
    qr = mx.random.normal((1, prompt + rows, args.q_lora_rank)).astype(mx.bfloat16)

    def primed():
        cache = KVCache()
        indexer(x[:, :prompt], qr[:, :prompt], None, cache)
        return cache

    cache = primed()
    decoded = [
        indexer(
            x[:, prompt + i : prompt + i + 1],
            qr[:, prompt + i : prompt + i + 1],
            None,
            cache,
            prompt + i,
        )
        for i in range(rows)
    ]
    cache = primed()
    mask = mx.arange(prompt, prompt + rows)[:, None] >= mx.arange(prompt + rows)
    exact.set_armed(True)
    try:
        with exact.scope(mx.zeros((1, rows), dtype=mx.int32), [cache]):
            block = indexer(x[:, prompt:], qr[:, prompt:], mask, cache, prompt)
    finally:
        exact.set_armed(False)
    assert len(block) == rows
    for got, want in zip(block, decoded):
        assert mx.array_equal(got, want).item()


def _generate(model, prompts, count, use_mtp):
    """Greedy tokens per prompt from mlx-lm's BatchGenerator with the MTP loop."""
    from types import SimpleNamespace

    from mlx_lm.generate import BatchGenerator

    from omlx.models.vlm import VLMModelAdapter

    lm = model.language_model
    lm.configure_mtp(use_mtp, 3)
    lm._omlx_mtp_depth_fixed = True
    adapter = VLMModelAdapter(
        SimpleNamespace(
            config=SimpleNamespace(model_type=lm.model_type), language_model=lm
        )
    )
    generator = BatchGenerator(
        adapter,
        max_tokens=count,
        prefill_step_size=512,
        sampler=lambda lp: mx.argmax(lp, -1),
    )
    tokens = {}
    try:
        uids = generator.insert([list(map(int, p)) for p in prompts])
        for _ in range(8 * count):
            _, responses = generator.next()
            for r in responses:
                tokens.setdefault(r.uid, []).append(r.token)
            if all(len(tokens.get(u, [])) >= count for u in uids):
                break
    finally:
        generator.close()
    return [tokens[u][:count] for u in uids]


@pytest.mark.parametrize(("kind", "bits"), [("hy4", 4), ("flash", 3)])
def test_mtp_generation_matches_greedy(monkeypatch, kind, bits):
    """Through the generator (draft, exact verify, accept, rollback), alone and
    with a second request, MTP emits each request's own greedy tokens."""
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    assert apply_mlx_lm_mtp_patch()
    monkeypatch.setattr(language, "EXACT_VERIFY", True)
    model = _quantized_model(kind, bits, dfly=True)
    lm = model.language_model
    drafts = []
    original = type(lm).dspark_forward

    def tracked(self, *args, **kwargs):
        drafts.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(lm), "dspark_forward", tracked)
    prompt = _ids(11, n=26)
    expected = _generate(model, [prompt], 12, use_mtp=False)
    assert _generate(model, [prompt], 12, use_mtp=True) == expected
    assert drafts, "the MTP loop must draft"


@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_mtp_drafts_for_concurrent_requests(monkeypatch, kind):
    """Two requests draft at once, each from its own caches.

    In fp32 the batched and single-request arithmetic agree far below the
    logit gaps, so each request must emit what it emits alone.
    """
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    assert apply_mlx_lm_mtp_patch()
    model = _quantized_model(kind, None, dfly=True, calm=True, dtype=mx.float32)
    lm = model.language_model
    drafts = []
    original = type(lm).dspark_forward

    def tracked(self, *args, **kwargs):
        drafts.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(lm), "dspark_forward", tracked)
    prompts = [_ids(11, n=26), _ids(12, n=19)]
    expected = [_generate(model, [p], 10, use_mtp=False)[0] for p in prompts]
    drafts.clear()
    assert _generate(model, prompts, 10, use_mtp=True) == expected
    assert len(drafts) > 2


def _checkpoint(path, cfg, model):
    """Save a tiny model as an oMLX-layout BF16 checkpoint."""
    weights = {
        k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
        for k, v in tree_flatten(model.parameters())
    }
    path.mkdir()
    (path / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(path / "model.safetensors"), weights)


def test_calibration_walk_matches_native_forward(model):
    # Past the index top-k, so shared indexer layers read the carried top-k.
    lm, core = model.language_model, model.language_model.model
    tokens = mx.array(_ids(5, n=24)[None].astype(np.int32))
    inputs = core.embed_tokens(tokens)
    inputs, masks, state = oq._prepare_layer_inputs(model, model.layers, tokens, inputs)
    for i, layer in enumerate(model.layers):
        inputs, aux = oq._forward_layer_result(
            layer, inputs, masks[i], state, layer_idx=i
        )
        oq._commit_layer_forward_aux(state, i, aux)
    engram = lm._engram_inputs(tokens, None) if lm.config.engram_layer_ids else None
    native = core(tokens, engram_inputs=engram)
    assert mx.array_equal(core.norm(core.hc_head(inputs)), native).item()


@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_streamed_calibration_matches_resident(tmp_path, monkeypatch, kind):
    from mlx_vlm.utils import load_model

    from omlx.patches.hy_v4 import apply_patch

    apply_patch()
    cfg = _hy4_config() if kind == "hy4" else _flash_config()
    schedule = "indexer_types" if kind == "hy4" else "layer_types"
    cfg["text_config"].update(
        {"num_hidden_layers": 4, schedule: cfg["text_config"][schedule][:4]}
    )
    source = tmp_path / "source"
    _checkpoint(source, cfg, _model(cfg))
    resident_model = load_model(source, lazy=False, strict=True)
    tokens = mx.array(np.stack([_ids(10 + i, n=12) for i in range(4)]).astype(np.int32))
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    resident, _ = oq._collect_imatrix_from_model(
        resident_model,
        object(),
        cfg,
        calib_dataset=oq._OQE_CALIB_DATASET,
        num_samples=2,
        seq_length=12,
    )
    sensitivity = oq._measure_sensitivity_from_model(
        resident_model, object(), cfg, 4, num_samples=2, seq_length=12
    )
    streamed = oq._load_or_collect_imatrix(
        str(source),
        cfg,
        cache_path=str(tmp_path / "imatrix.npz"),
        reuse_cache=False,
        num_samples=2,
        seq_length=12,
        strict=True,
        trust_remote_code=False,
        stream_calibration=True,
        measure_sensitivity=True,
        sensitivity_oq_level=4,
        sensitivity_num_samples=2,
        sensitivity_seq_length=12,
        require_mtp_entries=False,
    )
    assert "language_model.lm_head" in streamed.entries
    assert streamed.entries.keys() == resident.keys()
    for name, entry in resident.items():
        np.testing.assert_array_equal(streamed.entries[name].counts, entry.counts)
        np.testing.assert_allclose(
            streamed.entries[name].in_sum2, entry.in_sum2, rtol=1e-5
        )
    measured = streamed.metadata["collection"]["sensitivity_map"]
    assert measured == pytest.approx(sensitivity, abs=1e-6)


@pytest.mark.skipif(not indexing.available(), reason="needs sparse_index kernels")
@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_native_index_selection_matches_mlx(monkeypatch, kind):
    # Release geometry: 128-dim heads, top-k 2048 (512 flash groups).
    config = _hy4_config() if kind == "hy4" else _flash_config()
    config["text_config"].update(
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        compressed_indexer_num_attention_heads=16,
        compressed_indexer_head_dim=128,
    )
    args = ModelConfig.from_dict(config).text_config
    indexer = Indexer(args) if kind == "hy4" else CompressedIndexer(args)
    mx.random.seed(3)
    indexer.update(
        tree_unflatten(
            [
                (k, (mx.random.normal(v.shape) * v.shape[-1] ** -0.5).astype(v.dtype))
                for k, v in tree_flatten(indexer.parameters())
            ]
        )
    )
    total = 2600
    x = mx.random.normal((2, total, args.hidden_size)).astype(mx.bfloat16)
    qr = mx.random.normal((2, total, args.q_lora_rank)).astype(mx.bfloat16)
    mask = mx.arange(total)[:, None] >= mx.arange(total)
    native = np.array(indexer(x, qr, mask, None))
    monkeypatch.setattr(indexing, "available", lambda: False)
    reference = np.array(indexer(x, qr, mask, None))
    assert native.shape == reference.shape
    overlap = []
    for a, b in zip(
        native.reshape(-1, native.shape[-1]), reference.reshape(-1, native.shape[-1])
    ):
        a, b = set(a[a < total]), set(b[b < total])
        assert len(a) == len(b)
        overlap.append(len(a & b) / max(len(b), 1))
    # bf16 scores reorder only near-ties at the cutoff.
    assert np.mean(overlap) > 0.995 and np.min(overlap) > 0.95


def _sparse_reference(q, kv, idx, scale, q_pe, k_pe, sinks):
    """fp32 sparse MLA with sinks: softmax over each row's usable slots."""
    total, length = kv.shape[2], q.shape[2]
    rows = []
    for i in range(length):
        sel = idx[0, 0, i]
        valid = (sel >= 0) & (sel < total) & (sel <= total - length + i)
        safe = mx.where(valid, sel, 0)
        keys = kv[0, 0][safe].astype(mx.float32)
        s = q[0, :, i].astype(mx.float32) @ keys.T
        s = s + q_pe[0, :, i].astype(mx.float32) @ k_pe[0, 0][safe].astype(mx.float32).T
        s = mx.where(valid[None], s * scale, -mx.inf)
        p = mx.softmax(mx.concatenate([s, sinks[:, None]], axis=1), axis=-1)
        rows.append(p[:, :-1] @ keys)
    return mx.stack(rows, axis=1)[None]


@pytest.mark.skipif(not nax.is_nax_available(), reason="needs M5 tensor units")
def test_nax_paths_match_their_references():
    from omlx.patches.glm_moe_dsa.sparse_mla_nax import sparse_mla_attention_nax

    def rel(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        return (mx.abs(a - b).max() / mx.abs(b).max()).item()

    mx.random.seed(0)
    heads, length, total = 32, 16, 3000
    q = (mx.random.normal((1, heads, length, 512)) * 0.3).astype(mx.bfloat16)
    q_pe = (mx.random.normal((1, heads, length, 64)) * 0.3).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, total, 512)) * 0.3).astype(mx.bfloat16)
    k_pe = (mx.random.normal((1, 1, total, 64)) * 0.3).astype(mx.bfloat16)
    idx = mx.stack(
        [
            mx.random.permutation(total - length + i + 1)[:2048].astype(mx.int32)
            for i in range(length)
        ]
    )[None, None]
    sinks = mx.random.normal((heads,))
    scale = 576**-0.5
    out = sparse_mla_attention_nax(q, kv, idx, scale, q_pe=q_pe, k_pe=k_pe, sinks=sinks)
    assert rel(out, _sparse_reference(q, kv, idx, scale, q_pe, k_pe, sinks)) < 2e-2

    # Index scores: tensor units against the SIMD kernel (both bf16 out).
    k = (mx.random.normal((1, 1, total, 128)) * 0.3).astype(mx.bfloat16)
    w = mx.random.normal((1, 64, 32)).astype(mx.bfloat16)
    qi = (mx.random.normal((1, 32, 64, 128)) * 0.3).astype(mx.bfloat16)
    for causal in (True, False):
        got = indexing._nax_scores(qi, k, w, causal)
        want = indexing.index_fast.dsa_indexer_scores(qi, k, w, causal=causal)
        finite = want > -mx.inf
        assert mx.array_equal(got > -mx.inf, finite).item()
        assert rel(mx.where(finite, got, 0), mx.where(finite, want, 0)) < 1e-2

    # 256-wide heads with sinks: the fused SDPA against MLX's reference path.
    qs = (mx.random.normal((1, 8, 64, 256)) * 0.3).astype(mx.bfloat16)
    ks = (mx.random.normal((1, 8, 600, 256)) * 0.3).astype(mx.bfloat16)
    vs = mx.random.normal((1, 8, 600, 256)).astype(mx.bfloat16)
    mask = mx.arange(536, 600)[:, None] >= mx.arange(600)
    sink = mx.random.normal((8,)).astype(mx.bfloat16)
    fused = layers._nax_attention(qs, ks, vs, 0.0625, mask, sink)
    plain = mx.fast.scaled_dot_product_attention(
        qs, ks, vs, scale=0.0625, mask=mask, sinks=sink
    )
    assert rel(fused, plain) < 1e-2


def test_sanitize_dequantizes_fp8_and_stacks_experts():
    config = _hy4_config()
    model = Model(ModelConfig.from_dict(config))
    rng = np.random.default_rng(0)
    w = rng.normal(size=(2, 192, 256)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=(2, 2, 2)).astype(np.float32)
    codes = mx.to_fp8(mx.array(w))
    expected = np.array(mx.from_fp8(codes, dtype=mx.float32)) * np.repeat(
        np.repeat(scale, 128, axis=1)[:, :192], 128, axis=2
    )
    got = dequantize_block_fp8(codes, mx.array(scale))
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), expected, rtol=1e-2)

    experts, mid, hidden = 16, 64, 128
    gate_up = mx.random.normal((experts, 2 * mid, hidden))
    down = mx.random.normal((experts, hidden, mid))
    per_expert = {
        f"model.layers.2.mlp.experts.{e}.{name}.weight": tensor[e]
        for e in range(experts)
        for name, tensor in (
            ("gate_proj", gate_up[:, :mid]),
            ("up_proj", gate_up[:, mid:]),
            ("down_proj", down),
        )
    }
    fused = {
        "model.layers.1.mlp.experts.gate_up_proj": gate_up,
        "model.layers.1.mlp.experts.down_proj": down,
    }
    out = model.sanitize({**fused, **per_expert})
    for layer in (1, 2):
        prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        assert mx.array_equal(out[f"{prefix}.gate_proj.weight"], gate_up[:, :mid])
        assert mx.array_equal(out[f"{prefix}.up_proj.weight"], gate_up[:, mid:])
        assert mx.array_equal(out[f"{prefix}.down_proj.weight"], down)


def test_tool_calls_and_thinking_switches():
    text = (
        "<tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>Seoul</arg_value:opensource>"
        "<arg_key:opensource>days</arg_key:opensource>"
        "<arg_value:opensource>3</arg_value:opensource></tool_call:opensource>"
        "<tool_call:opensource>now</tool_call:opensource>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    }
                },
            },
        }
    ]
    assert parse_tool_call(text, tools) == [
        {"name": "get_weather", "arguments": {"city": "Seoul", "days": 3}},
        {"name": "now", "arguments": {}},
    ]
    template = (
        "{%- set HYTK = ':opensource' %}{%- set k = '<arg_key{}>'.format(HYTK) %}"
    )
    assert correct_tool_parser_type(None, template) == "hy_v4"
    assert correct_tool_parser_type("glm47", template + "<tool_sep>") == "glm47"

    for kwargs, expected in (
        ({"reasoning_effort": "none"}, "no_think"),
        ({"enable_thinking": False}, "no_think"),
        ({"reasoning_effort": "high", "enable_thinking": False}, "high"),
        ({"enable_thinking": True}, None),
    ):
        kwargs = dict(kwargs)
        _apply_template_thinking_mode("hy_v4_flash", kwargs)
        assert kwargs.get("reasoning_effort") == expected
        assert "enable_thinking" not in kwargs


_TINY_VISION = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    out_hidden_size=16,
    patch_size=4,
    spatial_merge_size=2,
    max_image_size=64,
)
_IMAGE_ID = 300
# One 8 x 12 patch image: 4 merged rows of 6 columns plus a newline slot each.
_GRID = [[1, 8, 12]]
_IMAGE_SLOTS = 4 * (6 + 1)


def _vision_value(name, shape):
    """Portable per-tensor weights, also used to record the reference values."""
    x = np.random.default_rng(zlib.crc32(name.encode())).standard_normal(shape)
    x = x.astype(np.float32)
    if name.endswith("norm.weight") or "layernorm.weight" in name:
        return 1 + 0.1 * x
    if name.endswith(".bias"):
        return 0.1 * x
    if name.endswith("image_newline"):
        return x
    if name.endswith("position_embedding.weight"):
        return 0.5 * x
    return x * shape[-1] ** -0.5


@pytest.mark.parametrize(
    ("grid", "seed", "row_sums", "row_norms"),
    [
        # The 16 x 16 position table is resampled down to 4 x 6 patches.
        (
            (1, 4, 6),
            1,
            [-3.4509, -6.5598, 8.1656, -4.0989, 0.9282, 0.9485, 4.2884, -4.0989],
            [5.5319, 5.4043, 4.6774, 3.5577, 3.2572, 5.5758, 7.4323, 3.5577],
        ),
        # And up to 24 columns: a wide image is not limited to the table size.
        (
            (1, 2, 24),
            2,
            [3.9254, -0.5263, 4.648, -0.2905, 0.2562, 1.8193, -2.6161]
            + [-3.2218, -5.719, 0.8748, 0.9433, 1.9788, -4.0989],
            [5.136, 5.0762, 5.8957, 4.9402, 5.3277, 5.8483, 4.3241]
            + [4.7421, 5.6187, 3.4543, 3.7057, 3.1879, 3.5577],
        ),
    ],
)
def test_vision_tower_matches_the_transformers_reference(
    grid, seed, row_sums, row_norms
):
    tower = VisionModel(VisionConfig.from_dict(_TINY_VISION))
    params = [
        (name, mx.array(_vision_value(name, value.shape)))
        for name, value in tree_flatten(tower.parameters())
    ]
    tower.update(tree_unflatten(params))
    shape = (grid[1] * grid[2], 3 * 4 * 4)
    pixels = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    out = np.array(tower(mx.array(pixels), [list(grid)]))
    # Merged rows of columns / 2 slots, each closed by the newline vector.
    assert out.shape == (len(row_sums), 16)
    for newline in out[grid[2] // 2 :: grid[2] // 2 + 1]:
        np.testing.assert_array_equal(newline, out[-1])
    # Recorded from transformers' HYV4VLVisionModel with the same weights.
    np.testing.assert_allclose(out.sum(-1), row_sums, atol=5e-4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), row_norms, atol=5e-4)


def _vision_config(config):
    config = json.loads(json.dumps(config))
    config.update(image_token_id=_IMAGE_ID, video_token_id=_IMAGE_ID + 1)
    config["vision_config"] = dict(_TINY_VISION, out_hidden_size=128)
    return config


@pytest.mark.parametrize("effort", ["high", "no_think"])
def test_loaded_processor_detects_prompt_opened_thinking(tmp_path, effort):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "prompt": 1}, "[UNK]")),
        unk_token="[UNK]",
        chat_template=(
            "prompt<think:opensource>"
            "{% if reasoning_effort == 'no_think' %}</think:opensource>{% endif %}"
        ),
    )
    tokenizer.add_tokens(["<think:opensource>", "</think:opensource>"])
    tokenizer.save_pretrained(tmp_path)
    processor = HyV4Processor.from_pretrained(tmp_path)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
    )
    detected, _ = prompt_opens_thinking(processor.tokenizer, prompt)
    assert detected == (effort == "high")


def test_sanitize_maps_checkpoint_vision_weights():
    model = Model(ModelConfig.from_dict(_vision_config(_hy4_config())))
    own = dict(tree_flatten(model.vision_tower.parameters()))
    raw = {}
    for name, value in own.items():
        # The checkpoint stores torch conv layouts and numbered projector layers.
        key = "vit." + name.replace("proj_conv", "proj.0").replace("proj_out", "proj.2")
        if name.endswith("patch_embedding.weight"):
            value = value.reshape(value.shape[0], 3, 4, 4)
        elif name.endswith("proj_conv.weight"):
            value = value.transpose(0, 3, 1, 2)
        elif name.endswith("proj_out.weight"):
            value = value[:, :, None, None]
        raw[key] = value
    out = model.sanitize(dict(raw))
    assert out.keys() == {"vision_tower." + name for name in own}
    for name, value in own.items():
        assert mx.array_equal(out["vision_tower." + name], value).item()

    text_only = Model(ModelConfig.from_dict(_hy4_config()))
    assert text_only.vision_tower is None
    assert text_only.sanitize(dict(raw)) == {}


def _image(width, height, mode="RGB"):
    channels = 4 if mode == "RGBA" else 3
    rng = np.random.default_rng(width + height)
    return Image.fromarray(
        rng.integers(0, 256, (height, width, channels), dtype=np.uint8), mode
    )


@pytest.mark.parametrize(
    ("size", "mode", "grid", "tokens", "total", "magnitude"),
    [
        ((31, 17), "RGB", [1, 4, 6], 8, -172.5095, 7472.8944),
        ((200, 150), "RGB", [1, 10, 12], 35, -243.1197, 35381.4603),
        ((1000, 700), "RGB", [1, 44, 62], 704, 1313.2888, 820166.5362),
        ((4000, 40), "RGB", [1, 2, 250], 126, -640.0552, 151049.4651),
        # Transparent pixels are composited onto white.
        ((40, 40), "RGBA", [1, 4, 4], 6, 6053.4905, 6701.1297),
    ],
)
def test_image_processor_matches_the_transformers_reference(
    size, mode, grid, tokens, total, magnitude
):
    processor = HyV4ImageProcessor()
    out = processor(_image(*size, mode))
    assert out["image_grid_thw"].tolist() == [grid]
    assert out["pixel_values"].shape == (grid[1] * grid[2], 3 * 16 * 16)
    # Reference sums over the float32 pixel values (rescale, mean/std 0.5).
    pixels = out["pixel_values"].astype(np.float64)
    assert pixels.sum() == pytest.approx(total, abs=1e-3)
    assert np.abs(pixels).sum() == pytest.approx(magnitude, abs=1e-3)
    height, width = size[1], size[0]
    assert processor.count_image_tokens(height, width) == tokens
    assert tokens <= processor.max_image_tokens
    # The request-size guard counts the row newline slots too.
    guard = _count_image_tokens_real(
        [], HyV4Processor(processor, _Tokenizer()), images=[_image(*size, mode)]
    )
    assert guard == tokens


class _Tokenizer:
    image_token = "<img>"
    model_input_names = ["input_ids", "attention_mask"]

    def convert_tokens_to_ids(self, token):
        return 7 if token == self.image_token else 1

    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        words = [t.replace("<img>", " <img> ").split() for t in texts]
        ids = [[self.convert_tokens_to_ids(w) for w in row] for row in words]
        return {"input_ids": ids, "attention_mask": [[1] * len(r) for r in ids]}


def test_processor_expands_image_placeholders_to_the_token_count():
    processor = HyV4Processor(HyV4ImageProcessor(), _Tokenizer())
    images = [_image(31, 17), _image(40, 40, "RGBA")]
    out = processor(
        text="a <img> b <img> c", images=images, padding=True, return_tensors="np"
    )
    # Grids 4 x 6 and 4 x 4 take 2 x 4 and 2 x 3 slots, newline slots included.
    ids = out["input_ids"][0]
    assert (ids == 7).sum() == 8 + 6
    assert ids.tolist() == [1] + [7] * 8 + [1] + [7] * 6 + [1]
    assert out["mm_token_type_ids"][0].tolist() == [int(i == 7) for i in ids]
    assert out["image_grid_thw"].tolist() == [[1, 4, 6], [1, 4, 4]]
    with pytest.raises(ValueError, match="image placeholders"):
        processor(text="a <img> b", images=images)
    with pytest.raises(ValueError, match="no images"):
        processor(text="a <img> b")
    with pytest.raises(ValueError, match="image input only"):
        processor(text="a", videos=[object()])


def test_rope_index_follows_the_reference_rule():
    # Text counts all three planes together. A 2 x 3 merged grid fills 2 x 4
    # slots (a newline closes every row): rows and columns count from the
    # image's first position and text resumes after the largest one.
    ids = np.array([[11, 12] + [_IMAGE_ID] * 8 + [13, 14, 15]])
    planes, deltas = rope_index(ids, _IMAGE_ID, [[1, 4, 6]], 2)
    assert planes[:, 0].tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 6, 7, 8],
        [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 8],
    ]
    assert deltas.tolist() == [[-4]]
    # A left-padded row ignores its padding.
    padded = np.concatenate([np.zeros((1, 2), int), ids], axis=1)
    mask = np.array([[0, 0] + [1] * 13])
    shifted, _ = rope_index(padded, _IMAGE_ID, [[1, 4, 6]], 2, mask)
    assert shifted[:, 0, 2:].tolist() == planes[:, 0].tolist()
    assert shifted[:, 0, :2].tolist() == [[0, 0]] * 3
    with pytest.raises(ValueError, match="image_grid_thw"):
        rope_index(ids, _IMAGE_ID, [[1, 8, 12]], 2)


@pytest.fixture(params=["hy4", "flash"], scope="module")
def image_model(request):
    config = _hy4_config() if request.param == "hy4" else _flash_config()
    return _model(_vision_config(config))


def _image_prompt(model):
    ids = np.concatenate(
        [_ids(5, n=6), np.full(_IMAGE_SLOTS, _IMAGE_ID), _ids(6, n=10)]
    ).astype(np.int32)
    pixels = mx.array(
        np.random.default_rng(3).standard_normal((96, 48)).astype(np.float32)
    )
    merged = model.get_input_embeddings(
        mx.array(ids[None]), pixels, image_grid_thw=mx.array(_GRID)
    )
    return ids, merged


def test_image_features_replace_the_placeholder_slots(image_model):
    ids, merged = _image_prompt(image_model)
    slots = slice(6, 6 + _IMAGE_SLOTS)
    pixels = np.random.default_rng(3).standard_normal((96, 48)).astype(np.float32)
    features = image_model.encode_image(mx.array(pixels), mx.array(_GRID))
    assert features.shape == (_IMAGE_SLOTS, 128)
    embeds = np.array(merged.inputs_embeds[0])
    np.testing.assert_array_equal(embeds[slots], np.array(features))
    text = image_model.language_model.model.embed_tokens(mx.array(ids[None]))
    np.testing.assert_array_equal(embeds[:6], np.array(text[0, :6]))
    np.testing.assert_array_equal(embeds[slots.stop :], np.array(text[0, slots.stop :]))
    # The image starts at position 6; its largest position is 12 (the newline
    # column), so the 10 text tokens after it run 13 .. 22.
    planes = np.array(merged.position_ids[:, 0])
    assert planes[0, slots].tolist() == [6] * _IMAGE_SLOTS
    assert planes[1, slots].tolist() == [6 + i // 7 for i in range(_IMAGE_SLOTS)]
    assert planes[2, slots].tolist() == [6 + i % 7 for i in range(_IMAGE_SLOTS)]
    assert planes[:, slots.stop :].tolist() == [list(range(13, 23))] * 3
    assert np.array(merged.rope_deltas).tolist() == [[-21]]
    with pytest.raises(ValueError, match="image tokens"):
        image_model.get_input_embeddings(
            mx.array(np.delete(ids, 10)[None]),
            mx.array(pixels),
            image_grid_thw=mx.array(_GRID),
        )


def test_image_prompt_chunks_and_decode_follow_the_positions(image_model):
    lm = image_model.language_model
    ids, merged = _image_prompt(image_model)
    embeds, planes = merged.inputs_embeds, merged.position_ids
    full = np.array(
        lm(
            mx.array(ids[None]),
            cache=image_model.make_cache(),
            inputs_embeds=embeds,
            position_ids=planes,
        ).logits[0]
    )
    # Chunks split the image and its compressed-indexer groups anywhere.
    for chunks in ([11, 5, 28], [7, 9, 3, 25]):
        cache, outs, start = image_model.make_cache(), [], 0
        for size in chunks:
            part = slice(start, start + size)
            outs.append(
                lm(
                    mx.array(ids[None, part]),
                    cache=cache,
                    inputs_embeds=embeds[:, part],
                    position_ids=planes[..., part],
                ).logits
            )
            start += size
        np.testing.assert_allclose(
            np.array(mx.concatenate(outs, axis=1))[0], full, atol=2e-4
        )
    # Counting the image slots one by one instead gives other logits.
    sequential = mx.broadcast_to(mx.arange(len(ids))[None, None], (3, 1, len(ids)))
    other = lm(
        mx.array(ids[None]),
        cache=image_model.make_cache(),
        inputs_embeds=embeds,
        position_ids=sequential,
    ).logits
    assert np.abs(np.array(other[0]) - full).max() > 0.1

    # Decode continues at slot + delta through the server adapter.
    extra = _ids(9, n=12).astype(np.int32)
    extended = np.concatenate([ids, extra])
    grown = mx.concatenate([embeds, lm.model.embed_tokens(mx.array(extra[None]))], 1)
    extended_planes, _ = rope_index(extended[None], _IMAGE_ID, _GRID, 2)
    reference = np.array(
        lm(
            mx.array(extended[None]),
            cache=image_model.make_cache(),
            inputs_embeds=grown,
            position_ids=mx.array(extended_planes),
        ).logits[0]
    )
    cache = image_model.make_cache()
    lm(mx.array(ids[None]), cache=cache, inputs_embeds=embeds, position_ids=planes)
    adapter = VLMModelAdapter(image_model)
    adapter.set_step_rope_deltas(merged.rope_deltas.reshape(-1), [0])
    steps = [
        np.array(adapter(mx.array([[int(token)]]), cache=cache))[0, 0]
        for token in extra
    ]
    np.testing.assert_allclose(np.stack(steps), reference[len(ids) :], atol=2e-4)


def test_text_decode_through_the_adapter_is_unchanged(model):
    # Text requests reach the model with plain position ids and delta 0.
    ids = _ids(1).astype(np.int32)
    lm = model.language_model
    cache, plain = model.make_cache(), model.make_cache()
    lm(mx.array(ids[None, :24]), cache=cache)
    positions = mx.broadcast_to(mx.arange(24)[None, None], (3, 1, 24))
    lm(mx.array(ids[None, :24]), cache=plain, position_ids=positions)
    adapter = VLMModelAdapter(model)
    adapter.set_step_rope_deltas(mx.array([0]), [0])
    for token in ids[24:]:
        step = mx.array([[int(token)]])
        expected = lm(step, cache=cache).logits
        assert mx.array_equal(adapter(step, cache=plain), expected).item()


@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_dfly_anchor_position_includes_the_image_delta(kind):
    config = _hy4_config() if kind == "hy4" else _flash_config()
    model = _model(_with_dfly(_vision_config(config)))
    lm = model.language_model
    lm.configure_mtp(True, 3)
    ids, merged = _image_prompt(model)
    out = lm(
        mx.array(ids[None]),
        cache=model.make_cache(),
        inputs_embeds=merged.inputs_embeds,
        position_ids=merged.position_ids,
        return_hidden=True,
    )
    assert lm._position_delta.value.tolist() == [-21]
    hidden, anchor = out.hidden_states[0], mx.array([int(ids[-1])])
    shifted = np.array(lm.dspark_forward(hidden, anchor, draft_length=3)[0])
    lm._position_delta.value = None
    plain = np.array(lm.dspark_forward(hidden, anchor, draft_length=3)[0])
    assert np.abs(shifted - plain).max() > 1e-3


def test_flash_history_restored_without_positions_still_serves_text():
    model = _model(_flash_config())
    lm = model.language_model
    ids = _ids(1).astype(np.int32)
    expected = _run(model, ids, [24] + [1] * 16)
    cache = model.make_cache()
    lm(mx.array(ids[None, :24]), cache=cache)
    # Cache blocks written before positions were stored carry zero-width values.
    history = cache[0].caches[2]
    history.values = history.values[..., :0]
    steps = [
        np.array(lm(mx.array([[int(t)]]), cache=cache).logits)[0, 0] for t in ids[24:]
    ]
    np.testing.assert_allclose(np.stack(steps), expected[24:], atol=2e-4)
