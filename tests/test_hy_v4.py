# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hy4 / Hy4-flash text models on tiny synthetic checkpoints."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models.cache import (  # noqa: E402
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
)

from omlx import oq  # noqa: E402
from omlx.adapter.output_parser import correct_tool_parser_type  # noqa: E402
from omlx.engine.vlm import _apply_template_thinking_mode  # noqa: E402
from omlx.patches.hy_v4 import indexing  # noqa: E402
from omlx.patches.hy_v4.flash import CompressedIndexer  # noqa: E402
from omlx.patches.hy_v4.language import set_dspark_verify_armed  # noqa: E402
from omlx.patches.hy_v4.layers import Indexer  # noqa: E402
from omlx.patches.hy_v4.model import (  # noqa: E402
    Model,
    ModelConfig,
    dequantize_block_fp8,
)
from omlx.patches.hy_v4.tool_parser import parse_tool_call  # noqa: E402
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
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
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
            "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1e7},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
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


def test_incremental_matches_full_forward(model):
    ids = _ids(1)
    full = _run(model, ids, [SEQ])
    # Prefill, ragged chunks (open compressed-indexer groups) and decode steps.
    for chunks in ([24] + [1] * 16, [7, 7, 7, 7, 7, 5], [1] * SEQ):
        np.testing.assert_allclose(_run(model, ids, chunks), full, atol=2e-4)


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
        nn.quantize(
            model.language_model,
            group_size=64,
            bits=bits,
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


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("bits", [None, 4, 3])
@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_exact_verify_rows_match_decode(kind, bits, batched):
    # Past the index top-k and the 8-slot sliding window, so sparse selection
    # and a rotated ring are both in play. mlx-lm's generator hands even one
    # request one-row batch caches.
    apply_verify_qmm_patch()
    cache_rollback.apply()
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
    tail = ids[prompt + 3 : prompt + 9] + 1
    for got, want in zip(
        _decode_logits(lm, cache, tail), _decode_logits(lm, reference, tail)
    ):
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


@pytest.mark.parametrize("bits", [4, 3])
@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_mtp_generation_matches_greedy(monkeypatch, kind, bits):
    """Through the generator (draft, exact verify, accept, rollback), alone and
    with a second request, MTP emits each request's own greedy tokens."""
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    assert apply_mlx_lm_mtp_patch()
    model = _quantized_model(kind, bits, dfly=True)
    lm = model.language_model
    drafts = []
    original = type(lm).dspark_forward

    def tracked(self, *args, **kwargs):
        drafts.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(lm), "dspark_forward", tracked)
    prompt = _ids(11, n=26)
    expected = _generate(model, [prompt], 24, use_mtp=False)
    assert _generate(model, [prompt], 24, use_mtp=True) == expected
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
    expected = [_generate(model, [p], 20, use_mtp=False)[0] for p in prompts]
    drafts.clear()
    assert _generate(model, prompts, 20, use_mtp=True) == expected
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
