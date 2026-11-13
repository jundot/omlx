"""Synthetic packed GGUF checks; no model downloads or gguf dependency."""

import json
import struct

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.gguf import GgufError
from omlx.patches.deepseek_v41.gguf import (
    _LAYER,
    _ROOT,
    backbone_tensors,
    prepare,
    read_header,
)
from omlx.patches.deepseek_v41.gguf_quant import (
    FORMATS,
    GGUFProjection,
    dequantize,
    row_bytes,
)

# Recorded independently with llama.cpp gguf-py CPU dequantize + NumPy dot.
# Each seed exercises every block's grid/sign/scale bytes, two expert choices,
# two blocks per row and tail output rows. No upstream checkout is needed.
_EXPECTED = {
    8: [[1.1008341, -34.3133774, 29.1364708], [-5.4555197, -11.1512699, -2.8665476]],
    10: [[-3.2740026, 0.9873986, 3.904026], [-9.4498959, 1.3240428, -0.8963728]],
    13: [
        [-90.5944824, -260.4677429, -1.8368226],
        [126.7198334, 22.7707634, 159.5210266],
    ],
    14: [
        [-742.0026855, 144.9895782, -18.4693451],
        [290.0865479, -188.8960266, 354.407074],
    ],
    16: [[-17.837059, 3.6467731, 0.0851333], [6.784008, 1.6515222, -11.0491123]],
    17: [[6.7956161, -0.8044491, -0.6864071], [17.9276123, 9.4605827, -16.1722946]],
    18: [[28.2220955, 49.6144447, 1.2364564], [32.4849396, 48.3264542, 26.399353]],
    19: [[0.9451886, 1.4143728, 0.4916597], [0.6244406, -0.0008309, 0.2853332]],
    22: [[4.7803683, 11.6624737, -5.4103909], [-0.5960388, -6.1177526, 11.5483284]],
}


def packed(qtype, shape, seed):
    block, size, _ = FORMATS[qtype]
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (*shape[:-1], shape[-1] // block, size), np.uint8)
    offset = {10: 80, 14: 208}.get(qtype, 0)
    data[..., offset : offset + 2] = np.array([0.01], "<f2").view(np.uint8)
    if qtype in (10, 13):
        offset = 82 if qtype == 10 else 2
        data[..., offset : offset + 2] = np.array([0.003], "<f2").view(np.uint8)
    return data.reshape(*shape[:-1], -1), rng


@pytest.mark.parametrize("qtype", FORMATS)
def test_packed_projection_matches_cpu_golden(qtype):
    data, rng = packed(qtype, (2, 3, 512), 4100 + qtype)
    x = rng.normal(size=(2, 512)).astype(np.float32)
    p = GGUFProjection(mx.array(data), qtype=qtype, input_dims=512)
    y = p(mx.array(x[:, None]), mx.array([1, 0]))
    np.testing.assert_allclose(y[:, 0], _EXPECTED[qtype], atol=2e-4, rtol=2e-5)
    dense = dequantize(p.weight, qtype, (2, 3, 512), mx.float32)
    decoded_dot = np.einsum("rk,rnk->rn", x, np.asarray(dense)[[1, 0]])
    np.testing.assert_allclose(decoded_dot, _EXPECTED[qtype], atol=2e-4, rtol=2e-5)
    # Gather broadcasting (decode) and already sorted rows (prefill) agree.
    ids = mx.array([[1, 0], [0, 1]])
    broadcast = p(mx.array(x[:, None, None]), ids)
    selected = mx.repeat(mx.array(x), 2, axis=0)[:, None]
    sorted_rows = p(selected, ids.reshape(-1), sorted_indices=True)
    np.testing.assert_array_equal(broadcast.reshape(4, 1, 3), sorted_rows)
    shared = GGUFProjection(p.weight[0], qtype=qtype, input_dims=512)
    bf16 = mx.array(x).astype(mx.bfloat16)
    np.testing.assert_array_equal(
        shared(bf16).astype(mx.float32),
        shared(bf16, sorted_indices=True).astype(mx.float32),
    )
    np.testing.assert_allclose(
        shared(bf16).astype(mx.float32),
        (bf16.astype(mx.float32) @ dense[0].T).astype(mx.bfloat16).astype(mx.float32),
        atol=0.01,
        rtol=0.01,
    )
    assert p.weight.nbytes == data.nbytes


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def part(path, number, count, total, tensors, config):
    meta = {
        "general.architecture": "deepseek41",
        "split.no": number,
        "split.count": count,
        "split.tensors.count": total,
        "deepseek41.block_count": config.n_layers,
        "deepseek41.embedding_length": config.dim,
        "deepseek41.expert_count": config.n_routed_experts,
        "deepseek41.attention.head_count": config.n_heads,
        "deepseek41.expert_feed_forward_length": config.moe_inter_dim,
    }
    if number:
        # Real split GGUFs put model/tokenizer metadata only in part zero.
        meta = {key: value for key, value in meta.items() if key.startswith("split.")}
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for key, value in meta.items():
        data += string(key) + struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack("<I", value)
    payload = bytearray()
    for name, shape, qtype, value in tensors:
        payload += bytes(-len(payload) % 32)
        data += string(name) + struct.pack("<I", len(shape))
        data += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        data += struct.pack("<IQ", qtype, len(payload))
        payload += value
    data += bytes(-len(data) % 32) + payload
    file = path / f"synthetic-{number + 1:05}-of-{count:05}.gguf"
    file.write_bytes(data)
    return file


@pytest.mark.parametrize("preserve_mtp", [False, True])
def test_load_without_final_shard_retains_original_engram(
    tmp_path, monkeypatch, preserve_mtp
):
    from test_deepseek_v41 import write_checkpoint

    from omlx.patches import mlx_lm_mtp
    from omlx.patches.deepseek_v41.loading import load
    from omlx.utils.model_loading import lightning_mtp_supported

    source, reference = write_checkpoint(
        tmp_path,
        vision=True,
        dim=256,
        moe_inter_dim=256,
        n_layers=1,
        compress_ratios=(0, 0),
        preserve_mtp=True,
        n_mtp_layers=1,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(0,),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        kv_source_layers=(),
        index_source_layers=(),
        candidate_source_layer=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        engram_layer_ids=(0,),
        engram_num_embeddings=(72,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    before = (source / "model.safetensors").read_bytes()
    folder, dest = tmp_path / "UD-IQ2_XXS", tmp_path / "mlx"
    # This must work before the GGUF download starts.
    prepare(folder, source, dest)
    spec = json.loads((dest / "config.json").read_text())["omlx_deepseek_v41"]
    assert spec["engram_tables"]["language_model.layers.0.engram.embed"][
        "weight_file"
    ] == str(source / "model.safetensors")
    with pytest.raises(FileNotFoundError, match="No GGUF parts"):
        backbone_tensors(folder, reference.config)
    folder.mkdir()
    inverse = {v: k for k, v in _ROOT.items()}
    inverse_layer = {v: k for k, v in _LAYER.items()}
    tensors, expected_packed = [], {}
    for name, value in tree_flatten(reference.parameters()):
        if (
            not name.startswith("language_model.")
            or name.startswith("language_model.mtp.")
            or ".engram.embed." in name
            or name.endswith(".engram.k_weight")
        ):
            continue
        original = name.removeprefix("language_model.")
        if original.startswith("layers."):
            _, layer, rest = original.split(".", 2)
            ggname = f"blk.{layer}." + inverse_layer[rest]
        else:
            ggname = inverse[original]
        qtype = 0
        raw = np.asarray(value.astype(mx.float32))
        if ".experts.w" in name:
            qtype = {"w1": 16, "w2": 22, "w3": 19}[name.split(".")[-2]]
            raw, _ = packed(qtype, value.shape, len(tensors))
            expected_packed[name] = raw
        tensors.append((ggname, value.shape, qtype, raw.tobytes()))
    count = len(tensors) + 1
    part(folder, 0, 3, count, [], reference.config)
    weights = part(folder, 1, 3, count, tensors, reference.config)
    assert len(backbone_tensors(folder, reference.config)) == len(tensors)
    from omlx.model_discovery import _register_model, estimate_model_size
    from omlx.patches.deepseek_v41.residency import deepseek_v41_residency_estimate

    estimate = deepseek_v41_residency_estimate(dest)
    extra_bytes = sum(
        v.nbytes
        for k, v in tree_flatten(reference.parameters())
        if k.startswith(("vision.", "aligner.", "image_", "language_model.mtp."))
    )
    table_bytes = reference.language_model.layers[0].engram.embed.weight.nbytes
    expected_bytes = sum(len(t[3]) for t in tensors) + extra_bytes + table_bytes
    assert estimate.resident_bytes == int(expected_bytes * 1.05)
    assert estimate.engram_bytes == table_bytes
    assert estimate_model_size(dest) == estimate.resident_bytes
    registered = {}
    _register_model(registered, dest, "packed-iq2")
    assert registered["packed-iq2"].engine_type == "vlm"
    assert registered["packed-iq2"].estimated_size == estimate.resident_bytes
    assert lightning_mtp_supported(dest)
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", preserve_mtp)
    # The engine relies on the construction flag, without a preserve_mtp override.
    model, _ = load(dest)
    try:
        for name, raw in expected_packed.items():
            value = dict(tree_flatten(model.parameters()))[name]
            np.testing.assert_array_equal(value, raw)
        table = model.language_model.layers[0].engram.embed
        ids = mx.array([0, 3, 71])
        np.testing.assert_array_equal(
            table(ids).astype(mx.float32),
            reference.language_model.layers[0]
            .engram.embed(ids)
            .astype(mx.bfloat16)
            .astype(mx.float32),
        )
        y = model(mx.array([[3, 4, 3]]))
        assert y.shape == (1, 3, 64) and bool(mx.all(mx.isfinite(y)))
        assert ("mtp" in model.language_model) == preserve_mtp
        assert model.language_model._omlx_dspark_decode_enabled == preserve_mtp
        if preserve_mtp:
            actual = dict(tree_flatten(model.parameters()))
            for name, value in tree_flatten(reference.parameters()):
                if name.startswith(
                    ("vision.", "aligner.", "image_", "language_model.mtp.")
                ):
                    np.testing.assert_array_equal(actual[name], value)
    finally:
        model.close()
        reference.close()
    assert (source / "model.safetensors").read_bytes() == before
    # Present final Engram part is ignored, with no payload read.
    final = part(
        folder,
        2,
        3,
        count,
        [("per_layer_token_embd.weight", (144, 32), 13, b"")],
        reference.config,
    )
    assert len(backbone_tensors(folder, reference.config)) == len(tensors)
    assert deepseek_v41_residency_estimate(dest) == estimate
    # Neither a missing backbone part nor a truncated tensor is excused.
    saved = weights.read_bytes()
    weights.unlink()
    with pytest.raises(GgufError, match="Missing required GGUF backbone"):
        backbone_tensors(folder, reference.config)
    weights.write_bytes(saved[:-1])
    with pytest.raises(GgufError, match="Truncated or overlapping"):
        read_header(weights)
    final.unlink()


def test_packed_row_validation():
    with pytest.raises(ValueError, match="not divisible"):
        row_bytes(255, 16)
    with pytest.raises(ValueError, match="row length"):
        GGUFProjection(mx.zeros((2, 65), mx.uint8), qtype=16, input_dims=256)


@pytest.mark.parametrize("qtype", FORMATS)
def test_packed_matrix_and_gather_tails(qtype):
    raw, rng = packed(qtype, (4, 11, 512), 6500 + qtype)
    p = GGUFProjection(mx.array(raw), qtype=qtype, input_dims=512)
    dense = dequantize(p.weight, qtype, (4, 11, 512), mx.float32)
    ids = np.repeat(np.arange(4, dtype=np.int32), [5, 17, 0, 45])
    x = rng.normal(size=(len(ids), 512)).astype(np.float32)
    actual = p(mx.array(x[:, None]), mx.array(ids), sorted_indices=True)[:, 0]
    expected = np.einsum("rk,rnk->rn", x, np.asarray(dense)[ids])
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-4)
    shared = GGUFProjection(p.weight[0], qtype=qtype, input_dims=512)
    np.testing.assert_allclose(
        shared(mx.array(x[:33])), x[:33] @ np.asarray(dense[0]).T, atol=1e-3, rtol=1e-4
    )
    np.testing.assert_allclose(
        shared(mx.array(x)), x @ np.asarray(dense[0]).T, atol=1e-3, rtol=1e-4
    )


def gguf_module(module, qtype, seed):
    shape = (
        *module.weight.shape[:-1],
        getattr(module, "input_dims", module.weight.shape[-1]),
    )
    raw, _ = packed(qtype, shape, seed)
    blocks = raw.reshape(*shape[:-1], shape[-1] // FORMATS[qtype][0], FORMATS[qtype][1])
    offset = {10: 80, 14: 208}.get(qtype, 0)
    blocks[..., offset : offset + 2] = np.array([0.0002], "<f2").view(np.uint8)
    if qtype in (10, 13):
        offset = 82 if qtype == 10 else 2
        blocks[..., offset : offset + 2] = np.array([0.00003], "<f2").view(np.uint8)
    return GGUFProjection(mx.array(raw), qtype=qtype, input_dims=shape[-1])


@pytest.mark.parametrize(
    "formats", [(19, 19, 10, 8), (16, 16, 17, 8), (22, 22, 18, 13)]
)
@pytest.mark.parametrize("rows", [1, 6])
def test_gguf_fused_moe_keeps_unquantized_activations(formats, rows):
    from test_deepseek_v41 import tiny

    from omlx.patches.deepseek_v41 import decode_moe as dm
    from omlx.patches.deepseek_v41.language import MoE

    c = tiny(dim=1024, moe_inter_dim=768, swiglu_limit=10)
    moe = MoE(c)
    for n, qt in zip(("w1", "w3", "w2"), formats[:3]):
        setattr(moe.experts, n, gguf_module(getattr(moe.experts, n), qt, 92 + qt))
        setattr(
            moe.shared_experts,
            n,
            gguf_module(getattr(moe.shared_experts, n), formats[3], 71 + qt),
        )
    mx.random.seed(21)
    moe.gate.weight = mx.random.normal(moe.gate.weight.shape) * 0.05
    moe.gate.bias = mx.array([0.01, -0.02, 0.03, -0.04])
    x = mx.random.normal((rows, c.dim)).astype(mx.bfloat16)
    assert dm.supported(moe, x[:1], c)
    expected = moe(x[None], None)[0]
    actual, ids, weights = dm.fused_decode_moe_nb(moe, x, c)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), atol=2e-4, rtol=0.015
    )
    wanted_ids, wanted_weights = moe.gate(x, None)
    np.testing.assert_array_equal(ids, wanted_ids)
    np.testing.assert_allclose(weights, wanted_weights, atol=1e-6, rtol=1e-5)
    individual = mx.concatenate(
        [dm.fused_decode_moe_nb(moe, x[r : r + 1], c)[0] for r in range(rows)]
    )
    exact = dm.fused_decode_moe_nb(moe, x, c, mma=False)[0]
    np.testing.assert_array_equal(
        exact.astype(mx.float32), individual.astype(mx.float32)
    )
    np.testing.assert_allclose(
        actual.astype(mx.float32), exact.astype(mx.float32), atol=2e-4, rtol=0.015
    )


@pytest.mark.parametrize("inter", [768, 800])
def test_gguf_fused_moe_rejects_incompatible_shared_input_and_k_split(inter):
    from test_deepseek_v41 import tiny

    from omlx.patches.deepseek_v41 import decode_moe as dm
    from omlx.patches.deepseek_v41.language import MoE

    c = tiny(dim=1024, moe_inter_dim=inter)
    moe = MoE(c)
    for n in ("w1", "w3", "w2"):
        setattr(moe.experts, n, gguf_module(getattr(moe.experts, n), 8, 10))
    x = mx.zeros((1, c.dim), mx.bfloat16)
    assert not dm.supported(moe, x, c)
    for n in ("w1", "w3", "w2"):
        setattr(
            moe.shared_experts, n, gguf_module(getattr(moe.shared_experts, n), 8, 11)
        )
    assert dm.supported(moe, x, c) == (inter % 64 == 0)


@pytest.mark.parametrize("batch,rows", [(1, 1), (1, 6), (2, 6)])
def test_gguf_fused_block_and_mtp_verify_match_unfused(monkeypatch, batch, rows):
    from test_deepseek_v41_fused_decode import build

    from omlx.patches.deepseek_v41 import decode_fused

    model = build(
        n_layers=1,
        compress_ratios=(0,),
        kv_source_layers=(),
        index_source_layers=(),
        candidate_source_layer=-1,
    )
    c, blk = model._config, model.layers[0]
    for n in ("wq_a", "wkv", "wq_b", "wo_b"):
        setattr(blk.attn, n, gguf_module(getattr(blk.attn, n), 8, 11))
    for n, qt in zip(("w1", "w3", "w2"), (19, 16, 22)):
        setattr(
            blk.ffn.experts, n, gguf_module(getattr(blk.ffn.experts, n), qt, 71 + qt)
        )
        setattr(
            blk.ffn.shared_experts,
            n,
            gguf_module(getattr(blk.ffn.shared_experts, n), 8, 77 + qt),
        )
    assert decode_fused.supported(blk, c)
    expected_caches = [model.make_cache()[0] for _ in range(batch)]
    actual_caches = [model.make_cache()[0] for _ in range(batch)]
    mx.random.seed(91)
    h = mx.random.normal((batch, rows, c.hc_mult, c.dim)).astype(mx.bfloat16)
    pre = mx.ones((batch, rows, c.hc_mult), mx.float32)
    with monkeypatch.context() as patch:
        patch.setattr(decode_fused, "supported", lambda *args: False)
        expected = [
            blk(h[b : b + 1], pre[b : b + 1], expected_caches[b], {}, 0, None)
            for b in range(batch)
        ]
        expected_h = mx.concatenate([v[0] for v in expected])
        expected_pre = mx.concatenate([v[1] for v in expected])
        mx.eval(expected_h, expected_pre)
    actual_h, actual_pre = decode_fused.decode_block_verify_nb(
        blk, h, pre, actual_caches, [{} for _ in range(batch)], [0] * batch, c
    )
    np.testing.assert_allclose(
        actual_h.astype(mx.float32), expected_h.astype(mx.float32), atol=0.02, rtol=0.01
    )
    np.testing.assert_allclose(actual_pre, expected_pre, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("qtype", [8, 14])
def test_gguf_verify_projection_reuses_weights_without_changing_rows(qtype):
    raw, rng = packed(qtype, (8192, 512), 381 + qtype)
    p = GGUFProjection(mx.array(raw), qtype=qtype, input_dims=512)
    x = mx.array(rng.normal(size=(6, 512)).astype(np.float32)).astype(mx.bfloat16)
    expected = mx.concatenate([p(x[r : r + 1]) for r in range(6)])
    np.testing.assert_array_equal(p(x).astype(mx.float32), expected.astype(mx.float32))


def test_gguf_mtp_generation_matches_greedy(monkeypatch):
    from mlx_lm.generate import BatchGenerator
    from test_deepseek_v41_fused_decode import build

    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    model = build(
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
    )
    for i, blk in enumerate(model.layers):
        for n in ("wq_a", "wkv", "wq_b", "wo_b"):
            setattr(blk.attn, n, gguf_module(getattr(blk.attn, n), 8, 11 + i))
        for n, qt in zip(("w1", "w3", "w2"), (19, 16, 22)):
            setattr(
                blk.ffn.experts,
                n,
                gguf_module(getattr(blk.ffn.experts, n), qt, 71 + i + qt),
            )
            setattr(
                blk.ffn.shared_experts,
                n,
                gguf_module(getattr(blk.ffn.shared_experts, n), 8, 77 + i + qt),
            )
    prompt, wanted, cache = [3, 4, 5, 6, 7, 8, 9], [], model.make_cache()
    ids = mx.array([prompt])
    for _ in range(8):
        logits = model(ids, cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        wanted.append(token)
        ids = mx.array([[token]])
    model.configure_mtp(True, 3)
    calls = []
    original = type(model).dspark_forward

    def tracked(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "dspark_forward", tracked)
    generator = BatchGenerator(
        model,
        max_tokens=8,
        prefill_step_size=32,
        sampler=lambda logits: mx.argmax(logits, -1),
    )
    result = []
    try:
        generator.insert([prompt])
        for _ in range(64):
            _, responses = generator.next()
            result.extend(r.token for r in responses)
            if len(result) == 8:
                break
    finally:
        generator.close()
    assert calls, "Exercise actual DSpark proposal execution"
    assert result == wanted
