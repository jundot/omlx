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
def test_load_without_final_shard_retains_original_engram(tmp_path, preserve_mtp):
    from test_deepseek_v41 import write_checkpoint

    from omlx.patches.deepseek_v41.loading import load

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
    model, _ = load(dest, preserve_mtp=preserve_mtp)
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
