# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper import checkpoint, oqh, tensors  # noqa: E402
from omlx_cuda_helper.families import qwen3_5  # noqa: E402
from omlx_cuda_helper.families.qwen3_5.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/qwen3_5"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)


def _case():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {name: (data[name], dtype) for name, dtype, _, _ in meta["manifest"]}
    return meta, arrays


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


def _replay(bounds, capture=None, model=FIXTURE / "model"):
    """The fixture's chunks through stages cut at ``bounds``.

    Yields each chunk's index, the boundary at the end of every stage (by
    its last layer), the state of the whole chain and what the stages
    captured for ``capture``. Only the first stage gets the host's inputs.
    """
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == qwen3_5.CONTRACT
    assert all(a >= qwen3_5.FIRST_STAGE_MIN for a in bounds[1:-1])
    stages = [load_stage(model, first, last) for first, last in zip(bounds, bounds[1:])]
    for stage in stages:
        # A layer loaded again after a long prompt runs as at the first load.
        stage.drop_layer()
        stage.add_layer()
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage, job in zip(stages, jobs):
        # On CUDA the replay then runs through the self-checked kernels.
        stage.enable_kernels()
        stage.load_prefix(job, _group(arrays, "p:"))
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start = 0
    for index, size in enumerate(sizes[skip:]):
        boundary, cuts, state, captured = None, {}, {}, {}
        for stage, job in zip(stages, jobs):
            inputs = {} if boundary else _group(arrays, f"in{index}:")
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs=inputs,
                following=0,
                capture=capture,
            )
            assert all(stage.first <= int(n.split(".")[0]) < stage.last for n in own)
            state.update(own)
            captured.update({n: boundary.pop(n) for n in list(boundary) if "." in n})
            cuts[min(stage.last, k)] = boundary
        start += size
        yield index, cuts, state, captured


@pytest.mark.parametrize(
    "bounds",
    [(0, 5), (0, 1, 5), (0, 2, 5), (0, 3, 5), (0, 4, 5), (0, 6)],
    ids=["one-stage", "cut-1", "cut-2", "cut-3", "cut-4", "fewer-layers-than-held"],
)
def test_stages_reproduce_the_contract_fixture(bounds):
    meta, arrays = _case()
    for index, cuts, state, _ in _replay(bounds):
        boundary = cuts[meta["layers"]]
        for tag, got in ((f"b{index}:", boundary), (f"s{index}:", state)):
            want = _group(arrays, tag)
            assert sorted(got) == sorted(want), tag
            for name, (data, dtype) in got.items():
                expected, expected_dtype = want[name]
                assert (dtype, data.dtype, data.shape) == (
                    expected_dtype,
                    expected.dtype,
                    expected.shape,
                ), (
                    tag + name
                )
                np.testing.assert_allclose(
                    data,
                    expected,
                    rtol=meta["rtol"],
                    atol=meta["atol"],
                    err_msg=tag + name,
                )


@pytest.mark.skipif(not mx.cuda.is_available(), reason="a GPU load runs on CUDA")
def test_weights_loaded_into_the_gpu_pool_give_the_same_stages(monkeypatch):
    monkeypatch.setattr(tensors, "weights_on_gpu", True)
    test_stages_reproduce_the_contract_fixture((0, 2, 5))


_MIXED = {
    "mxfp4": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    "mxfp8": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    "nvfp4": {"group_size": 16, "bits": 4, "mode": "nvfp4"},
}


def _quantized(folder, bits, mode=None):
    """The fixture model with its linears in affine ``bits``, group 32, and its
    full-attention q_proj and GDN in_proj_qkv in ``_MIXED[mode]``."""
    folder.mkdir()
    weights = mx.load(str(FIXTURE / "model/model.safetensors"))
    config = json.loads((FIXTURE / "model/config.json").read_text())
    config["quantization"] = {"group_size": 32, "bits": bits, "mode": "affine"}
    for name in [n for n in weights if "_proj" in n and n.endswith(".weight")]:
        base = name.removesuffix(".weight")
        quant = {"group_size": 32, "bits": bits, "mode": "affine"}
        if mode and base.endswith(("self_attn.q_proj", "linear_attn.in_proj_qkv")):
            quant = config["quantization"][base] = _MIXED[mode]
        # The floating point modes have no biases.
        parts = mx.quantize(weights[name].astype(mx.bfloat16), **quant)
        for suffix, part in zip(("weight", "scales", "biases"), parts):
            weights[f"{base}.{suffix}"] = part
    mx.save_safetensors(str(folder / "model.safetensors"), weights)
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def _linears(obj, depth=4):
    """The ``Linear`` layers reachable from *obj*."""
    if isinstance(obj, tensors.Linear):
        return [obj]
    if depth == 0:
        return []
    if isinstance(obj, (list, tuple)):
        return [x for item in obj for x in _linears(item, depth - 1)]
    if hasattr(obj, "__dict__"):
        return [x for v in vars(obj).values() for x in _linears(v, depth - 1)]
    return []


@pytest.mark.parametrize("bits", [8, 5])
def test_packed_weights_give_the_same_stages(tmp_path, monkeypatch, bits):
    model = _quantized(tmp_path / "model", bits)

    def hidden():
        return [
            np.array(tensors.from_host(cuts[5]["h"]).astype(mx.float32))
            for _, cuts, _, _ in _replay((0, 2, 5), model=model)
        ]

    unpacked = hidden()
    monkeypatch.setattr(tensors, "packed_weights", True)
    for want, got in zip(unpacked, hidden()):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 3e-2
    files = checkpoint.memory_plan(model, qwen3_5.resident)[1]
    assert qwen3_5.plan(model, True)[1] == files
    stage = load_stage(model, 0, 5)
    assert {lin.quant[2]["bits"] for lin in _linears(stage) if lin.quant} == {bits}


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mode", list(_MIXED))
def test_the_plan_counts_what_a_mixed_stage_holds(tmp_path, monkeypatch, mode, packed):
    model = _quantized(tmp_path / "model", 8, mode)
    monkeypatch.setattr(tensors, "packed_weights", packed)
    files = checkpoint.memory_plan(model, qwen3_5.resident)[1]
    planned = qwen3_5.plan(model, packed)[1]
    index = checkpoint.tensors(model)
    for i, layer in enumerate(load_stage(model, 0, 6).layers):
        # The floating point modes stay dense; the affine linears follow the stage.
        floating = layer.mixer.qkv if layer.linear else layer.mixer.q_proj
        assert floating.quant is None and (layer.down.quant is not None) is packed
        assert floating.dtype == layer.down.dtype == mx.bfloat16
        # Every linear of a layer, and nothing else, is named *_proj*.
        on_file = sum(
            t.nbytes
            for name, t in index.items()
            if checkpoint.layer_of(name) == i and "_proj" in name
        )
        linears = {id(lin): lin for lin in _linears(layer)}.values()
        held = sum(p.nbytes for lin in linears for p in lin.parameters())
        assert planned[i] - files[i] == held - on_file


def test_a_stage_returns_the_layers_a_block_drafter_reads():
    # The window starts at job row 8, inside the second chunk (rows 6..12).
    # Layer 5 is past the job's layers.
    chunks = list(_replay((0, 2, 5), {"layers": [1, 4, 5], "from": 8}))
    assert [sorted(c[3]) for c in chunks] == [[], ["cap.1", "cap.4"]]
    _, cuts, _, captured = chunks[1]
    for layer in (1, 4):
        got, dtype = captured[f"cap.{layer}"]
        want, want_dtype = cuts[layer + 1]["h"]
        assert dtype == want_dtype
        np.testing.assert_array_equal(got, want[:, 2:])


def test_a_repeated_kernel_self_check_still_compares_with_plain_ops(monkeypatch):
    attention = load_stage(FIXTURE / "model", 3, 4).layers[0].mixer

    def wrong(queries, keys, values, scale, mask):
        return mx.ones_like(queries)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", wrong)
    # As an earlier self-check had left fused attention in use.
    attention.fused = True
    attention.enable_kernels()
    assert attention.fused is False


def test_plan_counts_only_what_a_stage_loads():
    shared, layers = qwen3_5.plan(FIXTURE / "model")
    assert len(layers) == 6 and min(layers) > 0
    # The embedding is shared; the output head and the final norm stay on
    # the host.
    assert shared == 64 * 32 * 4
    assert not qwen3_5.resident("language_model.mtp.layers.0.mlp.up_proj.weight")
    # A Hugging Face checkpoint needs the host's conversion first.
    with pytest.raises(ValueError, match="MLX layout"):
        qwen3_5.check_layout({"model.language_model.layers.0.mlp.up_proj.weight": 0})


def _pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """Codes [rows, k] in MLX affine packing: one little-endian bit stream a row."""
    words = np.zeros((codes.shape[0], codes.shape[1] * bits // 32), np.uint64)
    for j in range(codes.shape[1]):
        word, shift = divmod(bits * j, 32)
        words[:, word] |= codes[:, j].astype(np.uint64) << shift
        if shift + bits > 32:
            words[:, word + 1] |= codes[:, j].astype(np.uint64) >> (32 - shift)
    return (words & 0xFFFFFFFF).astype(np.uint32)


def _bf16(a: np.ndarray) -> mx.array:
    return mx.array(a.astype(np.float32)).astype(mx.bfloat16)


@pytest.mark.skipif(not mx.cuda.is_available(), reason="oQh kernels run on CUDA")
@pytest.mark.parametrize(
    "bits, group, block", [(3, 0, 256), (6, 0, 128), (4, 128, 256)]
)
def test_an_oqh_linear_equals_its_weight_rotated_back(bits, group, block):
    rng = np.random.default_rng(bits)
    n, k = 64, 512
    codes = rng.integers(0, 2**bits, (n, k))
    signs = rng.choice([-1.0, 1.0], k)
    prefix = "layers.0.mlp.down_proj"
    weights = {
        f"{prefix}.weight": mx.array(_pack(codes, bits)),
        f"{prefix}.oqh_signs": _bf16(signs.reshape(-1, block)),
    }
    if group:
        scales = np.array(
            _bf16(rng.uniform(0.01, 0.02, (n, k // group))).astype(mx.float32)
        )
        biases = np.array(
            _bf16(rng.normal(0, 0.05, (n, k // group))).astype(mx.float32)
        )
        rotated = codes * scales.repeat(group, 1) + biases.repeat(group, 1)
        weights[f"{prefix}.oqh_scales"] = _bf16(scales)
        weights[f"{prefix}.oqh_biases"] = _bf16(biases)
    else:
        rscale = np.array(_bf16(rng.uniform(0.01, 0.02, n)).astype(mx.float32))
        rotated = rscale[:, None] * (codes - (2**bits - 1) / 2)
        weights[f"{prefix}.rscale"] = _bf16(rscale)
    # x is rotated as x diag(signs) H / sqrt(b) per block, so the plain weight
    # is W H / sqrt(b) diag(signs).
    hadamard = np.ones((1, 1))
    while hadamard.shape[0] < block:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    blocks = np.kron(np.eye(k // block), hadamard / np.sqrt(block))
    plain = rotated @ blocks * signs
    x = rng.normal(0, 1, (8, k))
    got = np.array(tensors.Linear(weights, prefix, None)(_bf16(x)).astype(mx.float32))
    want = x @ plain.T
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-2


@pytest.mark.skipif(not mx.cuda.is_available(), reason="oQh kernels run on CUDA")
def test_the_oqh_rotation_keeps_the_host_butterfly_order():
    # The host rotates in float32 from bit 0 up and rounds once: equal bits.
    signs = _bf16(np.random.default_rng(0).choice([-1.0, 1.0], 1024))
    for block in (128, 256):
        for dtype in (mx.bfloat16, mx.float32):
            x = (mx.random.normal((64, 1024), key=mx.random.key(block)) * 3).astype(
                dtype
            )
            got = oqh.rotate(x, signs, block, dtype=mx.bfloat16)
            want = oqh.reference_rotate(x, signs, block, mx.bfloat16)
            assert mx.array_equal(got, want).item()


def test_the_next_conv_window_does_not_hold_the_chunk():
    mixer = load_stage(FIXTURE / "model", 0, 1).layers[0].mixer
    qkv = mx.random.normal((1, 8192, mixer.taps.shape[0])).astype(mx.bfloat16)
    mx.eval(qkv)
    mx.synchronize()
    before = mx.get_active_memory()
    *outputs, window = mixer.prework(qkv, None)
    mx.eval(*outputs, window)
    del outputs
    mx.synchronize()
    assert mx.get_active_memory() - before < qkv.nbytes // 16
