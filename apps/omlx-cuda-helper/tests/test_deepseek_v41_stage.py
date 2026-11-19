# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper import mx_linear  # noqa: E402
from omlx_cuda_helper.families import deepseek_v41  # noqa: E402
from omlx_cuda_helper.families.deepseek_v41 import (  # noqa: E402
    kernels,
    layers,
    ops,
    quant,
)
from omlx_cuda_helper.families.deepseek_v41.config import Config  # noqa: E402
from omlx_cuda_helper.families.deepseek_v41.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/deepseek_v41"
MODEL = FIXTURE / "model"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)


@cache
def _case():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {name: (data[name], dtype) for name, dtype, _, _ in meta["manifest"]}
    return meta, arrays


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


@cache
def _stage(first, last):
    return load_stage(MODEL, first, last)


def _replay(bounds, layers):
    """Chunks of the recorded prompt through chained stages, like the worker:
    (boundary at ``layers``, state of every layer below it) per chunk."""
    meta, arrays = _case()
    sizes, skip = meta["sizes"], meta["prefix_chunks"]
    stages = [_stage(a, b) for a, b in zip(bounds, bounds[1:])]
    for stage in stages:
        # A layer loaded again after a long prompt runs as at the first load.
        stage.drop_layer()
        stage.add_layer()
    jobs = []
    for stage in stages:
        own = max(0, min(layers, stage.last) - stage.first)
        jobs.append(stage.new_job(own) if own else None)
        prefix = _group(arrays, "p:")
        mine = {n: v for n, v in prefix.items() if int(n.split(".")[0]) >= stage.first}
        if own and mine:
            stage.load_prefix(jobs[-1], mine)
        below = {f"p:{n}": v for n, v in prefix.items() if n not in mine}
        if own and stage.first:
            stage.load_upstream(jobs[-1], below)
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    out, start = [], 0
    for index, size in enumerate(sizes[skip:]):
        boundary, state = None, {}
        for stage, job in zip(stages, jobs):
            if job is None:
                continue
            if stage.first:
                stage.load_upstream(job, {f"s:{n}": v for n, v in state.items()})
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs=_group(arrays, f"in{index}:"),
                following=0,
                capture=None,
            )
            state.update(own)
        out.append((boundary, state))
        start += size
    return out


def _assert_same(got, expected, tolerance, label):
    assert sorted(got) == sorted(expected), label
    for name, (want, dtype) in expected.items():
        data, name_of_dtype = got[name]
        where = f"{label} {name}"
        assert (name_of_dtype, data.dtype, data.shape) == (
            dtype,
            want.dtype,
            want.shape,
        ), where
        if want.dtype.kind == "f":
            np.testing.assert_allclose(data, want, *tolerance, err_msg=where)
        else:
            np.testing.assert_array_equal(data, want, err_msg=where)


@pytest.mark.parametrize("layers", [6, 4])
@pytest.mark.parametrize("bounds", [(0, 6), *((0, a, 6) for a in range(1, 6))])
def test_stages_reproduce_the_contract_fixture(bounds, layers):
    """One stage and every two-stage split, also for a job that uses fewer
    layers than the stages hold. A second stage may start at an Engram layer
    or a layer that reads the compressed KV of a layer before it."""
    meta, arrays = _case()
    assert meta["contract"] == deepseek_v41.CONTRACT
    tolerance = (meta["rtol"], meta["atol"])
    whole = _replay((0, meta["layers"]), layers)
    for index, (boundary, state) in enumerate(_replay(bounds, layers)):
        below = {
            n: v
            for n, v in _group(arrays, f"s{index}:").items()
            if int(n.split(".")[0]) < layers
        }
        _assert_same(state, below, tolerance, f"chunk {index} state")
        # The fixture has the boundary of its own cut only; a lower cut must
        # give what a single stage gives there.
        expected = (
            _group(arrays, f"b{index}:")
            if layers == meta["layers"]
            else whole[index][0]
        )
        _assert_same(boundary, expected, tolerance, f"chunk {index} boundary")


def test_plan_and_warm():
    shared, per_layer = deepseek_v41.plan(MODEL)
    # The embedding only: the head and the final norm are never loaded.
    assert shared == 64 * 32 * 4 and len(per_layer) == 12 and min(per_layer) > 0
    # Stages that start at a KV source, an Engram layer and a KV reader.
    assert all(
        _stage(*bounds).warm(12) > 0 for bounds in ((0, 3), (3, 6), (1, 6), (2, 6))
    )


@pytest.mark.skipif(not kernels.available(), reason="the tile kernels run on CUDA")
@pytest.mark.parametrize(
    "spec",
    [{"mode": "affine", "bits": 3, "group_size": 64}, {"mode": "mxfp4", "bits": 4}],
    ids=["affine3", "mxfp4"],
)
def test_expert_tiles_read_banks_as_gather_qmm(spec):
    """Gate/up with SwiGLU and FP8, then down, against MLX's expert kernel."""

    def bank(name, rows, width, seed):
        w = mx.random.normal((2, rows, width), key=mx.random.key(seed)) * 0.1
        group = spec.get("group_size", 32)
        q = mx.quantize(w.astype(mx.bfloat16), group, spec["bits"], mode=spec["mode"])
        tensors = dict(zip((".weight", ".scales", ".biases"), q))
        weights = {name + part: value for part, value in tensors.items()}
        return layers.ExpertBank(weights, name, Config(quantized={name: spec}))

    gate, up, down = (
        bank("g", 128, 256, 1),
        bank("u", 128, 256, 2),
        bank("d", 256, 128, 3),
    )
    assert kernels.experts_fit([gate, up, down])
    keys = mx.random.split(mx.random.key(4), 3)
    # 200 routes on two experts: passes of two 64-route tiles and a partial one.
    x = quant.quantize(mx.random.normal((200, 256), key=keys[0]).astype(mx.bfloat16))
    experts = mx.sort(mx.random.randint(0, 2, (200,), key=keys[1])).astype(mx.uint32)
    weights = mx.random.uniform(0.1, 1.0, (200,), key=keys[2])
    rows = x[:, None]
    want = layers._swiglu(gate(rows, experts), up(rows, experts), weights, 10.0, True)
    got = kernels.expert_gate_up(
        x, mx.arange(200, dtype=mx.uint32), experts, gate, up, weights, 10.0, True
    )
    assert kernels.relative_error(got, want.reshape(got.shape)) < layers._TILE_ERROR
    got = kernels.expert_down(want.reshape(200, -1), experts, down)
    want_down = down(want, experts)
    assert (
        kernels.relative_error(got, want_down.reshape(got.shape)) < layers._TILE_ERROR
    )
    if not kernels.mx_experts_fit([gate, up, down]):
        return

    # The block-scaled tiles: the round trip as codes, the SwiGLU output too.
    def values(codes, scales):
        scale = mx.repeat(mx.power(2.0, scales.astype(mx.float32) - 127), 32, -1)
        return quant.from_fp8(codes) * scale

    raw = mx.random.normal((200, 256), key=keys[0]).astype(mx.bfloat16)
    codes, scales = kernels.fp8_codes(raw)
    assert mx.array_equal(values(codes, scales).astype(mx.bfloat16), x).item()
    codes, scales = kernels.expert_gate_up_mx(
        codes, scales, mx.arange(200, dtype=mx.uint32), experts, gate, up, weights, 10.0
    )
    got = values(codes, scales)
    assert kernels.relative_error(got, want.reshape(got.shape)) < layers._TILE_ERROR
    got = kernels.expert_down_mx(codes, scales, experts, down)
    assert (
        kernels.relative_error(got, want_down.reshape(got.shape)) < layers._TILE_ERROR
    )


@pytest.mark.skipif(not kernels.available(), reason="the FP8 kernel runs on CUDA")
def test_fp8_pack_matches_quant_pack_for_each_input_dtype():
    """The FP8 kernel compiled for one input dtype must not serve another."""
    x = mx.random.normal((2, 4, 64), key=mx.random.key(5)) * 3
    for dtype in (mx.float32, mx.bfloat16):
        rows = x.astype(dtype)
        assert mx.array_equal(kernels.pack_fp8(rows), quant.pack(rows)).item()


@pytest.mark.skipif(not mx_linear.available(), reason="needs the block-scaled mma")
def test_an_mxfp8_projection_stays_packed_on_the_block_scaled_matmul():
    """Over a partial row tile and two stages of depth, its codes times the
    input's FP8 codes give the product of the unpacked weight."""
    w = mx.random.normal((256, 512), key=mx.random.key(1)) * 0.05
    packed, scales = mx.quantize(w.astype(mx.bfloat16), 32, 8, mode="mxfp8")[:2]
    weights = {"w.weight": packed, "w.scales": scales}
    config = Config(quantized={"w": {"mode": "mxfp8", "bits": 8}})
    projection = layers.Projection(weights, "w", config)
    assert projection.mx is not None and projection.weight_t is None
    x = mx.random.normal((1, 130, 512), key=mx.random.key(2)).astype(mx.bfloat16)
    want = ops.matmul(layers._quantize(x), projection.unpacked().T)
    assert kernels.relative_error(projection(x), want) < layers._TILE_ERROR
    projection.use_unpacked()
    assert mx.array_equal(projection(x), want).item()
