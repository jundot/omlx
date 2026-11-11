# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper import tensors  # noqa: E402
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


def _replay(bounds, capture=None):
    """The fixture's chunks through stages cut at ``bounds``.

    Yields each chunk's index, the boundary at the end of every stage (by
    its last layer), the state of the whole chain and what the stages
    captured for ``capture``. Only the first stage gets the host's inputs.
    """
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == qwen3_5.CONTRACT
    assert all(a >= qwen3_5.FIRST_STAGE_MIN for a in bounds[1:-1])
    stages = [
        load_stage(FIXTURE / "model", first, last)
        for first, last in zip(bounds, bounds[1:])
    ]
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
