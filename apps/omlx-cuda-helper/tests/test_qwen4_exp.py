# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash-Next stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper.families import qwen4_exp  # noqa: E402
from omlx_cuda_helper.families.qwen4_exp.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/qwen4_exp"
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


@pytest.mark.parametrize(
    "bounds",
    [(0, 3), (0, 2, 3), (0, 4)],
    ids=["one-stage", "two-stages", "fewer-layers-than-held"],
)
def test_stages_reproduce_the_contract_fixture(bounds):
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == qwen4_exp.CONTRACT
    assert all(a >= qwen4_exp.FIRST_STAGE_MIN for a in bounds[1:-1])
    stages = [
        load_stage(FIXTURE / "model", first, last)
        for first, last in zip(bounds, bounds[1:])
    ]
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage in stages:
        # On CUDA the replay then runs through the self-checked kernels.
        stage.enable_kernels()
    for stage, job in zip(stages, jobs):
        stage.load_prefix(job, _group(arrays, "p:"))

    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start = 0
    for index, size in enumerate(sizes[skip:]):
        boundary, state = None, {}
        for stage, job in zip(stages, jobs):
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs=_group(arrays, f"in{index}:"),
                following=0,
                capture=None,
            )
            assert all(stage.first <= int(n.split(".")[0]) < stage.last for n in own)
            state.update(own)
        start += size
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
                if data.dtype.kind == "f":
                    np.testing.assert_allclose(
                        data,
                        expected,
                        rtol=meta["rtol"],
                        atol=meta["atol"],
                        err_msg=tag + name,
                    )
                else:
                    np.testing.assert_array_equal(data, expected, err_msg=tag + name)


def test_plan_counts_only_what_a_stage_loads():
    shared, layers = qwen4_exp.plan(FIXTURE / "model")
    assert len(layers) == 4 and min(layers) > 0
    # The embedding is shared; the output head, the mixer and the PLE table
    # stay on the host.
    assert shared == 64 * 32 * 4
    assert not qwen4_exp.resident(
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"
    )
