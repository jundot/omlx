# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper.families import deepseek_v41  # noqa: E402
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
