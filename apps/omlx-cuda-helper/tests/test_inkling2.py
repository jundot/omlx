# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper.families import inkling2  # noqa: E402
from omlx_cuda_helper.families.inkling2 import kernels  # noqa: E402
from omlx_cuda_helper.families.inkling2.layers import Config  # noqa: E402
from omlx_cuda_helper.families.inkling2.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/inkling2"
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
    """The fixture's chunks through stages cut at ``bounds``. Yields each
    chunk's index, the boundary at the end of the job's layers, the state of
    the whole chain and what the stages captured for ``capture``."""
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == inkling2.CONTRACT
    stages = [load_stage(model, a, b) for a, b in zip(bounds, bounds[1:])]
    for stage in stages:
        # A layer loaded again after a long prompt runs as at the first load.
        stage.drop_layer()
        stage.add_layer()
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage, job in zip(stages, jobs):
        stage.enable_kernels()
        stage.load_prefix(job, _group(arrays, "p:"))
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start = 0
    for index, size in enumerate(sizes[skip:]):
        boundary, state, captured = None, {}, {}
        for stage, job in zip(stages, jobs):
            if not job:
                continue
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs={},
                following=0,
                capture=capture,
            )
            state.update(own)
            captured.update({n: boundary.pop(n) for n in list(boundary) if "cap." in n})
        start += size
        yield index, boundary, state, captured


def _assert_matches(got, want, meta, tag):
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
            data, expected, rtol=meta["rtol"], atol=meta["atol"], err_msg=tag + name
        )


@pytest.mark.parametrize(
    "bounds",
    [(0, 7), (0, 1, 7), (0, 5, 7), (0, 6, 7), (0, 3, 6, 7), (0, 8)],
    ids=["one-stage", "cut-1", "cut-5", "cut-6", "cuts-3-6", "fewer-than-held"],
)
def test_stages_reproduce_the_contract_fixture(bounds):
    meta, arrays = _case()
    for index, boundary, state, _ in _replay(bounds):
        _assert_matches(boundary, _group(arrays, f"b{index}:"), meta, f"b{index}:")
        _assert_matches(state, _group(arrays, f"s{index}:"), meta, f"s{index}:")


def test_a_stage_returns_the_partial_sums_a_block_drafter_reads():
    # The window starts at job row 8, inside the second chunk (rows 6..12).
    # Layer 5 runs on the second stage; a stage that ends at 5 hands on the
    # partial sum entering it.
    chunks = list(_replay((0, 3, 7), {"layers": [1, 5], "from": 8}))
    assert [sorted(c[3]) for c in chunks] == [[], ["cap.1", "cap.5"]]
    _, cut, _, _ = list(_replay((0, 5)))[1]
    np.testing.assert_array_equal(chunks[1][3]["cap.5"][0], cut["h"][0][:, 2:])


def _source_layout(model: Path, out: Path) -> None:
    """The fixture checkpoint under the published names (``model.llm.*``)."""
    config = json.loads((model / "config.json").read_text())
    text = config["text_config"]
    weights = mx.load(str(model / "model.safetensors"))
    hidden, conv = text["hidden_size"], text["sconv_kernel_size"]
    local = set(text["local_layer_ids"])
    t = {}
    root = "model.llm."
    t[root + "embed.weight"] = weights["model.embed.weight"]
    t[root + "embed_norm.weight"] = weights["model.embed_norm.weight"]
    for i in range(text["num_hidden_layers"]):
        w = {
            n[len(f"model.layers.{i}.") :]: v
            for n, v in weights.items()
            if n.startswith(f"model.layers.{i}.")
        }
        p = f"{root}layers.{i}."
        heads, dim = text["num_attention_heads"], text["head_dim"]
        dq, dkv = heads * dim, text["num_key_value_heads"] * dim
        dr = heads * text["d_rel"]
        if i in local:
            parts = zip(
                ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj"),
                np.cumsum([0, dq, dkv, dkv, dr, dq, heads])[:-1],
                (dq, dkv, dkv, dr, dq, heads),
            )
            stacked = w["attn.in_proj.weight"]
            t[p + "attn.k_sconv.weight"] = w["attn.k_conv"].reshape(-1, 1, conv)
            t[p + "attn.v_sconv.weight"] = w["attn.v_conv"].reshape(-1, 1, conv)
            t[p + "attn_norm.weight"] = w["attn_norm.weight"]
        else:
            parts = zip(
                ("wq_du", "wr_du", "gate_attn", "attn_sink_proj"),
                np.cumsum([0, dq, dr, dq, heads])[:-1],
                (dq, dr, dq, heads),
            )
            stacked = w["attn.q_proj.weight"]
            for n in "kv":
                both = w[f"attn.{n}_proj.weight"]
                t[p + f"attn.w{n}_dv.weight"] = both[:dkv]
                t[p + f"attn.{n}_sconv.weight_proj"] = both[dkv:]
                t[p + f"attn.{n}_sconv.static_w"] = w[f"attn.{n}_static"]
            for n in "qkv":
                t[p + f"{n}_input_norm.weight"] = w[f"{n}_input_norm.weight"]
        for name, begin, size in parts:
            t[p + f"attn.{name}.weight"] = stacked[int(begin) : int(begin) + size]
        t[p + "attn.wo_ud.weight"] = w["attn.o_proj.weight"]
        t[p + "attn.q_l2_clip.scale"] = w["attn.q_scale"]
        t[p + "attn.rel_l2_clip.scale"] = w["attn.r_scale"]
        t[p + "attn.rel_logits_proj.proj"] = w["attn.rel_proj"]
        t[p + "attn_sconv.weight"] = w["attn_conv"].reshape(hidden, 1, conv)
        t[p + "mlp_sconv.weight"] = w["mlp_conv"].reshape(hidden, 1, conv)
        t[p + "mlp_norm.weight"] = w["mlp_norm.weight"]
        t[p + "mlp.gate.weight"] = w["mlp.router_weight"]
        t[p + "mlp.gate.bias"] = w["mlp.router_bias"]
        for which in ("w13", "w2"):
            base = f"mlp.experts.{which}"
            t[p + f"{base}_weight"] = w[f"{base}.weight"].view(mx.uint8)
            t[p + f"{base}_weight_scale"] = w[f"{base}.scales"]
            if f"{base}.global_scale" in w:
                # A denominator other than 6 * 448 takes the rescale path.
                t[p + f"{base}_weight_amax"] = w[f"{base}.global_scale"] * 0.5
                t[p + f"{base}_weight_outer_scale_denominator"] = mx.array(
                    [1344.0], dtype=mx.float32
                )
        if i > 0:
            t[f"{root}attn_res.attn_queries.{i - 1}"] = w["attn_input.query"]
        t[f"{root}attn_res.mlp_queries.{i}"] = w["mlp_input.query"]
    out.mkdir()
    mx.save_safetensors(str(out / "model.safetensors"), t)
    shutil.copy(model / "config.json", out / "config.json")


def test_the_published_layout_loads_as_the_runtime_layout(tmp_path):
    source = tmp_path / "source"
    _source_layout(FIXTURE / "model", source)
    runtime = list(_replay((0, 3, 7)))
    published = list(_replay((0, 3, 7), model=source))
    for (_, b1, s1, _), (_, b2, s2, _) in zip(runtime, published):
        for got, want in ((b2, b1), (s2, s1)):
            assert sorted(got) == sorted(want)
            for name in got:
                np.testing.assert_array_equal(got[name][0], want[name][0], name)


@pytest.mark.skipif(not mx.cuda.is_available(), reason="the kernels run on CUDA")
def test_every_expert_carrier_takes_an_expert_gemm():
    # The published checkpoint (NVFP4, MXFP8) and the affine codes oQ writes
    # below 4 bits, each checked against MLX's gather_qmm.
    kernels.enable(Config.from_text({}), mx.bfloat16)
    carriers = {f"experts:{c}" for c in ("nvfp4", "mxfp8", "affine2", "affine3")}
    assert carriers <= kernels._on
