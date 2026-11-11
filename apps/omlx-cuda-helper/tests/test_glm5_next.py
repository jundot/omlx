# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper.families import glm5_next  # noqa: E402
from omlx_cuda_helper.families.glm5_next.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/glm5_next"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)
LAYERS = "language_model.model.layers."


def _case():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {name: (data[name], dtype) for name, dtype, _, _ in meta["manifest"]}
    return meta, arrays


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


def _replay(model, bounds, capture=None):
    """The fixture prompt through chained stages: per chunk (boundary, state)."""
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    stages = [load_stage(model, a, b) for a, b in zip(bounds, bounds[1:])]
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage, job in zip(stages, jobs):
        held = range(stage.first, stage.first + len(job))
        prefix = _group(arrays, "p:")
        stage.load_prefix(
            job, {n: v for n, v in prefix.items() if int(n.split(".")[0]) in held}
        )
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start, chunks = 0, []
    for size in sizes[skip:]:
        boundary, state, captured = None, {}, {}
        for stage, job in zip(stages, jobs):
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary and {"h": boundary["h"]},
                inputs={},
                following=0,
                capture=capture,
            )
            assert all(stage.first <= int(n.split(".")[0]) < stage.last for n in own)
            state.update(own)
            captured.update({n: v for n, v in boundary.items() if n != "h"})
        start += size
        chunks.append(({**captured, "h": boundary["h"]}, state))
    return chunks


def _assert_fixture(chunks):
    meta, arrays = _case()
    for index, (boundary, state) in enumerate(chunks):
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


@pytest.mark.parametrize(
    "bounds",
    [(0, 3), (0, 1, 3), (0, 2, 3), (0, 4)],
    ids=["one-stage", "cut-1", "cut-2", "fewer-layers-than-held"],
)
def test_stages_reproduce_the_contract_fixture(bounds):
    assert _case()[0]["contract"] == glm5_next.CONTRACT
    assert all(a >= glm5_next.FIRST_STAGE_MIN for a in bounds[1:-1])
    _assert_fixture(_replay(FIXTURE / "model", bounds))


def test_captured_layers_are_the_stream_mean_from_the_wanted_row():
    # Layer 2 ends the helper and layer 0 ends a first stage cut at 1, so the
    # boundaries of those runs hold the captured layers. Row 8 of the job is
    # row 2 of the second chunk.
    capture = {"layers": [0, 2], "from": 8}
    chunks = _replay(FIXTURE / "model", (0, 1, 3), capture)
    first = _replay(FIXTURE / "model", (0, 1))
    assert not [n for n in chunks[0][0] if n.startswith("cap.")]
    boundary = chunks[1][0]
    for name, streams in (("cap.2", boundary["h"]), ("cap.0", first[1][0]["h"])):
        np.testing.assert_allclose(
            boundary[name][0], streams[0][:, 2:].mean(axis=2), rtol=1e-6, atol=1e-7
        )


def _hugging_face_layout(weights, config):
    """The fixture weights under the names and shapes of the original release."""
    out = {}
    for stored, value in weights.items():
        name = stored.replace("language_model.model.", "model.language_model.")
        name = name.replace(".attn_hc.", ".hc_attn_").replace(".ffn_hc.", ".hc_ffn_")
        name = name.replace(".forget_gate.", ".")
        if name.endswith(".conv1d.weight"):
            parts = mx.split(value.moveaxis(1, 2), 3, axis=0)
            for letter, part in zip("qkv", parts):
                out[name.replace("conv1d", f"{letter}_conv1d")] = part
        elif ".switch_mlp." in name:
            for expert in range(value.shape[0]):
                key = name.replace("switch_mlp.", f"experts.{expert}.")
                out[key] = value[expert]
        elif name.endswith(".unembed_out.weight"):
            keys = weights[stored.replace("unembed_out", "embed_q")]
            joint = mx.concatenate([keys.swapaxes(-1, -2), value], axis=1)
            out[name.replace("unembed_out", "kv_b_proj")] = joint.flatten(0, 1)
        elif not name.endswith(".embed_q.weight"):
            out[name] = value
    return out


def _quantized(weights, config):
    """The fixture weights packed like a real checkpoint: 8-bit linears and
    embedding, 4-bit expert banks, dense norms, gates and mixes."""
    config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    out = dict(weights)
    for name, value in weights.items():
        module = name[: -len(".weight")]
        dense = ("norm", ".mlp.gate.", "conv1d", "_hc.")
        if not name.endswith(".weight") or any(mark in name for mark in dense):
            continue
        if value.shape[-1] % 32:
            continue
        bits = 4 if ".switch_mlp." in name else 8
        out[name], out[f"{module}.scales"], out[f"{module}.biases"] = mx.quantize(
            value, 32, bits
        )
        if bits == 8:
            config["quantization"][module] = {"group_size": 32, "bits": 8}
    return out


def _unpacked(weights, config):
    """The float weights the tensors of ``_quantized`` stand for."""
    packed = _quantized(weights, config)
    table = config.pop("quantization")
    out = {}
    for name, value in packed.items():
        module, _, leaf = name.rpartition(".")
        if f"{module}.scales" not in packed:
            out[name] = value
        elif leaf == "weight":
            out[name] = mx.dequantize(
                value,
                packed[f"{module}.scales"],
                packed[f"{module}.biases"],
                group_size=32,
                bits=table.get(module, table)["bits"],
            )
    return out


def _variant(folder, layout):
    weights = mx.load(str(FIXTURE / "model" / "model.safetensors"))
    config = json.loads((FIXTURE / "model" / "config.json").read_text())
    folder.mkdir()
    mx.save_safetensors(str(folder / "model.safetensors"), layout(weights, config))
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def test_the_hugging_face_layout_gives_the_same_arrays(tmp_path):
    _assert_fixture(_replay(_variant(tmp_path / "hf", _hugging_face_layout), (0, 3)))


def test_quantized_weights_run_like_their_dense_values(tmp_path):
    packed = _replay(_variant(tmp_path / "q", _quantized), (0, 1, 3))
    dense = _replay(_variant(tmp_path / "d", _unpacked), (0, 1, 3))
    for (boundary, state), (want_boundary, want_state) in zip(packed, dense):
        for name, (data, _) in {**boundary, **state}.items():
            expected = {**want_boundary, **want_state}[name][0]
            np.testing.assert_allclose(
                data, expected, rtol=1e-4, atol=1e-5, err_msg=name
            )


def test_plan_counts_what_a_stage_holds(tmp_path):
    shared, layers = glm5_next.plan(FIXTURE / "model")
    stored = mx.load(str(FIXTURE / "model" / "model.safetensors"))
    assert shared == stored["language_model.model.embed_tokens.weight"].nbytes
    assert layers == [
        sum(v.nbytes for n, v in stored.items() if n.startswith(f"{LAYERS}{i}."))
        for i in range(4)
    ]
    assert not glm5_next.resident("language_model.lm_head.weight")
    assert not glm5_next.resident("language_model.model.norm.weight")
    # A packed linear is unpacked to its scales' dtype; an expert bank and the
    # embedding stay packed.
    packed_shared, packed_layers = glm5_next.plan(_variant(tmp_path / "q", _quantized))
    assert packed_shared == 64 * (64 + 2 * 2 * 4)
    bank = sum(
        v.nbytes for n, v in stored.items() if n.startswith(f"{LAYERS}1.mlp.switch_")
    )
    assert packed_layers[1] == layers[1] - bank + bank // 8 + bank // 16
