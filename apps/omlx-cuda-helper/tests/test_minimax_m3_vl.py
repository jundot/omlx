# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 stages must reproduce the oMLX contract fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx_cuda_helper import families  # noqa: E402
from omlx_cuda_helper.families import minimax_m3_vl  # noqa: E402
from omlx_cuda_helper.families.minimax_m3_vl.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/minimax_m3_vl"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)
MOE = "language_model.model.layers.{}.block_sparse_moe."


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
    for stage in stages:
        # A layer loaded again after a long prompt runs as at the first load.
        stage.drop_layer()
        stage.add_layer()
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage, job in zip(stages, jobs):
        stage.load_prefix(job, _group(arrays, "p:"))
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
                spec = (dtype, data.dtype, data.shape)
                assert spec == (expected_dtype, expected.dtype, expected.shape), name
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
    assert _case()[0]["contract"] == minimax_m3_vl.CONTRACT
    assert families.contracts()["minimax_m3_vl"] == minimax_m3_vl.CONTRACT
    assert all(a >= minimax_m3_vl.FIRST_STAGE_MIN for a in bounds[1:-1])
    _assert_fixture(_replay(FIXTURE / "model", bounds))


def test_captured_layers_are_the_layer_outputs_from_the_wanted_row():
    # Layer 2 ends the helper and layer 0 ends a first stage cut at 1, so the
    # boundaries of those runs are the expected captures. Row 8 of the job is
    # row 2 of the second chunk.
    capture = {"layers": [0, 2], "from": 8}
    chunks = _replay(FIXTURE / "model", (0, 1, 3), capture)
    first = _replay(FIXTURE / "model", (0, 1))
    assert not [n for n in chunks[0][0] if n.startswith("cap.")]
    boundary = chunks[1][0]
    np.testing.assert_array_equal(boundary["cap.2"][0], boundary["h"][0][:, 2:])
    np.testing.assert_array_equal(boundary["cap.0"][0], first[1][0]["h"][0][:, 2:])


def _fold_scales(weights, layer):
    """Multiply each expert's global scale into its dense weights."""
    switch = MOE.format(layer) + "switch_mlp."
    for name in ("gate", "up", "down"):
        scale = weights.pop(f"{switch}{name}_scale")[:, None, None]
        weights[f"{switch}{name}_proj.weight"] = (
            weights[f"{switch}{name}_proj.weight"] * scale
        )


def _global_scale_names(weights, text):
    """Expert scales as mlx-lm writes them next to an NVFP4 bank."""
    for layer in (1, 2, 3):
        switch = MOE.format(layer) + "switch_mlp."
        for name in ("gate", "up", "down"):
            scale = weights.pop(f"{switch}{name}_scale")
            weights[f"{switch}{name}_proj.global_scale"] = scale * 2688.0


def _unpacked_file_packed_config(weights, text):
    for layer in (1, 2, 3):
        _fold_scales(weights, layer)
    text.update(pack_shared_expert=None, routed_expert_scales=False)


def _packed_bank(weights, text):
    """The shared expert as the last row of a fused gate/up bank."""
    _unpacked_file_packed_config(weights, text)
    for layer in (1, 2, 3):
        moe = MOE.format(layer)
        routed = {
            name: weights.pop(f"{moe}switch_mlp.{name}_proj.weight")
            for name in ("gate", "up", "down")
        }
        shared = {
            name: weights.pop(f"{moe}shared_experts.{name}_proj.weight")[None]
            for name in ("gate", "up", "down")
        }
        weights[f"{moe}switch_mlp.gate_up_proj.weight"] = mx.concatenate(
            [
                mx.concatenate([routed["gate"], routed["up"]], axis=1),
                mx.concatenate([shared["gate"], shared["up"]], axis=1),
            ]
        )
        weights[f"{moe}switch_mlp.down_proj.weight"] = mx.concatenate(
            [routed["down"], shared["down"]]
        )


@pytest.mark.parametrize(
    "layout", [_global_scale_names, _unpacked_file_packed_config, _packed_bank]
)
def test_other_expert_layouts_of_the_same_weights_give_the_same_arrays(
    tmp_path, layout
):
    weights = mx.load(str(FIXTURE / "model" / "model.safetensors"))
    config = json.loads((FIXTURE / "model" / "config.json").read_text())
    layout(weights, config["text_config"])
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    _assert_fixture(_replay(tmp_path, (0, 3)))


def _quantized(source, folder, dense):
    """The fixture model with the real checkpoint's quantization modes, or
    (``dense``) the float weights those quantized tensors stand for."""
    modes = {
        "embed_tokens": (64, 8, "affine"),
        "switch_mlp": (16, 4, "nvfp4"),
        "shared_experts": (32, 8, "mxfp8"),
        "index_": (32, 8, "mxfp8"),
        "_proj": (64, 5, "affine"),
    }
    weights = mx.load(str(source / "model.safetensors"))
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    for name in [n for n in weights if n.endswith(".weight")]:
        module = name[: -len(".weight")]
        match = next((m for m in modes if m in module), None)
        if match is None or weights[name].ndim < 2 or module.endswith(".gate"):
            continue
        group, bits, mode = modes[match]
        packed, scales, *biases = mx.quantize(weights[name], group, bits, mode=mode)
        if dense:
            weights[name] = mx.dequantize(
                packed, scales, *biases, group_size=group, bits=bits, mode=mode
            ).astype(mx.float32)
            continue
        weights[name], weights[f"{module}.scales"] = packed, scales
        if biases:
            weights[f"{module}.biases"] = biases[0]
        config["quantization"][module] = {
            "group_size": group,
            "bits": bits,
            "mode": mode,
        }
    folder.mkdir()
    mx.save_safetensors(str(folder / "model.safetensors"), weights)
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def test_quantized_weights_run_like_their_dense_values(tmp_path):
    packed = _replay(_quantized(FIXTURE / "model", tmp_path / "q", False), (0, 3))
    dense = _replay(_quantized(FIXTURE / "model", tmp_path / "d", True), (0, 3))
    for (boundary, state), (want_boundary, want_state) in zip(packed, dense):
        for name, (data, _) in {**boundary, **state}.items():
            expected = {**want_boundary, **want_state}[name][0]
            np.testing.assert_allclose(
                data, expected, rtol=1e-4, atol=1e-5, err_msg=name
            )


def test_plan_counts_what_a_stage_holds(tmp_path):
    shared, layers = minimax_m3_vl.plan(FIXTURE / "model")
    assert len(layers) == 4 and min(layers) > 0
    # The embedding is shared; the vocabulary head and final norm stay behind.
    assert shared == 64 * 64 * 4
    assert not minimax_m3_vl.resident("language_model.lm_head.weight")
    # A quantized linear is unpacked to bfloat16 and an expert bank stays packed.
    model = _quantized(FIXTURE / "model", tmp_path / "q", False)
    packed_shared, packed_layers = minimax_m3_vl.plan(model)
    assert packed_shared == 64 * (64 + 2 * 4)
    dense_layer = 2 * (64 * (64 + 32 + 32 + 64 + 32 + 16) + 3 * 64 * 128)
    assert packed_layers[0] == dense_layer + (4 * 16 + 2 * 64) * 4
