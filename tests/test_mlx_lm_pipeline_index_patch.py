# SPDX-License-Identifier: Apache-2.0
"""Pipeline shard selection must not reject parameters the loader tolerates."""

import io
import json
import struct

from omlx.patches.mlx_lm_pipeline_index import (
    TolerantWeightMap,
    apply_mlx_lm_pipeline_index_patch,
    is_applied,
)


def _upstream_loop(weight_index, parameters):
    """The exact guard from mlx_lm/utils.py:586, so the test pins real behaviour."""

    local_files = set()
    for k in parameters:
        if file_name := weight_index.get(k, None) is None:  # noqa: F841
            raise ValueError(
                "Pipeline loading is only supported for MLX converted models."
            )
        local_files.add(weight_index[k])
    return local_files


def test_the_unpatched_guard_rejects_a_missing_parameter():
    """Establish the failure we are fixing, so the fix is demonstrably needed."""

    index = {"model.layers.0.q.weight": "shard-1.safetensors"}
    params = ["model.layers.0.q.weight", "model.layers.0.self_attn.indexer.wk.weight"]

    try:
        _upstream_loop(index, params)
        raise AssertionError("expected the upstream guard to reject this")
    except ValueError as exc:
        assert "MLX converted models" in str(exc)


def test_a_tolerant_map_lets_the_same_loop_through():
    index = TolerantWeightMap({"model.layers.0.q.weight": "shard-1.safetensors"})
    params = ["model.layers.0.q.weight", "model.layers.0.self_attn.indexer.wk.weight"]

    files = _upstream_loop(index, params)

    assert "shard-1.safetensors" in files
    assert None not in files, "a None would break the download pattern list"
    assert all(isinstance(f, str) for f in files)


def test_present_parameters_still_map_to_their_real_shard():
    """The fix must not change where existing weights are loaded from."""

    index = TolerantWeightMap({"a": "shard-1.safetensors", "b": "shard-2.safetensors"})
    assert index["a"] == "shard-1.safetensors"
    assert index["b"] == "shard-2.safetensors"
    assert index.get("a") == "shard-1.safetensors"


def test_an_empty_index_still_yields_a_usable_name():
    assert isinstance(TolerantWeightMap({})["anything"], str)


def test_the_patch_only_touches_safetensors_indexes():
    """config.json and every other json read in that module must be unaffected."""

    from omlx.patches.mlx_lm_pipeline_index import _JsonProxy

    proxy = _JsonProxy()
    config = proxy.load(io.StringIO(json.dumps({"model_type": "glm_moe_dsa"})))
    assert config == {"model_type": "glm_moe_dsa"}
    assert not isinstance(config, TolerantWeightMap)

    index = proxy.load(io.StringIO(json.dumps({"weight_map": {"a": "s.safetensors"}})))
    assert isinstance(index["weight_map"], TolerantWeightMap)

    # Everything else on the module still resolves.
    assert proxy.dumps({"x": 1}) == '{"x": 1}'


def test_applying_is_idempotent_and_reports_state():
    assert apply_mlx_lm_pipeline_index_patch() is True
    assert is_applied() is True
    assert apply_mlx_lm_pipeline_index_patch() is True

    from mlx_lm import utils as mlx_lm_utils

    # The module keeps working as a json provider after patching.
    assert mlx_lm_utils.json.loads('{"ok": true}') == {"ok": True}


def test_single_file_model_gets_an_in_memory_index(tmp_path):
    """A valid one-file export must not need a generated file on disk."""

    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
        "__metadata__": {"format": "mlx"},
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0\0"
    )
    missing_index = tmp_path / "model.safetensors.index.json"

    apply_mlx_lm_pipeline_index_patch()
    from mlx_lm import utils as mlx_lm_utils

    with mlx_lm_utils.open(missing_index, "r") as stream:
        index = mlx_lm_utils.json.load(stream)

    assert not missing_index.exists(), "compatibility must not mutate the model"
    assert index["weight_map"]["model.layers.0.self_attn.q_proj.weight"] == (
        "model.safetensors"
    )
    assert isinstance(index["weight_map"], TolerantWeightMap)
