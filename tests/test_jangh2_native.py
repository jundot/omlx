"""Native JANGH2 dispatch, explicit storage and strict-load contracts."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from omlx.models.jangh2 import is_jangh2_config
from omlx.models.jangh2.loader import (
    LoadOptions,
    _model_config,
    _runtime_config,
    compare_parameters,
    recenter_residual_norms,
)


def test_format_recognition_does_not_require_quant_method():
    assert is_jangh2_config({"jangtq": {"version": 2}})
    assert is_jangh2_config({"jang_config": {"format": "jangtq2"}})
    assert is_jangh2_config({"quantization": {"expert": {"mode": "jangtq2"}}})
    assert not is_jangh2_config(
        {"model_type": "qwen4_exp", "quantization": {"bits": 4}}
    )
    # Incomplete declarations belong to the strict loader, not affine fallback.
    assert is_jangh2_config({"jangtq": None})


def test_native_options_ignore_environment(monkeypatch):
    monkeypatch.setenv("OMLX_JANGH_METAL_GATE_ROWS", "invalid")
    monkeypatch.setenv("OMLX_JANGH_PREFILL", "grouped")
    options = LoadOptions()
    assert options.backend == "metal"
    assert options.performance.gate_rows == 4
    assert options.performance.prefill == "legacy"
    assert options.performance.ple_device_mib == 0


def test_native_metadata_and_options_do_not_import_legacy_or_gpu_runtime():
    script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jangh2_bridge', 'mlx', 'mlx_lm', 'mlx_vlm'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from omlx.models.jangh2 import is_jangh2_config
from omlx.models.jangh2.loader import LoadOptions
assert is_jangh2_config({'jangtq': {'version': 2}})
assert LoadOptions().backend == 'metal'
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_native_dispatch_precedes_quant_method_gate(tmp_path, monkeypatch):
    from omlx.models.jangh2 import loader
    from omlx.utils.model_loading import maybe_load_custom_quantization

    (tmp_path / "config.json").write_text(json.dumps({"jangtq": {"version": 2}}))
    sentinel = (object(), object())
    native = Mock(return_value=sentinel)
    monkeypatch.setattr(loader, "load", native)
    assert (
        maybe_load_custom_quantization(
            str(tmp_path), is_vlm=True, trust_remote_code=True
        )
        is sentinel
    )
    native.assert_called_once_with(str(tmp_path), trust_remote_code=True)


def test_ordinary_qwen_keeps_normal_loader(tmp_path, monkeypatch):
    from omlx.models.jangh2 import loader
    from omlx.utils.model_loading import maybe_load_custom_quantization

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    native = Mock()
    monkeypatch.setattr(loader, "load", native)
    assert maybe_load_custom_quantization(str(tmp_path), is_vlm=True) is None
    native.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [SimpleNamespace(mtp_enabled=True), SimpleNamespace(vlm_mtp_enabled=True)],
)
def test_native_dispatch_rejects_unbound_mtp(tmp_path, settings):
    from omlx.utils.model_loading import maybe_load_custom_quantization

    (tmp_path / "config.json").write_text(json.dumps({"jangtq": {}}))
    with pytest.raises(ValueError, match="packed-expert MTP is not supported"):
        maybe_load_custom_quantization(
            str(tmp_path), is_vlm=True, model_settings=settings
        )


def test_native_dispatch_refuses_text_only_fallback(tmp_path):
    from omlx.utils.model_loading import maybe_load_custom_quantization

    (tmp_path / "config.json").write_text(json.dumps({"jangtq": {}}))
    with pytest.raises(ValueError, match="require the Qwen4 VLM engine"):
        maybe_load_custom_quantization(str(tmp_path), is_vlm=False)


def test_malformed_native_format_fails_before_model_construction(tmp_path, monkeypatch):
    from omlx.models.jangh2 import loader

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "jangtq": {}})
    )
    construct = Mock()
    monkeypatch.setattr(loader, "_load_model", construct)
    with pytest.raises(ValueError, match="Unsupported jangtq.version"):
        loader.load_model(tmp_path)
    construct.assert_not_called()


def _checkpoint(tmp_path):
    return SimpleNamespace(
        root=tmp_path,
        config={
            "model_type": "qwen4_exp",
            "jang_config": {"norm_convention": "runtime_plus1_applied"},
            "text_config": {"num_hidden_layers": 2, "ple_layer_ids": [2]},
        },
        weight_map={
            "language_model.layers.1.ple.ngram_embedding.shards.0.weight": "model.safetensors"
        },
    )


def test_model_local_ple_mode_prefix_and_mtp_do_not_mutate_source(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    original = copy.deepcopy(checkpoint.config)
    config = _model_config(checkpoint)
    assert config["text_config"]["ple_storage_mode"] == "mmap"
    assert config["text_config"]["ple_model_path"] == str(tmp_path)
    assert config["text_config"]["ple_tensor_prefixes"] == {
        "1": "language_model.layers.1.ple.ngram_embedding"
    }
    assert config["mtp_enabled"] is False
    assert checkpoint.config == original


def test_explicit_norm_metadata_is_required(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    checkpoint.config["jang_config"].clear()
    with pytest.raises(ValueError, match="explicit.*norm_convention"):
        _model_config(checkpoint)


def test_parameter_audit_rejects_missing_extra_and_wrong_shapes():
    expected = {"weight": np.empty((2, 3))}
    compare_parameters(expected, dict(expected))
    for supplied in (
        {},
        {"weight": np.empty((3, 2))},
        {**expected, "invented": np.empty(1)},
    ):
        with pytest.raises(ValueError, match="Strict JANGH2 parameter audit"):
            compare_parameters(expected, supplied)


def test_explicit_norm_recenter_preserves_unrelated_weights():
    class Residual:
        pass

    model = SimpleNamespace(named_modules=lambda: [("norm", Residual())])
    ordinary = np.array([0.7], dtype=np.float32)
    weights = {
        "norm.weight": np.array([1.0, 0.25, 1.5], dtype=np.float32),
        "ordinary": ordinary,
    }
    assert recenter_residual_norms(model, weights, Residual, np) == 1
    np.testing.assert_array_equal(weights["norm.weight"], [0.0, -0.75, 0.5])
    assert weights["ordinary"] is ordinary


def test_processor_failure_closes_loaded_model(tmp_path, monkeypatch):
    import mlx_vlm.utils as utils

    from omlx.models.jangh2 import loader

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=1), close=Mock())
    monkeypatch.setattr(
        loader, "inspect_checkpoint", lambda _: SimpleNamespace(root=tmp_path)
    )
    monkeypatch.setattr(loader, "_load_model", lambda *a, **k: model)
    monkeypatch.setattr(
        utils, "load_processor", Mock(side_effect=ValueError("processor failed"))
    )
    with pytest.raises(ValueError, match="processor failed"):
        loader.load(tmp_path)
    model.close.assert_called_once()


def test_legacy_shims_alias_the_shared_runtime(monkeypatch):
    import sys

    source_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(source_root / "contrib" / "jangh2"))
    for name in ("metadata", "metal_ops", "performance"):
        production = importlib.import_module(f"omlx.models.jangh2.{name}")
        diagnostic = importlib.import_module(f"jangh2_bridge.{name}")
        assert diagnostic is production
        assert sys.modules[f"jangh2_bridge.{name}"] is production


def test_generation_sidecar_defaults_match_ordinary_vlm_loader(tmp_path):
    import mlx_vlm.utils as utils

    checkpoint = _checkpoint(tmp_path)
    checkpoint.config.update(eos_token_id=1, temperature=0.2, top_p=0.4, top_k=1)
    original = copy.deepcopy(checkpoint.config)
    (tmp_path / "config.json").write_text(json.dumps(checkpoint.config))
    generation = {
        "eos_token_id": [248046, 248044],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "do_sample": True,
    }
    (tmp_path / "generation_config.json").write_text(json.dumps(generation))

    config = _runtime_config(checkpoint, utils)
    runtime = utils.apply_generation_config_defaults(SimpleNamespace(), config)

    for key, value in generation.items():
        assert config[key] == value
        assert getattr(runtime, key) == value
    assert config["generation_config"] == generation
    assert config["text_config"]["ple_storage_mode"] == "mmap"
    assert config["mtp_enabled"] is False
    assert checkpoint.config == original


def test_generation_merge_cannot_replace_audited_model_structure(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    checkpoint.config["quantization"] = {"bits": 8, "group_size": 32}
    original = copy.deepcopy(checkpoint.config)
    defaults = {
        "eos_token_id": [2, 3],
        "model_type": "other",
        "quantization": {"bits": 2},
        "jang_config": {},
        "text_config": {"ple_storage_mode": "resident"},
        "mtp_enabled": True,
    }
    utils = SimpleNamespace(
        load_config=Mock(return_value=defaults),
        GENERATION_CONFIG_DEFAULT_KEYS=("eos_token_id",),
    )

    config = _runtime_config(checkpoint, utils)

    utils.load_config.assert_called_once_with(tmp_path)
    assert config["eos_token_id"] == [2, 3]
    assert config["model_type"] == "qwen4_exp"
    assert config["quantization"] == original["quantization"]
    assert config["jang_config"] == original["jang_config"]
    assert config["text_config"]["ple_storage_mode"] == "mmap"
    assert config["mtp_enabled"] is False
    assert checkpoint.config == original
