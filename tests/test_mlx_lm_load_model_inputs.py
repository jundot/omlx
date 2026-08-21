# SPDX-License-Identifier: Apache-2.0
"""``load_model`` must build from a caller's config and weights without ever
reaching the filesystem, and must behave exactly as mlx-lm does when neither
is supplied."""

import json
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_lm_load_model_inputs as patch_module
from omlx.patches.mlx_lm_load_model_inputs import (
    install_load_model_inputs,
    load_model,
    register_config_transform,
    register_quant_method,
)

DIMS = 64
QUANTIZATION = {"group_size": 32, "bits": 4, "mode": "affine"}


@dataclass
class TinyArgs:
    model_type: str = "tiny"
    dims: int = DIMS

    @classmethod
    def from_dict(cls, config):
        fields = {"model_type", "dims"}
        return cls(**{k: v for k, v in config.items() if k in fields})


class TinyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear = nn.Linear(args.dims, args.dims, bias=False)


def tiny_classes(config):
    return TinyModel, TinyArgs


def tiny_config(**overrides):
    return {"model_type": "tiny", "dims": DIMS, **overrides}


def tiny_weights():
    return {"linear.weight": mx.zeros((DIMS, DIMS), dtype=mx.float32)}


@pytest.fixture
def registries():
    """Restore the registration seams so a test cannot leak into the next."""
    transforms = dict(patch_module._CONFIG_TRANSFORMS)
    methods = dict(patch_module._QUANT_METHODS)
    yield
    patch_module._CONFIG_TRANSFORMS.clear()
    patch_module._CONFIG_TRANSFORMS.update(transforms)
    patch_module._QUANT_METHODS.clear()
    patch_module._QUANT_METHODS.update(methods)


class TestPreloadedInputs:
    """The in-memory path must not touch disk."""

    def test_weights_and_config_build_a_model_with_no_files_present(self, tmp_path):
        model, config = load_model(
            tmp_path,
            lazy=True,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights=tiny_weights(),
        )
        assert isinstance(model, TinyModel)
        assert config["model_type"] == "tiny"
        assert not list(tmp_path.iterdir())

    def test_model_path_is_optional_once_both_are_supplied(self):
        model, _ = load_model(
            None,
            lazy=True,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights=tiny_weights(),
        )
        assert isinstance(model, TinyModel)

    def test_supplied_weights_bind_to_the_model(self):
        weights = {"linear.weight": mx.full((DIMS, DIMS), 3.0, dtype=mx.float32)}
        model, _ = load_model(
            None,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights=weights,
        )
        assert mx.array_equal(model.linear.weight, weights["linear.weight"])

    def test_supplied_config_is_used_instead_of_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "on-disk"}))
        _, config = load_model(
            tmp_path,
            lazy=True,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights=tiny_weights(),
        )
        assert config["model_type"] == "tiny"

    def test_model_config_still_merges_over_a_supplied_config(self):
        model, config = load_model(
            None,
            lazy=True,
            model_config={"dims": 32},
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights={"linear.weight": mx.zeros((32, 32), dtype=mx.float32)},
        )
        assert config["dims"] == 32
        assert model.args.dims == 32

    def test_supplied_weights_drive_the_quantization_predicate(self):
        """The predicate must read the in-memory dict, not a disk glob."""
        weights = {
            "linear.scales": mx.zeros((DIMS, DIMS // 32), dtype=mx.float32),
        }
        model, _ = load_model(
            None,
            lazy=True,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(quantization=dict(QUANTIZATION)),
            weights=weights,
        )
        assert isinstance(model.linear, nn.QuantizedLinear)

    def test_absent_scales_leave_the_layer_unquantized(self):
        model, _ = load_model(
            None,
            lazy=True,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(quantization=dict(QUANTIZATION)),
            weights=tiny_weights(),
        )
        assert not isinstance(model.linear, nn.QuantizedLinear)


class TestEmptyWeightsAreNotAbsentWeights:
    """``weights={}`` says 'nothing to bind', not 'go and look on disk'."""

    def test_empty_dict_does_not_fall_back_to_disk(self, tmp_path):
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"),
            {"linear.weight": mx.full((DIMS, DIMS), 7.0, dtype=mx.float32)},
        )
        model, _ = load_model(
            tmp_path,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights={},
        )
        assert not mx.array_equal(
            model.linear.weight, mx.full((DIMS, DIMS), 7.0, dtype=mx.float32)
        )

    def test_empty_dict_does_not_raise_the_missing_safetensors_error(self, tmp_path):
        model, _ = load_model(
            tmp_path,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights={},
        )
        assert isinstance(model, TinyModel)


class TestMissingInputsAreReported:
    """A path that still needs disk must say which half it is missing."""

    def test_no_path_and_no_config_names_the_config(self):
        with pytest.raises(ValueError, match="config.json"):
            load_model(None, weights=tiny_weights(), get_model_classes=tiny_classes)

    def test_no_path_and_no_weights_names_the_safetensors(self):
        with pytest.raises(ValueError, match="safetensors"):
            load_model(None, config=tiny_config(), get_model_classes=tiny_classes)

    def test_custom_model_file_without_a_path_is_refused(self):
        with pytest.raises(ValueError, match="no model_path"):
            load_model(
                None,
                trust_remote_code=True,
                config=tiny_config(model_file="custom_arch.py"),
                weights=tiny_weights(),
            )

    def test_custom_model_file_is_refused_before_weights_are_read(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "tiny", "model_file": "custom_arch.py"})
        )
        # Unreadable as safetensors: opening it would raise something other
        # than the refusal this test is about.
        (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")
        with pytest.raises(ValueError, match="trust_remote_code=True"):
            load_model(tmp_path, lazy=True, get_model_classes=tiny_classes)


class TestDiskPathIsUnchanged:
    """Supplying neither parameter must reproduce mlx-lm exactly."""

    def test_weights_are_read_from_safetensors(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"),
            {"linear.weight": mx.full((DIMS, DIMS), 5.0, dtype=mx.float32)},
        )
        model, config = load_model(tmp_path, get_model_classes=tiny_classes)
        assert config["model_type"] == "tiny"
        assert mx.array_equal(
            model.linear.weight, mx.full((DIMS, DIMS), 5.0, dtype=mx.float32)
        )

    def test_strict_load_with_no_safetensors_still_raises(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
        with pytest.raises(FileNotFoundError, match="No safetensors found"):
            load_model(tmp_path, get_model_classes=tiny_classes)

    def test_the_default_model_class_resolver_is_mlx_lm(self, tmp_path):
        """``get_model_classes=None`` resolves ``mlx_lm.utils._get_classes``."""
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "not-a-real-architecture"})
        )
        with pytest.raises(ValueError, match="not supported"):
            load_model(tmp_path, weights={})


class TestKeywordOnly:
    """PR 1002 made these positional, which silently captures other arguments."""

    def test_config_and_weights_cannot_be_passed_positionally(self, tmp_path):
        with pytest.raises(TypeError):
            load_model(
                tmp_path,
                False,
                True,
                None,
                tiny_classes,
                False,
                tiny_config(),
                tiny_weights(),
            )


class TestRegistrationSeams:
    """Architecture patches reach both paths through one registration."""

    def test_a_config_transform_runs_on_a_supplied_config(self, registries):
        register_config_transform("test_marker", lambda c: c.update(marker=True))
        _, config = load_model(
            None,
            lazy=True,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights=tiny_weights(),
        )
        assert config["marker"] is True

    def test_a_config_transform_runs_on_a_config_read_from_disk(
        self, tmp_path, registries
    ):
        (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
        register_config_transform("test_marker", lambda c: c.update(marker=True))
        _, config = load_model(
            tmp_path, lazy=True, strict=False, get_model_classes=tiny_classes
        )
        assert config["marker"] is True

    def test_a_transform_can_steer_model_construction(self, registries):
        register_config_transform("test_dims", lambda c: c.update(dims=32))
        model, _ = load_model(
            None,
            lazy=True,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights={},
        )
        assert model.args.dims == 32

    def test_re_registering_a_name_replaces_it(self, registries):
        register_config_transform("test_marker", lambda c: c.update(marker="first"))
        register_config_transform("test_marker", lambda c: c.update(marker="second"))
        _, config = load_model(
            None,
            lazy=True,
            get_model_classes=tiny_classes,
            config=tiny_config(),
            weights=tiny_weights(),
        )
        assert config["marker"] == "second"

    def test_an_unknown_quant_method_reaches_its_handler(self, registries):
        register_quant_method("chunked-fp8", lambda model, config: dict(QUANTIZATION))
        model, config = load_model(
            None,
            lazy=True,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(
                quantization_config={"quant_method": "chunked-fp8"},
            ),
            weights={"linear.scales": mx.zeros((DIMS, DIMS // 32), dtype=mx.float32)},
        )
        assert isinstance(model.linear, nn.QuantizedLinear)
        assert config["quantization"] == QUANTIZATION

    def test_a_handler_declining_leaves_the_model_unquantized(self, registries):
        register_quant_method("chunked-fp8", lambda model, config: None)
        model, config = load_model(
            None,
            lazy=True,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(
                quantization_config={"quant_method": "chunked-fp8"},
            ),
            weights={"linear.scales": mx.zeros((DIMS, DIMS // 32), dtype=mx.float32)},
        )
        assert not isinstance(model.linear, nn.QuantizedLinear)
        assert "quantization" not in config

    def test_an_unregistered_quant_method_is_ignored_as_upstream_does(self):
        model, config = load_model(
            None,
            lazy=True,
            strict=False,
            get_model_classes=tiny_classes,
            config=tiny_config(quantization_config={"quant_method": "invented"}),
            weights=tiny_weights(),
        )
        assert not isinstance(model.linear, nn.QuantizedLinear)
        assert "quantization" not in config


class TestInstall:
    def test_install_binds_mlx_lm_utils_and_is_idempotent(self):
        import mlx_lm.utils as utils_mod

        install_load_model_inputs()
        assert utils_mod.load_model is load_model
        assert install_load_model_inputs() is False

    def test_the_pre_load_dispatch_installs_it(self, tmp_path):
        import mlx_lm.utils as utils_mod

        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        maybe_apply_pre_load_patches(str(tmp_path))
        assert utils_mod.load_model is load_model
