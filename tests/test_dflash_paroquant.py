# SPDX-License-Identifier: Apache-2.0
"""ParoQuant loading, eligibility, and rotation/cache correctness contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omlx.patches.dflash_paroquant import (
    is_paroquant_config,
    load_target_bundle,
    paroquant_dflash_compatibility,
    validate_paroquant_draft,
)


def config():
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_experts": 0,
        },
        "quantization_config": {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 128,
            "krot": 8,
        },
    }


def test_compatibility_and_case_normalization():
    cfg = config()
    cfg["quantization_config"]["quant_method"] = "ParoQuant"
    assert is_paroquant_config(cfg)
    assert paroquant_dflash_compatibility(cfg) == (True, "")
    assert not is_paroquant_config({})


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_experts", 8),
        ("hidden_size", 4096),
        ("num_hidden_layers", 32),
        ("vocab_size", 100),
    ],
)
def test_reject_other_target_layouts(field, value):
    cfg = config()
    cfg["text_config"][field] = value
    assert not paroquant_dflash_compatibility(cfg)[0]


@pytest.mark.parametrize("field,value", [("bits", 8), ("group_size", 64), ("krot", 4)])
def test_reject_other_quantization_layouts(field, value):
    cfg = config()
    cfg["quantization_config"][field] = value
    assert not paroquant_dflash_compatibility(cfg)[0]


def test_standard_loader_keeps_options(tmp_path, monkeypatch):
    from dflash_mlx.runtime import loading

    original = Mock(return_value=object())
    monkeypatch.setattr(loading, "load_target_bundle", original)
    assert (
        load_target_bundle(tmp_path, lazy=False, quantize_kv_cache=True)
        is original.return_value
    )
    original.assert_called_once_with(tmp_path, lazy=False, quantize_kv_cache=True)


def test_paroquant_loader_preserves_rotations_and_disables_verify_rewrites(
    tmp_path, monkeypatch
):
    loader = pytest.importorskip("paroquant.inference.backends.mlx.load")
    from dflash_mlx import verify_linear
    from dflash_mlx.engine import target_ops
    from dflash_mlx.runtime import loading

    (tmp_path / "config.json").write_text(json.dumps(config()))
    model, tokenizer = object(), object()
    load = Mock(return_value=(model, tokenizer, False))
    ops = SimpleNamespace(
        install_speculative_hooks=Mock(), family=lambda _: "hybrid_gdn"
    )
    monkeypatch.setattr(loader, "load", load)
    monkeypatch.setattr(target_ops, "resolve_target_ops", lambda _: ops)
    standard = Mock(side_effect=AssertionError("standard loader used"))
    rewrite = Mock(side_effect=AssertionError("rotation rewrite attempted"))
    monkeypatch.setattr(loading, "load_target_bundle", standard)
    monkeypatch.setattr(verify_linear, "install_verify_linears", rewrite)
    bundle = load_target_bundle(tmp_path, verify_config=SimpleNamespace(mode="on"))
    load.assert_called_once_with(str(tmp_path), lazy=True, force_text=True)
    assert bundle.model is model and bundle.tokenizer is tokenizer
    assert bundle.meta["verify_linear_enabled"] is False
    ops.install_speculative_hooks.assert_called_once_with(model)
    standard.assert_not_called()
    rewrite.assert_not_called()


def test_draft_shape_and_capture_validation():
    target = {"quant_method": "paroquant", "config": config()}
    draft = {
        "config": {
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_target_layers": 64,
            "dflash_config": {"target_layer_ids": [5, 19, 33, 47, 61]},
        }
    }
    validate_paroquant_draft(target, draft)
    draft["config"]["dflash_config"]["target_layer_ids"] = [64]
    with pytest.raises(ValueError, match="capture layers"):
        validate_paroquant_draft(target, draft)
    validate_paroquant_draft({}, {})


def test_admin_and_engine_agree(tmp_path):
    from omlx.admin.routes import _dflash_compat_for_model
    from omlx.engine.dflash import is_dflash_compatible

    (tmp_path / "config.json").write_text(json.dumps(config()))
    assert is_dflash_compatible(tmp_path) == (True, "")
    assert _dflash_compat_for_model({"model_path": str(tmp_path)}) == (True, "")
    cfg = config()
    cfg["text_config"]["num_experts"] = 8
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert _dflash_compat_for_model(
        {"model_path": str(tmp_path)}
    ) == is_dflash_compatible(tmp_path)
    assert not is_dflash_compatible(tmp_path)[0]


@pytest.fixture
def rotated_qwen():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    paro = pytest.importorskip("paroquant.inference.backends.mlx.modules")
    from mlx.utils import tree_unflatten
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from omlx.patches.dflash_lifecycle import (
        install_dflash_lifecycle_wrap,
        restore_dflash_class_patches,
    )

    restore_dflash_class_patches()
    mx.random.seed(7)
    model = Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config=dict(
                model_type="qwen3_5_text",
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                vocab_size=128,
                linear_num_value_heads=4,
                linear_num_key_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
            ),
        )
    )
    replacements = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and not name.endswith("lm_head"):
            out_dim, in_dim = layer.weight.shape
            rotated = paro.RotateQuantizedLinear(
                in_dim, out_dim, bias=False, group_size=128, krot=8
            )
            rotated.weight, rotated.scales, rotated.biases = mx.quantize(
                layer.weight, group_size=128, bits=4
            )
            rotated.theta = mx.random.normal((8, in_dim // 2)) * 0.2
            rotated.pairs = mx.broadcast_to(
                mx.arange(in_dim, dtype=mx.int16) % 128, (8, in_dim)
            )
            rotated.channel_scales = mx.ones((1, in_dim)) * 0.9
            replacements.append((name, rotated))
    model.update_modules(tree_unflatten(replacements))
    model.eval()
    mx.eval(model.parameters())
    install_dflash_lifecycle_wrap()
    yield model
    restore_dflash_class_patches()


def test_rotated_target_hidden_capture_matches_unpatched_forward(rotated_qwen):
    import mlx.core as mx
    from dflash_mlx.engine.target_ops import resolve_target_ops

    model = rotated_qwen
    ids = mx.array([[1, 2, 3, 4]])
    expected = model(ids, cache=model.make_cache())
    mx.eval(expected)
    ops = resolve_target_ops(model)
    ops.install_speculative_hooks(model)
    actual, hidden = ops.forward_with_hidden_capture(
        model,
        input_ids=ids,
        cache=ops.make_cache(model, enable_speculative_linear_cache=True),
        capture_layer_ids={1, 4},
    )
    mx.eval(actual)
    assert set(hidden) == {1, 4}
    assert float(mx.max(mx.abs(actual - expected))) < 1e-4


@pytest.mark.parametrize("accepted", range(5))
def test_rotated_target_rejection_restores_recurrent_and_attention_state(
    rotated_qwen, accepted
):
    import mlx.core as mx
    from dflash_mlx.engine.target_ops import resolve_target_ops

    model = rotated_qwen
    prefix, proposed, next_token = [1, 2, 3], [4, 5, 6, 7, 8], [9]
    expected = model(
        mx.array([prefix + proposed[: accepted + 1] + next_token]),
        cache=model.make_cache(),
    )[:, -1:, :]
    mx.eval(expected)
    ops = resolve_target_ops(model)
    ops.install_speculative_hooks(model)
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    out, _ = ops.forward_with_hidden_capture(
        model, input_ids=mx.array([prefix]), cache=cache
    )
    mx.eval(out)
    ops.arm_rollback(cache, prefix_len=len(prefix))
    out, _ = ops.verify_block(
        target_model=model,
        verify_ids=mx.array([proposed]),
        target_cache=cache,
        capture_layer_ids={1, 4},
    )
    mx.eval(out)
    ops.restore_after_acceptance(
        cache,
        target_len=len(prefix) + accepted + 1,
        acceptance_length=accepted,
        drafted_tokens=len(proposed) - 1,
    )
    actual, _ = ops.forward_with_hidden_capture(
        model, input_ids=mx.array([next_token]), cache=cache
    )
    mx.eval(actual)
    assert float(mx.max(mx.abs(actual - expected))) < 1e-4


def test_unsupported_paroquant_never_reaches_standard_loader(tmp_path, monkeypatch):
    from dflash_mlx.runtime import loading

    cfg = config()
    cfg["text_config"]["num_experts"] = 8
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    standard = Mock(side_effect=AssertionError("standard loader used"))
    monkeypatch.setattr(loading, "load_target_bundle", standard)
    with pytest.raises(ValueError, match="dense Qwen3.8"):
        load_target_bundle(tmp_path)
    standard.assert_not_called()


def test_missing_paroquant_reports_install_hint(tmp_path, monkeypatch):
    import sys

    (tmp_path / "config.json").write_text(json.dumps(config()))
    monkeypatch.setitem(sys.modules, "paroquant.inference.backends.mlx.load", None)
    with pytest.raises(ImportError, match=r"omlx\[paroquant\]"):
        load_target_bundle(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("hidden_size", 4096), ("vocab_size", 100), ("num_target_layers", 32)],
)
def test_draft_dimension_mismatch(field, value):
    draft = {
        "config": {
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_target_layers": 64,
            "dflash_config": {"target_layer_ids": [5, 19, 33, 47, 61]},
        }
    }
    draft["config"][field] = value
    with pytest.raises(ValueError, match="dimensions"):
        validate_paroquant_draft(
            {"quant_method": "paroquant", "config": config()}, draft
        )


def test_unregistered_text_architecture_is_not_advertised():
    cfg = config()
    cfg["model_type"] = "qwen3_5_text"
    assert not paroquant_dflash_compatibility(cfg)[0]
