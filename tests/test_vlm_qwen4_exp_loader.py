# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen4-Exp multimodal admission in the mlx-vlm load path."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.utils.model_loading import (
    lightning_mtp_supported,
    maybe_apply_pre_load_patches,
)


def test_qwen4_exp_runtime_rejects_audio_only():
    engine = VLMBatchedEngine("qwen4")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.QWEN4_EXP_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="not audio"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "hello"}],
            images=[],
            audio=[("samples", 16000)],
        )


@pytest.mark.parametrize("symlinked", [False, True], ids=["plain", "hf-symlink"])
def test_qwen4_exp_mlx_metadata_is_hidden_only_for_model_shards(
    tmp_path, monkeypatch, symlinked
):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    weight_file = model_dir / "model.safetensors"
    if symlinked:
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        blob = blob_dir / "content-hash"
        blob.touch()
        weight_file.symlink_to(blob)
    else:
        weight_file.touch()
    outside_file = tmp_path / "outside.safetensors"
    outside_file.touch()

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return {"format": "mlx", "source": "test"}

    import safetensors

    def fake_safe_open(*_args, **_kwargs):
        return FakeHandle()

    monkeypatch.setattr(safetensors, "safe_open", fake_safe_open)

    with vlm_module._force_qwen4_exp_sanitize_on_load(model_dir):
        target_handle = safetensors.safe_open(weight_file)
        outside_handle = safetensors.safe_open(outside_file)
        assert target_handle.metadata() == {"source": "test"}
        assert outside_handle.metadata() == {"format": "mlx", "source": "test"}

    assert safetensors.safe_open is fake_safe_open


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "expected_lazy"),
    [("qwen4_exp", True), ("qwen2_vl", None)],
)
async def test_only_qwen4_exp_loader_defers_parameter_eval_to_materialize(
    tmp_path, monkeypatch, model_type, expected_lazy
):
    import mlx_vlm.utils as vlm_utils

    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    captured = {}

    def stop_after_load(model_name, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after load")

    monkeypatch.setattr(vlm_utils, "load", stop_after_load)
    monkeypatch.setattr(vlm_module, "_patch_video_processor_bug", lambda: None)
    monkeypatch.setattr(vlm_module, "_patch_torch_free_image_processor", lambda: None)
    monkeypatch.setattr(vlm_module, "apply_pixtral_torch_free_patch", lambda: None)
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *a, **k: None
    )
    monkeypatch.setattr(
        model_loading, "maybe_load_custom_quantization", lambda *a, **k: None
    )

    with pytest.raises(RuntimeError, match="stop after load"):
        await VLMBatchedEngine(model_name=str(tmp_path)).start()

    if expected_lazy is None:
        assert "lazy" not in captured
    else:
        assert captured["lazy"] is expected_lazy


def test_qwen4_exp_loader_enables_adaptive_depth_three_lightning_mtp(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(mtp_enabled=True, mtp_adaptive_max_depth=None)

    maybe_apply_pre_load_patches(str(tmp_path), settings, for_vlm=True)

    from mlx_vlm.models.qwen4_exp.language import get_mtp_runtime

    from omlx.patches.mlx_lm_mtp import get_mtp_depth, is_mtp_active

    assert get_mtp_runtime().enabled is True
    assert get_mtp_runtime().checkpoint_prefix == "mtp."
    assert get_mtp_depth() == 3
    assert is_mtp_active() is True

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(mtp_enabled=False),
        for_vlm=True,
    )
    assert get_mtp_runtime().enabled is False
    assert is_mtp_active() is False


def test_qwen4_exp_loader_uses_explicit_ple_ssd_offload_setting(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(mtp_enabled=False, qwen4_ple_ssd_offload=False),
        for_vlm=True,
    )
    from mlx_vlm.models.qwen4_exp.language import get_ple_runtime_mode

    assert get_ple_runtime_mode() == "resident"

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(mtp_enabled=False, qwen4_ple_ssd_offload=True),
        for_vlm=True,
    )
    assert get_ple_runtime_mode() == "mmap"


@pytest.fixture
def qwen4_checkpoint(tmp_path):
    """Tiny public-architecture checkpoint, with no downloaded model assets."""
    from mlx.utils import tree_flatten

    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp import Model
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpMTPModule,
        Qwen4ExpRMSNorm,
        configure_mtp_runtime,
        configure_ple_runtime,
    )

    from omlx.patches.mlx_lm_mtp import set_mtp_active

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    configure_mtp_runtime(tmp_path, enabled=False)
    configure_ple_runtime(tmp_path, mode="resident")

    def write(*, layout="sidecar", centered=True, quantized=True):
        # Eight base anchors exercise the checkpoint-wide centering decision.
        config.text_config.num_hidden_layers = 8
        config.text_config.layer_types *= 4
        config.quantization = {"group_size": 32, "bits": 4} if quantized else None
        model = Model(config)
        for _, module in model.named_modules():
            if isinstance(module, Qwen4ExpRMSNorm):
                module.weight = mx.full(
                    module.weight.shape, float(centered), dtype=mx.bfloat16
                )
        if quantized:
            embedding = model.language_model.model.embed_tokens
            model.language_model.model.embed_tokens = embedding.to_quantized(32, 4)
        root = {
            key: (
                value.astype(mx.bfloat16)
                if mx.issubdtype(value.dtype, mx.floating)
                else value
            )
            for key, value in tree_flatten(model.parameters())
        }
        head = Qwen4ExpMTPModule(config.text_config)
        head_weights = {
            key: mx.full(value.shape, (i + 1) / 64, dtype=mx.bfloat16)
            for i, (key, value) in enumerate(tree_flatten(head.parameters()))
        }
        for path, module in head.named_modules():
            if isinstance(module, Qwen4ExpRMSNorm):
                # Deliberately not near zero: MTP norms cannot decide centering.
                head_weights[path + ".weight"] = mx.full(
                    module.weight.shape, 1.25 + centered, dtype=mx.bfloat16
                )
        checkpoint_config = asdict(config)
        if not quantized:
            checkpoint_config.pop("quantization")
            checkpoint_config.pop("quantization_config")
        (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
        if layout == "embedded":
            root.update(("mtp." + key, value) for key, value in head_weights.items())
        else:
            sidecar_dir = tmp_path / "mtp"
            sidecar_dir.mkdir()
            (sidecar_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen4_exp_mtp",
                        "text_config": asdict(config.text_config),
                    }
                )
            )
            mx.save_safetensors(str(sidecar_dir / "model.safetensors"), head_weights)
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"), root, metadata={"format": "mlx"}
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors" for key in root}})
        )
        return tmp_path, head_weights

    try:
        yield write
    finally:
        configure_mtp_runtime(tmp_path, enabled=False)
        configure_ple_runtime(tmp_path, mode="resident")
        set_mtp_active(False)
        mx.set_default_device(previous_device)


def _load_qwen4_checkpoint(path, *, mtp_enabled=True):
    from mlx_vlm.utils import load_model

    maybe_apply_pre_load_patches(
        str(path),
        SimpleNamespace(mtp_enabled=mtp_enabled, qwen4_ple_ssd_offload=False),
        for_vlm=True,
    )
    with vlm_module._force_qwen4_exp_sanitize_on_load(path):
        model = load_model(path, lazy=True)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize(
    ("layout", "enabled", "centered"),
    [
        ("sidecar", True, True),
        ("sidecar", True, False),
        ("sidecar", False, True),
        ("embedded", True, True),
        ("embedded", False, True),
    ],
)
def test_qwen4_mtp_checkpoint_loads_values_and_precision(
    qwen4_checkpoint, layout, enabled, centered
):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

    from omlx.admin.routes import _mtp_compat_for_model
    from omlx.model_discovery import estimate_model_size
    from omlx.patches.mlx_lm_mtp import is_mtp_active
    from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import (
        qwen4_exp_residency_estimate,
    )

    path, head_weights = qwen4_checkpoint(layout=layout, centered=centered)
    original_files = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    checkpoint_bytes = sum(
        len(data) for p, data in original_files.items() if p.suffix == ".safetensors"
    )
    estimate = qwen4_exp_residency_estimate(path)
    assert estimate.checkpoint_bytes == checkpoint_bytes
    assert (
        estimate.resident_bytes
        == estimate_model_size(path)
        == int(checkpoint_bytes * 1.05)
    )
    assert _mtp_compat_for_model({"model_path": str(path)}) == (True, "")

    model = _load_qwen4_checkpoint(path, mtp_enabled=enabled)

    assert isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    assert model.language_model.model.embed_tokens.bits == 4
    base_norm = model.language_model.model.layers[0].attn_hyper_connection.hc_norm
    assert mx.array_equal(base_norm.weight, mx.zeros_like(base_norm.weight)).item()
    assert hasattr(model, "mtp") is enabled
    assert is_mtp_active() is enabled
    if enabled:
        assert model.language_model.get_mtp_module() is model.mtp
        norms = {
            name + ".weight"
            for name, module in model.mtp.named_modules()
            if isinstance(module, Qwen4ExpRMSNorm)
        }
        loaded_head = dict(tree_flatten(model.mtp.parameters()))
        assert loaded_head.keys() == head_weights.keys()
        for key, value in loaded_head.items():
            expected = (
                head_weights[key] - float(centered)
                if key in norms
                else head_weights[key]
            )
            assert value.dtype == mx.bfloat16, key
            assert mx.array_equal(value, expected).item(), key
        assert isinstance(model.mtp.fc_hidden, nn.Linear)
    assert lightning_mtp_supported(path)
    assert original_files == {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["sidecar", "embedded"])
async def test_qwen4_quantization_advertises_only_preservable_heads(
    qwen4_checkpoint, monkeypatch, layout
):
    from omlx.admin.oq_manager import OQManager

    path, _ = qwen4_checkpoint(layout=layout, quantized=False)
    manager = OQManager(model_dirs=[str(path.parent)])
    sources, _ = await manager.list_quantizable_models()
    source = next(model for model in sources if model["path"] == str(path))

    async def no_conversion(task_id):
        pass

    monkeypatch.setattr(manager, "_run_quantization", no_conversion)
    task = await manager.start_quantization(str(path), 4, preserve_mtp=True)
    await manager._active_tasks[task.task_id]

    preservable = layout == "embedded"
    suffix = "-mtp" if layout == "embedded" else ""
    assert (source["has_mtp_heads"], task.preserve_mtp, task.output_name) == (
        preservable,
        preservable,
        path.name + "-oQ4" + suffix,
    )


@pytest.mark.asyncio
async def test_qwen4_mtp_admin_toggle_accepts_standalone_head(qwen4_checkpoint):
    from omlx.admin.routes import ModelSettingsRequest
    from omlx.model_settings import ModelSettings
    from tests.test_admin_model_settings import _failed_pool, _update_settings

    path, _ = qwen4_checkpoint()
    pool, entry = _failed_pool()
    entry.model_path = str(path)
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()

    await _update_settings(pool, settings, ModelSettingsRequest(mtp_enabled=True))

    assert settings.mtp_enabled is True


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape"])
def test_qwen4_mtp_sidecar_keeps_strict_weight_validation(qwen4_checkpoint, damage):
    path, weights = qwen4_checkpoint()
    key = "fc_hidden.weight"
    if damage == "missing":
        weights.pop(key)
        message = r"Missing .*parameters"
    elif damage == "unexpected":
        weights["unexpected.weight"] = mx.zeros((1,), dtype=mx.bfloat16)
        key = "unexpected.weight"
        message = r"parameters not in model"
    else:
        weights[key] = weights[key].reshape(1, -1)
        message = r"Expected shape"
    mx.save_safetensors(str(path / "mtp" / "model.safetensors"), weights)

    with pytest.raises(ValueError, match=message) as exc:
        _load_qwen4_checkpoint(path)
    assert "mtp." + key in str(exc.value)


@pytest.mark.parametrize("damage", ["model_type", "text_config", "unreadable"])
def test_qwen4_mtp_rejects_incompatible_sidecar(qwen4_checkpoint, damage):
    path, _ = qwen4_checkpoint()
    config_path = path / "mtp" / "config.json"
    config = json.loads(config_path.read_text())
    if damage == "unreadable":
        config_path.write_text("{")
    else:
        if damage == "model_type":
            config["model_type"] = "other_mtp"
        else:
            config["text_config"]["hidden_size"] *= 2
        config_path.write_text(json.dumps(config))

    assert not lightning_mtp_supported(path)
    model = _load_qwen4_checkpoint(path)
    assert not hasattr(model, "mtp")
