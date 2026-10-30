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
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"mtp.fc_hidden.weight": mx.ones((1,))}
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

    def write(*, layout="sidecar", centered=True, head_centered=None, quantized=True):
        # Eight base anchors exercise the checkpoint-wide centering decision.
        if head_centered is None:
            head_centered = centered
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
                    module.weight.shape, -0.75 + head_centered, dtype=mx.bfloat16
                )
        checkpoint_config = asdict(config)
        if not quantized:
            checkpoint_config.pop("quantization")
            checkpoint_config.pop("quantization_config")
        (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
        if layout in ("embedded", "both"):
            root.update(("mtp." + key, value) for key, value in head_weights.items())
        if layout in ("sidecar", "both"):
            sidecar_dir = tmp_path / "mtp"
            sidecar_dir.mkdir()
            sidecar_config = {
                "model_type": "qwen4_exp_mtp",
                "text_config": asdict(config.text_config),
            }
            if head_centered:
                sidecar_config["rms_norm_weight_format"] = "ones_centered"
            (sidecar_dir / "config.json").write_text(json.dumps(sidecar_config))
            sidecar_weights = (
                {key: value + 3 for key, value in head_weights.items()}
                if layout == "both"
                else head_weights
            )
            mx.save_safetensors(str(sidecar_dir / "model.safetensors"), sidecar_weights)
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
        ("both", True, True),
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


@pytest.mark.parametrize(
    ("centered", "head_centered", "explicit_zero"),
    [(True, False, False), (True, False, True), (False, True, False)],
)
def test_qwen4_sidecar_uses_its_own_norm_convention(
    qwen4_checkpoint, centered, head_centered, explicit_zero
):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

    path, head_weights = qwen4_checkpoint(
        centered=centered, head_centered=head_centered
    )
    if explicit_zero:
        config_path = path / "mtp" / "config.json"
        config = json.loads(config_path.read_text())
        config["rms_norm_weight_format"] = "zero_centered"
        config_path.write_text(json.dumps(config))

    model = _load_qwen4_checkpoint(path)

    for name, module in model.mtp.named_modules():
        if isinstance(module, Qwen4ExpRMSNorm):
            expected = head_weights[name + ".weight"].astype(mx.float32)
            if not head_centered:
                expected = expected + 1.0
            assert mx.array_equal(
                1.0 + module.weight.astype(mx.float32), expected
            ).item()


def test_qwen4_direct_gamma_sidecar_keeps_nonrepresentable_norms(qwen4_checkpoint):
    path, head_weights = qwen4_checkpoint(centered=False, head_centered=True)
    head_weights["pre_fc_norm_embedding.weight"] = mx.full(
        head_weights["pre_fc_norm_embedding.weight"].shape,
        0.2001953125,
        dtype=mx.bfloat16,
    )
    mx.save_safetensors(str(path / "mtp" / "model.safetensors"), head_weights)

    model = _load_qwen4_checkpoint(path)

    weight = model.mtp.pre_fc_norm_embedding.weight
    assert weight.dtype == mx.float32
    assert mx.array_equal(
        1.0 + weight, head_weights["pre_fc_norm_embedding.weight"].astype(mx.float32)
    ).item()


@pytest.mark.parametrize("prefix", ["mtp.", "model.language_model.mtp."])
def test_qwen4_sidecar_loads_checkpoint_prefixed_head(qwen4_checkpoint, prefix):
    path, head_weights = qwen4_checkpoint(centered=False)
    mx.save_safetensors(
        str(path / "mtp" / "model.safetensors"),
        {prefix + key: value for key, value in head_weights.items()},
    )

    model = _load_qwen4_checkpoint(path)

    assert mx.array_equal(
        model.mtp.fc_hidden.weight, head_weights["fc_hidden.weight"]
    ).item()


def test_qwen4_sidecar_rejects_colliding_weight_names(qwen4_checkpoint):
    path, head_weights = qwen4_checkpoint(centered=False)
    head_weights["mtp.fc_hidden.weight"] = head_weights["fc_hidden.weight"]
    mx.save_safetensors(str(path / "mtp" / "model.safetensors"), head_weights)

    with pytest.raises(ValueError, match="duplicate.*mtp.fc_hidden.weight"):
        _load_qwen4_checkpoint(path)


def test_qwen4_sidecar_rejects_colliding_expert_layouts(qwen4_checkpoint):
    path, head_weights = qwen4_checkpoint(centered=False)
    prefix = "layers.0.mlp."
    gate = head_weights.pop(prefix + "switch_mlp.gate_proj.weight")
    up = head_weights[prefix + "switch_mlp.up_proj.weight"]
    head_weights[prefix + "experts.gate_up_proj"] = mx.concatenate(
        [gate, mx.full(up.shape, 7, dtype=up.dtype)], axis=-2
    )
    head_weights[prefix + "switch_mlp.up_proj.weight"] = mx.full(
        up.shape, 3, dtype=up.dtype
    )
    mx.save_safetensors(str(path / "mtp" / "model.safetensors"), head_weights)

    with pytest.raises(ValueError, match="duplicate.*switch_mlp.up_proj.weight"):
        _load_qwen4_checkpoint(path)


@pytest.mark.parametrize("prefix", ["", "mtp."])
@pytest.mark.parametrize("enabled", [False, True])
def test_qwen4_indexed_sidecar_uses_its_own_norm_convention(
    qwen4_checkpoint, prefix, enabled
):
    path, head_weights = qwen4_checkpoint(centered=True, head_centered=False)
    mx.save_safetensors(
        str(path / "mtp" / "model.safetensors"),
        {prefix + key: value for key, value in head_weights.items()},
    )
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update(
        {prefix + key: "mtp/model.safetensors" for key in head_weights}
    )
    index_path.write_text(json.dumps(index))

    model = _load_qwen4_checkpoint(path, mtp_enabled=enabled)

    assert hasattr(model, "mtp") is enabled
    if enabled:
        assert mx.array_equal(
            model.mtp.pre_fc_norm_embedding.weight,
            head_weights["pre_fc_norm_embedding.weight"],
        ).item()


@pytest.mark.parametrize(
    ("prefix", "index_all"), [("", False), ("mtp.", False), ("mtp.", True)]
)
def test_qwen4_indexed_sidecar_does_not_override_embedded_head(
    qwen4_checkpoint, prefix, index_all
):
    path, head_weights = qwen4_checkpoint(layout="both", centered=False)
    sidecar_path = path / "mtp" / "model.safetensors"
    sidecar_weights = mx.load(str(sidecar_path))
    mx.save_safetensors(
        str(sidecar_path),
        {prefix + key: value for key, value in sidecar_weights.items()},
    )
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    keys = sidecar_weights if index_all else ["fc_hidden.weight"]
    index["weight_map"].update({prefix + key: "mtp/model.safetensors" for key in keys})
    index_path.write_text(json.dumps(index))

    model = _load_qwen4_checkpoint(path)

    assert mx.array_equal(
        model.mtp.fc_hidden.weight, head_weights["fc_hidden.weight"]
    ).item()


def test_qwen4_nested_embedded_shard_is_loaded_and_available_to_oq(qwen4_checkpoint):
    from omlx.oq import _source_weight_files
    from omlx.utils.model_loading import _checkpoint_has_mtp_weights

    path, head_weights = qwen4_checkpoint(layout="embedded", centered=False)
    root_path = path / "model.safetensors"
    root = mx.load(str(root_path))
    embedded = {key: root.pop(key) for key in list(root) if key.startswith("mtp.")}
    mx.eval(root, embedded)
    mx.save_safetensors(str(root_path), root, metadata={"format": "mlx"})
    nested = path / "nested" / "embedded.safetensors"
    nested.parent.mkdir()
    mx.save_safetensors(str(nested), embedded)
    # The index selects a whole shard, even when it lists only one tensor.
    index = {"weight_map": {key: "model.safetensors" for key in root}}
    index["weight_map"]["mtp.fc_hidden.weight"] = "nested/embedded.safetensors"
    (path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert _checkpoint_has_mtp_weights(path)
    assert nested in _source_weight_files(path)
    model = _load_qwen4_checkpoint(path)
    assert mx.array_equal(
        model.mtp.fc_hidden.weight, head_weights["fc_hidden.weight"]
    ).item()


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen4_exp"},
        {"model_type": "other"},
        {"model_type": "other", "text_config": {"hidden_size": 32}},
        {"note": "metadata"},
    ],
)
def test_qwen4_embedded_mtp_directory_is_not_a_standalone_head(
    qwen4_checkpoint, config
):
    path, head_weights = qwen4_checkpoint(layout="embedded", centered=False)
    root_path = path / "model.safetensors"
    root = mx.load(str(root_path))
    embedded = {key: root.pop(key) for key in list(root) if key.startswith("mtp.")}
    mx.eval(root, embedded)
    mx.save_safetensors(str(root_path), root, metadata={"format": "mlx"})
    nested = path / "mtp"
    nested.mkdir()
    (nested / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(nested / "model.safetensors"), embedded)
    index = {
        "weight_map": {
            **{key: "model.safetensors" for key in root},
            **{key: "mtp/model.safetensors" for key in embedded},
        }
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))

    model = _load_qwen4_checkpoint(path)
    assert mx.array_equal(
        model.mtp.fc_hidden.weight, head_weights["fc_hidden.weight"]
    ).item()


@pytest.mark.parametrize("missing_shard", [False, True])
def test_qwen4_ignores_stale_index_entries(qwen4_checkpoint, missing_shard):
    path, _ = qwen4_checkpoint(layout="embedded", centered=False)
    root_path = path / "model.safetensors"
    root = mx.load(str(root_path))
    for key in [key for key in root if key.startswith("mtp.")]:
        root.pop(key)
    mx.eval(root)
    mx.save_safetensors(str(root_path), root, metadata={"format": "mlx"})
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if missing_shard:
        for key in index["weight_map"]:
            if key.startswith("mtp."):
                index["weight_map"][key] = "missing.safetensors"
    index_path.write_text(json.dumps(index))

    assert not lightning_mtp_supported(path)
    assert not hasattr(_load_qwen4_checkpoint(path), "mtp")


def test_qwen4_oq_uses_indexed_shards_in_loader_order(tmp_path):
    from omlx.oq import _LazyTensorIndex, _source_weight_files

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    nested = tmp_path / "a" / "nested.safetensors"
    nested.parent.mkdir()
    last = tmp_path / "z.safetensors"
    stale = tmp_path / "stale.safetensors"
    mx.save_safetensors(str(nested), {"mtp.fc_hidden.weight": mx.array([1])})
    mx.save_safetensors(
        str(last),
        {"mtp.fc_hidden.weight": mx.array([2]), "trunk.weight": mx.array([1])},
    )
    mx.save_safetensors(str(stale), {"unselected.weight": mx.array([3])})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.fc_hidden.weight": "a/nested.safetensors",
                    "trunk.weight": "z.safetensors",
                }
            }
        )
    )

    files = _source_weight_files(tmp_path)
    assert files == [nested, last]
    weights = _LazyTensorIndex(files)
    assert weights["mtp.fc_hidden.weight"].item() == 2
    assert "unselected.weight" not in weights


def test_qwen4_dense_sidecar_ignores_target_head_quantization_override(
    qwen4_checkpoint,
):
    import mlx.nn as nn

    path, head_weights = qwen4_checkpoint(centered=False)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"]["mtp.fc_hidden"] = {"group_size": 32, "bits": 4}
    config_path.write_text(json.dumps(config))

    model = _load_qwen4_checkpoint(path)

    assert isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.mtp.fc_hidden, nn.Linear)
    assert mx.array_equal(
        model.mtp.fc_hidden.weight, head_weights["fc_hidden.weight"]
    ).item()


@pytest.mark.parametrize(
    ("layout", "disabled"), [("embedded", False), ("sidecar", False), ("sidecar", True)]
)
def test_qwen4_quantized_head_honors_target_quantization_override(
    qwen4_checkpoint, layout, disabled
):
    import mlx.nn as nn

    path, _ = qwen4_checkpoint(layout=layout, centered=False)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"]["mtp.fc_hidden"] = (
        False if disabled else {"group_size": 32, "bits": 4}
    )
    config_path.write_text(json.dumps(config))
    weight_path = path / (
        "mtp/model.safetensors" if layout == "sidecar" else "model.safetensors"
    )
    prefix = "" if layout == "sidecar" else "mtp."
    weights = mx.load(str(weight_path))
    weight, scales, biases = mx.quantize(
        weights[prefix + "fc_hidden.weight"], group_size=32, bits=4
    )
    weights.update(
        {
            prefix + "fc_hidden.weight": weight,
            prefix + "fc_hidden.scales": scales,
            prefix + "fc_hidden.biases": biases,
        }
    )
    mx.save_safetensors(str(weight_path), weights, metadata={"format": "mlx"})

    model = _load_qwen4_checkpoint(path)

    assert isinstance(model.mtp.fc_hidden, nn.QuantizedLinear)
    assert model.mtp.fc_hidden.bits == 4
    assert mx.array_equal(model.mtp.fc_hidden.weight, weight).item()


@pytest.mark.parametrize(
    ("target_quantized", "disable_target_head"),
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize("per_module", [False, True])
def test_qwen4_quantized_sidecar_uses_its_own_recipe(
    qwen4_checkpoint, target_quantized, disable_target_head, per_module
):
    import mlx.nn as nn

    path, _ = qwen4_checkpoint(centered=False, quantized=target_quantized)
    target_config_path = path / "config.json"
    target_config = json.loads(target_config_path.read_text())
    if disable_target_head:
        target_config["quantization"]["mtp.fc_hidden"] = False
    target_config_path.write_text(json.dumps(target_config))
    sidecar_config_path = path / "mtp" / "config.json"
    sidecar_config = json.loads(sidecar_config_path.read_text())
    sidecar_config["quantization"] = (
        {"group_size": 32, "bits": 4, "fc_hidden": {"group_size": 32, "bits": 8}}
        if per_module
        else {"group_size": 32, "bits": 8}
    )
    sidecar_config_path.write_text(json.dumps(sidecar_config))
    weight_path = path / "mtp" / "model.safetensors"
    weights = mx.load(str(weight_path))
    weight, scales, biases = mx.quantize(
        weights["fc_hidden.weight"], group_size=32, bits=8
    )
    weights.update(
        {
            "fc_hidden.weight": weight,
            "fc_hidden.scales": scales,
            "fc_hidden.biases": biases,
        }
    )
    mx.save_safetensors(str(weight_path), weights)

    model = _load_qwen4_checkpoint(path)

    assert (
        isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)
        is target_quantized
    )
    if target_quantized:
        assert model.language_model.model.embed_tokens.bits == 4
    assert isinstance(model.mtp.fc_hidden, nn.QuantizedLinear)
    assert model.mtp.fc_hidden.bits == 8
    assert mx.array_equal(model.mtp.fc_hidden.weight, weight).item()


def test_qwen4_sidecar_rejects_unconverted_external_quantization(qwen4_checkpoint):
    path, _ = qwen4_checkpoint(centered=False)
    config_path = path / "mtp" / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "compressed-tensors"}
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="not a native MLX recipe"):
        _load_qwen4_checkpoint(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "layout",
    ["sidecar", "prefixed_sidecar", "indexed_sidecar", "embedded", "both_indexed"],
)
async def test_qwen4_quantization_advertises_only_preservable_heads(
    qwen4_checkpoint, monkeypatch, layout
):
    from omlx.admin.oq_manager import OQManager

    fixture_layout = (
        "sidecar"
        if layout.endswith("sidecar")
        else "both"
        if layout == "both_indexed"
        else layout
    )
    path, head_weights = qwen4_checkpoint(layout=fixture_layout, quantized=False)
    if layout in ("prefixed_sidecar", "indexed_sidecar", "both_indexed"):
        mx.save_safetensors(
            str(path / "mtp" / "model.safetensors"),
            {"mtp." + key: value for key, value in head_weights.items()},
        )
    if layout in ("indexed_sidecar", "both_indexed"):
        index_path = path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"].update(
            {"mtp." + key: "mtp/model.safetensors" for key in head_weights}
        )
        index_path.write_text(json.dumps(index))
    manager = OQManager(model_dirs=[str(path.parent)])
    sources, _ = await manager.list_quantizable_models()
    source = next(model for model in sources if model["path"] == str(path))

    async def no_conversion(task_id):
        pass

    monkeypatch.setattr(manager, "_run_quantization", no_conversion)
    task = await manager.start_quantization(str(path), 4, preserve_mtp=True)
    await manager._active_tasks[task.task_id]

    preservable = layout in ("embedded", "both_indexed")
    suffix = "-mtp" if preservable else ""
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


@pytest.mark.parametrize(
    "damage",
    [
        "model_type",
        "text_config",
        "unreadable",
        "prefixed_mismatch",
        "norm_format",
        "indexed_norm_format",
        "indexed_text_config",
        "indexed_model_type",
    ],
)
def test_qwen4_mtp_rejects_incompatible_sidecar(qwen4_checkpoint, damage):
    from omlx.admin.routes import _mtp_compat_for_model
    from omlx.patches.mlx_lm_mtp import is_mtp_active

    path, weights = qwen4_checkpoint()
    config_path = path / "mtp" / "config.json"
    config = json.loads(config_path.read_text())
    if damage == "unreadable":
        config_path.write_text("{")
    else:
        if damage in ("model_type", "prefixed_mismatch", "indexed_model_type"):
            config["model_type"] = "other_mtp"
        elif damage in ("norm_format", "indexed_norm_format"):
            config["rms_norm_weight_format"] = "unrecognized"
        else:
            config["text_config"]["hidden_size"] *= 2
        config_path.write_text(json.dumps(config))
    if damage == "prefixed_mismatch" or damage.startswith("indexed_"):
        mx.save_safetensors(
            str(path / "mtp" / "model.safetensors"),
            {"mtp." + key: value for key, value in weights.items()},
        )
    if damage.startswith("indexed_"):
        index_path = path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"].update(
            {"mtp." + key: "mtp/model.safetensors" for key in weights}
        )
        index_path.write_text(json.dumps(index))

    assert not lightning_mtp_supported(path)
    assert not _mtp_compat_for_model({"model_path": str(path)})[0]
    model = _load_qwen4_checkpoint(path)
    assert not hasattr(model, "mtp")
    assert not is_mtp_active()
