# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen4-Exp multimodal admission in the mlx-vlm load path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.utils.model_loading import maybe_apply_pre_load_patches


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
    settings = SimpleNamespace(mtp_enabled=True, mtp_num_draft_tokens=None)

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

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(
            mtp_enabled=False,
            qwen4_ple_ssd_offload=False,
            expert_streaming_enabled=True,
        ),
        for_vlm=True,
    )
    assert get_ple_runtime_mode() == "mmap"
