# SPDX-License-Identifier: Apache-2.0
"""Keep configured Gemma4 unified image resolution through mlx-vlm loading."""

import json
from types import SimpleNamespace

import mlx_vlm.utils as vlm_utils
import pytest
from mlx_vlm.models.gemma4_unified import Gemma4UnifiedImageProcessor
from PIL import Image

from omlx.engine.vlm import _preserve_gemma4_unified_image_processor_on_load


def _model_dir(tmp_path, model_type="gemma4_unified"):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return tmp_path


@pytest.mark.parametrize(
    ("budget", "expected_tokens", "expected_size"),
    [(280, 266, (672, 912)), (560, 540, (960, 1296)), (1120, 1064, (1344, 1824))],
)
def test_load_preserves_configured_image_processor(
    tmp_path, monkeypatch, budget, expected_tokens, expected_size
):
    model_dir = _model_dir(tmp_path)
    configured = Gemma4UnifiedImageProcessor(max_soft_tokens=budget, do_normalize=True)
    processor = SimpleNamespace(image_processor=configured)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=1))
    # Exercise the real mlx-vlm load() and load_image_processor() sequence.
    # Only weight/tokenizer loading is replaced; no checkpoint download is needed.
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(vlm_utils, "load_processor", lambda *a, **kw: processor)
    original = vlm_utils.load_image_processor

    with _preserve_gemma4_unified_image_processor_on_load(model_dir):
        loaded_model, loaded_processor = vlm_utils.load(
            model_dir, trust_remote_code=False
        )

    assert vlm_utils.load_image_processor is original
    assert loaded_model is model
    assert loaded_processor.image_processor is configured
    assert configured.do_normalize is True
    data, tokens = configured.preprocess(Image.new("RGB", (2400, 3200)))
    assert tokens == [expected_tokens]
    assert data["pixel_values"].shape == (1, budget, 6912)
    positions = data["image_position_ids"][0]
    valid = positions[positions[:, 0] >= 0]
    assert tuple((valid.max(axis=0) + 1) * 48) == expected_size


@pytest.mark.parametrize("model_type", ["gemma4", "qwen3_5", "qwen3_vl"])
def test_other_model_types_keep_their_loader(tmp_path, monkeypatch, model_type):
    model_dir = _model_dir(tmp_path, model_type)
    sentinel = object()
    calls = []

    def original(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(vlm_utils, "load_image_processor", original)
    with _preserve_gemma4_unified_image_processor_on_load(model_dir):
        assert vlm_utils.load_image_processor is original
        assert (
            vlm_utils.load_image_processor(model_dir, trust_remote_code=False)
            is sentinel
        )
    assert calls == [(model_dir, {"trust_remote_code": False})]


def test_wrapper_is_scoped_to_model_and_restored_on_failure(tmp_path, monkeypatch):
    model_dir = _model_dir(tmp_path)
    other_dir = tmp_path / "other"
    sentinel = object()
    calls = []

    def original(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(vlm_utils, "load_image_processor", original)
    with (
        pytest.raises(RuntimeError, match="load failed"),
        _preserve_gemma4_unified_image_processor_on_load(model_dir),
    ):
        assert vlm_utils.load_image_processor(str(model_dir)) is None
        assert (
            vlm_utils.load_image_processor(other_dir, trust_remote_code=False)
            is sentinel
        )
        raise RuntimeError("load failed")
    assert vlm_utils.load_image_processor is original
    assert calls == [(other_dir, {"trust_remote_code": False})]


@pytest.mark.asyncio
async def test_engine_start_preserves_configured_processor_during_load(
    tmp_path, monkeypatch
):
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.utils import model_loading

    model_dir = _model_dir(tmp_path)
    original = vlm_utils.load_image_processor

    class LoadReachedError(Exception):
        pass

    def check_load(model_path, **kwargs):
        assert vlm_utils.load_image_processor(model_path, **kwargs) is None
        raise LoadReachedError

    monkeypatch.setattr(vlm_utils, "load", check_load)
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        model_loading, "maybe_load_custom_quantization", lambda *a, **kw: None
    )
    engine = VLMBatchedEngine(model_name=str(model_dir))
    with pytest.raises(LoadReachedError):
        await engine.start()
    assert vlm_utils.load_image_processor is original
