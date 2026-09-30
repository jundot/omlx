# SPDX-License-Identifier: Apache-2.0
"""Regression tests for admin chat model dropdown filtering."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omlx.model_settings import ModelSettings
from omlx.settings import GlobalSettings

CHAT_TEMPLATE = (
    Path(__file__).parent.parent / "omlx" / "admin" / "templates" / "chat.html"
)


def _chat_model_selectable(model_type, engine_type=""):
    """Python equivalent of the chat.html availableModels filter."""
    t = (model_type or "").lower()
    e = (engine_type or "").lower()
    return not (
        t.startswith("audio_")
        or t == "embedding"
        or t == "reranker"
        or t == "image"
        or e == "image"
    )


def _status_metadata_maps(models):
    """Python equivalent of the chat.html /v1/models/status metadata mapping."""
    model_type_map = {}
    model_engine_type_map = {}
    for model in models:
        model_id = model.get("id")
        if model_id:
            model_type_map[model_id] = model.get("model_type") or "llm"
            model_engine_type_map[model_id] = model.get("engine_type") or ""
    for model in models:
        model_id = model.get("model_alias")
        if model_id and model_id not in model_type_map:
            model_type_map[model_id] = model.get("model_type") or "llm"
            model_engine_type_map[model_id] = model.get("engine_type") or ""
    return model_type_map, model_engine_type_map


def test_chat_dropdown_filter_keeps_chat_capable_model_types():
    """LLM and VLM models should remain selectable in the chat UI."""
    assert _chat_model_selectable("llm")
    assert _chat_model_selectable("vlm")


def test_chat_dropdown_filter_excludes_non_chat_model_types():
    """Dedicated non-chat engines should not be offered in the chat UI."""
    assert not _chat_model_selectable("image")
    assert not _chat_model_selectable("llm", engine_type="image")
    assert not _chat_model_selectable("audio_transcription")
    assert not _chat_model_selectable("audio_speech")
    assert not _chat_model_selectable("embedding")
    assert not _chat_model_selectable("reranker")


def test_chat_dropdown_filter_excludes_aliased_image_model():
    """/v1/models aliases should still receive image metadata from status."""
    model_type_map, engine_type_map = _status_metadata_maps(
        [
            {
                "id": "raw-image-model",
                "model_alias": "friendly-image-name",
                "model_type": "llm",
                "engine_type": "image",
            },
            {
                "id": "raw-vlm-model",
                "model_alias": "friendly-vlm-name",
                "model_type": "vlm",
                "engine_type": "vlm",
            },
        ]
    )

    assert not _chat_model_selectable(
        model_type_map["friendly-image-name"],
        engine_type_map["friendly-image-name"],
    )
    assert _chat_model_selectable(
        model_type_map["friendly-vlm-name"],
        engine_type_map["friendly-vlm-name"],
    )


def test_status_metadata_prefers_model_id_when_alias_collides():
    """A model alias should not overwrite another model's canonical metadata."""
    model_type_map, engine_type_map = _status_metadata_maps(
        [
            {
                "id": "chat-model",
                "model_type": "llm",
                "engine_type": "batched",
            },
            {
                "id": "raw-image-model",
                "model_alias": "chat-model",
                "model_type": "llm",
                "engine_type": "image",
            },
        ]
    )

    assert _chat_model_selectable(
        model_type_map["chat-model"],
        engine_type_map["chat-model"],
    )


@pytest.mark.asyncio
async def test_models_status_exposes_model_alias_metadata():
    """/v1/models/status should include aliases for clients listing alias IDs."""
    from omlx.server import ServerState, list_models_status

    state = ServerState()
    state.engine_pool = MagicMock()
    state.engine_pool.get_status.return_value = {
        "models": [
            {
                "id": "raw-image-model",
                "engine_type": "image",
                "model_type": "llm",
            }
        ]
    }
    state.engine_pool.get_active_model_aliases.return_value = {
        "raw-image-model": "friendly-image-name",
    }
    state.settings_manager = MagicMock()
    state.settings_manager.get_settings.return_value = ModelSettings(
        model_alias="friendly-image-name"
    )

    with patch("omlx.server._server_state", state):
        status = await list_models_status()

    assert status["models"][0]["id"] == "raw-image-model"
    assert status["models"][0]["model_alias"] == "friendly-image-name"
    assert status["models"][0]["engine_type"] == "image"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "discovered_default", "expected"),
    [
        (ModelSettings(enable_thinking=False), True, False),
        (ModelSettings(enable_thinking=True), False, True),
        (ModelSettings(chat_template_kwargs={"enable_thinking": True}), False, True),
        (
            ModelSettings(
                enable_thinking=False, chat_template_kwargs={"enable_thinking": True}
            ),
            True,
            False,
        ),
        (ModelSettings(), True, True),
        (ModelSettings(), None, None),
        (None, False, False),
    ],
)
async def test_models_status_exposes_effective_thinking(
    settings, discovered_default, expected
):
    from omlx.server import ServerState, list_models_status

    state = ServerState()
    state.engine_pool = MagicMock()
    state.engine_pool.get_status.return_value = {
        "models": [
            {"id": "model", "model_type": "llm", "thinking_default": discovered_default}
        ]
    }
    state.engine_pool.get_active_model_aliases.return_value = {}
    if settings is not None:
        state.settings_manager = MagicMock()
        state.settings_manager.get_settings_for_request.return_value = settings
        state.settings_manager.get_settings.return_value = settings
    with patch("omlx.server._server_state", state):
        status = await list_models_status()
    assert status["models"][0]["enable_thinking"] is expected


@pytest.mark.asyncio
async def test_models_endpoints_omit_inactive_colliding_aliases():
    """Inactive aliases should not create duplicate /v1/models display IDs."""
    from omlx.server import ServerState, list_models, list_models_status

    state = ServerState()
    state.engine_pool = MagicMock()
    state.engine_pool.get_status.return_value = {
        "models": [
            {
                "id": "chat-model",
                "engine_type": "batched",
                "model_type": "llm",
            },
            {
                "id": "raw-image-model",
                "engine_type": "image",
                "model_type": "llm",
            },
        ]
    }
    state.engine_pool.get_active_model_aliases.return_value = {}
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_expose_model = False
    state.settings_manager = MagicMock()
    state.settings_manager.get_settings.side_effect = lambda model_id: ModelSettings(
        model_alias="chat-model" if model_id == "raw-image-model" else None
    )

    with patch("omlx.server._server_state", state):
        models = await list_models()
        status = await list_models_status()

    assert [m.id for m in models.data] == ["chat-model", "raw-image-model"]
    raw_image = next(m for m in status["models"] if m["id"] == "raw-image-model")
    assert "model_alias" not in raw_image


def test_chat_template_uses_status_metadata_for_filtering():
    """The UI filter should use /v1/models/status model and engine metadata."""
    source = CHAT_TEMPLATE.read_text()

    assert "modelTypeMap" in source
    assert "modelEngineTypeMap" in source
    assert "fetch('/v1/models/status'" in source
    assert "map[m.id] = m.model_type || 'llm';" in source
    assert "engineMap[m.id] = m.engine_type || '';" in source
    assert "map[m.model_alias] === undefined" in source
    assert source.index("await this.fetchModelTypes();") < source.index(
        "this.availableModels = this.dedupeAvailableModels("
    )
    assert "t !== 'image'" in source
    assert "e !== 'image'" in source
    assert "t !== 'embedding'" in source
    assert "t !== 'reranker'" in source
    assert "t !== 'audio_tts'" in source
    assert "t !== 'audio_sts'" in source
