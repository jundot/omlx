# SPDX-License-Identifier: Apache-2.0
"""Verify extreme-aspect-ratio images are rejected with HTTP 400.

Regression tests for the aspect-ratio pre-check added to
``VLMBatchedEngine.preflight_chat`` and the defense-in-depth catch in
``_prepare_vision_inputs``.  The upstream ``_smart_resize_image`` (Qwen3-VL,
Hunyuan-VL, MiniMax-M3-VL, GLM-OCR, PaddleOCR-VL families) raises
``ValueError`` for aspect ratios > 200, which previously propagated as an
unhandled HTTP 500.

The 1093×5 fixture below mirrors the original repro: a horizontal
divider line cropped out of a page scan.
"""

import base64
import io
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.exceptions import InvalidRequestError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(width: int, height: int) -> bytes:
    """Create a minimal in-memory PNG with the given dimensions."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _b64_data_uri(png_bytes: bytes) -> str:
    """Encode raw PNG bytes as a base64 data URI."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _chat_message_with_image(width: int, height: int) -> list[dict]:
    """Build an OpenAI-style messages list with a single inline image."""
    png = _make_png(width, height)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": _b64_data_uri(png)},
                },
            ],
        }
    ]


def _chat_message_with_remote_image(
    url: str = "https://example.com/thin.png",
) -> list[dict]:
    """Build messages with a remote-URL image (dimensions unreadable)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe."},
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                },
            ],
        }
    ]


# A model_type that enforces the aspect-ratio limit.
_AFFECTED_MODEL = "qwen3_vl"

# A model_type that does NOT enforce the aspect-ratio limit.
_UNAFFECTED_MODEL = "llava"


# ---------------------------------------------------------------------------
# Unit tests for _validate_image_aspect_ratios
# ---------------------------------------------------------------------------


class TestValidateImageAspectRatios:
    """Direct tests for the module-level validation helper."""

    def test_extreme_wide_image_rejected(self):
        """1093×5 image (ratio ~218.6) is rejected for an affected model."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(1093, 5)
        with pytest.raises(InvalidRequestError, match=r"aspect ratio.*218"):
            _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_extreme_tall_image_rejected(self):
        """5×1093 image (ratio ~218.6) should also be rejected."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(5, 1093)
        with pytest.raises(InvalidRequestError, match=r"aspect ratio.*218"):
            _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_valid_image_passes(self):
        """A normal 17×17 image should not raise."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(17, 17)
        _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_borderline_199_passes(self):
        """199:1 ratio should pass (below the 200 limit)."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(199, 1)
        _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_exact_200_passes(self):
        """200:1 ratio passes — upstream rejects on > 200, not >= 200."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(200, 1)
        _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_just_above_200_rejected(self):
        """201:1 ratio (just above limit) should be rejected."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(201, 1)
        with pytest.raises(InvalidRequestError, match=r"aspect ratio.*201"):
            _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_text_only_messages_ignored(self):
        """Messages without images should pass without error."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = [{"role": "user", "content": "Just text, no images."}]
        _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_remote_url_skipped_at_helper_level(self):
        """Helper-level: remote URLs can't be dimension-checked, so the guard skips.

        Production preflight never reaches here for remote URLs:
        extract_images_from_messages runs first and raises InvalidRequestError.
        """
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_remote_image()
        # Remote URLs can't be dimension-checked decode-free; skip silently.
        _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_remote_url_rejected_at_extraction(self):
        """Production-level: remote URLs raise before the aspect guard runs."""
        from omlx.utils.image import extract_images_from_messages

        messages = _chat_message_with_remote_image()
        with pytest.raises(InvalidRequestError, match="base64 data URIs"):
            extract_images_from_messages(messages)

    def test_error_message_includes_dimensions(self):
        """The error message should include the actual image dimensions."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(1093, 5)
        with pytest.raises(InvalidRequestError, match=r"1093\u00d75"):
            _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)

    def test_error_field_is_messages(self):
        """The InvalidRequestError should have field='messages'."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(1093, 5)
        with pytest.raises(InvalidRequestError) as exc_info:
            _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)
        assert exc_info.value.field == "messages"

    # --- Model scoping ---

    def test_unaffected_model_skips_validation(self):
        """An extreme-aspect image is not rejected for an unaffected model."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(1093, 5)
        # Must not raise — the guard is a no-op for unaffected models.
        _validate_image_aspect_ratios(messages, model_type=_UNAFFECTED_MODEL)

    def test_none_model_type_skips_validation(self):
        """When model_type is None (engine not loaded), skip validation."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = _chat_message_with_image(1093, 5)
        _validate_image_aspect_ratios(messages, model_type=None)

    def test_all_affected_models_covered(self):
        """Every model_type in _ASPECT_RATIO_MODELS triggers rejection."""
        from omlx.engine.vlm import (
            _ASPECT_RATIO_MODELS,
            _validate_image_aspect_ratios,
        )

        messages = _chat_message_with_image(1093, 5)
        for mt in _ASPECT_RATIO_MODELS:
            with pytest.raises(InvalidRequestError):
                _validate_image_aspect_ratios(messages, model_type=mt)

    def test_guard_uses_top_level_model_types(self):
        """Bind the guard to external sources of truth, not to itself.

        Regression anchor for dead allowlist keys: mlx-vlm config modules
        define TextConfig and ModelConfig types in the same file, and the
        text type never reaches VLMBatchedEngine.model_type.
        """
        from omlx.engine.vlm import (
            _ASPECT_RATIO_MODELS,
            MINIMAX_M3_VL_MODEL_TYPE,
            OCR_MODEL_TYPES,
        )

        # Top-level types must be guarded.
        assert MINIMAX_M3_VL_MODEL_TYPE in _ASPECT_RATIO_MODELS
        assert "glm_ocr" in _ASPECT_RATIO_MODELS
        assert "glm_ocr" in OCR_MODEL_TYPES
        assert "qwen2_vl" in _ASPECT_RATIO_MODELS
        assert "qwen2_5_vl" in _ASPECT_RATIO_MODELS
        assert "glm4v" in _ASPECT_RATIO_MODELS
        # Nested TextConfig types must not be used as guard keys.
        assert "glm_ocr_text" not in _ASPECT_RATIO_MODELS
        assert "minimax_m3" not in _ASPECT_RATIO_MODELS


# ---------------------------------------------------------------------------
# Integration tests: preflight_chat wiring
# ---------------------------------------------------------------------------


def _make_preflight_engine(model_type: str):
    """Build a VLMBatchedEngine wired enough for preflight_chat to run.

    Exercises the real preflight_chat method (not a substitute route),
    confirming the production wiring passes model_type to the guard.
    """
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True

    # Wire a minimal processor, tokenizer, model config
    mock_processor = MagicMock()
    mock_processor.apply_chat_template = MagicMock(return_value="<prompt>")
    # Ensure _derive_image_token_upper_bound() gets a fallback value
    mock_processor.image_processor = None
    engine._processor = mock_processor

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")
    engine._tokenizer = mock_tokenizer

    mock_vlm_model = MagicMock()
    mock_vlm_model.config = MagicMock()
    mock_vlm_model.config.model_type = model_type
    engine._vlm_model = mock_vlm_model

    # is_diffusion_model must be False
    engine._diffusion_family = None  # is_diffusion_model property reads this

    return engine


class TestPreflightChatWiring:
    """Test the real preflight_chat method, not a substitute route."""

    @pytest.mark.asyncio
    async def test_affected_model_rejects_extreme_aspect(self):
        """preflight_chat rejects an extreme-aspect image for an affected model."""
        engine = _make_preflight_engine(_AFFECTED_MODEL)
        messages = _chat_message_with_image(1093, 5)

        with pytest.raises(InvalidRequestError, match="aspect ratio"):
            await engine.preflight_chat(messages)

    @pytest.mark.asyncio
    async def test_unaffected_model_passes_extreme_aspect(self):
        """preflight_chat does not reject an extreme-aspect image for an unaffected model."""
        engine = _make_preflight_engine(_UNAFFECTED_MODEL)
        messages = _chat_message_with_image(1093, 5)

        # Should not raise — the guard is a no-op for this model type.
        # The method may raise other errors (e.g. scheduler unreachable)
        # which are fine — we only check that InvalidRequestError is not raised.
        try:
            await engine.preflight_chat(messages)
        except InvalidRequestError:
            pytest.fail(
                "preflight_chat should not reject extreme-aspect "
                "images for unaffected models"
            )
        except Exception:
            pass  # Other errors (scheduler, etc.) are expected in this mock

    @pytest.mark.asyncio
    async def test_affected_model_accepts_valid_image(self):
        """preflight_chat does not raise for a normal image on an affected model."""
        engine = _make_preflight_engine(_AFFECTED_MODEL)
        messages = _chat_message_with_image(17, 17)

        try:
            await engine.preflight_chat(messages)
        except InvalidRequestError:
            pytest.fail("preflight_chat should not reject valid images")
        except Exception:
            pass  # Other errors are expected in this mock


# ---------------------------------------------------------------------------
# HTTP integration tests via TestClient with production exception handler
# ---------------------------------------------------------------------------


def _build_test_app():
    """Build a minimal app reusing the production InvalidRequestError handler.

    Exercises _validate_image_aspect_ratios against a fixed model_type.
    """
    from fastapi import FastAPI

    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(InvalidRequestError, srv.invalid_request_error_handler)

    @app.post("/v1/chat/completions")
    def mock_chat(request_body: dict):
        """Validate, then return a placeholder success (never hits a real engine)."""
        from omlx.engine.vlm import _validate_image_aspect_ratios

        messages = request_body.get("messages", [])
        # Use affected model type so the guard actually fires
        _validate_image_aspect_ratios(messages, model_type=_AFFECTED_MODEL)
        return {"id": "chatcmpl-test", "choices": []}

    return app


class TestAspectRatioHTTP:
    """HTTP-level tests verifying the error reaches clients as 400."""

    def test_json_request_returns_400(self):
        """Non-streaming request with extreme-aspect image → HTTP 400."""
        messages = _chat_message_with_image(1093, 5)
        with TestClient(_build_test_app()) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": messages, "stream": False},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert "aspect ratio" in body["error"]["message"].lower()

    def test_valid_image_returns_200(self):
        """A normal 17×17 image should get a 200 success response."""
        messages = _chat_message_with_image(17, 17)
        with TestClient(_build_test_app()) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": messages, "stream": False},
            )
        assert resp.status_code == 200

    def test_400_body_has_openai_error_format(self):
        """The 400 response should use the OpenAI error body shape."""
        messages = _chat_message_with_image(1093, 5)
        with TestClient(_build_test_app()) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": messages},
            )
        body = resp.json()
        # OpenAI-style error body: {"error": {"message": ..., "type": ..., "param": ...}}
        err = body["error"]
        assert "message" in err
        assert err.get("type") == "invalid_request_error"
        assert err.get("param") == "messages"

    @pytest.mark.asyncio
    async def test_remote_image_rejected_before_guard(self):
        """Production preflight rejects remote URLs at extraction (HTTP 400).

        The substitute route in _build_test_app skips extraction, so it
        cannot reproduce this: exercise the real preflight_chat wiring
        instead, which runs extract_images_from_messages before the guard.
        """
        messages = _chat_message_with_remote_image()
        engine = _make_preflight_engine(_AFFECTED_MODEL)
        with pytest.raises(InvalidRequestError, match="base64 data URIs"):
            await engine.preflight_chat(messages)


# ---------------------------------------------------------------------------
# Production endpoint wiring: real /v1/chat/completions route
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpointWiring:
    """Hit the production ``/v1/chat/completions`` route with a real
    ``VLMBatchedEngine.preflight_chat`` (model_type ``qwen3_5`` — the type
    OvisOCR2 resolves to) and an inline extreme-aspect image.

    Unlike ``TestAspectRatioHTTP`` (substitute route) and
    ``TestPreflightChatWiring`` (direct method call), this exercises the
    full handler wiring: message extraction → ``count_chat_tokens`` →
    ``preflight_chat`` → ``InvalidRequestError`` handler, parameterized
    over ``stream=false/true``.  Both must return HTTP 400 with a JSON
    body — preflight runs before ``StreamingResponse`` commits headers.
    """

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_completions_rejects_extreme_aspect(self, stream):
        import omlx.server as srv

        engine = _make_preflight_engine("qwen3_5")
        engine.count_chat_tokens = MagicMock(return_value=128)

        async def _get_engine_for_model(model_id, *, lease=None):
            return engine

        original_get_engine = srv.get_engine_for_model
        original_overrides = dict(srv.app.dependency_overrides)
        original_engine_pool = srv._server_state.engine_pool
        try:
            srv.app.dependency_overrides[srv.verify_api_key] = lambda: True
            srv.get_engine_for_model = _get_engine_for_model  # type: ignore[assignment]
            fake_pool = MagicMock()
            fake_pool.get_entry = MagicMock(return_value=None)
            fake_pool.preload_pinned_models = AsyncMock()
            fake_pool.check_ttl_expirations = AsyncMock()
            fake_pool.shutdown = AsyncMock()
            srv._server_state.engine_pool = fake_pool
            messages = _chat_message_with_image(1093, 5)
            with TestClient(srv.app, raise_server_exceptions=False) as client:
                with (
                    patch.object(srv, "resolve_model_id", lambda name: name),
                    patch.object(srv, "validate_context_window", lambda *a, **k: None),
                ):
                    resp = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": messages,
                            "stream": stream,
                        },
                    )
            assert resp.status_code == 400, (
                f"expected 400, got {resp.status_code}: {resp.text}"
            )
            body = resp.json()
            assert "error" in body, body
            assert "aspect ratio" in body["error"]["message"].lower()
            assert body["error"].get("type") == "invalid_request_error"
        finally:
            srv.get_engine_for_model = original_get_engine
            srv._server_state.engine_pool = original_engine_pool
            srv.app.dependency_overrides.clear()
            srv.app.dependency_overrides.update(original_overrides)

    def test_ovis_model_type_is_guarded(self):
        """OvisOCR2 resolves to ``qwen3_5`` — it must stay in the guard set.

        Regression anchor for the model that triggered this fix: if the
        allowlist ever drops ``qwen3_5``, the endpoint test above would
        silently pass (no rejection) for the original repro.
        """
        from omlx.engine.vlm import _ASPECT_RATIO_MODELS

        assert "qwen3_5" in _ASPECT_RATIO_MODELS
        assert "qwen3_5_moe" in _ASPECT_RATIO_MODELS


# ---------------------------------------------------------------------------
# Defense-in-depth: _prepare_vision_inputs catch
# ---------------------------------------------------------------------------


@contextmanager
def _prepare_inputs_raising(engine, message: str):
    """Patch the vision path so ``prepare_inputs`` raises ``ValueError(message)``.

    Also stubs ``_format_messages_for_vlm_template`` so the call reaches
    ``prepare_inputs`` without a real processor.
    """
    with (
        patch("mlx_vlm.utils.prepare_inputs", side_effect=ValueError(message)),
        patch.object(
            engine,
            "_format_messages_for_vlm_template",
            return_value=([{"role": "user", "content": "test"}], [(0, 1)]),
        ),
    ):
        yield


def _call_prepare_vision_inputs(engine):
    """Invoke ``_prepare_vision_inputs`` with a minimal message/image pair."""
    return engine._prepare_vision_inputs(
        messages=[{"role": "user", "content": "test"}],
        images=["fake_image"],
    )


def _make_vision_inputs_engine():
    """Build a VLMBatchedEngine wired for _prepare_vision_inputs."""
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._processor = MagicMock()
    engine._processor.apply_chat_template = MagicMock(return_value="<prompt>")
    engine._vlm_model = MagicMock()
    engine._vlm_model.config = MagicMock()
    engine._vlm_model.config.model_type = "test_model"
    engine._tokenizer = MagicMock()
    engine._tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    engine._enable_thinking = None
    engine._model_name = "test-vlm"
    return engine


class TestPrepareVisionInputsDefenseInDepth:
    """Verify the ValueError→InvalidRequestError re-mapping.

    Only the specific upstream aspect-ratio error string is converted.
    """

    def test_aspect_ratio_valueerror_remapped(self):
        """A ValueError naming the aspect-ratio limit becomes InvalidRequestError."""
        engine = _make_vision_inputs_engine()

        message = (
            "Failed to process inputs with error: absolute aspect ratio "
            "must be smaller than 200, got 218.6"
        )
        with _prepare_inputs_raising(engine, message):
            with pytest.raises(InvalidRequestError, match="aspect ratio"):
                _call_prepare_vision_inputs(engine)

    def test_unrelated_valueerror_not_caught(self):
        """A ValueError not mentioning aspect ratio propagates unchanged."""
        engine = _make_vision_inputs_engine()

        with _prepare_inputs_raising(engine, "Unrelated processing error"):
            with pytest.raises(ValueError, match="Unrelated"):
                _call_prepare_vision_inputs(engine)

    def test_defense_in_depth_documents_streaming_limitation(self):
        """The defense-in-depth catch in _prepare_vision_inputs runs
        inside the engine's chat/stream_chat path — i.e. AFTER streaming
        headers are committed for SSE.

        For model types outside _ASPECT_RATIO_MODELS (or any image whose
        dimensions preflight couldn't read), this means an SSE request may
        receive a partial stream followed by an error, not a clean HTTP
        400.  This test documents that limitation: the pre-header HTTP 400
        guarantee for SSE is limited to listed models with readable inline
        images.
        """
        engine = _make_vision_inputs_engine()

        message = "absolute aspect ratio must be smaller than 200, got 300.0"
        with _prepare_inputs_raising(engine, message):
            with pytest.raises(InvalidRequestError) as exc_info:
                _call_prepare_vision_inputs(engine)

        # InvalidRequestError, not ValueError: the server handler maps this to
        # 400, but for SSE the status may already be committed.
        assert exc_info.value.field == "messages"
