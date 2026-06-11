# SPDX-License-Identifier: Apache-2.0
"""Regression guards for the chat image generation UI logic in chat.html.

The chat page is a single Alpine.js template with no JS test runner, so
these tests assert on the template source the same way a reviewer would
(same approach as test_chat_video_ui.py): every send path that can fire
with an image model selected must branch on isImageModel() instead of
unconditionally calling streamResponse(), since the server rejects image
models on /v1/chat/completions with 400 "Use POST /v1/images.".
"""
import json
import re
from pathlib import Path

import pytest

CHAT_TEMPLATE = (
    Path(__file__).parent.parent / "omlx" / "admin" / "templates" / "chat.html"
)
I18N_DIR = Path(__file__).parent.parent / "omlx" / "admin" / "i18n"


def _function_body(source: str, name: str) -> str:
    """Extract the body of an Alpine method by brace matching.

    Matches both ``async name(args) {`` and plain ``name(args) {``
    definitions. Requiring the opening brace after the parameter list
    skips call sites, which are never followed by a brace.
    """
    match = re.search(rf"(?:async )?\b{name}\([^)]*\)\s*{{", source)
    assert match, f"{name} not found in chat.html"
    start = match.end() - 1
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


@pytest.fixture(scope="module")
def chat_source() -> str:
    return CHAT_TEMPLATE.read_text(encoding="utf-8")


class TestImageModelRouting:
    """All send paths must route image models to generateImage()."""

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_path_branches_on_image_model(self, chat_source, fn):
        body = _function_body(chat_source, fn)
        assert "isImageModel()" in body, (
            f"{fn} must branch on isImageModel(); an image model on "
            "/v1/chat/completions is rejected by the server with 400"
        )
        assert "generateImage(" in body, (
            f"{fn} must call generateImage for image models"
        )

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_stream_response_is_inside_else_branch(self, chat_source, fn):
        """streamResponse() may only appear after the isImageModel() check."""
        body = _function_body(chat_source, fn)
        stream_pos = body.index("streamResponse()")
        branch_pos = body.index("isImageModel()")
        assert branch_pos < stream_pos, (
            f"{fn}: streamResponse() runs before the isImageModel() branch"
        )

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_path_debounces_image_submit(self, chat_source, fn):
        """Every send path must respect the imageSubmitting debounce flag."""
        body = _function_body(chat_source, fn)
        assert "imageSubmitting" in body, (
            f"{fn} must check imageSubmitting like sendMessage does"
        )

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_paths_pass_images_to_generate_image(self, chat_source, fn):
        """Every send path must forward the attached images, or edit
        generation silently loses its input image."""
        body = _function_body(chat_source, fn)
        assert re.search(r"generateImage\([^)]+,", body), (
            f"{fn} must pass an images argument to generateImage"
        )


class TestImageAutoRouting:
    """generateImage routes between the installed image checkpoints by
    request shape (attached image -> edit-capable, text -> t2i-capable)
    so users never have to know which checkpoint does what."""

    def test_resolve_image_model_exists(self, chat_source):
        body = _function_body(chat_source, "resolveImageModel")
        assert "findEditImageModel()" in body
        assert "'edit'" in body

    def test_generate_image_uses_routed_model(self, chat_source):
        body = _function_body(chat_source, "generateImage")
        assert "resolveImageModel(" in body
        assert "model: routedModel" in body, (
            "both the POST body and the bubble must carry the routed model"
        )
        assert "model: this.currentModel," not in body.split("_image:")[1], (
            "the request must not fall back to the raw selection"
        )

    def test_generate_image_posts_async_job(self, chat_source):
        """The chat flow must use sync=false and poll, or the fetch blocks
        for the whole render and a tab close loses the submission."""
        body = _function_body(chat_source, "generateImage")
        assert "'/v1/images'" in body
        assert "sync: false" in body

    def test_generate_image_sends_input_images(self, chat_source):
        """Attached images go as the JSON "image" field (data URLs)."""
        body = _function_body(chat_source, "generateImage")
        assert "body.image = imageUrls" in body

    def test_picker_badges_show_pipeline(self, chat_source):
        assert 'x-text="imageBadge(model.id)"' in chat_source
        body = _function_body(chat_source, "imageBadge")
        for key in ("badge_t2i", "badge_edit"):
            assert key in body

    def test_picker_allowlist_includes_image(self, chat_source):
        body = _function_body(chat_source, "loadModels")
        assert "t === 'image'" in body, (
            "the picker allowlist must include model_type 'image' or the "
            "models exist on the server but never appear in the UI"
        )

    def test_fetch_model_types_populates_image_pipeline_map(self, chat_source):
        """fetchModelTypes must build imagePipelineMap from the
        image_pipeline field emitted by /v1/models/status."""
        body = _function_body(chat_source, "fetchModelTypes")
        assert "image_pipeline" in body
        assert "imagePipelineMap" in body


class TestImageIntakeGates:
    """Image attach must open up for image models when an edit-capable
    model exists (generateImage auto-routes), mirroring the video gates."""

    @pytest.mark.parametrize("fn", ["handleDrop", "handlePaste"])
    def test_image_intake_open_for_image_models(self, chat_source, fn):
        body = _function_body(chat_source, fn)
        assert "imageCanUseImage()" in body, (
            f"{fn} must allow images when imageCanUseImage(), or image "
            "edit models can never receive an input image"
        )

    def test_upload_button_gated_on_image_capability(self, chat_source):
        assert (
            'x-show="hasVisionSupport() || videoCanUseImage() || imageCanUseImage()"'
            in chat_source
        )

    def test_load_image_file_format_gate_covers_image_models(self, chat_source):
        """The server magic-sniffs PNG/JPEG/WebP for image inputs too;
        macOS pastes HEIC, which would 400 after the message commits."""
        body = _function_body(chat_source, "loadImageFile")
        assert "isImageModel()" in body
        assert "chat.image.bad_input_type" in body

    def test_upload_caps_use_shared_helper(self, chat_source):
        """Button gate and loadImageFile must share one cap source."""
        assert _function_body(chat_source, "maxUploadImages")
        body = _function_body(chat_source, "loadImageFile")
        assert "maxUploadImages()" in body
        assert "uploadImages.length >= maxUploadImages()" in chat_source


class TestImagePersistence:
    """Blob object URLs are session-scoped: saveCurrentChat must strip
    them and loadChat must rehydrate from the server by job id."""

    def test_save_strips_image_blob_urls(self, chat_source):
        body = _function_body(chat_source, "saveCurrentChat")
        assert "_image" in body and "urls: []" in body, (
            "saveCurrentChat must blank _image.urls like the _video.url arm"
        )

    def test_save_still_strips_attached_image_base64(self, chat_source):
        """The VLM attachment strip arm must survive the image-gen work."""
        body = _function_body(chat_source, "saveCurrentChat")
        assert "image_url: { url: '' }" in body

    def test_load_chat_rehydrates_image_bubbles(self, chat_source):
        body = _function_body(chat_source, "loadChat")
        assert "rehydrateImage(" in body
        assert "rehydrateVideo(" in body, "video rehydration must survive"

    def test_rehydrate_image_handles_expiry(self, chat_source):
        body = _function_body(chat_source, "rehydrateImage")
        assert "artifact_expired" in body, (
            "purged artifacts must surface the expired placeholder"
        )
        assert "pollImage(" in body, (
            "still-running jobs must resume polling on chat reopen"
        )

    def test_expired_placeholder_in_template(self, chat_source):
        assert "msg._image.error === 'artifact_expired'" in chat_source
        assert "chat.image.expired" in chat_source


class TestImageChatSwitchPreservesInflight:
    """Switching chats while an image renders must not lose content
    (mirrors the video fix in PR #68)."""

    def test_generate_image_saves_after_job_id(self, chat_source):
        body = _function_body(chat_source, "generateImage")
        job_id_pos = body.index("msg._image.job_id = job.id")
        poll_pos = body.index("await this.pollImage(msg)")
        save_pos = body.find("this.saveCurrentChat()", job_id_pos, poll_pos)
        assert save_pos != -1, (
            "the bubble must be persisted between job-id assignment and the "
            "poll await, or a chat switch/reload mid-render loses it"
        )

    def test_poll_image_is_chat_scoped(self, chat_source):
        body = _function_body(chat_source, "pollImage")
        assert "pollChatId" in body and "currentChatId !== pollChatId" in body, (
            "pollImage must exit when its chat is no longer open"
        )

    def test_truncation_cancels_inflight_image_jobs(self, chat_source):
        """regenerate/edit truncation must cancel orphaned image jobs the
        same way it cancels video jobs."""
        body = _function_body(chat_source, "cancelRemovedVideos")
        assert "_image" in body and "cancelImage(" in body


class TestImageI18nKeys:
    """The template references these keys via tImage(); a missing key
    falls back to the inline English string, but every shipped bundle
    must still define them so localized UIs stay localized. Keys MUST be
    flat dotted strings (see TestVideoI18nKeys for the history)."""

    LOCALES = ["en", "es", "fr", "ja", "ko", "ru", "zh", "zh-TW"]
    KEYS = [
        "badge_t2i",
        "badge_edit",
        "generating",
        "cancel",
        "cancelled",
        "need_prompt",
        "need_image",
        "no_edit_model",
        "expired",
        "bad_input_type",
    ]

    @pytest.mark.parametrize("lang", LOCALES)
    @pytest.mark.parametrize("key", KEYS)
    def test_chat_image_key_present(self, lang, key):
        data = json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))
        flat_key = f"chat.image.{key}"
        assert flat_key in data, f"{lang}.json missing flat key {flat_key}"
        assert isinstance(data[flat_key], str) and data[flat_key].strip()

    def test_zh_badges_use_canonical_terms(self):
        data = json.loads((I18N_DIR / "zh.json").read_text(encoding="utf-8"))
        assert data["chat.image.badge_t2i"] == "文生图"
        assert data["chat.image.badge_edit"] == "图生图"
