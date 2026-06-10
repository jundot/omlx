# SPDX-License-Identifier: Apache-2.0
"""Regression guards for the chat video UI logic in chat.html.

The chat page is a single Alpine.js template with no JS test runner, so
these tests assert on the template source the same way a reviewer would:
every send path that can fire with a video model selected must branch on
isVideoModel() instead of unconditionally calling streamResponse().

Regression context: after cancelling an in-chat video generation, editing
the original prompt and re-running used to POST /v1/chat/completions and
the server answered 400 "is a video generation model. Use POST /v1/videos."
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
    skips call sites (``handlePaste($event)"`` in attributes, ``this.
    findExtendModel();`` in JS), which are never followed by a brace.
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


class TestVideoModelRouting:
    """All send paths must route video models to generateVideo()."""

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_path_branches_on_video_model(self, chat_source, fn):
        body = _function_body(chat_source, fn)
        assert "isVideoModel()" in body, (
            f"{fn} must branch on isVideoModel(); a video model on "
            "/v1/chat/completions is rejected by the server with 400"
        )
        assert "generateVideo(" in body, f"{fn} must call generateVideo for video models"

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_stream_response_is_inside_else_branch(self, chat_source, fn):
        """streamResponse() may only appear after the isVideoModel() check."""
        body = _function_body(chat_source, fn)
        stream_pos = body.index("streamResponse()")
        branch_pos = body.index("isVideoModel()")
        assert branch_pos < stream_pos, (
            f"{fn}: streamResponse() runs before the isVideoModel() branch"
        )

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_path_debounces_video_submit(self, chat_source, fn):
        """Every send path must respect the videoSubmitting debounce flag."""
        body = _function_body(chat_source, fn)
        assert "videoSubmitting" in body, (
            f"{fn} must check videoSubmitting like sendMessage does"
        )

    def test_regenerate_reuses_last_user_prompt(self, chat_source):
        """regenerateMessage must extract the prompt from message history,
        since unlike sendMessage it has no input box text to use."""
        body = _function_body(chat_source, "regenerateMessage")
        assert "getTextContent(" in body


class TestVideoImageToVideoUI:
    """Guards for the I2V / extend / upscale wiring in the chat page.

    Server contract (video_routes.py): input_reference and extend_video_id
    are mutually exclusive, t2v models reject both, i2v models require one,
    and upscale_resolution triggers the SeedVR2 pass. The template must
    mirror those gates so users do not hit raw 400s.
    """

    def test_generate_video_wires_extend_reference_and_upscale(self, chat_source):
        """generateVideo must carry all three new request fields."""
        body = _function_body(chat_source, "generateVideo")
        for field in ("extend_video_id", "input_reference", "upscale_resolution"):
            assert field in body, f"generateVideo must wire {field} into the POST body"

    def test_extend_wins_over_input_reference(self, chat_source):
        """The server 400s when both extend_video_id and input_reference
        are sent, so the client must branch on videoExtendFrom first and
        only attach the conditioning image in the else path."""
        body = _function_body(chat_source, "generateVideo")
        assert body.index("extend_video_id") < body.index("input_reference"), (
            "generateVideo must attach extend_video_id before considering "
            "input_reference (extend wins)"
        )
        assert re.search(
            r"if\s*\(\s*this\.videoExtendFrom\s*\)"
            r"[\s\S]*?else\s+if\s*\([^)]*videoAcceptsImage\(\)[^)]*\)",
            body,
        ), (
            "input_reference must only be attached in the else branch of "
            "the videoExtendFrom check"
        )

    @pytest.mark.parametrize("fn", ["handleDrop", "handlePaste"])
    def test_image_intake_open_for_i2v_models(self, chat_source, fn):
        """Drag-drop and paste must accept images for i2v-capable video
        models, not just VLMs (these handlers are not async)."""
        body = _function_body(chat_source, fn)
        assert "videoAcceptsImage()" in body, (
            f"{fn} must allow images when videoAcceptsImage(), or i2v "
            "models can never receive a conditioning image"
        )

    def test_upload_button_gated_on_vision_or_video_image(self, chat_source):
        """The image upload button must show for VLMs and i2v video models."""
        assert 'x-show="hasVisionSupport() || videoAcceptsImage()"' in chat_source

    @pytest.mark.parametrize("fn", ["startVideoExtend", "findExtendModel"])
    def test_extend_helpers_exist(self, chat_source, fn):
        """The continuation entry points must exist as Alpine methods."""
        body = _function_body(chat_source, fn)
        assert body  # brace-matched body extracted

    def test_completed_bubble_offers_extend(self, chat_source):
        """The finished-video bubble must expose the Continue button."""
        assert '@click="startVideoExtend(' in chat_source, (
            "completed video bubble must call startVideoExtend(msg)"
        )

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_paths_pass_images_to_generate_video(self, chat_source, fn):
        """Every send path must forward the attached images, or i2v
        generation silently loses its conditioning image."""
        body = _function_body(chat_source, fn)
        assert re.search(r"generateVideo\([^)]+,", body), (
            f"{fn} must pass an images argument to generateVideo"
        )

    def test_fetch_model_types_populates_pipeline_map(self, chat_source):
        """fetchModelTypes must build videoPipelineMap from the
        video_pipeline field emitted by /v1/models/status."""
        body = _function_body(chat_source, "fetchModelTypes")
        assert "video_pipeline" in body
        assert "videoPipelineMap" in body


class TestVideoI18nKeys:
    """The template references these keys via tVideo(); a missing key
    falls back to the inline English string, but both shipped bundles
    must still define them so localized UIs stay localized.

    Keys MUST be flat dotted strings: both lookups are flat dict gets
    (routes.py _make_t server-side, window.t in base.html client-side).
    Nested {"chat": {"video": {...}}} blocks are unreachable dead data --
    that exact bug shipped once and made settings labels render as raw
    dotted keys."""

    @pytest.mark.parametrize("lang", ["en", "zh"])
    @pytest.mark.parametrize("key", ["extend", "extend_chip", "upscale", "need_image"])
    def test_chat_video_key_present(self, lang, key):
        data = json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))
        flat_key = f"chat.video.{key}"
        assert flat_key in data, f"{lang}.json missing flat key {flat_key}"
        assert isinstance(data[flat_key], str) and data[flat_key].strip()

    @pytest.mark.parametrize("path", sorted(I18N_DIR.glob("*.json")))
    def test_bundle_is_flat(self, path):
        """Every bundle value must be a string; nested dicts never resolve."""
        data = json.loads(path.read_text(encoding="utf-8"))
        nested = [k for k, v in data.items() if not isinstance(v, str)]
        assert not nested, f"{path.name} has unreachable nested blocks: {nested}"
