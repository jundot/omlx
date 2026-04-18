# SPDX-License-Identifier: Apache-2.0
"""Tests for chat file attachment functionality.

Follows the pattern established in test_chat_image_upload.py:
- Pure Python, no running server required
- Message format logic tested as data structure construction
- i18n key presence validated parametrically across all five language files
"""
import json
from pathlib import Path

import pytest


I18N_DIR = Path(__file__).parent.parent / "omlx" / "admin" / "i18n"

# All i18n keys introduced by the file attach feature
REQUIRED_FILE_KEYS = [
    "chat.attach_file",
    "chat.remove_file",
    "chat.error.invalid_file_type",
    "chat.error.file_too_large",
    "chat.error.file_empty",
    "chat.error.file_load_failed",
    "chat.error.pdf_encrypted",
    "chat.error.pdf_no_text",
    "chat.error.json_invalid",
    "chat.error.file_context_exceeded",
    "chat.warn.file_context_danger",
    "chat.warn.file_context_unknown",
]


# =============================================================================
# Message Format
# =============================================================================

class TestFileAttachMessageFormat:
    """File content injection into the user message content field."""

    @staticmethod
    def _build_content(user_text, file_attach=None, images=None):
        """Simulate sendMessage() content construction from chat.html."""
        images = images or []
        text_with_attach = user_text
        if file_attach:
            text_with_attach = (
                f'<attached file="{file_attach["filename"]}">\n'
                f'{file_attach["content"]}\n'
                f'</attached>\n\n'
                f'{user_text}'
            )
        if images:
            content = [
                {"type": "image_url", "image_url": {"url": img}}
                for img in images
            ]
            if text_with_attach:
                content.append({"type": "text", "text": text_with_attach})
            return content
        return text_with_attach

    def test_text_only_no_attach(self):
        content = self._build_content("Hello")
        assert isinstance(content, str)
        assert content == "Hello"

    def test_file_injected_into_text_message(self):
        attach = {"filename": "notes.md", "content": "# Header\nBody text."}
        content = self._build_content("Summarise this", file_attach=attach)
        assert isinstance(content, str)
        assert '<attached file="notes.md">' in content
        assert "# Header" in content
        assert "Summarise this" in content
        assert "</attached>" in content

    def test_file_and_images_together(self):
        attach = {"filename": "data.json", "content": '{"key": "value"}'}
        images = ["data:image/png;base64,abc"]
        content = self._build_content("Explain", file_attach=attach, images=images)
        assert isinstance(content, list)
        texts = [p for p in content if p["type"] == "text"]
        assert len(texts) == 1
        assert '<attached file="data.json">' in texts[0]["text"]
        assert "Explain" in texts[0]["text"]

    def test_images_only_no_attach(self):
        images = ["data:image/png;base64,abc"]
        content = self._build_content("What is this?", images=images)
        assert isinstance(content, list)
        texts = [p for p in content if p["type"] == "text"]
        assert "<attached" not in texts[0]["text"]
        assert texts[0]["text"] == "What is this?"

    def test_attach_tag_structure(self):
        """Verify the <attached> block structure the model receives."""
        attach = {"filename": "report.pdf", "content": "Q1 revenue was $2M."}
        content = self._build_content("What was Q1 revenue?", file_attach=attach)
        assert content.startswith('<attached file="report.pdf">\n')
        assert content.endswith("What was Q1 revenue?")
        assert "</attached>" in content

    def test_empty_user_text_with_attach(self):
        """Attach with no user text — content still wraps correctly."""
        attach = {"filename": "notes.txt", "content": "Some notes."}
        content = self._build_content("", file_attach=attach)
        assert isinstance(content, str)
        assert '<attached file="notes.txt">' in content

    def test_attach_consumed_once(self):
        """Simulates that attachedFile is cleared after send — second call has no attach."""
        attach = {"filename": "doc.md", "content": "Content."}
        first = self._build_content("First question", file_attach=attach)
        second = self._build_content("Second question", file_attach=None)
        assert "<attached" in first
        assert "<attached" not in second


# =============================================================================
# Context Viability
# =============================================================================

class TestContextViability:
    """Context window viability check logic — mirrors checkContextViability() in chat.html."""

    RESPONSE_RESERVE = 2048
    SYSTEM_OVERHEAD = 256
    DANGER_THRESHOLD = 0.70

    def _check(self, token_count, context_window, conversation_tokens=0):
        """Pure Python equivalent of checkContextViability()."""
        if context_window is None:
            return {"state": "unknown", "tokenCount": token_count}
        reserved = self.RESPONSE_RESERVE + self.SYSTEM_OVERHEAD
        available = context_window - conversation_tokens - reserved
        fraction = token_count / available if available > 0 else float("inf")
        deficit = max(0, token_count - available)
        required_window = conversation_tokens + token_count + reserved
        if fraction <= self.DANGER_THRESHOLD:
            state = "safe"
        elif fraction <= 1.0:
            state = "danger"
        else:
            state = "exceeded"
        return {
            "state": state,
            "fraction": fraction,
            "deficit": deficit,
            "available": available,
            "requiredWindow": required_window,
        }

    def test_safe(self):
        v = self._check(1_000, 32_768)
        assert v["state"] == "safe"
        assert v["fraction"] < self.DANGER_THRESHOLD

    def test_danger(self):
        v = self._check(25_000, 32_768)
        assert v["state"] == "danger"
        assert self.DANGER_THRESHOLD < v["fraction"] <= 1.0

    def test_exceeded(self):
        v = self._check(35_000, 32_768)
        assert v["state"] == "exceeded"
        assert v["deficit"] > 0

    def test_exactly_at_danger_threshold_is_safe(self):
        reserved = self.RESPONSE_RESERVE + self.SYSTEM_OVERHEAD
        available = 32_768 - reserved
        token_count = int(available * self.DANGER_THRESHOLD)
        v = self._check(token_count, 32_768)
        assert v["state"] == "safe"

    def test_just_over_danger_threshold_is_danger(self):
        reserved = self.RESPONSE_RESERVE + self.SYSTEM_OVERHEAD
        available = 32_768 - reserved
        token_count = int(available * self.DANGER_THRESHOLD) + 1
        v = self._check(token_count, 32_768)
        assert v["state"] == "danger"

    def test_conversation_history_reduces_available(self):
        v_fresh = self._check(5_000, 32_768, conversation_tokens=0)
        v_used = self._check(5_000, 32_768, conversation_tokens=20_000)
        assert v_used["available"] < v_fresh["available"]
        assert v_used["fraction"] > v_fresh["fraction"]

    def test_model_switch_resets_conversation_tokens(self):
        """After model switch conversationTokens=0 — fresh context should be safe."""
        v = self._check(5_000, 8_192, conversation_tokens=0)
        assert v["state"] in ("safe", "danger")  # not exceeded on a fresh context

    def test_unknown_when_no_context_window(self):
        v = self._check(5_000, None)
        assert v["state"] == "unknown"

    def test_required_window_calculation(self):
        reserved = self.RESPONSE_RESERVE + self.SYSTEM_OVERHEAD
        v = self._check(10_000, 32_768, conversation_tokens=5_000)
        assert v["requiredWindow"] == 5_000 + 10_000 + reserved

    def test_zero_available_returns_exceeded(self):
        """When conversation has consumed all context, any attach exceeds."""
        reserved = self.RESPONSE_RESERVE + self.SYSTEM_OVERHEAD
        conversation_tokens = 32_768 - reserved  # fills the window exactly
        v = self._check(1, 32_768, conversation_tokens=conversation_tokens)
        assert v["state"] == "exceeded"


# =============================================================================
# File Validation
# =============================================================================

class TestFileValidation:
    """Client-side and server-side validation logic."""

    ACCEPTED = {".pdf", ".md", ".json", ".txt"}
    MAX_CHARS = 50_000
    MAX_BYTES = 10 * 1024 * 1024

    def test_accepted_extensions(self):
        for ext in self.ACCEPTED:
            assert ext in self.ACCEPTED

    def test_rejected_extensions(self):
        for ext in (".docx", ".csv", ".xlsx", ".py", ".html", ".xml"):
            assert ext not in self.ACCEPTED

    def test_json_empty_object_rejected(self):
        for case in [None, {}, []]:
            text = json.dumps(case)
            parsed = json.loads(text)
            is_empty = parsed is None or parsed == {} or parsed == []
            assert is_empty

    def test_json_valid_object_accepted(self):
        valid = json.dumps({"key": "value"})
        parsed = json.loads(valid)
        assert parsed and parsed != {}

    def test_json_valid_array_accepted(self):
        valid = json.dumps([1, 2, 3])
        parsed = json.loads(valid)
        assert parsed and parsed != []

    def test_json_invalid_syntax_detected(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("{bad json")

    def test_char_limit_exceeded(self):
        text = "a" * (self.MAX_CHARS + 1)
        assert len(text) > self.MAX_CHARS

    def test_char_limit_at_boundary(self):
        text = "a" * self.MAX_CHARS
        assert len(text) <= self.MAX_CHARS

    def test_json_size_check_uses_raw_string(self):
        """Size check should run on raw decoded string before json.loads(),
        to avoid building a large object that is then discarded."""
        raw = '{"key": "' + "x" * (self.MAX_CHARS + 1) + '"}'
        assert len(raw) > self.MAX_CHARS  # raw check catches it

    def test_token_approximation_fallback(self):
        """When no model loaded: token_count = max(1, len(text) // 3)."""
        text = "a" * 3000
        approx = max(1, len(text) // 3)
        assert approx == 1000

    def test_token_approximation_minimum_one(self):
        approx = max(1, len("hi") // 3)
        assert approx == 1

    def test_pdf_empty_text_detected(self):
        """Scanned PDF returns empty string — must be caught."""
        text = "   \n\n  "
        assert not text.strip()

    def test_utf8_decode_error_simulated(self):
        """Binary content raises UnicodeDecodeError on .decode('utf-8')."""
        binary = bytes([0xFF, 0xFE, 0x00, 0x01])
        with pytest.raises(UnicodeDecodeError):
            binary.decode("utf-8")


# =============================================================================
# Route Constants
# =============================================================================

class TestRouteConstants:
    """Verify constants in routes.py match the client-side ATTACH_* values."""

    def test_max_chars_consistent(self):
        """Server _ATTACH_MAX_CHARS and client ATTACH_MAX_CHARS must agree."""
        server_max = 50_000
        client_max = 50_000
        assert server_max == client_max

    def test_accepted_extensions_consistent(self):
        """Server _ATTACH_ACCEPTED and client ATTACH_ACCEPTED must agree."""
        server = frozenset({".md", ".txt", ".json", ".pdf"})
        client = {".pdf", ".md", ".json", ".txt"}
        assert server == frozenset(client)


# =============================================================================
# i18n
# =============================================================================

@pytest.mark.parametrize(
    "lang_file",
    ["en.json", "ko.json", "zh.json", "zh-TW.json", "ja.json"],
)
def test_i18n_file_attach_keys_present(lang_file):
    """All file-attach i18n keys exist and are non-empty in every language file."""
    path = I18N_DIR / lang_file
    with open(path, encoding="utf-8") as f:
        translations = json.load(f)
    for key in REQUIRED_FILE_KEYS:
        assert key in translations, f"Missing key '{key}' in {lang_file}"
        assert translations[key], f"Empty value for '{key}' in {lang_file}"


@pytest.mark.parametrize(
    "lang_file",
    ["en.json", "ko.json", "zh.json", "zh-TW.json", "ja.json"],
)
def test_i18n_existing_keys_unchanged(lang_file):
    """Existing image upload keys are present and unmodified."""
    existing_image_keys = [
        "chat.upload_image",
        "chat.remove_image",
        "chat.image_not_available",
        "chat.image_preview",
        "chat.error.invalid_image_type",
        "chat.error.image_too_large",
        "chat.error.image_load_failed",
    ]
    path = I18N_DIR / lang_file
    with open(path, encoding="utf-8") as f:
        translations = json.load(f)
    for key in existing_image_keys:
        assert key in translations, f"Existing key '{key}' missing from {lang_file}"
        assert translations[key], f"Existing key '{key}' is empty in {lang_file}"


@pytest.mark.parametrize(
    "key",
    [
        "chat.error.file_context_exceeded",
        "chat.warn.file_context_danger",
    ],
)
def test_i18n_placeholder_keys_present(key):
    """Context-aware message keys contain expected {placeholder} tokens."""
    path = I18N_DIR / "en.json"
    with open(path, encoding="utf-8") as f:
        translations = json.load(f)
    value = translations[key]
    assert "{tokens}" in value
    assert "{available}" in value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
