"""Tests for the enable_thinking toggle and detect_thinking_default heuristic."""

import json
from pathlib import Path

import pytest

from omlx.model_discovery import detect_thinking_default
from omlx.model_settings import ModelSettings


# ---------------------------------------------------------------------------
# detect_thinking_default
# ---------------------------------------------------------------------------


class TestDetectThinkingDefault:
    """Test chat template heuristic for thinking default detection."""

    def test_qwen_pattern_returns_true(self, tmp_path):
        """Qwen3 pattern: thinking is ON by default, only suppressed when
        enable_thinking is explicitly false."""
        template = (
            "{%- if enable_thinking is false -%}\n"
            "  ... suppress thinking ...\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is True

    def test_gemma_default_false_pattern_returns_false(self, tmp_path):
        """Gemma4 pattern: thinking is OFF by default, requires explicit enable."""
        template = "{%- set thinking = enable_thinking | default(false) -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_enable_thinking_paren_pattern_returns_false(self, tmp_path):
        """Template that references enable_thinking) returns False."""
        template = "{%- if default(enable_thinking) -%}think{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_no_enable_thinking_returns_none(self, tmp_path):
        """Template without enable_thinking reference returns None."""
        template = "{{ messages[0].content }}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is None

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_thinking_default(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        # Jinja file says Qwen pattern (True)
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        )
        # tokenizer_config says Gemma pattern (False)
        tc = {"chat_template": "{%- set t = enable_thinking | default(false) -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        assert detect_thinking_default(tmp_path) is True

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """When no jinja file exists, reads from tokenizer_config.json."""
        tc = {"chat_template": "{%- if enable_thinking is false -%}ok{%- endif -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert detect_thinking_default(tmp_path) is True

    def test_tokenizer_config_without_chat_template_key(self, tmp_path):
        """tokenizer_config.json without chat_template key returns None."""
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_thinking_default(tmp_path) is None

    def test_unrecognized_pattern_returns_none(self, tmp_path):
        """Template with enable_thinking but no recognized pattern returns None."""
        template = "{%- if enable_thinking == 'maybe' -%}hmm{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is None

    def test_malformed_tokenizer_config_returns_none(self, tmp_path):
        """Malformed JSON in tokenizer_config.json returns None gracefully."""
        (tmp_path / "tokenizer_config.json").write_text("not valid json{{{")
        assert detect_thinking_default(tmp_path) is None


# ---------------------------------------------------------------------------
# ModelSettings.enable_thinking field
# ---------------------------------------------------------------------------


class TestModelSettingsEnableThinking:
    """Test enable_thinking field on ModelSettings dataclass."""

    def test_default_is_none(self):
        ms = ModelSettings()
        assert ms.enable_thinking is None

    def test_set_to_true(self):
        ms = ModelSettings(enable_thinking=True)
        assert ms.enable_thinking is True

    def test_set_to_false(self):
        ms = ModelSettings(enable_thinking=False)
        assert ms.enable_thinking is False
