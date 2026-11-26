"""Tests for the enable_thinking toggle and detect_thinking_default probe."""

import json

import pytest

from omlx.model_discovery import (
    detect_preserve_thinking,
    detect_reasoning_effort,
    detect_thinking_default,
)
from omlx.model_settings import ModelSettings

# ---------------------------------------------------------------------------
# detect_thinking_default
# ---------------------------------------------------------------------------


class TestDetectThinkingDefault:
    """Test the rendered chat template probe for the thinking default."""

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            # Qwen 3.x: thinking is suppressed only when explicitly false.
            (
                "{%- if enable_thinking is false -%}<think></think>{%- endif -%}",
                True,
            ),
            # Gemma 4: thinking requires an explicit opt-in.
            (
                "{%- if enable_thinking | default(false) -%}<think>{%- endif -%}",
                False,
            ),
            # Another flag defaulting to false must not decide the result.
            (
                "{%- set enable_thinking = enable_thinking | default(true) -%}\n"
                "{%- set preserve_thinking = preserve_thinking | default(false) -%}"
                "{%- if enable_thinking -%}<think>{%- endif -%}",
                True,
            ),
            # GLM-4.6V: the closing paren in the guard does not mean opt-in.
            (
                "{{- '<think></think>' if (enable_thinking is defined "
                "and not enable_thinking) else '' -}}",
                True,
            ),
            # Nemotron Cascade: default assigned with an inline if/else.
            (
                "{%- set enable_thinking = enable_thinking "
                "if enable_thinking is defined else True -%}"
                "{%- if enable_thinking -%}<think>{%- endif -%}",
                True,
            ),
        ],
    )
    def test_rendered_default(self, tmp_path, template, expected):
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is expected

    def test_no_enable_thinking_returns_none(self, tmp_path):
        """Template without enable_thinking reference returns None."""
        template = "{{ messages[0].content }}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is None

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_thinking_default(tmp_path) is None

    def test_laguna_template_uses_recommended_serving_default(self, tmp_path):
        """Laguna's effective default follows Poolside's serving recommendation."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}))
        (tmp_path / "chat_template.jinja").write_text(
            "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
            "{%- if not enable_thinking -%}</think>{%- else -%}<think>{%- endif -%}"
        )

        assert detect_thinking_default(tmp_path) is True

    def test_laguna_without_thinking_template_returns_none(self, tmp_path):
        """A Laguna config alone is not evidence that its template accepts the flag."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}))

        assert detect_thinking_default(tmp_path) is None

    def test_non_laguna_config_without_thinking_template_returns_none(self, tmp_path):
        """Reading config metadata must not enable thinking for other models."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

        assert detect_thinking_default(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        # Jinja file says Qwen pattern (True)
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        )
        # tokenizer_config says Gemma pattern (False)
        tc = {
            "chat_template": (
                "{%- if enable_thinking | default(false) -%}<think>{%- endif -%}"
            )
        }
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

    def test_toggle_without_prompt_effect_returns_none(self, tmp_path):
        """A template whose prompt does not change with the flag returns None."""
        template = "{%- if enable_thinking == 'maybe' -%}hmm{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is None

    def test_template_render_error_returns_none(self, tmp_path):
        """A template that fails to render must not break discovery."""
        template = (
            "{%- if enable_thinking is false -%}{{ raise_exception('no') }}"
            "{%- endif -%}"
        )
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


# ---------------------------------------------------------------------------
# detect_preserve_thinking
# ---------------------------------------------------------------------------


class TestDetectPreserveThinking:
    """Test chat template heuristic for preserve_thinking support detection."""

    def test_qwen36_pattern_returns_true(self, tmp_path):
        """Qwen 3.6+ pattern: template references preserve_thinking kwarg."""
        template = (
            "{%- if (preserve_thinking is defined and preserve_thinking is true) "
            "or (loop.index0 > ns.last_query_index) -%}\n"
            "  <think>{{ reasoning_content }}</think>\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_preserve_thinking(tmp_path) is True

    def test_no_preserve_thinking_returns_none(self, tmp_path):
        """Template without preserve_thinking reference returns None."""
        template = "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_preserve_thinking(tmp_path) is None

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_preserve_thinking(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if preserve_thinking -%}keep{%- endif -%}"
        )
        tc = {"chat_template": "{{ messages[0].content }}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        assert detect_preserve_thinking(tmp_path) is True

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """When no jinja file exists, reads from tokenizer_config.json."""
        tc = {"chat_template": "{%- if preserve_thinking -%}keep{%- endif -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert detect_preserve_thinking(tmp_path) is True

    def test_tokenizer_config_without_chat_template_key(self, tmp_path):
        """tokenizer_config.json without chat_template key returns None."""
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_preserve_thinking(tmp_path) is None

    def test_malformed_tokenizer_config_returns_none(self, tmp_path):
        """Malformed JSON in tokenizer_config.json returns None gracefully."""
        (tmp_path / "tokenizer_config.json").write_text("not valid json{{{")
        assert detect_preserve_thinking(tmp_path) is None


# ---------------------------------------------------------------------------
# ModelSettings.preserve_thinking field
# ---------------------------------------------------------------------------


class TestModelSettingsPreserveThinking:
    """Test preserve_thinking field on ModelSettings dataclass."""

    def test_default_is_none(self):
        ms = ModelSettings()
        assert ms.preserve_thinking is None

    def test_set_to_true(self):
        ms = ModelSettings(preserve_thinking=True)
        assert ms.preserve_thinking is True

    def test_set_to_false(self):
        ms = ModelSettings(preserve_thinking=False)
        assert ms.preserve_thinking is False


@pytest.mark.parametrize(
    "template, levels",
    [
        ("{{ messages }}", None),
        ("Reasoning: {{ reasoning_effort | default('medium') }}", []),
        # Qwen3.8 raises on anything outside its three levels.
        (
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
            "{%- elif resolved_reasoning_effort == 'low' %}",
            ["low", "medium", "xhigh"],
        ),
        # GLM-5.3 maps every other value to max; Hunyuan adds no_think.
        (
            "{%- set e = reasoning_effort if reasoning_effort in ['low', 'high'] %}",
            ["low", "high"],
        ),
        (
            "{%- if reasoning_effort not in ['high', 'low', 'no_think'] %}",
            ["low", "high"],
        ),
        # Muse Spark Open checks the keys of a level-to-strength map.
        (
            "{%- set reasoning_effort_map = "
            "{'low': 8, 'medium': 32, 'high': 256, 'xhigh': 512} -%}"
            "{%- if not (reasoning_effort is string and "
            "reasoning_effort in reasoning_effort_map) -%}",
            ["low", "medium", "high", "xhigh"],
        ),
    ],
)
def test_detect_reasoning_effort_reads_the_accepted_levels(tmp_path, template, levels):
    tc = {"chat_template": template}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
    assert detect_reasoning_effort(tmp_path) == levels
