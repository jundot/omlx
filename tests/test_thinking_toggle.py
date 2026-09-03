"""Tests for the enable_thinking toggle and detect_thinking_default heuristic."""

import json

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

    def test_explicit_default_true_pattern_returns_true(self, tmp_path):
        """Laguna S-2.1 pattern: enable_thinking | default(true) means ON even
        when another flag in the same template defaults to false."""
        template = (
            "{%- set enable_thinking = enable_thinking | default(true) -%}\n"
            "{%- set preserve_thinking = preserve_thinking | default(false) -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is True

    def test_enable_thinking_paren_pattern_returns_false(self, tmp_path):
        """Template that references enable_thinking) returns False."""
        template = "{%- if default(enable_thinking) -%}think{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_ternary_else_true_returns_true(self, tmp_path):
        """Nemotron / fixed-template repos: a fallback ternary states the ON
        default without a ``| default(true)`` filter."""
        template = (
            "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True -%}\n"
            "{%- if enable_thinking %}...{% endif %}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is True

    def test_opt_in_is_defined_and_true_returns_false(self, tmp_path):
        """Qwen 3.5 pattern: thinking is emitted only when explicitly asked."""
        template = "{%- if enable_thinking is defined and enable_thinking is true %}...{% endif %}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_opt_in_equality_returns_false(self, tmp_path):
        """LongCat pattern: ``== true`` / ``== false`` comparisons, which the
        ``default(false)`` scan does not see."""
        template = (
            "{%- if enable_thinking == true %}...{% elif enable_thinking == false %}...{% endif %}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_qwen_pattern_wins_over_later_opt_in_test(self, tmp_path):
        """A template that defaults ON (``is false`` guard first) stays ON even
        if it also tests for truth elsewhere."""
        template = (
            "{%- if enable_thinking is false %}...{% endif %}\n"
            "{%- if enable_thinking is defined and enable_thinking is true %}...{% endif %}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is True

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


# ---------------------------------------------------------------------------
# detect_reasoning_effort
# ---------------------------------------------------------------------------


class TestDetectReasoningEffort:
    """Test chat template detection of the reasoning_effort contract."""

    def test_qwen38_whitelist_and_default(self, tmp_path):
        """Qwen3.8 pattern: strict tuple whitelist + |default('xhigh')."""
        template = (
            "{%- set reasoning_effort = reasoning_effort | default('xhigh') -%}\n"
            "{%- if reasoning_effort not in ('xhigh', 'medium', 'low') -%}\n"
            "  {{- raise_exception('Invalid reasoning effort') -}}\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_reasoning_effort(tmp_path) == (["xhigh", "medium", "low"], "xhigh")

    def test_gpt_oss_free_form_names_only_its_default(self, tmp_path):
        """gpt-oss pattern: free-form value with a defined-check default. The
        template names exactly one level, and that is what is reported — the
        client decides whether one value is a usable menu."""
        template = (
            "{%- if reasoning_effort is not defined -%}\n"
            "  {%- set reasoning_effort = 'medium' -%}\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_reasoning_effort(tmp_path) == (["medium"], "medium")

    def test_off_requires_a_thinking_knob_or_an_explicit_level(self, tmp_path):
        """Discovery reports only what the template names. ``none`` appears here
        because the template itself lists it, not because it was inferred."""
        template = (
            "{{- 'Reasoning: ' + reasoning_effort }}\n"
            "{%- if reasoning_effort == 'low' %}...{% endif %}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        options, _ = detect_reasoning_effort(tmp_path)
        assert "none" not in options

    def test_normalising_ternary_enumerates_levels(self, tmp_path):
        """A template that coerces rather than raises still discloses its
        levels: the compared value and the coerced value are both real."""
        template = (
            "{%- if reasoning_effort is defined and reasoning_effort != 'high' -%}\n"
            "  {%- set reasoning_effort = 'max' -%}\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_reasoning_effort(tmp_path) == (["high", "max"], None)

    def test_list_membership_and_ternary_default(self, tmp_path):
        """GLM-5.3 pattern: list literal (not tuple) plus an ``else`` fallback.
        The fallback is both the default and an accepted level."""
        template = (
            "{%- set effective_reasoning_effort = reasoning_effort "
            "if reasoning_effort is defined and reasoning_effort in ['low', 'high'] "
            "else 'max' -%}\n"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_reasoning_effort(tmp_path) == (["low", "high", "max"], "max")

    def test_multi_branch_normalisation_unions_all_levels(self, tmp_path):
        """Qwen-fixed-template pattern: several membership tests alias many
        inputs onto a few canonical efforts. Every one is a level the template
        recognises, so the union is the menu."""
        template = (
            "{%- set _effort_raw = (reasoning_effort | string | lower) "
            "if reasoning_effort is defined else 'medium' -%}\n"
            "{%- if _effort_raw in ('none', 'off') -%}\n"
            "{%- elif _effort_raw in ('minimal', 'low') -%}\n"
            "{%- elif _effort_raw in ('high', 'xhigh', 'max') -%}\n"
            "{%- endif -%}\n"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        options, default = detect_reasoning_effort(tmp_path)
        # Template order, verbatim, nothing invented.
        assert options == ["none", "off", "minimal", "low", "high", "xhigh", "max", "medium"]
        assert default == "medium"

    def test_levels_keep_template_order(self, tmp_path):
        """Ordering is the template's, deliberately unmodified — sorting and
        vocabulary collapse are the client's concern."""
        template = (
            "{%- if reasoning_effort in ('extreme', 'ultracode', 'low') -%}\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        options, _ = detect_reasoning_effort(tmp_path)
        assert options == ["extreme", "ultracode", "low"]

    def test_dict_map_keys_are_levels(self, tmp_path):
        """Inkling pattern: a dict map's keys are the levels it knows about,
        even though the map is not an enforcement whitelist."""
        template = (
            "{%- set effort_map = {'none': 0.0, 'low': 0.2, 'high': 0.9} -%}\n"
            "{%- if key not in effort_map -%}\n"
            "  ...\n"
            "{%- endif -%}\n"
            "{%- if reasoning_effort is not defined -%}\n"
            "  {%- set reasoning_effort = 0.9 -%}\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_reasoning_effort(tmp_path) == (["none", "low", "high"], None)

    def test_no_reasoning_effort_returns_none(self, tmp_path):
        """Template without reasoning_effort (enable_thinking-only models)."""
        template = "{%- if enable_thinking is false -%}...{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_reasoning_effort(tmp_path) == (None, None)

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns (None, None)."""
        assert detect_reasoning_effort(tmp_path) == (None, None)

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """Template embedded in tokenizer_config.json is detected."""
        template = (
            "{%- set reasoning_effort = reasoning_effort | default('high') -%}\n"
            "{%- if reasoning_effort not in ('high', 'low') -%}{% endif -%}"
        )
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": template})
        )
        assert detect_reasoning_effort(tmp_path) == (["high", "low"], "high")
