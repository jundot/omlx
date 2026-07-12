"""Tests for the enable_thinking toggle and detect_thinking_default heuristic."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from omlx.model_discovery import detect_preserve_thinking, detect_thinking_default
from omlx.model_settings import ModelSettings
from omlx.server import _apply_laguna_enable_thinking_default

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
# Laguna serving default
# ---------------------------------------------------------------------------


class TestLagunaThinkingDefault:
    """Test Poolside's recommended Laguna request default and precedence."""

    def test_enables_recommended_laguna_thinking_default(self):
        template_kwargs = {}

        _apply_laguna_enable_thinking_default(
            template_kwargs,
            SimpleNamespace(config_model_type="laguna", thinking_default=True),
        )

        assert template_kwargs == {"enable_thinking": True}

    def test_does_not_override_non_laguna_thinking_default(self):
        template_kwargs = {}

        _apply_laguna_enable_thinking_default(
            template_kwargs,
            SimpleNamespace(config_model_type="gemma4", thinking_default=False),
        )

        assert template_kwargs == {}

    def test_preserves_explicit_thinking_disable(self):
        template_kwargs = {"enable_thinking": False}

        _apply_laguna_enable_thinking_default(
            template_kwargs,
            SimpleNamespace(config_model_type="laguna", thinking_default=True),
        )

        assert template_kwargs == {"enable_thinking": False}

    def test_preserves_explicit_thinking_enable(self):
        template_kwargs = {"enable_thinking": True}

        _apply_laguna_enable_thinking_default(
            template_kwargs,
            SimpleNamespace(config_model_type="laguna", thinking_default=True),
        )

        assert template_kwargs == {"enable_thinking": True}

    def test_ignores_laguna_without_thinking_capability(self):
        template_kwargs = {}

        _apply_laguna_enable_thinking_default(
            template_kwargs,
            SimpleNamespace(config_model_type="laguna", thinking_default=None),
        )

        assert template_kwargs == {}


def _server_route_node(route_name: str) -> ast.AsyncFunctionDef:
    """Return an API route AST node without importing or running the server."""
    server_path = Path(__file__).resolve().parents[1] / "omlx" / "server.py"
    server_source = server_path.read_text()
    for node in ast.walk(ast.parse(server_source)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == route_name:
            return node
    raise AssertionError(f"{route_name} not found in server.py")


def test_all_chat_apis_apply_laguna_thinking_default():
    """Every chat API must use the shared caller-precedence-safe policy."""
    route_names = (
        "create_chat_completion",
        "create_anthropic_message",
        "create_response",
    )

    for route_name in route_names:
        helper_calls = [
            call
            for call in ast.walk(_server_route_node(route_name))
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_apply_laguna_enable_thinking_default"
            )
        ]

        assert any(
            len(call.args) == 2
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "merged_ct_kwargs"
            for call in helper_calls
        ), f"{route_name} must apply the Laguna thinking default"


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
