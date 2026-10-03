# SPDX-License-Identifier: Apache-2.0
"""Explicit reasoning controls flow from templates to the model catalog."""

import json

import pytest

from omlx.model_discovery import detect_reasoning_effort, discover_models

QWEN_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort') }}
    {%- endif %}
    {{- resolved_reasoning_effort }}
{%- endif %}
"""


@pytest.mark.parametrize("standalone", [True, False])
def test_discovers_native_efforts_and_default(tmp_path, standalone):
    model_dir = tmp_path / "custom-alias"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_8"}))
    (model_dir / "model.safetensors").write_bytes(b"0" * 100)
    if standalone:
        (model_dir / "chat_template.jinja").write_text(QWEN_TEMPLATE)
    else:
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "chat_template": QWEN_TEMPLATE,
                }
            )
        )
    model = discover_models(tmp_path)["custom-alias"]
    assert model.reasoning_effort_options == ["xhigh", "medium", "low"]
    assert model.reasoning_effort_default == "xhigh"


@pytest.mark.parametrize(
    "template",
    [
        "{{ enable_thinking }}",
        "{{ reasoning_effort }}",
        QWEN_TEMPLATE.replace(
            "('xhigh', 'medium', 'low')", "('high', 'medium', 'low')"
        ),
        QWEN_TEMPLATE.replace("('xhigh', 'medium', 'low')", "('xhigh', 1)"),
        QWEN_TEMPLATE.replace("('xhigh', 'medium', 'low')", "(native_options)"),
        QWEN_TEMPLATE.replace(
            "resolved_reasoning_effort not in", "other_variable not in"
        ),
    ],
)
def test_does_not_invent_efforts_without_explicit_enum(tmp_path, template):
    (tmp_path / "chat_template.jinja").write_text(template)
    assert detect_reasoning_effort(tmp_path) == ([], None)


@pytest.mark.parametrize("config", [None, [], {"chat_template": []}, "invalid"])
def test_missing_or_malformed_template_is_conservative(tmp_path, config):
    if config is not None:
        text = "{" if config == "invalid" else json.dumps(config)
        (tmp_path / "tokenizer_config.json").write_text(text)
    assert detect_reasoning_effort(tmp_path) == ([], None)


def test_standalone_template_takes_precedence(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{{ enable_thinking }}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": QWEN_TEMPLATE,
            }
        )
    )
    assert detect_reasoning_effort(tmp_path) == ([], None)


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_native_levels_reach_the_template_without_alias_retry(effort):
    from jinja2 import Environment

    from omlx.reasoning_effort import apply_chat_template_with_reasoning_effort_fallback

    class Template:
        calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return Environment().from_string(QWEN_TEMPLATE).render(**kwargs)

    target = Template()
    prompt = apply_chat_template_with_reasoning_effort_fallback(
        target, [], {"reasoning_effort": effort}
    )
    assert prompt.strip() == effort
    assert target.calls == 1
