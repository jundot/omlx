"""Exercise Chat request settings for K2 and other model families."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from omlx.admin.routes import _model_options
from omlx.model_settings import ModelSettings


@pytest.mark.parametrize(
    "model_type, enabled, budget, expected_mode, expected_generation",
    [
        ("k2_horizon", None, True, "on_limit", {"thinking_budget": 4096}),
        ("k2_horizon", False, True, "on_limit", {"thinking_budget": 4096}),
        ("k2_horizon", True, True, "on_limit", {"thinking_budget": 4096}),
        ("k2_horizon", False, False, "auto", {}),
        ("k2_horizon", True, False, "auto", {}),
        ("qwen3", True, True, "on_limit", {"chat_template_kwargs": {"enable_thinking": True}, "thinking_budget": 4096}),
        ("qwen3", True, False, "on_unlimit", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("qwen3", False, True, "off", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("qwen3", False, False, "off", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("qwen3", None, True, "auto", {}),
        ("qwen3", None, False, "auto", {}),
    ],
)
def test_saved_thinking_settings_respect_model_capabilities(
    model_type, enabled, budget, expected_mode, expected_generation
):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = (Path(__file__).parents[1] / "omlx_web/templates/chat.html").read_text()
    names = [
        "currentModelInfo",
        "thinkingModes",
        "thinkingModeValue",
        "normalizeThinkingBudgetTokens",
        "reasoningLevels",
        "samplingSource",
        "snapshotGenerationSettings",
    ]
    methods = [
        re.search(
            r"^            " + name + r"\([^\n]*\) \{.*?^            \},",
            source,
            re.M | re.S,
        ).group()
        for name in names
    ]
    model = {
        "id": "base",
        "config_model_type": model_type,
        "thinking_default": True,
        "settings": {},
    }

    model.update(_model_options(model, ModelSettings()))
    # Presentation must follow the response even if a future model has another name.
    model["config_model_type"] = "future_model"
    script = (
        "const app = {"
        + "\n".join(methods)
        + "};\n"
        + f"""
app.currentModel = 'alias';
app.aliasToGateway = {{alias: 'base'}};
app._adminModelsList = [{json.dumps(model)}];
app.modelSettings = {{max_tokens: 128, enable_thinking: {json.dumps(enabled)},
    thinking_budget_enabled: {json.dumps(budget)}, thinking_budget_tokens: 4096}};
app.onModelSettingsChange = () => {{}};
const mode = app.thinkingModeValue();
const generation = app.snapshotGenerationSettings();
console.log(JSON.stringify({{mode, generation}}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["mode"] == expected_mode
    assert result["generation"] == {"max_tokens": 128, **expected_generation}


@pytest.mark.parametrize(
    "model_info, stops, level, expected_generation",
    [
        (
            {"config_model_type": "qwen3_5_moe", "thinking_default": True},
            ["off", "auto", "on"],
            "on",
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
        (
            {"config_model_type": "qwen3_5_moe", "thinking_default": True},
            ["off", "auto", "on"],
            "off",
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            {
                "config_model_type": "qwen3_5",
                "thinking_default": True,
                "reasoning_efforts": ["low", "medium", "xhigh"],
            },
            ["off", "auto", "low", "medium", "xhigh"],
            "medium",
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_effort": "medium",
            },
        ),
        (
            {"config_model_type": "gpt_oss", "reasoning_efforts": []},
            ["auto", "low", "medium", "high"],
            "high",
            {"reasoning_effort": "high"},
        ),
        (
            {"config_model_type": "deepseek_v4_mtp"},
            ["off", "auto", "low", "high", "max"],
            "max",
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_effort": "max",
            },
        ),
        (
            {"config_model_type": "k2_horizon", "thinking_default": True},
            ["auto", "low", "medium", "high"],
            "low",
            {"reasoning_effort": "low"},
        ),
        (
            {"config_model_type": "glm5_next", "reasoning_efforts": ["low", "high"]},
            ["auto", "low", "high"],
            "high",
            {"reasoning_effort": "high"},
        ),
        (
            {"config_model_type": "cohere2_moe", "reasoning_efforts": []},
            ["auto", "low", "medium", "high", "xhigh", "max"],
            "xhigh",
            {"reasoning_effort": "xhigh"},
        ),
        ({"config_model_type": "llama"}, [], "auto", {}),
    ],
)
def test_reasoning_slider_maps_model_stops_to_the_request(
    model_info, stops, level, expected_generation
):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = (Path(__file__).parents[1] / "omlx_web/templates/chat.html").read_text()
    names = [
        "currentModelInfo",
        "thinkingModes",
        "thinkingModeValue",
        "normalizeThinkingBudgetTokens",
        "reasoningLevels",
        "reasoningStops",
        "reasoningLevel",
        "setReasoningLevel",
        "setThinkingBudgetEnabled",
        "samplingSource",
        "snapshotGenerationSettings",
    ]
    methods = [
        re.search(
            r"^            " + name + r"\([^\n]*\) \{.*?^            \},",
            source,
            re.M | re.S,
        ).group()
        for name in names
    ]

    model = {"id": "base", "settings": {}, **model_info}
    model.update(_model_options(model, ModelSettings()))
    script = (
        "const app = {"
        + "\n".join(methods)
        + "};\n"
        + f"""
app.currentModel = 'base';
app.aliasToGateway = {{}};
app._adminModelsList = [{json.dumps(model)}];
app.modelSettings = {{enable_thinking: null, reasoning_effort: null,
    thinking_budget_enabled: false, thinking_budget_tokens: null}};
app.onModelSettingsChange = () => {{}};
app.modelSettings.enable_thinking = false;
app.modelSettings.reasoning_effort = 'max';
const stale = app.snapshotGenerationSettings();
app.modelSettings.enable_thinking = null;
app.modelSettings.reasoning_effort = null;
const auto = app.snapshotGenerationSettings();
app.setThinkingBudgetEnabled(true);
app.setThinkingBudgetEnabled(false);
const budgetCycle = app.snapshotGenerationSettings();
app.setReasoningLevel('auto');
app.setReasoningLevel({json.dumps(level)});
const picked = app.reasoningLevel();
const generation = app.snapshotGenerationSettings();
app.setReasoningLevel('auto');
console.log(JSON.stringify({{stops: app.reasoningStops(), stale, budgetCycle, auto,
    picked, generation, reset: app.snapshotGenerationSettings()}}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["stops"] == stops
    if not stops:
        # Saved thinking state stays local when the model shows no control.
        assert result["stale"] is None
    if "off" not in stops:
        # A saved Off only goes out while the slider has an Off stop.
        assert "chat_template_kwargs" not in (result["stale"] or {})
    # A budget turns thinking on; with an On stop the slider shows it and keeps it.
    on = {"chat_template_kwargs": {"enable_thinking": True}}
    assert result["budgetCycle"] == (on if "on" in stops else None)
    assert result["auto"] is None
    assert result["picked"] == level
    assert (result["generation"] or {}) == expected_generation
    assert result["reset"] is None
