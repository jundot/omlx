"""Exercise Chat request settings for K2 and other model families."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


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
    source = (Path(__file__).parents[1] / "omlx/admin/templates/chat.html").read_text()
    names = [
        "currentModelInfo",
        "thinkingModes",
        "thinkingModeValue",
        "normalizeThinkingBudgetTokens",
        "setThinkingMode",
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
        "settings": {},
    }
    from omlx.admin.routes import _model_options
    from omlx.model_settings import ModelSettings

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
app.setThinkingMode('off');
console.log(JSON.stringify({{mode, generation, afterOff: app.modelSettings.enable_thinking}}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["mode"] == expected_mode
    assert result["generation"] == {"max_tokens": 128, **expected_generation}
    assert result["afterOff"] is (enabled if model_type == "k2_horizon" else False)
