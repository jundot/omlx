"""Exercise Chat request settings for K2 and other model families."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "model_type,expected_mode,expected_generation",
    [
        ("k2_horizon", "on_limit", {"max_tokens": 128, "thinking_budget": 4096}),
        (
            "qwen3",
            "on_limit",
            {
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_budget": 4096,
            },
        ),
    ],
)
def test_saved_thinking_settings_respect_model_capabilities(
    model_type, expected_mode, expected_generation
):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = (Path(__file__).parents[1] / "omlx/admin/templates/chat.html").read_text()
    names = [
        "currentModelInfo",
        "isK2Model",
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
    script = (
        "const app = {"
        + "\n".join(methods)
        + "};\n"
        + f"""
app.currentModel = 'alias';
app.aliasToGateway = {{alias: 'base'}};
app._adminModelsList = [{json.dumps(model)}];
app.modelSettings = {{max_tokens: 128, enable_thinking: true,
    thinking_budget_enabled: true, thinking_budget_tokens: 4096}};
app.onModelSettingsChange = () => {{}};
const mode = app.thinkingModeValue();
const generation = app.snapshotGenerationSettings();
app.setThinkingMode('off');
console.log(JSON.stringify({{mode, generation, afterOff: app.modelSettings.enable_thinking}}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["mode"] == expected_mode
    assert result["generation"] == expected_generation
    assert result["afterOff"] is (model_type == "k2_horizon")
