"""Exercise the chat sidebar's thinking-budget input and persistence."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_TEMPLATE = Path(__file__).parents[1] / "omlx/admin/templates/chat.html"

METHODS = [
    "currentModelInfo",
    "thinkingModes",
    "thinkingModeValue",
    "normalizeThinkingBudgetTokens",
    "stepThinkingBudgetTokens",
    "onThinkingBudgetTokensInput",
    "clampThinkingBudgetTokens",
    "captureSessionModelSettings",
    "syncSessionModelSettingsFromUi",
    "onModelSettingsChange",
    "cancelPersistModelSettingsTimer",
    "schedulePersistModelSettings",
    "snapshotGenerationSettings",
]


def _method_sources():
    source = CHAT_TEMPLATE.read_text()
    return [
        re.search(
            r"^( +)" + name + r"\([^\n]*\) \{.*?^\1\},",
            source,
            re.M | re.S,
        ).group()
        for name in METHODS
    ]


def _run(body: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    script = (
        "let pendingSave;\n"
        "const setTimeout = callback => { pendingSave = callback; return 1; };\n"
        "const clearTimeout = () => { pendingSave = null; };\n"
        "const app = {" + "\n".join(_method_sources()) + "};\n" + """
app.modelSettings = {enable_thinking: true, thinking_budget_enabled: true,
                     thinking_budget_tokens: null};
app.currentChatId = 'test';
app.session = {};
app.getChatSession = () => app.session;
app.cloneData = value => JSON.parse(JSON.stringify(value));
app.persistChatSettingsToHistory = () => {};
app.el = {value: ''};
// The field only shows for a model whose thinking modes include `on_limit`.
app.currentModel = 'base';
app.aliasToGateway = {base: 'base'};
app._adminModelsList = [{id: 'base', thinking_modes: ['auto', 'on_limit', 'off'],
                         thinking_forced: true}];
""" + body
    )
    return json.loads(subprocess.check_output([node, "-e", script], text=True))


def _type():
    """Replace the field's contents one character at a time, as a person would.

    The first keystroke overwrites the selection; the rest append.
    """
    return """
const type = (text) => {
    for (let i = 0; i < text.length; i++) {
        app.el.value = i === 0 ? text[i] : app.el.value + text[i];
        app.onThinkingBudgetTokensInput({target: app.el});
    }
};
"""


def test_typed_value_lands_verbatim():
    """Typing a small budget must not be re-stepped into 1024 + delta."""
    result = _run(_type() + """
app.modelSettings.thinking_budget_tokens = 4096;
type('512');
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens,
                            field: app.el.value}));
""")
    assert result["model"] == 512
    assert result["field"] == "512"


def test_typing_a_four_digit_budget_is_not_re_stepped():
    result = _run(_type() + """
app.modelSettings.thinking_budget_tokens = 4096;
type('2048');
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
""")
    assert result["model"] == 2048


def test_clearing_the_field_records_no_budget():
    result = _run("""
app.modelSettings.thinking_budget_tokens = 4096;
app.el.value = '';
app.onThinkingBudgetTokensInput({target: app.el});
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
""")
    assert result["model"] is None


def test_blur_keeps_a_small_typed_value():
    """Blur preserves a valid budget below the spinner step."""
    result = _run("""
app.modelSettings.thinking_budget_tokens = 512;
app.el.value = '512';
app.clampThinkingBudgetTokens();
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
""")
    assert result["model"] == 512


def test_blur_repairs_an_emptied_field():
    result = _run("""
app.modelSettings.thinking_budget_tokens = null;
app.el.value = '';
app.clampThinkingBudgetTokens();
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
""")
    assert result["model"] == 4096


def test_arrow_keys_still_step_by_one_unit():
    result = _run("""
app.modelSettings.thinking_budget_tokens = 4096;
app.stepThinkingBudgetTokens(1);
const up = app.modelSettings.thinking_budget_tokens;
app.stepThinkingBudgetTokens(-1);
const down = app.modelSettings.thinking_budget_tokens;
console.log(JSON.stringify({up: up, down: down}));
""")
    assert result["up"] == 5120
    assert result["down"] == 4096


def test_native_spinner_grid_matches_the_step():
    source = CHAT_TEMPLATE.read_text()
    field = re.search(
        r"<input type=\"number\"\n(?:.*\n)*?.*thinking-budget-input", source
    ).group()
    assert 'step="1024"' in field
    assert 'min="0"' in field


def test_no_second_writer_is_left_on_the_field():
    source = CHAT_TEMPLATE.read_text()
    field = re.search(
        r"<input type=\"number\"\n(?:.*\n)*?.*thinking-budget-input", source
    ).group()
    assert "x-model" not in field
    assert ":value=\"modelSettings.thinking_budget_tokens ?? ''\"" in field


def test_autosave_preserves_empty_input_and_saves_normalized_budget():
    result = _run(_type() + """
app.modelSettings.thinking_budget_tokens = 4096;
app.el.value = '';
app.onThinkingBudgetTokensInput({target: app.el});
pendingSave();
const empty = app.modelSettings.thinking_budget_tokens;
const savedDefault = app.session.modelSettings.thinking_budget_tokens;
type('512');
pendingSave();
console.log(JSON.stringify({empty, savedDefault,
    typed: app.modelSettings.thinking_budget_tokens,
    saved: app.session.modelSettings.thinking_budget_tokens,
    requested: app.snapshotGenerationSettings().thinking_budget}));
""")
    assert result == {
        "empty": None,
        "savedDefault": 4096,
        "typed": 512,
        "saved": 512,
        "requested": 512,
    }
