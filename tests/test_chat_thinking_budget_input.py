"""Exercise the chat sidebar's thinking-budget number field.

The field used to fight the person typing in it: `x-model.number` wrote the
parsed digit to the model and then the `@input` handler re-stepped any change
smaller than the step size, so typing "512" into an empty field produced 1029
(and, before `x-model.number` was dropped, 1025). These tests run the real
handlers out of `chat.html` and pin what the field does now.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_TEMPLATE = Path(__file__).parents[1] / "omlx/admin/templates/chat.html"

# The handlers are read straight out of the template so the test cannot drift
# from what the browser runs.
METHODS = [
    "currentModelInfo",
    "thinkingModes",
    "thinkingModeValue",
    "normalizeThinkingBudgetTokens",
    "stepThinkingBudgetTokens",
    "onThinkingBudgetTokensInput",
    "clampThinkingBudgetTokens",
]


def _method_sources():
    source = CHAT_TEMPLATE.read_text()
    return [
        re.search(
            r"^            " + name + r"\([^\n]*\) \{.*?^            \},",
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
        "const app = {"
        + "\n".join(_method_sources())
        + "};\n"
        + """
app.modelSettings = {enable_thinking: true, thinking_budget_enabled: true,
                     thinking_budget_tokens: null};
app._thinkingBudgetBeforeInput = null;
app.onModelSettingsChange = () => {};
app.el = {value: ''};
// The field only shows for a model whose thinking modes include `on_limit`.
app.currentModel = 'base';
app.aliasToGateway = {base: 'base'};
app._adminModelsList = [{id: 'base', thinking_modes: ['auto', 'on_limit', 'off'],
                         thinking_forced: true}];
"""
        + body
    )
    return json.loads(subprocess.check_output([node, "-e", script], text=True))


def _type(value: str):
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
    result = _run(
        _type("512")
        + """
app.modelSettings.thinking_budget_tokens = 4096;
app._thinkingBudgetBeforeInput = 4096;
type('512');
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens,
                            field: app.el.value}));
"""
    )
    assert result["model"] == 512
    assert result["field"] == "512"


def test_typing_a_four_digit_budget_is_not_re_stepped():
    result = _run(
        _type("2048")
        + """
app.modelSettings.thinking_budget_tokens = 4096;
app._thinkingBudgetBeforeInput = 4096;
type('2048');
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
"""
    )
    assert result["model"] == 2048


def test_clearing_the_field_records_no_budget():
    result = _run(
        """
app.modelSettings.thinking_budget_tokens = 4096;
app._thinkingBudgetBeforeInput = 4096;
app.el.value = '';
app.onThinkingBudgetTokensInput({target: app.el});
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens,
                            before: app._thinkingBudgetBeforeInput}));
"""
    )
    assert result["model"] is None
    assert result["before"] is None


def test_blur_keeps_a_small_typed_value():
    """The clamp only repairs empty/garbage input; 512 is a valid budget."""
    result = _run(
        """
app.modelSettings.thinking_budget_tokens = 512;
app.el.value = '512';
app.clampThinkingBudgetTokens();
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
"""
    )
    assert result["model"] == 512


def test_blur_repairs_an_emptied_field():
    result = _run(
        """
app.modelSettings.thinking_budget_tokens = null;
app.el.value = '';
app.clampThinkingBudgetTokens();
console.log(JSON.stringify({model: app.modelSettings.thinking_budget_tokens}));
"""
    )
    assert result["model"] == 4096


def test_arrow_keys_still_step_by_one_unit():
    result = _run(
        """
app.modelSettings.thinking_budget_tokens = 4096;
app._thinkingBudgetBeforeInput = 4096;
app.stepThinkingBudgetTokens(1);
const up = app.modelSettings.thinking_budget_tokens;
app.stepThinkingBudgetTokens(-1);
const down = app.modelSettings.thinking_budget_tokens;
console.log(JSON.stringify({up: up, down: down, before: app._thinkingBudgetBeforeInput}));
"""
    )
    assert result["up"] == 5120
    assert result["down"] == 4096
    assert result["before"] == 4096


def test_native_spinner_grid_matches_the_step():
    """The input's own step must move in the same unit as the arrow keys.

    The field is the only writer, so the native spinner can no longer be
    re-stepped by the handler; its grid has to be 1024-aligned instead.
    """
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
    assert ':value="modelSettings.thinking_budget_tokens ?? \'\'"' in field


def test_the_re_stepping_helper_is_gone():
    """`_thinkingBudgetStepping` existed only to guard the removed branch."""
    source = CHAT_TEMPLATE.read_text()
    assert "_thinkingBudgetStepping" not in source
