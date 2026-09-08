"""Check percentage presets preserve saved allocation values."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("current", [0, 1 / 3, 0.5, 1, 0.42, 0.33333333])
def test_fraction_choices_preserve_current_value(shared, current):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise dashboard JavaScript")
    source = (
        Path(__file__).parents[1] / "omlx/admin/static/js/dashboard.js"
    ).read_text()
    method = re.search(
        r"^            k2AneFractionOptions\([^\n]*\) \{.*?^            \},",
        source,
        re.M | re.S,
    ).group()
    script = (
        "const app = {"
        + method
        + "}; console.log(JSON.stringify("
        + f"app.k2AneFractionOptions({json.dumps(current)}, {json.dumps(shared)})"
        + "));"
    )
    options = json.loads(subprocess.check_output([node, "-e", script], text=True))
    expected = [0, 1 / 3, 1] if shared else [1 / 3, 0.5]
    if current not in expected:
        expected.insert(0, current)
    assert [option["value"] for option in options] == expected
    assert (
        next(option["value"] for option in options if option["label"] == "33%") == 1 / 3
    )
