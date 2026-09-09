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


@pytest.mark.parametrize("device", ["gpu", "ane"])
def test_skipped_ane_comparison_retains_measured_gpu_value(device):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise dashboard JavaScript")
    source = (
        Path(__file__).parents[1] / "omlx/admin/static/js/dashboard.js"
    ).read_text()
    method = re.search(
        r"^            aneComparisonValue\([^\n]*\) \{.*?^            \},",
        source,
        re.M | re.S,
    ).group()
    row = dict(gpu=50, ane=None, unit="tok/s", unavailable="not_tested")
    script = (
        "const window = {t: key => key}; const app = {"
        + method
        + "};"
        + f"console.log(app.aneComparisonValue({json.dumps(row)}, {json.dumps(device)}));"
    )
    actual = subprocess.check_output([node, "-e", script], text=True).strip()
    assert actual == (
        "50.0 tok/s" if device == "gpu" else "modal.model_settings.ane_eval_not_tested"
    )
