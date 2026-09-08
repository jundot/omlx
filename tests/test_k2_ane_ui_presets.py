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
    source = (Path(__file__).parents[1] / "omlx/admin/static/js/dashboard.js").read_text()
    method = re.search(
        r"^            aneFractionOptions\([^\n]*\) \{.*?^            \},",
        source,
        re.M | re.S,
    ).group()
    from omlx.admin.routes import _model_options

    metadata = _model_options({"config_model_type": "k2_horizon"}, None)
    presets = metadata[
        "ane_prefill_shared_fractions" if shared else "ane_prefill_mlp_fractions"
    ]
    script = (
        "const app = {"
        + method
        + "}; console.log(JSON.stringify("
        + f"app.aneFractionOptions({json.dumps(current)}, {json.dumps(presets)})"
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


@pytest.mark.parametrize("model_type", ["k2_horizon", "qwen3_5"])
@pytest.mark.parametrize("saved_fraction", [None, 0.53, 1 / 3])
def test_shared_ane_web_state_preserves_backend_default_and_saved_fraction(
    model_type, saved_fraction
):
    from omlx.admin.routes import _model_options

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = (
        Path(__file__).parents[1] / "omlx/admin/static/js/dashboard.js"
    ).read_text()
    method = re.search(
        r"^            buildModelSettingsState\([^\n]*\) \{.*?^            \},",
        source,
        re.M | re.S,
    ).group()
    metadata = _model_options({"config_model_type": model_type}, None)
    saved = {
        "qwen35_ane_prefill_enabled": True,
        "qwen35_ane_prefill_fraction": saved_fraction,
        "qwen35_ane_prefill_shared_fraction": 0,
    }
    script = (
        "const OCR_CONFIG_MODEL_TYPES = new Set(); const app = {isDiffusionModel: () => false, buildCtKwargEntries: () => [],"
        + method
        + "};"
    )
    script += f"console.log(JSON.stringify(app.buildModelSettingsState({json.dumps(metadata)}, {json.dumps(saved)})));"
    state = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert state["qwen35_ane_prefill_enabled"] is True
    assert state["qwen35_ane_prefill_fraction"] == (
        saved_fraction
        if saved_fraction is not None
        else metadata["ane_prefill_default_fraction"]
    )
    assert state["qwen35_ane_prefill_shared_fraction"] == 0
    assert not any(key.startswith("k2_ane_") for key in state)
