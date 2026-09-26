# SPDX-License-Identifier: Apache-2.0
"""Structure of the Models page after the console refactor.

Main's Models page had five status treatments for the same state (a blue chip
for a downloading task, an amber one for a queued upload, a third for a
cancelled quantize job), two number-only filter fields per dimension, a bare
paragraph where an empty state belongs and a model-settings modal framed as a
generic dialog.

This PR routes every status through one tone map, splits the model memory figure
into the measured value and the estimate behind it, marks resident rows, gives
the downloader a real empty state, turns the parameter/size filters into range
sliders that read their own value back, and frames the settings modal as a
sheet whose empty sampling fields show what they inherit.

These are static assertions (the tone map, the memory cell and the slider
read-out are covered at runtime by tests/admin_dashboard.test.cjs).
"""

import json
import re
from pathlib import Path

import jinja2

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
TEMPLATES = ADMIN / "templates"
MODELS = (TEMPLATES / "dashboard" / "_models.html").read_text(encoding="utf-8")
MODAL = (TEMPLATES / "dashboard" / "_modal_model_settings.html").read_text(encoding="utf-8")
UI = (TEMPLATES / "components" / "ui.html").read_text(encoding="utf-8")
COMPONENTS_CSS = (ADMIN / "static" / "css" / "components.css").read_text(encoding="utf-8")
DASHBOARD_JS = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
EN = json.loads((ADMIN / "i18n" / "en.json").read_text(encoding="utf-8"))
LOCALES = sorted((ADMIN / "i18n").glob("*.json"))

# Every tone the shared Badge spec declares (components.css): the map may not
# invent a sixth one.
TONES = {"green", "orange", "red", "blue", "neutral"}


def _env() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)))
    env.globals.update(t=lambda key: key, static=lambda path: path, version="0.0.0")
    return env


def _samples() -> dict:
    """The status -> tone pairs the helper maps, read out of dashboard.js."""
    block = DASHBOARD_JS.split("const STATUS_TONES = {", 1)[1].split("};", 1)[0]
    return dict(re.findall(r"(\w+):\s*'(\w+)'", block))


def _dashboard_helper(name: str) -> str:
    """The body of one dashboard() method or const, up to its closing brace."""
    body = DASHBOARD_JS.split(f"{name}(", 1)[1]
    return body.split("\n            },", 1)[0]


# === One status vocabulary ===


def test_every_status_tone_is_a_declared_badge_tone():
    tones = set(_samples().values())
    assert tones, "dashboard.js has no status tone map"
    assert tones <= TONES, f"tones the Badge spec does not declare: {sorted(tones - TONES)}"
    for tone in TONES - {"neutral"}:
        assert f"badge--{tone}" in COMPONENTS_CSS


def test_the_tone_map_carries_the_agreed_meanings():
    tones = _samples()
    assert tones["loaded"] == "green"
    assert tones["completed"] == "green"
    assert tones["failed"] == "red"
    assert tones["error"] == "red"
    assert tones["cancelled"] == "orange"
    assert tones["warning"] == "orange"
    assert tones["partial"] == "orange"
    for status in ("pending", "downloading", "uploading", "loading", "quantizing", "saving"):
        assert tones[status] == "blue", status
    # Anything unlisted is unknown, and unknown is neutral.
    helper = _dashboard_helper("statusTone")
    assert "'neutral'" in helper
    assert "hasOwnProperty" in helper, "a status like 'constructor' must not pick up a tone"


def test_every_table_asks_the_one_helper():
    assert MODELS.count("tone_expr='statusTone(task.status)'") == 4, (
        "the download queues (hf + ms), the quantizer and the uploader"
    )
    assert "tone_expr='statusTone(managerModelStatus(model.name))'" in MODELS
    assert "tone_expr='statusTone(" in MODELS
    # The colour decisions themselves are gone from the templates.
    assert "task.status === 'downloading'," not in MODELS
    assert 'x-text="task.status"' not in MODELS


def test_status_labels_are_localized_and_fall_back_to_the_raw_status():
    helper = _dashboard_helper("statusLabel")
    assert "window.t(key)" in helper
    assert "String(status" in helper, "an unknown status is shown verbatim"

    keys = set(re.findall(r"'(models\.status\.[a-z_]+)'", DASHBOARD_JS))
    assert len(keys) == 12, sorted(keys)
    for key in sorted(keys):
        assert key in EN, key
        for locale in LOCALES:
            catalogue = json.loads(locale.read_text(encoding="utf-8"))
            assert key in catalogue, f"{locale.name} is missing {key}"
            assert str(catalogue[key]).strip(), f"{locale.name} {key} is empty"
    # The manager rows use the two states a local model can be in.
    assert "if (info.is_loading) return 'loading';" in DASHBOARD_JS
    assert "return info.loaded ? 'loaded' : 'unloaded';" in DASHBOARD_JS


def test_the_badge_macro_can_bind_its_tone():
    assert "tone_expr" in UI and "label_expr" in UI
    module = _env().get_template("components/ui.html").module
    html = str(module.badge("", tone=None, tone_expr="statusTone(x)", label_expr="statusLabel(x)"))
    assert ":class=\"'badge--' + statusTone(x)\"" in html
    assert 'x-text="statusLabel(x)"' in html
    assert "badge--None" not in html, "a bound tone must not also be written"
    # A caller that passes a tone still gets the written class.
    assert "badge--green" in str(module.badge("Running", tone="green", dot=True))


# === Memory: measured first, estimate behind it ===


def test_the_memory_cell_separates_the_measurement_from_the_estimate():
    body = _dashboard_helper("modelMemoryCell")
    assert "actual_size_formatted" in body
    assert "estimated_size_formatted" in body
    assert "'~' + measured" in body, "the measured footprint is a rough delta"
    assert "status.memory.estimated" in body, "the secondary line says it is an estimate"
    assert "'—'" in body, "a model with no figure at all gets a dash, not a zero"


def test_the_manager_table_renders_the_dual_value_cell():
    assert "ui.metric_dual(" in MODELS
    assert "metric_dual" in UI
    assert "modelMemoryCell(model.name).footprint" in MODELS
    assert "modelMemoryCell(model.name).estimate" in MODELS
    assert "models.manager.table.memory" in MODELS
    for name in ("metric-dual", "metric-dual__value", "metric-dual__secondary"):
        assert f".{name} {{" in COMPONENTS_CSS, name
    # The measured value is the large one: the base size wins over the aux step.
    value = COMPONENTS_CSS.split(".metric-dual__value {", 1)[1].split("}", 1)[0]
    secondary = COMPONENTS_CSS.split(".metric-dual__secondary {", 1)[1].split("}", 1)[0]
    assert "var(--fs-body)" in value
    assert "var(--fs-aux)" in secondary
    # The tooltip keeps the Status tab's own wording for the pair.
    assert "modelSizeLabel(managerModelInfo(model.name))" in MODELS


def test_the_memory_cell_renders():
    html = _env().get_template("dashboard/_models.html").render()
    assert 'class="metric-dual "' in html
    assert 'x-text="modelMemoryCell(model.name).footprint"' in html


# === Resident rows ===


def test_loaded_rows_carry_the_glow_without_a_border():
    assert "model-row--loaded" in MODELS
    assert "'model-row--loaded' : ''" in MODELS
    assert "isModelLoaded(model.name)" in MODELS
    assert "isModelLoaded(name) {" in DASHBOARD_JS
    rule = COMPONENTS_CSS.split(".model-row--loaded {", 1)[1].split("}", 1)[0]
    assert "var(--shadow-glow)" in rule
    # The review asked every page's accent to be neutral, so the tint is the
    # text token rather than a system colour; blue stays inside the badges that
    # actually mean "in progress".
    assert "var(--text-" in rule, "the tint comes from a neutral token"
    assert "border" not in rule, "no hard border"
    assert re.search(r"#[0-9a-fA-F]{3,8}", rule) is None


# === Downloader empty state ===


def test_the_downloader_empty_state_is_a_component():
    assert "models.browse.empty_title" in MODELS
    assert MODELS.count("ui.empty_state('star'") == 2, "the HF and ModelScope browse lists"
    # The bare paragraph the state replaced is gone.
    assert '<p class="text-sm text-neutral-500">{{ t(\'models.browse.load_prompt\') }}</p>' not in MODELS
    html = _env().get_template("dashboard/_models.html").render()
    assert "empty-state__title" in html and "empty-state__desc" in html
    # The guide button runs the page's existing action, not a new one.
    assert '@click="loadRecommendedModels()"' in html
    assert '@click="loadMsRecommendedModels()"' in html
    for key in ("models.browse.empty_title", "models.browse.load_prompt", "models.browse.load_button"):
        assert key in EN, key


# === Filter sliders ===


def test_the_parameter_and_size_filters_are_sliders():
    assert MODELS.count('type="range"') == 4
    for field in ("hfSearchMinParams", "hfSearchMaxParams", "hfSearchMinSize", "hfSearchMaxSize"):
        assert f'x-model.number="{field}"' in MODELS, field
        # Same state the search request and the "clear" affordance read.
        assert f"{field}: 0" in DASHBOARD_JS, field
        assert f"this.{field} = 0;" in DASHBOARD_JS, field
    panel = MODELS.split("<!-- Filter panel -->", 1)[1].split("Clear filters", 1)[0]
    assert panel.count("@input=\"debounceSearch()\"") == 4
    assert 'type="number"' not in panel


def test_the_sliders_read_their_value_back():
    assert MODELS.count("filterSliderLabel(") == 4
    for kind in ("min_params", "max_params", "min_size", "max_size"):
        assert f"filterSliderLabel('{kind}'" in MODELS, kind
    helper = _dashboard_helper("filterSliderLabel")
    assert "window.formatParams(" in helper, "parameter counts go through the shared formatter"
    assert "'≥ '" in helper and "'≤ '" in helper
    assert "models.search.filter.any" in helper, "zero is the off position, not >= 0"
    for name in (".filter-slider {", ".filter-slider__value {"):
        assert name in COMPONENTS_CSS, name


def test_the_sliders_keep_feeding_the_search_request():
    body = DASHBOARD_JS.split("async searchHFModels()", 1)[1].split("async ", 1)[0]
    assert "min_params" in body and "max_params" in body
    assert "min_size" in body and "max_size" in body
    assert "hfSearchMinParams)" in body, "the request still reads the same state"


# === Model settings sheet ===


def test_the_modal_is_framed_as_a_sheet():
    for name in ("sheet", "sheet__titlebar", "sheet__body", "sheet__footer"):
        assert f'.{name} {{' in COMPONENTS_CSS, name
    assert 'class="relative sheet"' in MODAL
    assert '<header class="sheet__titlebar">' in MODAL
    assert '<div class="sheet__body">' in MODAL
    assert '<footer class="sheet__footer">' in MODAL
    assert MODAL.index("sheet__titlebar") < MODAL.index("sheet__body") < MODAL.index("sheet__footer")
    # The title bar is the one the dialog points its label at.
    titlebar = MODAL.split('class="sheet__titlebar"', 1)[1].split("</header>", 1)[0]
    assert 'id="model-settings-modal-title"' in titlebar
    assert "modal.model_settings.section_label" in titlebar
    assert "showModelSettingsModal = false" in titlebar
    # The save action is pinned in the footer, not at the end of the scroll.
    footer = MODAL.split('class="sheet__footer"', 1)[1]
    assert "saveModelSettings" in footer
    assert "modal.model_settings.cancel" in footer
    # One measure from the token layer, and Esc still closes.
    sheet = COMPONENTS_CSS.split(".sheet {", 1)[1].split("}", 1)[0]
    assert "var(--container-form)" in sheet
    assert MODAL.split("<dialog", 1)[1].split(">", 1)[0].count("@cancel.prevent") == 1


def test_the_sheet_does_not_touch_field_semantics():
    # The restyle is a frame: every field binding the offload tests read is
    # still in place.
    assert "modelSettings.max_context_window" in MODAL
    assert "modelSettings.temperature" in MODAL
    assert "modelSettings.top_p" in MODAL
    assert "<!-- DeepSeek V4.1 Engram SSD Offload -->" in MODAL
    assert "<!-- Thinking Budget -->" in MODAL


def test_empty_sampling_fields_show_the_inherited_value():
    assert "placeholder=\"{{ t('modal.model_settings.placeholder_default') }}\"" not in MODAL
    fields = (
        "max_context_window",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "min_p",
        "presence_penalty",
    )
    for field in fields:
        assert f":placeholder=\"samplingInherited('{field}')\"" in MODAL, field
    helper = _dashboard_helper("samplingInherited")
    assert "this.globalSettings.sampling" in helper, "the value comes from the global settings"
    assert "String(value)" in helper
    assert "modal.model_settings.inherit_global_value" in helper
    # min_p and presence_penalty have no global counterpart: they say so rather
    # than print a number the server never sent.
    assert "min_p" not in DASHBOARD_JS.split("sampling: {", 1)[0].split("globalSettings")[-1]
    for key in ("modal.model_settings.inherit_global_value", "modal.model_settings.sampling_inherit_hint"):
        assert key in EN, key
        assert "Default" not in EN[key], "the placeholder must not read as the word 'default'"


def test_the_modal_renders_with_its_frame():
    html = _env().get_template("dashboard/_modal_model_settings.html").render()
    assert 'class="relative sheet"' in html
    assert '<footer class="sheet__footer">' in html
    assert "samplingInherited('temperature')" in html
