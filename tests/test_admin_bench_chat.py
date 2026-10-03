# SPDX-License-Identifier: Apache-2.0
"""Benchmark configuration and chat page structure (console PR 7).

Two screens are covered here:

* the accuracy benchmark's configuration, which was a flat checklist: it now
  renders one `ui.card` per benchmark group (title, one-line explanation,
  sample count, group-level "Full"), with a quick/standard/full preset on top
  and the per-benchmark sample selector still inside its group;
* the chat page, where the profile picker and the model settings panel moved
  into a right-hand drawer that keeps the header summary, metrics without data
  read as an em dash, and the shortcut sheet lists only bindings the page
  really implements.

Everything is a static assertion over the templates, the stylesheet and
`dashboard.js`; the behaviour behind them (the preset → selection mapping, the
group helpers and the metric formatter) is pinned by
`tests/admin_dashboard.test.cjs` and `tests/admin_format.test.cjs`.
"""

import json
import re
from pathlib import Path

import jinja2
import pytest

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
TEMPLATES = ADMIN / "templates"
DASH = TEMPLATES / "dashboard"

BENCH = (DASH / "_bench.html").read_text(encoding="utf-8")
ACCURACY = (DASH / "_bench_accuracy.html").read_text(encoding="utf-8")
CONTEXT = (DASH / "_bench_context.html").read_text(encoding="utf-8")
CHAT = (TEMPLATES / "chat.html").read_text(encoding="utf-8")
COMPONENTS = (TEMPLATES / "components" / "ui.html").read_text(encoding="utf-8")
COMPONENTS_CSS = (ADMIN / "static" / "css" / "components.css").read_text(encoding="utf-8")
DASHBOARD_JS = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
FORMAT_JS = (ADMIN / "static" / "js" / "format.js").read_text(encoding="utf-8")
EN = json.loads((ADMIN / "i18n" / "en.json").read_text(encoding="utf-8"))

GROUP_SLUGS = ["knowledge", "commonsense", "math", "coding", "safety"]
BENCH_KINDS = ["throughput", "accuracy", "context"]

NEW_KEYS = [
    "acc_bench.presets.label",
    "acc_bench.presets.quick",
    "acc_bench.presets.standard",
    "acc_bench.presets.full",
    "acc_bench.presets.hint",
    "acc_bench.config.samples_unit",
    "acc_bench.config.group_full_hint",
    "bench.confirm.title",
    "bench.confirm.action",
    "bench.confirm.body.throughput",
    "bench.confirm.body.accuracy",
    "bench.confirm.body.context",
    "chat.drawer_title",
    "chat.profile_default",
    "chat.shortcut_focus_search",
    "chat.shortcut_new_chat",
    "chat.shortcut_close_drawer",
] + [f"acc_bench.benchmarks.group_{slug}_desc" for slug in GROUP_SLUGS]


def _env() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)))
    env.globals.update(
        t=lambda key: key,
        static=lambda path: path,
        version="0.0.0",
        api_key="dev",
        current_lang="en",
        locale_json="{}",
    )
    return env


def _render(name: str) -> str:
    return _env().get_template(name).render()


def _section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


# === A. Smart benchmark configuration ===


def test_benchmark_groups_render_as_cards():
    assert "{% call ui.card(" in ACCURACY
    assert "title_bind='group.name'" in ACCURACY, "the group name is render-time data"
    assert "subtitle_bind='group.desc'" in ACCURACY, "each group carries its explanation"
    assert "attrs='acc-group'" in ACCURACY
    # One card per group, inside the template that walks the catalogue.
    assert '<template x-for="group in accBenchmarkGroups"' in ACCURACY
    assert ACCURACY.index("{% call ui.card(") > ACCURACY.index(
        '<template x-for="group in accBenchmarkGroups"'
    )


def test_every_group_has_a_one_line_explanation():
    for slug in GROUP_SLUGS:
        key = f"acc_bench.benchmarks.group_{slug}_desc"
        assert key in EN, f"{key} is missing from English"
        assert f"desc: window.t('{key}')" in DASHBOARD_JS
        assert f"key: '{slug}'," in DASHBOARD_JS


def test_group_card_shows_the_sample_count_and_a_full_option():
    assert "accGroupSelected(group)" in ACCURACY
    assert "accGroupSamples(group)" in ACCURACY
    assert "window.formatCountExact(accGroupSamples(group))" in ACCURACY, (
        "counts go through the shared formatter"
    )
    assert "acc_bench.config.samples_unit" in ACCURACY
    assert "applyGroupFull(group)" in ACCURACY, "the group's Full option is wired"
    for helper in ("accGroupSelected(group)", "accGroupSamples(group)", "applyGroupFull(group)"):
        name = helper.split("(")[0]
        assert f"{name}(" in DASHBOARD_JS, f"{name} is not implemented"


def test_the_per_benchmark_sample_selector_stays_in_its_group():
    card = _section(ACCURACY, "{% call ui.card(", "{% endcall %}")
    assert 'x-model.number="accSampleSizes[b.key]"' in card
    assert "acc_bench.config.full_option" in card
    assert "{% endcall %}" in ACCURACY


def test_presets_are_a_segmented_control():
    assert "ui.segmented(" in ACCURACY
    preset_block = _section(ACCURACY, "{{ ui.segmented(", "}}")
    for preset in ("quick", "standard", "full"):
        assert f"('{preset}', t('acc_bench.presets.{preset}'))" in preset_block
    assert "active='accActivePreset'" in preset_block
    assert "applyAccPreset('{value}')" in preset_block
    for key in ("accPresetSelection", "applyAccPreset", "get accActivePreset"):
        assert key in DASHBOARD_JS


def test_a_preset_is_a_complete_selection():
    """The mapping itself is behavioural and lives in the node test; this only
    pins the shape that makes it impossible to leave the form empty."""
    body = _section(DASHBOARD_JS, "accPresetSelection(preset) {", "applyAccPreset(preset) {")
    assert "ACC_PRESETS" in body
    assert "preset === 'full'" in body, "full is built from the catalogue"
    assert "if (!wanted) wanted = ACC_PRESETS.standard;" in body, (
        "an unknown preset must not produce an empty selection"
    )
    assert "benchmarks[b.key] = sampled !== undefined;" in body

    presets = _section(DASHBOARD_JS, "const ACC_PRESETS = {", "};")
    catalogue_keys = set(re.findall(r"\{ key: '([a-z0-9_]+)'", DASHBOARD_JS))
    for name, table in re.findall(r"(quick|standard): \{([^}]*)\}", presets):
        for key in re.findall(r"([a-z0-9_]+):", table):
            assert key in catalogue_keys, f"{name} preset names an unknown benchmark {key}"


def test_context_target_is_a_segmented_control():
    assert "ui.segmented_each(" in CONTEXT
    block = _section(CONTEXT, "{{ ui.segmented_each(", "}}")
    assert "ctxBenchTargetChoices()" in block
    assert "active='ctxBenchTarget'" in block, "the existing state variable is kept"
    assert "ctxBenchTarget = {value}" in block
    assert "disabled='ctxBenchRunning'" in block
    # The old chip list is gone.
    assert "ctxBenchTarget === t" not in CONTEXT
    assert "x-text=\"(t / 1024) + 'k'\"" not in CONTEXT
    assert "ctxBenchTargetChoices() {" in DASHBOARD_JS
    assert "ctxBenchTargetOptions()" in DASHBOARD_JS, "the reachable-target filter is reused"


# === B. Destructive confirmation ===


def test_starting_a_benchmark_asks_through_the_shared_dialog():
    assert "<dialog" in BENCH
    dialog = _section(BENCH, "<!-- Destructive-action alert", "</dialog>")
    assert "dashboard-dialog" in dialog, "the console's existing dialog pattern"
    # The dialog is opened from JS (dashboard.js / syncDialog): <dialog> with
    # x-effect never ran, because the element sits outside Alpine's scope.
    assert 'id="bench-confirm-dialog"' in dialog
    assert "syncDialog('bench-confirm-dialog'" in DASHBOARD_JS
    assert 'role="alertdialog"' in dialog
    assert 'aria-modal="true"' in dialog
    assert '@keydown.tab="trapDialogFocus($event)"' in dialog
    assert '@cancel.prevent="cancelBenchConfirm()"' in dialog
    # Background dimmed, and the scrim cancels.
    assert '<div class="fixed inset-0 bg-black/50" @click="cancelBenchConfirm()"></div>' in dialog
    # One icon, a title, one paragraph, a destructive action and a cancel.
    assert "alert-dialog__icon" in dialog
    assert ".alert-dialog__icon {" in COMPONENTS_CSS
    assert "bench.confirm.title" in dialog
    assert "benchConfirmBody()" in dialog
    assert "variant='destructive'" in dialog
    assert "cancelBenchConfirm()" in dialog
    assert "attrs='autofocus'" in dialog, "focus starts on cancel"


def test_no_window_confirm_is_used_to_start_a_benchmark():
    for name, text in (("_bench.html", BENCH), ("_bench_accuracy.html", ACCURACY),
                       ("_bench_context.html", CONTEXT)):
        assert "confirm(" not in text, f"{name} asks through window.confirm"
        assert "window.confirm" not in text


@pytest.mark.parametrize("kind", BENCH_KINDS)
def test_each_run_button_is_gated_by_the_alert(kind):
    target = {"throughput": BENCH, "accuracy": ACCURACY, "context": CONTEXT}[kind]
    assert f"requestBenchConfirm('{kind}')" in target
    request = _section(DASHBOARD_JS, "requestBenchConfirm(kind) {", "benchConfirmBody() {")
    assert "this.benchConfirm = kind;" in request
    dispatch = _section(DASHBOARD_JS, "confirmBenchRun() {", "cancelBenchConfirm() {")
    assert f"kind === '{kind}'" in dispatch


def test_an_external_endpoint_run_does_not_ask():
    """An external run never unloads the local models, so it stays direct."""
    body = _section(DASHBOARD_JS, "requestBenchConfirm(kind) {", "benchConfirmBody() {")
    assert "this.accExternalEnabled) return this.addToAccQueue()" in body
    assert "this.benchExternalEnabled) return this.startBenchmark()" in body


def test_the_confirmation_body_is_per_kind():
    body = _section(DASHBOARD_JS, "benchConfirmBody() {", "confirmBenchRun() {")
    assert "window.t('bench.confirm.body.' + this.benchConfirm)" in body
    for kind in BENCH_KINDS:
        assert f"bench.confirm.body.{kind}" in EN


def test_dispatch_reaches_the_real_starter():
    dispatch = _section(DASHBOARD_JS, "confirmBenchRun() {", "cancelBenchConfirm() {")
    assert "this.startBenchmark()" in dispatch
    assert "this.addToAccQueue()" in dispatch
    assert "this.startContextBenchmark()" in dispatch


# === C. Chat page ===


def test_every_new_key_exists():
    missing = [key for key in NEW_KEYS if key not in EN]
    assert not missing, f"new keys not in the English catalogue: {missing}"


@pytest.mark.parametrize(
    "name",
    ["dashboard/_bench.html", "dashboard/_bench_accuracy.html",
     "dashboard/_bench_context.html"],
)
def test_touched_templates_render(name):
    html = _render(name)
    assert "Undefined" not in html
    if name.endswith("_bench_accuracy.html"):
        assert "acc-group" in html
        assert "card__title" in html and "card__subtitle" in html
        assert "segmented__item" in html
    if name.endswith("_bench_context.html"):
        assert "segmented__item" in html
        assert "ctxBenchTargetChoices()" in html
    if name.endswith("_bench.html"):
        assert "alert-dialog__icon" in html
    if name.endswith("chat.html"):
        assert "shortcut-row" in html
        assert "chat-drawer" in html


def test_the_templates_use_macros_that_exist():
    env = _env()
    module = env.get_template("components/ui.html").module
    for source in (ACCURACY, CONTEXT, BENCH, CHAT):
        for macro in set(re.findall(r"\bui\.([a-z_]+)\(", source)):
            assert hasattr(module, macro), f"components/ui.html has no {macro} macro"


def test_segmented_each_keeps_the_segmented_spec():
    module = _env().get_template("components/ui.html").module
    html = str(module.segmented_each(
        "choices()", active="target", setter="target = {value}", disabled="busy"
    ))
    assert 'class="segmented' in html
    assert html.count('class="segmented__item"') == 1
    assert "target === item.value" in html, "the item field is compared"
    assert "target = item.value" in html, "and assigned back"
    assert ':disabled="busy"' in html
    assert 'x-text="item.label"' in html
    assert "segmented__item--active" in html

    card = _env().from_string(
        '{% import "components/ui.html" as ui %}'
        "{% call ui.card(title_bind='group.name', subtitle_bind='group.desc', attrs='acc-group') %}"
        "BODY{% endcall %}"
    ).render()
    assert 'x-text="group.name"' in card
    assert 'x-text="group.desc"' in card
    assert "card__title" in card and "card__subtitle" in card
    assert "card acc-group" in card and "BODY" in card


