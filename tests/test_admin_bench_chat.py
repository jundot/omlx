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


def test_missing_metrics_read_as_an_em_dash():
    assert "function formatMetric(" in FORMAT_JS
    assert "global.formatMetric = formatMetric;" in FORMAT_JS
    assert "formatMetric: formatMetric," in FORMAT_JS, "node tests import it"

    for metric in ("avg_prefill_tps", "avg_generation_tps", "thinking_time", "total_time"):
        assert f"window.formatMetric(recentStats?.{metric})" in CHAT, (
            f"{metric} does not go through the shared metric helper"
        )
    assert CHAT.count("window.formatMetric(") == 4, "a new metric cell bypassed the helper"
    assert "toFixed(1) : '0.0'" not in CHAT, "a zero stand-in came back"
    assert "toFixed(1) : '-'" not in CHAT, "a second no-data marker came back"


def test_the_drawer_carries_the_profile_picker_and_model_settings():
    assert '{% import "components/ui.html" as ui %}' in CHAT, (
        "the chat page extends base.html and needs its own import"
    )
    assert "{{ ui.button(" in CHAT, "the drawer reuses the shared button spec"

    drawer = _section(CHAT, 'id="chat-settings-drawer"', "<!-- Image Modal -->")
    assert 'class="drawer chat-drawer"' in drawer
    assert ":class=\"{ 'drawer--open': rightSidebarOpen }\"" in drawer
    # A modal only while it floats over the page: at the column width it is a
    # panel beside the content, so the role, the aria-modal flag and the focus
    # trap all follow `drawerOverlay`.
    assert ":role=\"drawerOverlay ? 'dialog' : null\"" in drawer
    assert ":aria-modal=\"drawerOverlay ? 'true' : null\"" in drawer
    assert '@keydown.tab="drawerOverlay && trapDialogFocus($event)"' in drawer
    assert "rightSidebarTab === 'profile'" in drawer, "the profile picker moved in"
    assert "rightSidebarTab === 'settings'" in drawer, "so did the model settings"

    # Scrim and shared specs.
    assert 'class="drawer-scrim chat-drawer-scrim"' in CHAT
    for rule in (".drawer {", ".drawer--open {", ".drawer-scrim {", ".drawer-scrim--closed {"):
        assert rule in COMPONENTS_CSS, f"{rule} has no spec"
    assert ".chat-drawer {" in CHAT

    # Esc closes it, after the sheets that own the key — and only while the
    # drawer covers the page: as a column it does not own the key, which still
    # stops the stream.
    handler = _section(CHAT, "handleGlobalKeydown(e) {", "focusChatSearch() {")
    assert (
        "if (this.rightSidebarOpen && this.drawerOverlay) { this.closeRightDrawer(); return; }"
        in handler
    )
    assert handler.index("this.showShortcutsHelp = false") < handler.index(
        "this.closeRightDrawer()"
    ), "the help sheet closes before the drawer"

    # Focus moves in and comes back.
    assert "this.$refs.rightDrawer" in CHAT and 'x-ref="rightDrawer"' in CHAT
    assert "drawerReturnFocus" in CHAT
    assert "if (previous && typeof previous.focus === 'function')" in CHAT
    assert "trapDialogFocus(event) {" in CHAT

    # The background goes inert while the drawer covers it. Everything else
    # that can take focus — the image modal, the settings sheet, the shortcut
    # sheet, and base.html's palette and toast stack — is a sibling of these
    # two, so the scrim is the only thing left behind it. At the column width
    # the drawer sits beside the content instead of over it, and the content
    # must stay interactive: `drawerOverlay` is the one place that decides.
    assert CHAT.count(':inert="rightSidebarOpen && drawerOverlay"') == 2, (
        "the sidebar and the chat column go inert only under the overlay"
    )
    assert "drawerOverlay: window.innerWidth < 1100" in CHAT
    assert "this.drawerOverlay = window.innerWidth < 1100;" in CHAT, (
        "the flag has to follow the viewport"
    )
    assert "showRightToggle: window.innerWidth < 1100" in CHAT, (
        "the toggle appears at the same breakpoint as the column layout"
    )
    assert "@media (min-width: 1100px) {" in CHAT, (
        "the flag and the column media query are one breakpoint"
    )
    # The page's own bindings win over the shared palette's: both listen for
    # keydown, and the page's runs in the capture phase so the palette's
    # `defaultPrevented` check skips the keys this page documents.
    assert '@keydown.window.capture="handleGlobalKeydown($event)"' in CHAT
    assert 'aria-hidden="true"' in CHAT, "the scrim is a click surface, not content"


def test_the_settings_sheet_covers_the_right_drawer():
    """The sheet dims the whole window, and the right drawer is part of it: at
    z-50 the sheet sat *under* the drawer (z-60), which stayed lit behind the
    scrim — a second layer the sheet appeared not to cover."""
    sheet = _section(CHAT, "<!-- Chat Settings Modal", "<!-- Keyboard Shortcuts Help")
    assert "z-[70]" in sheet
    assert "bg-black/50" in sheet, "the scrim is the sheet's own"
    drawer_z = int(re.search(r"\.chat-drawer \{[^}]*?z-index: (\d+)", CHAT).group(1))
    assert drawer_z < 70, "the sheet has to sit above the drawer"


def test_the_header_keeps_the_summary():
    header = _section(CHAT, "<!-- Header summary (top-centre)", "<!-- Messages Container -->")
    assert "availableModels.find(m => m.id === currentModel)?.name" in header, (
        "the model stays in the header"
    )
    assert "activePromptProfile || window.t('chat.profile_default')" not in header, (
        "the profile pill moved into the drawer, leaving the model name centred"
    )
    assert "openRightDrawer('profile')" not in header
    assert "chat.profile_default" in EN


def test_the_drawer_keeps_the_existing_state_and_keys():
    for state in ("rightSidebarOpen:", "rightSidebarTab:"):
        assert state in CHAT
    for key in ("chat.model_tab", "chat.profile_tab", "chat.close_settings_tooltip"):
        assert key in CHAT, f"{key} was dropped while moving the panels"
    assert "chat.drawer_title" in CHAT and "chat.drawer_title" in EN


def test_shortcut_sheet_lists_only_real_bindings():
    sheet = _section(CHAT, "<!-- Keyboard Shortcuts Help", "</dialog>")
    assert "<dialog" in sheet
    assert 'x-effect="showShortcutsHelp ? $el.showModal() : $el.close()"' in sheet
    assert '@keydown.tab="trapDialogFocus($event)"' in sheet
    assert "shortcut-list__row" in sheet and "kbd" in sheet
    assert ".kbd {" in COMPONENTS_CSS and ".shortcut-list__row {" in COMPONENTS_CSS

    # Esc closes the drawer only while the drawer overlays the page, so the
    # row that promises it is conditional on the same condition.
    close_drawer = [
        row for row in sheet.split('<li class="shortcut-list__row"')
        if "chat.shortcut_close_drawer" in row
    ]
    assert len(close_drawer) == 1, close_drawer
    assert 'x-show="drawerOverlay"' in close_drawer[0], close_drawer[0]

    listed = set(re.findall(r'class="kbd">([^<]+)</kbd>', sheet))
    assert listed == {
        "{{ t('chat.shortcut_key_enter') }}",
        "{{ t('chat.shortcut_shift_enter') }}",
        "{{ t('chat.shortcut_key_esc') }}",
        "⌘K",
        "⇧⌘K",
        "⌘/",
        "?",
    }, listed

    handler = _section(CHAT, "handleGlobalKeydown(e) {", "focusChatSearch() {")
    assert "e.key === 'Escape'" in handler
    assert "e.key === '?'" in handler
    assert "e.key === '/'" in handler, "⌘/ is listed, so it must be handled"
    assert "e.key === 'k'" in handler and "e.shiftKey" in handler
    assert "onComposerEnter(e) {" in CHAT and "e.shiftKey" in _section(
        CHAT, "onComposerEnter(e) {", "sortChatHistory() {"
    )
    # ⌘K needs a target even when there is no chat history to search.
    focus = _section(CHAT, "focusChatSearch() {", "openRightDrawer() {")
    assert "this.$refs.chatSearchInput" in focus
    assert "this.$refs.messageInput" in focus, "the composer is the fallback"
    assert 'x-ref="messageInput"' in CHAT
    # A key the page does not implement must not be advertised.
    for invented in ("⌘J", "⇧⌘P", "⌘S", "⇧Esc"):
        assert invented not in sheet


def test_the_shortcut_sheet_opens_from_the_button_and_the_key():
    assert 'showShortcutsHelp = true' in CHAT
    assert "chat.view_shortcuts" in CHAT
    assert "this.showShortcutsHelp = !this.showShortcutsHelp;" in CHAT


# === Shared contracts ===


def test_every_new_key_exists():
    missing = [key for key in NEW_KEYS if key not in EN]
    assert not missing, f"new keys not in the English catalogue: {missing}"


@pytest.mark.parametrize(
    "name",
    ["chat.html", "dashboard/_bench.html", "dashboard/_bench_accuracy.html",
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
        assert "shortcut-list__row" in html
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


def test_the_old_flat_checklist_is_gone():
    # The grouping used to be a header line over a bare grid; the sample-size
    # select is the only piece that stayed.
    assert "border-t border-neutral-200\"></div>" not in ACCURACY
    assert "'(' + group.benchmarks.length + ')'" not in ACCURACY
    assert "right-sidebar-width" not in CHAT
    assert "right-sidebar-hidden" not in CHAT
