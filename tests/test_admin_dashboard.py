# SPDX-License-Identifier: Apache-2.0
"""Structure of the Status tab after the dashboard was rebuilt.

Main turned the tab into a GridStack block layout (`dashboard_blocks` in
`dashboard/_status.html`, one partial per block in `dashboard/blocks/`). This PR
adds the pieces the review asked for on top of that layout: a server status
header card above the grid, KPI cards with sparklines in the serving-stats
block, a memory watermark bar in the active-models block, and the hourly trend
in the usage-history block. The three groups the review described as sub-tabs
are expressed by the block registry and its tray instead of a second navigation
layer — see the PR body.

These are static assertions; the drawing math is covered by
tests/admin_dashboard.test.cjs.
"""

import json
from pathlib import Path

import jinja2

ADMIN_DIR = Path(__file__).resolve().parents[1] / "omlx" / "admin"
COMPONENTS_CSS = (ADMIN_DIR / "static" / "css" / "components.css").read_text(encoding="utf-8")
DASHBOARD_JS = (ADMIN_DIR / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
DASH = ADMIN / "templates" / "dashboard"
BLOCKS = DASH / "blocks"
STATUS = (DASH / "_status.html").read_text(encoding="utf-8")
SERVING = (BLOCKS / "_serving_stats.html").read_text(encoding="utf-8")
ACTIVE = (BLOCKS / "_active_models.html").read_text(encoding="utf-8")
USAGE = (DASH / "_usage.html").read_text(encoding="utf-8")
NAVBAR = (DASH / "_navbar.html").read_text(encoding="utf-8")
DASHBOARD_JS = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
USAGE_JS = (ADMIN / "static" / "js" / "usage.js").read_text(encoding="utf-8")
EN = json.loads((ADMIN / "i18n" / "en.json").read_text(encoding="utf-8"))


def _env() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ADMIN / "templates")))
    env.globals.update(t=lambda key: key, static=lambda path: path, version="0.0.0")
    return env


def test_blocks_are_registered_with_their_own_partials():
    for block_id in ("serving_stats", "usage_history", "active_models",
                     "cache_observability", "api_endpoints", "claude_code",
                     "applications", "engine_versions"):
        assert f"('{block_id}', '_{block_id}.html')" in STATUS
        assert (BLOCKS / f"_{block_id}.html").exists()


def test_each_block_partial_renders_on_its_own():
    """The layout editor and the template tests render a partial without the
    dashboard's context, so a partial that needs the shared macros imports them."""
    env = _env()
    for partial in sorted(BLOCKS.glob("_*.html")):
        env.get_template(f"dashboard/blocks/{partial.name}").render()


def test_server_status_header_card():
    header = STATUS[: STATUS.index("<!-- Layout toolbar")]
    # The mark is the oMLX logo the top bar draws, not an app icon in a tile:
    # no rounded-square background, and it swaps with the theme.
    assert "status-header__mark" in header
    assert "navbar-logo-light.svg" in header and "navbar-logo-dark.svg" in header
    assert "status-header__icon" not in header
    assert "{{ version }}" in header, "the header shows the running version"
    assert "ui.badge(t('status.header.running'), tone='green', dot=True)" in header
    assert "status.header.uptime" in header
    assert "restartServerStart()" in header, "restart is reachable from the header"
    assert "unloadAllModels()" in header
    assert "mainTab === 'status'" in STATUS


def test_the_scope_toggle_sets_the_scope_it_shows():
    """The two scopes arrive in one poll and the KPI cards read `statsScope`
    themselves, so the toggle only has to set it — and it does."""
    assert "active='statsScope'" in SERVING
    assert "statsScope = '{value}'" in SERVING
    assert "status.scope_label" in SERVING, "the toggle names itself for assistive tech"
    assert "kpiValue(" in DASHBOARD_JS and "statsScope === 'alltime'" in DASHBOARD_JS
    assert "loadStats()" in DASHBOARD_JS, "both payloads are fetched together"


def test_the_scope_toggle_cancels_a_pending_confirmation():
    """Changing scope drops both clear confirmations.

    They are per-scope rows, so a "Clear stats?" left open on Session used to
    reappear on its own when the user came back to Session from All-time.
    """
    setters = [line for line in SERVING.splitlines() if 'setter="statsScope' in line]
    assert len(setters) == 1, setters
    assert "showClearStatsConfirm = false" in setters[0]
    assert "showClearAlltimeConfirm = false" in setters[0]


def test_the_brand_pair_uses_the_type_scale():
    """The wordmark and the version under it sit at the top of every page; both
    are token steps, one step up from the point sizes they used to carry."""
    assert "text-2xl leading-none" in NAVBAR, "the wordmark is the page step (24)"
    assert "text-sm text-neutral-500" in NAVBAR, "the version is the body step (14)"
    assert "text-xl leading-none" not in NAVBAR, "the old 18pt wordmark is gone"
    assert "text-[12px] text-neutral-500" not in NAVBAR, "and the 12pt version with it"


def test_kpi_cards_are_left_aligned_without_a_sparkline():
    assert SERVING.count('class="kpi"') == 4
    assert 'class="kpi__value"' in SERVING
    assert "text-center" not in SERVING, "KPI cards are left aligned"
    # The review removed the line under the figure; nothing may draw it again.
    assert "kpi__spark" not in SERVING
    assert "sparkPath" not in DASHBOARD_JS
    assert "kpiHistory" not in DASHBOARD_JS


def test_memory_watermark_bar():
    assert "watermark__bar--estimated" in ACTIVE
    assert "watermark__bar--actual" in ACTIVE
    assert "watermark__marker" in ACTIVE
    for key in ("status.memory.actual", "status.memory.estimated",
                "status.memory.soft", "status.memory.hard"):
        assert f"t('{key}')" in ACTIVE
        assert key in EN
    for getter in ("memoryWatermark", "watermarkBarStyle", "watermarkMarkerStyle"):
        assert getter in DASHBOARD_JS


def test_memory_watermark_is_scaled_to_the_hard_limit():
    body = DASHBOARD_JS[DASHBOARD_JS.index("get memoryWatermark()"):][:1400]
    assert "hard_bytes" in body
    assert "soft_bytes" in body
    assert "current_bytes" in body
    assert "estimated_size" in body, "the estimate is summed from the resident models"


def test_hourly_trend_sits_above_the_heatmap():
    assert "heat-strip" in USAGE
    assert "hourlyTotals()" in USAGE
    assert "usage.trend" in EN
    assert USAGE.index("heat-strip") < USAGE.index("data.heatmap"), "the trend reads before the grid"
    assert "hourlyBarStyle" in USAGE_JS


def test_status_tab_helpers_exist_and_are_pure():
    for helper in ("formatUptime", "unloadAllModels"):
        assert helper in DASHBOARD_JS
    # unloadAllModels must reuse the per-model endpoint rather than invent one:
    # the task scope is front-end only.
    body = DASHBOARD_JS[DASHBOARD_JS.index("async unloadAllModels()"):][:600]
    assert "this.unloadModel(model.id)" in body
    assert "/api/" not in body, "no new endpoint"


def test_every_template_keeps_its_divs_balanced():
    """One stray </div> in a partial closes the page's Alpine root early, and
    everything after it — the modals — lands outside the scope, so their buttons
    do nothing. This is the guard for that class of bug."""
    env = _env()
    unbalanced = {}
    for partial in sorted(BLOCKS.glob("_*.html")):
        html = env.get_template(f"dashboard/blocks/{partial.name}").render()
        unbalanced[partial.name] = html.count("<div") - html.count("</div>")
    status = env.get_template("dashboard/_status.html").render()
    unbalanced["_status.html"] = status.count("<div") - status.count("</div>")
    dashboard = env.get_template("dashboard.html").render()
    unbalanced["dashboard.html"] = dashboard.count("<div") - dashboard.count("</div>")
    offenders = {name: value for name, value in unbalanced.items() if value != 0}
    assert not offenders, f"unbalanced <div> in: {offenders}"

def test_the_memory_meter_reads_through_the_enforcers_thresholds():
    """The meter's colour is not a decorative ramp: it follows the two limits the
    enforcer itself defines — fine under the soft guard, the guard's colour at it,
    the ceiling's at the hard limit — and the two limits are told apart by shape
    (dashed guard, solid ceiling with a caret), never by colour alone."""
    tones = {
        "ok": "--sys-green",
        "warn": "--sys-orange",
        "over": "--sys-red",
    }
    for tone, colour in tones.items():
        for rule in (f".watermark__bar--actual.meter--{tone}", f".pressure__bar.meter--{tone}"):
            assert rule in COMPONENTS_CSS, rule
            body = COMPONENTS_CSS[COMPONENTS_CSS.index(rule + " {"):]
            body = body[: body.index("}")]
            assert f"background-color: var({colour})" in body, (rule, body)

    # Two marks, two shapes, and each on its own status colour.
    soft = COMPONENTS_CSS[COMPONENTS_CSS.index(".watermark__marker--soft {"):]
    soft = soft[: soft.index("}")]
    hard = COMPONENTS_CSS[COMPONENTS_CSS.index(".watermark__marker--hard {"):]
    hard = hard[: hard.index("}")]
    assert "border-left-style: dashed" in soft and "var(--sys-orange)" in soft
    assert "border-left-style: solid" not in soft
    assert "var(--sys-red)" in hard and "dashed" not in hard
    assert ".watermark__marker--hard::after" in COMPONENTS_CSS, "the ceiling carries a caret"

    # The bands are 70 % and 90 % of the hard limit, and the JS owns no palette.
    assert "memoryTone(percent)" in DASHBOARD_JS
    assert "if (percent >= 90) return 'meter--over';" in DASHBOARD_JS
    assert "if (percent >= 70) return 'meter--warn';" in DASHBOARD_JS
    assert "actualOfLimit" in DASHBOARD_JS, "the colour reads the ratio to the hard limit"

    # The track is scaled to the machine's memory, so the two zones have a place:
    # the guarded band between soft and hard, and everything above the hard limit.
    assert "system?.total_memory_bytes" in DASHBOARD_JS
    assert "guardWidth" in DASHBOARD_JS and "unavailableWidth" in DASHBOARD_JS
    for rule in (".watermark__bar--guard", ".watermark__bar--unavailable"):
        assert rule in COMPONENTS_CSS, rule
    assert "color-mix(in srgb, var(--text-primary) 32%, transparent)" in COMPONENTS_CSS
    unavailable = COMPONENTS_CSS[COMPONENTS_CSS.index(".watermark__bar--unavailable {"):]
    assert "background-color: var(--text-primary)" in unavailable[: unavailable.index("}")]
    for hex in ("#ef4444", "#f97316", "#f59e0b", "#facc15", "#22c55e", "rgba(64, 64, 64"):
        assert hex not in DASHBOARD_JS, f"{hex} is a hand-picked colour in the meter"

