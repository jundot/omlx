"""The status page's memory watermark bar.

One track scaled to the machine's memory — the measured footprint inside it,
the estimate behind it, the guarded band between the two limits and the part
above the hard ceiling — driven by the enforcer's own numbers, so the bar and
the guard cannot drift apart. The bar replaces the little pressure bar the
card header used to carry, and it adds no i18n keys of its own: the two limit
words are read out of the pressure label the old bar used, and the rest of
the legend is hardcoded until the console's translation sweep covers it.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
ACTIVE = (ADMIN / "templates" / "dashboard" / "blocks" / "_active_models.html").read_text(
    encoding="utf-8"
)
DASHBOARD_JS = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ADMIN / "static" / "css" / "dashboard.css").read_text(encoding="utf-8")


def test_memory_watermark_bar():
    assert "watermark__bar--estimated" in ACTIVE
    assert "watermark__bar--actual" in ACTIVE
    assert "watermark__marker" in ACTIVE
    assert "Measured" in ACTIVE
    assert "Estimated" in ACTIVE
    for getter in ("memoryWatermark", "watermarkBarStyle", "watermarkMarkerStyle"):
        assert getter in DASHBOARD_JS


def test_the_old_header_bar_is_gone():
    """The watermark is the only memory gauge the card carries: the old mini
    bar and the getters it lived on leave no trace, and the bar invents no
    i18n keys of its own."""
    for gone in (
        "activeModelsPressureBarStyle",
        "activeModelsSoftMarkerStyle",
        "activeModelsPressureLabel",
        "activeModelsPressurePercent",
        "activeModelsSoftPercent",
    ):
        assert gone not in ACTIVE, gone
        assert gone not in DASHBOARD_JS, gone
    assert "status.memory." not in ACTIVE
    assert "status.memory." not in DASHBOARD_JS


def test_the_limit_words_are_reused_from_the_pressure_label():
    """The legend's two limit words are the words the old bar used: read out
    of `status.active_models.pressure_label`, so every locale's own wording
    carries over and the key must stay in every catalogue."""
    assert "status.active_models.pressure_label" in DASHBOARD_JS
    assert "status.active_models.pressure_label" not in ACTIVE, "the template names no key itself"
    for path in sorted((ADMIN / "i18n").glob("*.json")):
        template = json.loads(path.read_text(encoding="utf-8"))[
            "status.active_models.pressure_label"
        ]
        soft = re.search(r"\{soft\}\s*([^/]+)", template)
        hard = re.search(r"\{hard\}\s*([^/]+)", template)
        assert soft and hard, (path.name, template)
        assert soft.group(1).strip() and hard.group(1).strip(), (path.name, template)


def test_memory_watermark_is_scaled_to_the_hard_limit():
    body = DASHBOARD_JS[DASHBOARD_JS.index("get memoryWatermark()"):][:1400]
    assert "hard_bytes" in body
    assert "soft_bytes" in body
    assert "current_bytes" in body
    assert "estimated_size" in body, "the estimate is summed from the resident models"
    assert "system?.total_memory_bytes" in body
    assert "guardWidth" in body and "unavailableWidth" in body


def test_the_memory_meter_reads_through_the_enforcers_thresholds():
    """The meter's colour is not a decorative ramp: it follows the two limits
    the enforcer itself defines — fine under the soft guard, the guard's
    colour at it, the ceiling's at the hard limit — and the two limits are
    told apart by shape (dashed guard, solid ceiling with a caret), never by
    colour alone."""
    tones = {
        "ok": "--sys-green",
        "warn": "--sys-orange",
        "over": "--sys-red",
    }
    for tone, colour in tones.items():
        rule = f".watermark__bar--actual.meter--{tone}"
        assert rule in DASHBOARD_CSS, rule
        body = DASHBOARD_CSS[DASHBOARD_CSS.index(rule + " {"):]
        body = body[: body.index("}")]
        assert f"background-color: var({colour})" in body, (rule, body)

    soft = DASHBOARD_CSS[DASHBOARD_CSS.index(".watermark__marker--soft {"):]
    soft = soft[: soft.index("}")]
    hard = DASHBOARD_CSS[DASHBOARD_CSS.index(".watermark__marker--hard {"):]
    hard = hard[: hard.index("}")]
    assert "border-left-style: dashed" in soft and "var(--sys-orange)" in soft
    assert "border-left-style: solid" not in soft
    assert "var(--sys-red)" in hard and "dashed" not in hard
    assert ".watermark__marker--hard::after" in DASHBOARD_CSS, "the ceiling carries a caret"

    # The bands are 70 % and 90 % of the hard limit, and the JS owns no palette.
    assert "memoryTone(percent)" in DASHBOARD_JS
    assert "if (percent >= 90) return 'meter--over';" in DASHBOARD_JS
    assert "if (percent >= 70) return 'meter--warn';" in DASHBOARD_JS
    assert "actualOfLimit" in DASHBOARD_JS, "the colour reads the ratio to the hard limit"
    for hex in ("#ef4444", "#f97316", "#f59e0b", "#facc15", "#22c55e", "rgba(64, 64, 64"):
        assert hex not in DASHBOARD_JS, f"{hex} is a hand-picked colour in the meter"

    # The guarded band and the ceiling-only zone have their own fills.
    for rule in (".watermark__bar--guard", ".watermark__bar--unavailable"):
        assert rule in DASHBOARD_CSS, rule
    assert "color-mix(in srgb, var(--text-primary) 32%, transparent)" in DASHBOARD_CSS
    unavailable = DASHBOARD_CSS[DASHBOARD_CSS.index(".watermark__bar--unavailable {"):]
    assert "background-color: var(--text-primary)" in unavailable[: unavailable.index("}")]
