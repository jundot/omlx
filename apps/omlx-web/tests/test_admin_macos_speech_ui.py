# SPDX-License-Identifier: Apache-2.0
"""The macOS Speech card in Settings > Integrations.

Pins the contract the card was built to: the engine switch defaults to on,
every language gets one dropdown whose first entry is what macOS itself
answers, and the choice that reaches the server is the voice identifier the
engine addresses `say -v` with. Also guards the defects found while wiring it
up: an uncompiled Tailwind width, and a `:value` binding that lost the race
with the option list and silently showed 'macOS default' over a saved voice.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "omlx_web/templates/dashboard/_settings.html").read_text()
DASHBOARD_JS = (ROOT / "omlx_web/static/js/dashboard.js").read_text()
EN = json.loads((ROOT / "omlx_web/i18n/en.json").read_text())

CARD_START = TEMPLATE.index("settings.integrations.speech.section_label")
CARD = TEMPLATE[CARD_START: TEMPLATE.index("settings.integrations.markitdown.section_label")]


def test_speech_card_sits_inside_the_integrations_tab():
    tab_start = TEMPLATE.index("activeTab === 'integrations'")
    tab_end = TEMPLATE.index("activeTab === 'models'")
    assert tab_start < CARD_START < tab_end


def test_every_key_the_card_uses_exists_in_the_catalog():
    used = set(re.findall(r"settings\.integrations\.speech\.[a-z_]+", TEMPLATE))
    used |= set(re.findall(r"settings\.integrations\.speech\.[a-z_]+", DASHBOARD_JS))
    missing = sorted(k for k in used if k not in EN)
    assert not missing, f"keys used but never defined: {missing}"
    # The two plural forms the helpers look up by suffix.
    for base in ("voice_count", "languages_badge"):
        assert f"settings.integrations.speech.{base}" in EN
        assert f"settings.integrations.speech.{base}_one" in EN


def test_placeholders_match_between_catalog_and_card():
    assert "{count}" in EN["settings.integrations.speech.languages_badge"]
    assert "{count}" in EN["settings.integrations.speech.voice_count"]
    assert "{voice}" in EN["settings.integrations.speech.default_preference"]
    # The unnamed default must never claim a voice: it carries no placeholder.
    assert "{" not in EN["settings.integrations.speech.default_system"]


def test_engine_switch_is_bound_to_the_settings_field():
    assert "toggleSayEnabled()" in CARD
    assert "globalSettings.integrations.say_enabled" in CARD
    # A server without the synthesizer cannot be talked into offering it.
    assert ':disabled="!globalSettings.integrations.say_supported"' in CARD
    assert 'role="switch"' in CARD


def test_each_language_row_offers_the_macos_default_first():
    rows = re.findall(r'x-for="lang in[^"]*"[^>]*:key="lang\.locale"', CARD, re.DOTALL)
    assert len(rows) == 1, "the card must render one dropdown per language"
    default_option = CARD.index('x-text="sayDefaultLabel(lang)"')
    voice_option = CARD.index('<template x-for="v in lang.voices"')
    assert default_option < voice_option, "the empty (macOS default) entry comes first"
    assert '<option value="" :selected="!lang.configured"' in CARD[:voice_option]


def test_saved_choice_is_marked_selected_on_the_option_itself():
    """A select whose options come from x-for cannot be primed by `:value` or
    `x-effect`: Alpine initialises the select before its children exist, so
    the assignment lands on an empty list and a saved voice renders as
    'macOS default'. `:selected` on the option has no such ordering trap."""
    # The language dropdown, not the card's other selects.
    at = CARD.index("setSayVoice(lang,")
    start = CARD.rindex("<select", 0, at)
    select_tag = CARD[start : CARD.index(">", start)]
    assert ":value=" not in select_tag and "x-effect=" not in select_tag
    assert '@change="setSayVoice(lang, $event.target.value)"' in select_tag
    options = CARD[start : CARD.index("</select>", start)]
    assert ':selected="!lang.configured"' in options
    assert ':selected="v.value === lang.configured"' in options


def test_card_uses_only_compiled_tailwind_widths():
    """Tailwind here is a prebuilt catalog: a width class it never compiled
    (w-72) silently leaves the control full width."""
    compiled = (ROOT / "omlx_web/static/css/tailwind.css").read_text()
    for cls in set(re.findall(r"\bsm:(w-\w+)\b", CARD)):
        assert f".sm\\:{cls}" in compiled, f"sm:{cls} is not in tailwind.css"


def test_dashboard_state_defaults_to_the_engine_being_on():
    start = DASHBOARD_JS.index("integrations: {")
    block = DASHBOARD_JS[start: DASHBOARD_JS.index("external_engines: {", start)]
    assert "say_enabled: true" in block
    assert "say_voices: {}" in block
    assert "say_supported: false" in block


def test_voices_are_loaded_when_the_tab_opens():
    assert "async loadMacosSayVoices(force)" in DASHBOARD_JS
    assert "fetch('/admin/api/macos-say/voices')" in DASHBOARD_JS
    # Both doors into the tab: the click and a deep link that skips it.
    tab_handler = DASHBOARD_JS[DASHBOARD_JS.index("setSettingsTab(tab) {"):]
    tab_handler = tab_handler[: tab_handler.index("\n            },")]
    assert "this.loadMacosSayVoices()" in tab_handler
    assert "if (this.activeTab === 'integrations')" in DASHBOARD_JS


def test_saves_send_the_whole_voice_map():
    save = DASHBOARD_JS[DASHBOARD_JS.index("async saveIntegrationSettings()"):]
    save = save[: save.index("\n            },")]
    assert "say_enabled: this.globalSettings.integrations.say_enabled" in save
    assert "say_voices: this.sayVoicesMap()" in save
    # A language back on its default leaves the map instead of pinning ''.
    assert "if (lang.configured) out[lang.locale] = lang.configured;" in DASHBOARD_JS


def test_the_copy_describes_what_the_engine_actually_does():
    """The long-text control must name the limit it is about and say what it
    does to the text. It is not allowed to promise that nothing is split any
    more, because a long reply IS split - at sentence boundaries, into pieces
    that are stitched back into one file. The promise that still holds, and
    that has to stay in words the owner can rely on, is that nothing is cut."""
    label = EN["settings.integrations.speech.long_label"]
    assert "OpenAI" in label and "limit" in label.lower()
    hint = EN["settings.integrations.speech.long_hint"]
    assert "4096" in hint, "the hint lost the OpenAI character figure"
    assert "10000" not in hint, "the retired character ceiling must not reappear"
    assert "Nothing is ever cut short" in hint, "the promise the engine keeps"
    assert "sentence-sized pieces" in hint and "stitched" in hint, "how it keeps it"
    assert "refus" in hint.lower(), "the reject option must be described"
    # and the same promises must hold in the translated catalog, not just en
    cs = json.loads((ROOT / "omlx_web/i18n/cs.json").read_text())
    long_cs = cs["settings.integrations.speech.long_hint"]
    assert "4096" in long_cs and "10000" not in long_cs
    assert "dílc" in long_cs and "spoj" in long_cs, "cs must say split-and-stitch too"


def test_the_usage_example_covers_every_way_to_address_the_engine():
    """A bare "say" is the interesting case, because macOS then decides the
    language. And because display names are NOT unique on macOS (eight of them
    exist once per language on this Mac), the example has to teach the form that
    removes the doubt, not just the pretty one."""
    hint = EN["settings.integrations.speech.usage_hint"]
    assert '"say"' in hint, "the plain model name"
    assert "say:cs-CZ" in hint, "a language forced, macOS picks the voice"
    assert "say:Zuzana" in hint, "a voice named by itself"
    assert "say:Zuzana:cs-CZ" in hint, "the unambiguous voice-plus-language form"
    assert "detects the language" in hint, "what a bare say actually does"
    assert "belongs to a language" in hint, "the binding the names hide"
    assert "identifier" in hint, "the address that can never be ambiguous"
    # and the dropdown copy must say the same thing about shared names
    voices = EN["settings.integrations.speech.voices_hint"]
    assert "other languages" in voices and "different voice" in voices


def test_languages_with_an_observable_macos_default_are_listed_first():
    """Fifty-two rows, and only a handful carry a default macOS lets us see.
    Those answer the owner's question ('what will it pick here?'), so they lead;
    each group then follows the alphabet the operator reads, which is the
    dashboard's own language, not the raw tags."""
    loader = DASHBOARD_JS[DASHBOARD_JS.index("async loadMacosSayVoices("):]
    loader = loader[: loader.index("\n            },")]
    assert "(lang.default || {}).preference === true" in loader, \
        "the group split follows the OS pick"
    assert "Intl.DisplayNames" in loader and "Intl.Collator" in loader
    order = loader.index("filter(seen)")
    assert order < loader.index(".concat("), "detected defaults come before the rest"
    assert loader.count("sort(byName)") == 2, "both groups sort, not just the first"
    assert "localeCompare" in loader, "an engine without Intl still gets a stable order"


def test_long_input_policy_is_an_operator_choice_on_the_card():
    """Past OpenAI's input limit the engine either speaks everything or
    refuses. The caller does not pick: it asked for audio."""
    at = CARD.index("globalSettings.integrations.say_long_input")
    start = CARD.rindex("<select", 0, at)
    tag = CARD[start : CARD.index(">", start)]
    assert 'x-model="globalSettings.integrations.say_long_input"' in tag
    assert '@change="saveSayLongInput()"' in tag
    options = CARD[start : CARD.index("</select>", start)]
    assert '<option value="speak">' in options
    assert '<option value="reject">' in options
    # Both labels come from the catalog, never hardcoded in the template.
    for key in ("long_label", "long_hint", "long_speak", "long_reject"):
        assert f"settings.integrations.speech.{key}" in CARD
        assert EN[f"settings.integrations.speech.{key}"]


def test_dashboard_defaults_to_speaking_long_text():
    start = DASHBOARD_JS.index("integrations: {")
    block = DASHBOARD_JS[start : DASHBOARD_JS.index("external_engines: {", start)]
    assert "say_long_input: 'speak'" in block
    save = DASHBOARD_JS[DASHBOARD_JS.index("async saveIntegrationSettings()") :]
    save = save[: save.index("\n            },")]
    assert (
        "say_long_input: this.globalSettings.integrations.say_long_input" in save
    )


def test_the_segment_size_is_editable_and_the_ceiling_knob_is_gone():
    """Segment size belongs to the operator, like the policy above it. The
    audio ceiling does not: 256 MB is about ninety minutes of speech, no chat
    reply gets near it, so the card carries no knob for it (decided
    2026-10-29) - and the dead shape must stay dead."""
    at = CARD.index("speech.chunk_label")
    block = CARD[at - 400 : at + 900]
    assert 'x-model.number="globalSettings.integrations.say_chunk_chars"' in block
    assert 'type="number"' in block
    assert "@change=\"saveSayLimits()\"" in block
    assert "say_chunk_chars" in DASHBOARD_JS
    assert "say_max_audio_mb" not in DASHBOARD_JS, "the JS kept a dead field"
    assert "speech.max_label" not in CARD and "speech.max_hint" not in CARD


def test_the_ceiling_copy_is_out_of_every_locale():
    """The hint used to explain what the MB number bought. The number is no
    longer on the card, so no locale may still carry its copy - a translated
    string for a control that does not exist is noise waiting for a bug."""
    assert "settings.integrations.speech.max_label" not in EN
    assert "settings.integrations.speech.max_hint" not in EN
    for locale in sorted((ROOT / "omlx_web/i18n").glob("*.json")):
        data = json.loads(locale.read_text(encoding="utf-8"))
        assert "settings.integrations.speech.max_label" not in data, locale.name
        assert "settings.integrations.speech.max_hint" not in data, locale.name
    chunk = EN["settings.integrations.speech.chunk_hint"]
    assert "0 never splits" in chunk, "the escape hatch is documented where it is set"


def test_a_failed_save_puts_the_dropdown_back():
    handler = DASHBOARD_JS[DASHBOARD_JS.index("setSayVoice(lang, value) {"):]
    handler = handler[: handler.index("\n            },")]
    assert "const previous = lang.configured" in handler
    assert "lang.configured = previous;" in handler
