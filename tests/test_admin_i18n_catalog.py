# SPDX-License-Identifier: Apache-2.0
"""Catalogue contract for the admin console's translations.

`omlx/admin/i18n/*.json` is the only place UI copy lives. These checks are
static: every locale carries the same key set as English, nothing is left
untranslated outside the language-neutral list below, every key the templates
and the console's own scripts ask for exists, and no screen can reintroduce a
second number format next to the shared one.

The language-neutral list is deliberate, not an oversight: acronyms (API, CLI,
TTFT), product and protocol names (HuggingFace, SpecPrefill, DuckDuckGo, oQ),
keyboard key names (Enter, Esc), units (GB, GiB), sample values in placeholders
and log-line templates read the same in every language.
"""

import json
import re
from pathlib import Path

ADMIN = Path(__file__).resolve().parents[1] / "omlx" / "admin"
I18N = ADMIN / "i18n"
EN = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
LOCALES = sorted(path.stem for path in I18N.glob("*.json"))

TEMPLATES = sorted((ADMIN / "templates").rglob("*.html"))
OWN_SCRIPTS = [
    ADMIN / "static" / "js" / name
    for name in ("cluster_v2.js", "dashboard.js", "format.js", "logs.js", "settings_nav.js", "usage.js")
]
TRANSLATED_SOURCES = TEMPLATES + OWN_SCRIPTS

# Values that are language-neutral by design (see the module docstring).
# Values that are language-neutral by design (see the module docstring).
LANGUAGE_NEUTRAL = [
    "acc_bench.results.download_csv",
    "acc_bench.results.download_json",
    "acc_bench.results.download_txt",
    "acc_bench.results.text_export.question_header",
    "bench.metrics.tpot.name",
    "bench.metrics.ttft.name",
    "bench.results.test.plain",
    "bench.results.text_export.title",
    "chat.shortcut_key_enter",
    "chat.shortcut_key_esc",
    "chat.shortcut_shift_enter",
    "cluster.badge.pipeline",
    "cluster.badge.tensor",
    "cluster.v2.backend.jaccl",
    "cluster.v2.backend.ring",
    "cluster.v2.cuda.badge",
    "cluster.v2.link.thunderbolt",
    "cluster.v2.link.wifi",
    "cluster.v2.split.rank_label",
    "cluster.v2.strategy.pipeline",
    "cluster.v2.strategy.tensor",
    "js.ane_tune.cpu_gdn",
    "js.ane_tune.gdn",
    "js.ane_tune.mlp",
    "modal.model_settings.actions.recipe_placeholder",
    "modal.model_settings.dflash",
    "modal.model_settings.github",
    "modal.model_settings.guided_grammar_placeholder",
    "modal.model_settings.huggingface",
    "modal.model_settings.kwarg_enable_thinking",
    "modal.model_settings.kwarg_reasoning_effort",
    "modal.model_settings.lightning_mtp",
    "modal.model_settings.min_p",
    "modal.model_settings.specprefill",
    "modal.model_settings.top_k",
    "modal.model_settings.top_p",
    "modal.model_settings.vlm_mtp",
    "models.downloader.hf_token_placeholder",
    "models.downloader.ms_repo_placeholder",
    "models.downloader.repo_placeholder",
    "models.downloader.source.hf",
    "models.oq.about_table_oq",
    "models.oq.about_table_oqe",
    "navbar.brand",
    "settings.advanced.gdn_sidecar_state_precision_bf16",
    "settings.advanced.gdn_sidecar_state_precision_rht_int16",
    "settings.generation.top_k",
    "settings.generation.top_p",
    "settings.integrations.markitdown.formats_badge",
    "settings.integrations.markitdown.section_label",
    "settings.integrations.websearch.provider_duckduckgo",
    "settings.integrations.websearch.provider_searxng",
    "settings.integrations.websearch.providers_badge",
    "settings.language.en",
    "settings.language.es",
    "settings.language.fr",
    # Units and format examples stay as they are in every language.
    "chat.stats.token_suffix",
    "chat.stats.tokens_per_second_suffix",
    "models.downloader.ms_token_placeholder",
    "settings.language.cs",
    "settings.language.pt-BR",
    "settings.mcp.section_label",
    "settings.models.badge.ctx_window",
    "settings.models.badge.force_sampling",
    "settings.models.badge.max_tokens",
    "settings.models.badge.min_p",
    "settings.models.badge.presence_penalty",
    "settings.models.badge.rep_penalty",
    "settings.models.badge.temp",
    "settings.models.badge.tool_result_tokens",
    "settings.models.badge.top_k",
    "settings.models.badge.top_p",
    "settings.models.table.visible",
    "settings.resource.guard_tier.custom_input_placeholder",
    "status.active_models.dflash_label",
    "status.api.claude",
    "status.api.openai",
    "status.integrations.claude_code",
    "status.integrations.codex",
    "status.integrations.copilot_cli",
    # Product name, like the rows above it: dsh is DeepSeek Harness, not a
    # phrase to translate.
    "status.integrations.dsh",
    "status.integrations.hermes_agent",
    "status.integrations.openclaw",
    "status.integrations.opencode",
    "status.integrations.pi",
    "status.layout.block.claude_code",
    "status.layout.width_default",
    "status.layout.width_wide",
    "status.layout.width_wider",
]

# Literal text nodes in the templates that are identifiers, units or product
# names rather than copy.
TEMPLATE_LITERALS = [
    "API",
    "CLI",
    "GB)",
    "GiB",
    "LLM",
    "VLM",
    "bfloat16",
    "brew install jundot/omlx/omlx",
    "brew upgrade omlx",
    "ddtree",
    "dflash",
    "false",
    "float16",
    "high",
    "low",
    "max",
    "medium",
    "oMLX",
    "thinking",
    "Tok/s",
    "true",
    "xhigh",
]

KEY_CALL = re.compile(r"""(?:window\.)?\bt\(\s*'([^']*)'""")
TEXT_NODE = re.compile(r">([^<>]+)<")
ENGLISH_TEXT = re.compile(r"^[A-Za-z][A-Za-z0-9 ,.'’()/%+-]{2,}$")


def _strip_markup(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<script\b.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style\b.*?</style>", "", text, flags=re.S | re.I)
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _without_comments(text: str) -> str:
    text = re.sub(r"\{#.*?#\}", "", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def _keys_used() -> set:
    keys = set()
    for path in TRANSLATED_SOURCES:
        keys.update(KEY_CALL.findall(_without_comments(path.read_text(encoding="utf-8"))))
    return keys


# === Catalogue completeness ===


def test_locales_share_the_english_key_set():
    english = set(EN)
    assert LOCALES, "no locale files found"
    for name in LOCALES:
        catalogue = json.loads((I18N / f"{name}.json").read_text(encoding="utf-8"))
        missing = english - set(catalogue)
        extra = set(catalogue) - english
        assert not missing, f"{name}.json is missing {sorted(missing)[:5]}"
        assert not extra, f"{name}.json has keys English does not: {sorted(extra)[:5]}"


def test_no_locale_value_is_empty():
    for name in LOCALES:
        catalogue = json.loads((I18N / f"{name}.json").read_text(encoding="utf-8"))
        empty = [key for key, value in catalogue.items() if not str(value).strip()]
        assert not empty, f"{name}.json has empty values: {empty[:5]}"


def test_chinese_catalogue_is_translated():
    catalogue = json.loads((I18N / "zh.json").read_text(encoding="utf-8"))
    untranslated = sorted(
        key
        for key, value in catalogue.items()
        if value == EN.get(key)
        and re.search(r"[A-Za-z]", str(value))
        and key not in LANGUAGE_NEUTRAL
    )
    assert not untranslated, f"zh.json still shows English: {untranslated[:5]}"


# English is the source language, so a locale that has not caught up yet falls
# back to an English value rather than to a missing key. The gap is real and
# large for every locale except zh (the newest ~500 keys landed in English and
# Simplified Chinese only); these baselines are a ratchet, not an approval: a
# new key must ship translated, and the numbers may only go down.
UNTRANSLATED_BASELINE = {
    # Upstream's Czech catalogue arrived after this batch was cut, so the
    # strings this slice adds ship in English there until it is translated.
    "cs": 58,
    "es": 658,
    "fr": 660,
    "ja": 669,
    "ko": 595,
    "pt-BR": 601,
    "ru": 551,
    "zh-TW": 519,
    "zh": 7,
}


# Keys whose "token" is a credential rather than a count: an Hugging Face or
# ModelScope write token is a key, so its label keeps its case.
# Czech writes the loanword "token" in lower case inside a sentence. These
# keys hold Czech prose about tokens — not a token unit, and not a credential
# label — so the capital-T rule does not reach their Czech values.
LOWERCASE_TOKEN_ALLOWED = {
    "cs": {
        "status.active_models.last_token_ago",
        "bench.config.ane_aligned_prompt",
        "bench.metrics.tpot.full_name",
    },
}

CREDENTIAL_KEYS = {
    "models.downloader.hf_token",
    "models.downloader.ms_token",
    "models.downloader.token_warning",
    "models.uploader.hf_token_label",
    "models.uploader.hf_token_placeholder",
    "models.uploader.invalid_token",
    "settings.integrations.websearch.brave_api_key_hint",
}

# `tok/s`, `t/s`, `tok`, `token`, `tokens`: the copy writes them with a capital
# T wherever they mean the model's tokens, in every locale that uses the Latin
# word (the CJK and Cyrillic catalogues mostly use their own noun).
LOWERCASE_TOKEN = re.compile(r"\b(tok/s|t/s|tok|token|tokens)\b")


# The units that live in the markup and the scripts rather than in the
# catalogue: KPI suffixes, table cells, the bench text export and the heatmap's
# aria-label. A lower-case one of these is the same mistake in a different file.
LOWERCASE_TOKEN_UNITS = (
    "' tok/s'", "' tok'", "' tokens'", ">tok/s<", ">tok<",
    "prompt tok/s", "}M tokens`", "}K tokens`", "tok/s</span>",
)


def test_the_markup_draws_token_units_with_a_capital():
    for path in TRANSLATED_SOURCES:
        text = path.read_text(encoding="utf-8")
        for shape in LOWERCASE_TOKEN_UNITS:
            assert shape not in text, f"{path.name} still writes {shape!r}"


def test_token_words_are_capitalised():
    for locale in LOCALES:
        catalogue = json.loads((I18N / f"{locale}.json").read_text(encoding="utf-8"))
        allowed = LOWERCASE_TOKEN_ALLOWED.get(locale, set())
        offenders = [
            key
            for key, value in catalogue.items()
            if key not in CREDENTIAL_KEYS
            and key not in allowed
            and isinstance(value, str)
            and LOWERCASE_TOKEN.search(value)
        ]
        assert not offenders, f"{locale}.json writes token words in lower case: {offenders[:6]}"


def test_no_locale_falls_further_behind():
    for locale, baseline in UNTRANSLATED_BASELINE.items():
        catalogue = json.loads((I18N / f"{locale}.json").read_text(encoding="utf-8"))
        untranslated = [
            key
            for key, value in catalogue.items()
            if value == EN.get(key) and len(re.findall(r"[A-Za-z]{3,}", str(value))) >= 3
        ]
        assert len(untranslated) <= baseline, (
            f"{locale}.json regressed: {len(untranslated)} untranslated entries, "
            f"baseline {baseline}"
        )


# === Everything the console asks for exists ===


def test_every_translation_key_exists():
    missing = sorted(key for key in _keys_used() if key not in EN)
    # A key built by concatenation ("usage." + label) shows up as its prefix.
    dynamic = [key for key in missing if any(existing.startswith(key) for existing in EN)]
    leftovers = [key for key in missing if key not in dynamic]
    assert not leftovers, f"templates ask for keys English does not have: {leftovers}"


def test_dynamic_key_prefixes_resolve():
    for key in _keys_used():
        if key in EN:
            continue
        expanded = [existing for existing in EN if existing.startswith(key)]
        assert expanded, f"{key!r} expands to nothing"


# === No hardcoded copy ===


def test_templates_carry_no_hardcoded_copy():
    offenders = {}
    for path in TEMPLATES:
        for match in TEXT_NODE.finditer(_strip_markup(path)):
            text = match.group(1).strip()
            if not text or "{{" in text or "{%" in text or text.startswith("&"):
                continue
            if ENGLISH_TEXT.match(text) and text not in TEMPLATE_LITERALS:
                offenders.setdefault(str(path.relative_to(ADMIN)), set()).add(text)
    assert not offenders, f"hardcoded copy in templates: {offenders}"


# === One number format per screen ===


def test_only_the_shared_formatter_builds_compact_counts():
    patterns = [
        r"\+ *'[KMBT]'",  # string concatenation of a magnitude suffix
        r"/ *1e[369]\b",  # hand-rolled millions/billions
        r"/ *1000000\b",
        r"notation: *'compact'",  # a second Intl formatter
    ]
    offenders = {}
    for path in TRANSLATED_SOURCES:
        text = path.read_text(encoding="utf-8")
        if path.name == "format.js":
            continue
        for pattern in patterns:
            if re.search(pattern, text):
                offenders.setdefault(str(path.relative_to(ADMIN)), []).append(pattern)
    assert not offenders, f"ad-hoc count formatting: {offenders}"


def test_templates_call_the_shared_formatter():
    # The dashboard's cards live in dashboard/blocks/*.html; scan the lot.
    dashboard = "\n".join(
        path.read_text(encoding="utf-8") for path in TEMPLATES if "dashboard" in str(path)
    )
    assert "window.formatCount(" in dashboard
    for stale in ("formatNumber(", "formatTokenCount(", "formatDownloads(", "formatParamCount("):
        assert stale not in dashboard, f"{stale} bypasses the shared formatter"
    usage = (ADMIN / "templates" / "dashboard" / "_usage.html").read_text(encoding="utf-8")
    assert "toLocaleString()" not in usage


def test_console_language_reaches_the_formatter():
    base = (ADMIN / "templates" / "base.html").read_text(encoding="utf-8")
    assert "window.__omlxLang" in base
    assert "js/format.js" in base
