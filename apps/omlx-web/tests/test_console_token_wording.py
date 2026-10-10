# SPDX-License-Identifier: Apache-2.0
"""Guards for the console catalogues' Token wording.

The capital T is for the model's token used as a **unit** — the thing that
gets counted, cached and rate-limited, standing on a label or beside a number:
``cached ({count} Tokens)``, ``Output Tok/s``, ``Max Tokens``.

The lowercase form is kept everywhere the word is an ordinary English noun in
a sentence — ``Process tokens in batches internally``, ``Limit thinking
tokens`` — and everywhere it names a credential: a Hugging Face token, the
``HF_TOKEN`` environment variable and the HF write token. Capitalising those
reads as a different product name; capitalising a mid-sentence common noun
just reads as a mistake.

A case pass over the catalogues is easy to write and easy to overrun in either
direction, so the three rules below are pinned against ``en.json``.

The other locales are deliberately not held to them. Czech inflects
("tokenů", "tokeny") so a capital form does not exist, and the languages that
already had a translation of the credential — "Zadejte zapisovací Token HF",
"Insira o Token de escrita HF" — spell it with a capital on their own. Only
English is a style choice this repository makes.
"""

import json
import re
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parents[1] / "omlx_web" / "i18n"

# The counts these keys report are token counts, so their wording is where the
# capital belongs.
TOKEN_COUNT_KEYS = (
    "status.active_models.cached_tokens",
    "status.active_models.waiting_prompt_tokens",
    "status.active_models.last_token_ago",
    "status.active_models.dflash_draft_tokens_last_request",
)

# These name a Hugging Face credential and the HF_TOKEN environment variable.
# A case pass that reaches them has gone too far.
CREDENTIAL_KEYS = (
    "models.downloader.token_warning",
    "models.uploader.hf_token_placeholder",
)

# A prose value explains something in sentences. Past its first sentence, or
# past the " — " that opens a definition, "token" is a common noun again and
# takes the lowercase form.
PROSE_KEY = re.compile(
    r"(_hint|_description|_note|_positive|_help|_subtitle|\.tooltip)$"
)
SENTENCE_OPENERS = (" — ", ". ")

# A capitalised token word, whole-word on purpose: "Tokens" counts, and so does
# the "Tok" of "Tok/s", while "tokenizer", "max_tokens" and "HF_TOKEN" do not.
CAPITAL_TOKEN = re.compile(r"\b(?:Tok|Token|Tokens)\b")

# PP, TG and TTFT are abbreviations of prompt processing, token generation and
# time to first token. An abbreviation is written in capitals in every locale,
# and the Chinese and Japanese catalogues had always read that way.
ABBREVIATION_KEYS = (
    "bench.results.single.pp_tps",
    "bench.results.single.tg_tps",
    "bench.results.batch.pp_tps",
    "bench.results.batch.tg_tps",
    "bench.results.batch.pp_tps_req",
    "bench.metrics.pp_tps.name",
    "bench.metrics.tg_tps.name",
    "bench.metrics.pp_tps_req.name",
    "chat.status.ttft",
)
CAPITAL_PPTG = re.compile(r"\b(PP|TG|TTFT)\b")

# …and a test label is not prose: it is the string the template renders, so its
# case is the case on screen. `_bench.html` writes `'pp' + context_length`, so
# the label reads `pp1024` and the tooltip that explains the format has to
# spell it the same way or it describes a screen that does not exist.
FORMAT_EXAMPLE_KEY = "bench.metrics.test.tooltip"
LOWER_PPTG = re.compile(r"\bppN/tgM\b")


def _locales() -> dict:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(I18N_DIR.glob("*.json"))
    }


def _past_the_opening(value: str) -> str:
    """The part of a prose value that follows its first sentence opener."""

    cuts = [
        value.index(opener) + len(opener)
        for opener in SENTENCE_OPENERS
        if opener in value
    ]
    return value[min(cuts) :] if cuts else ""


def test_english_token_counts_carry_the_capital():
    english = _locales()["en"]

    for key in TOKEN_COUNT_KEYS:
        assert key in english, f"en.json is missing {key}"
        value = english[key]
        message = f"en.json: {key} = {value!r} keeps the lowercase token"
        assert CAPITAL_TOKEN.search(value), message


def test_english_credentials_keep_the_lowercase_token():
    english = _locales()["en"]

    for key in CREDENTIAL_KEYS:
        assert key in english, f"en.json is missing {key}"
        value = english[key]
        message = f"en.json: {key} = {value!r} capitalises a credential"
        assert not CAPITAL_TOKEN.search(value), message


def test_english_prose_keeps_the_common_noun_lowercase():
    """A sentence says "tokens"; only a label says "Tokens"."""

    english = _locales()["en"]
    offenders = []

    for key, value in english.items():
        if not PROSE_KEY.search(key):
            continue
        found = CAPITAL_TOKEN.search(_past_the_opening(value))
        if found:
            offenders.append(f"{key}: {found.group(0)!r} in {value!r}")

    report = "a prose value capitalises the common noun:\n" + "\n".join(offenders)
    assert not offenders, report


def test_the_abbreviations_are_capitalised():
    """An abbreviation is not a word, so it does not take the noun's case.

    A locale that translated the label away from the Latin form — zh reads
    `提示词处理 TPS` — has no abbreviation left to case, and upstream's
    translation is not this rule's business.
    """

    latin = re.compile(r"\bpp\b|\btg\b|\bttft\b", re.IGNORECASE)

    for locale, catalog in _locales().items():
        for key in ABBREVIATION_KEYS:
            assert key in catalog, f"{locale}.json is missing {key}"
            value = catalog[key]
            if not latin.search(value):
                continue
            message = (
                f"{locale}.json: {key} = {value!r} spells an abbreviation in lower case"
            )
            assert CAPITAL_PPTG.search(value), message


def test_the_test_label_stays_as_the_template_renders_it():
    """`pp1024` is a rendered string, not prose, so its case is the screen's.

    `_bench.html` builds the label with `'pp' + context_length`; a tooltip that
    spelled it `PP1024` would describe a screen nobody can see.
    """

    english = _locales()["en"]
    value = english[FORMAT_EXAMPLE_KEY]
    message = f"en.json: {FORMAT_EXAMPLE_KEY} describes a label the template does not render: {value!r}"
    assert LOWER_PPTG.search(value), message


def test_the_key_sets_stay_identical_across_locales():
    expected = None
    for locale, catalog in _locales().items():
        keys = set(catalog)
        if expected is None:
            expected = keys
        assert keys == expected, f"{locale} differs by {keys ^ expected}"
