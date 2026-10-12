"""PP, TG and TTFT are abbreviations, so they are written in capitals."""

import json
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "omlx_web"
I18N_DIR = WEB / "i18n"

LABEL_KEYS = (
    "bench.results.single.pp_tps",
    "bench.results.single.tg_tps",
    "bench.results.batch.pp_tps",
    "bench.results.batch.tg_tps",
    "bench.results.batch.pp_tps_req",
    "bench.metrics.pp_tps.name",
    "bench.metrics.tg_tps.name",
    "bench.metrics.pp_tps_req.name",
    "bench.metrics.tg_tps.tooltip",
    "bench.metrics.pp_tps_req.tooltip",
    "chat.status.ttft",
)

LOWERCASE = re.compile(r"\b(pp|tg) TPS\b|\bttft\b")


def _catalog(lang):
    return json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def test_english_spells_the_abbreviations_in_capitals():
    en = _catalog("en")
    for key in LABEL_KEYS:
        assert not LOWERCASE.search(en[key]), (key, en[key])
        assert re.search(r"\b(PP|TG) TPS\b|\bTTFT\b", en[key]), (key, en[key])


def test_no_catalog_keeps_the_old_english_spelling():
    # A locale that still shows the English text must show the new spelling.
    # Translated values are the translators' and are not checked here.
    old = {
        "bench.results.single.pp_tps": "pp TPS",
        "bench.results.single.tg_tps": "tg TPS",
        "bench.results.batch.pp_tps": "pp TPS",
        "bench.results.batch.tg_tps": "tg TPS",
        "bench.results.batch.pp_tps_req": "pp TPS/req",
        "bench.metrics.pp_tps.name": "pp TPS",
        "bench.metrics.tg_tps.name": "tg TPS",
        "bench.metrics.pp_tps_req.name": "pp TPS/req",
        "chat.status.ttft": "ttft {seconds}s",
    }
    for path in sorted(I18N_DIR.glob("*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        for key, value in old.items():
            assert catalog.get(key) != value, (path.name, key)


def test_the_text_export_uses_the_same_spelling_as_the_table():
    source = (WEB / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
    assert "'pp TPS" not in source and "'tg TPS" not in source
    assert "pad('PP TPS', 12)" in source and "pad('TG TPS', 12)" in source
