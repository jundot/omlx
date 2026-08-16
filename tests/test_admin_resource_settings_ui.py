import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (
    ROOT / "omlx/admin/templates/dashboard/_settings.html"
).read_text()


def test_decode_priority_copy_describes_prefill_behavior():
    translations = json.loads((ROOT / "omlx/admin/i18n/en.json").read_text())

    assert (
        translations["settings.resource.decode_fairness"]
        == "Prioritize Decoding During Prefill"
    )
    assert "Prefill pauses between chunks" in translations[
        "settings.resource.decode_fairness_description"
    ]


def test_decode_priority_toggle_keeps_its_fixed_width():
    binding = "globalSettings.scheduler.decode_fairness ="
    binding_start = SETTINGS.index(binding)
    button_start = SETTINGS.rindex("<button", 0, binding_start)
    button_end = SETTINGS.index("</button>", binding_start)
    button = SETTINGS[button_start:button_end]

    assert "flex-shrink-0" in button
