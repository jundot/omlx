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


def test_loading_cache_settings_preserves_size_until_slider_edit():
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');
(async () => {
    for (const size of ['auto', '1536MB']) {
        const context = {
            localStorage: {getItem: () => null},
            THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
            window: {t: key => key}, navigator: {language: 'en'}, document: {},
            console,
            fetch: async () => ({ok: true, json: async () => ({
                cache: {ssd_cache_max_size: size, ssd_cache_auto_size_bytes: 150 * 1024 ** 3},
                system: {ssd_total_bytes: 1000 * 1024 ** 3},
            })}),
        };
        const state = vm.runInNewContext(source + '\n dashboard;', context)();
        await state.loadGlobalSettings();
        assert.equal(state.globalSettings.cache.ssd_cache_max_size, size);
        assert.equal(state.loadingGlobalSettings, false);
        state.cachePercent = 20;
        state.updateCacheFromSlider();
        assert.equal(state.globalSettings.cache.ssd_cache_max_size, '200GB');
        state.globalSettings.cache.ssd_cache_max_size = 'auto';
        assert.equal(state.cacheSizeGB, 150);
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
