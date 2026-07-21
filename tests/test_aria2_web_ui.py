# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
TEMPLATE = (ROOT / "omlx/admin/templates/dashboard/_models.html").read_text()


def test_dashboard_loads_status_and_can_install_binary() -> None:
    assert "'/admin/api/aria2/status'" in JS
    assert "'/admin/api/aria2/install'" in JS
    assert "async installAria2()" in JS


def test_dashboard_saves_both_concurrency_controls() -> None:
    assert "aria2_connections_per_file" in JS
    assert "aria2_concurrent_files" in JS
    assert "aria2_proxy" in JS


def test_download_page_explains_controls_and_disables_proxy_for_mirror() -> None:
    assert 'x-show="downloaderSource === \'ms\'"' in TEMPLATE
    assert "Optional ModelScope acceleration" in TEMPLATE
    assert "Required for model downloads" not in TEMPLATE
    assert "Splits one large file into parallel ranges" in TEMPLATE
    assert "Controls how many files download at once" in TEMPLATE
    assert ':disabled="aria2MirrorActive"' in TEMPLATE
    assert "One-click installation uses Homebrew" in TEMPLATE
