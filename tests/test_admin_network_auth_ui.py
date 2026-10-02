import json
import shutil
import subprocess
from pathlib import Path

import pytest

from omlx.utils.network import is_loopback_bind

ROOT = Path(__file__).resolve().parents[1]

NETWORK_AUTH_I18N_KEYS = {
    "settings.auth.skip_verification_warning",
    "js.error.api_key_required_network",
}


def test_network_auth_ui_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    hosts = [
        "localhost",
        "LOCALHOST.",
        "127.0.0.1",
        "127.42.0.9",
        "::1",
        "0:0:0:0:0:0:0:1",
        "::0001",
        "::1%lo0",
        "::ffff:127.0.0.1",
        "::ffff:7f2a:9",
        "0:0:0:0:0:ffff:7f00:1",
        "127.0.0.1, ::1",
        "127.0.0.1, 192.168.1.1",
        "0.0.0.0",
        "::",
        "::ffff:192.168.1.1",
        "host.local",
        "127.999.0.1",
        "127.00.0.1",
        "127.0.0.1.",
        "::1.",
        "127.1",
        "",
        "::1/path",
        "[::1]",
    ]
    result = subprocess.run(
        [node, str(ROOT / "tests/network_auth_ui.test.cjs")],
        input=json.dumps([(host, is_loopback_bind(host)) for host in hosts]),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_network_auth_i18n_keys_exist_in_every_locale():
    i18n_dir = ROOT / "omlx/admin/i18n"

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        missing_keys = NETWORK_AUTH_I18N_KEYS - translations.keys()
        assert not missing_keys, f"{locale_path.name} is missing {sorted(missing_keys)}"
