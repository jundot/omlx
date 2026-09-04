# SPDX-License-Identifier: Apache-2.0

import base64
import stat

import pytest

from omlx.cluster.ssh_keys import pin_enrolled_host_key


def _key(material: bytes) -> str:
    return "ssh-ed25519 " + base64.b64encode(material).decode()


def test_enrolled_host_key_is_pinned_once_with_private_permissions(tmp_path):
    known_hosts = tmp_path / ".ssh" / "known_hosts"
    public_key = _key(b"cuda-worker-1-host-key")

    added = pin_enrolled_host_key(
        hostname="10.42.0.21",
        public_key=public_key,
        known_hosts_path=known_hosts,
    )
    duplicate = pin_enrolled_host_key(
        hostname="10.42.0.21",
        public_key=public_key,
        known_hosts_path=known_hosts,
    )

    assert added is True
    assert duplicate is False
    assert known_hosts.read_text() == f"10.42.0.21 {public_key}\n"
    assert stat.S_IMODE(known_hosts.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(known_hosts.stat().st_mode) == 0o600


def test_enrolled_host_key_never_overwrites_a_changed_identity(tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(f"10.42.0.21 {_key(b'original')}\n")

    with pytest.raises(RuntimeError, match="refusing changed SSH host key"):
        pin_enrolled_host_key(
            hostname="10.42.0.21",
            public_key=_key(b"replacement"),
            known_hosts_path=known_hosts,
        )

    assert "replacement" not in known_hosts.read_text()


@pytest.mark.parametrize(
    "hostname",
    (
        "cuda-worker-1; touch /tmp/pwned",
        "omlxworker@cuda-worker-1 && id",
        "-oProxyCommand=id",
    ),
)
def test_enrolled_host_key_rejects_shell_or_ssh_option_injection(tmp_path, hostname):
    with pytest.raises(ValueError, match="invalid SSH target"):
        pin_enrolled_host_key(
            hostname=hostname,
            public_key=_key(b"worker"),
            known_hosts_path=tmp_path / "known_hosts",
        )
