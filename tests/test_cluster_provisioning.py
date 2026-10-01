# SPDX-License-Identifier: Apache-2.0
"""Provisioning: push the managed cluster key to hosts."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from omlx.cluster.provisioning import (
    ProvisioningError,
    build_authorized_keys_command,
    install_managed_key,
    provision_hosts,
    verify_managed_login,
)


def test_build_authorized_keys_command_embeds_pubkey_securely():
    command = build_authorized_keys_command("ssh-ed25519 AAAA pubkey")
    assert "ssh-ed25519 AAAA pubkey" in command
    assert command.startswith("mkdir -p")
    assert "authorized_keys" in command


def test_build_authorized_keys_command_rejects_newlines():
    with pytest.raises(ValueError):
        build_authorized_keys_command("ssh-ed25519 AAAA evil\nrm -rf /")


    # A single quote in a trailing comment is the single-quote breakout vector:
    # 'ssh-ed25519 AAAA 'x'; rm -rf /' would inject into the remote shell.
    with pytest.raises(ValueError):
        build_authorized_keys_command("ssh-ed25519 AAAA 'comment'; echo pwned")
    with pytest.raises(ValueError):
        build_authorized_keys_command("ssh-ed25519 AAAA \"quoted\"")
    with pytest.raises(ValueError):
        build_authorized_keys_command("ssh-ed25519 AAAA `whoami`")
    with pytest.raises(ValueError):
        build_authorized_keys_command("ssh-ed25519 AAAA $(id)")


def test_build_authorized_keys_command_accepts_real_keygen_comments():
    """A dot in the comment is not an injection vector — ssh-keygen emits one.

    The default comment is ``user@host``, and hostnames/emails contain dots,
    so rejecting ``.`` would refuse every operator-supplied key.
    """

    for key in (
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample operator@node-a.local",
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQAB== operator@node-a.example.com",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample omlx-cluster",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample",
    ):
        assert f"grep -qF '{key}'" in build_authorized_keys_command(key)


def test_build_authorized_keys_command_emits_idempotent_install():
    command = build_authorized_keys_command("ssh-ed25519 AAAA pubkey")
    assert command.startswith("mkdir -p ~/.ssh && chmod 700 ~/.ssh && ")
    assert "grep -qF 'ssh-ed25519 AAAA pubkey'" in command
    assert "echo 'ssh-ed25519 AAAA pubkey' >> ~/.ssh/authorized_keys" in command
    assert command.endswith("chmod 600 ~/.ssh/authorized_keys")


def test_install_managed_key_pushes_key_and_verifies(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pubkey = "ssh-ed25519 AAAA test-pubkey"
    monkeypatch.setattr(
        "omlx.cluster.provisioning.managed_public_key",
        lambda: pubkey,
    )

    result = install_managed_key(
        host="192.168.1.64",
        user="operator",
        admin_key_path="/home/me/.ssh/id_ed25519",
        port=22,
        connect_timeout=10,
    )
    assert result is True
    assert calls, "expected an ssh subprocess call"
    ssh_argv = calls[0]
    assert ssh_argv[0].endswith("ssh")
    assert "operator@192.168.1.64" in ssh_argv
    assert "id_ed25519" in " ".join(ssh_argv)


def test_install_managed_key_fails_on_ssh_error(monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, "", "permission denied")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "omlx.cluster.provisioning.managed_public_key",
        lambda: "ssh-ed25519 AAAA test-pubkey",
    )

    with pytest.raises(ProvisioningError, match="permission denied"):
        install_managed_key(
            host="192.168.1.5",
            user="operator",
            admin_key_path="/home/me/.ssh/id_ed25519",
            port=22,
        )


def test_install_managed_key_rejects_shell_metacharacters():
    with pytest.raises(ValueError):
        install_managed_key(host="evil;rm -rf /", user="operator", port=22)


def test_install_managed_key_rejects_invalid_port():
    with pytest.raises(ValueError):
        install_managed_key(host="192.168.1.64", user="operator", port=0)
    with pytest.raises(ValueError):
        install_managed_key(host="192.168.1.64", user="operator", port=65536)


def test_verify_managed_login_uses_managed_identity(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert verify_managed_login(host="192.168.1.64", user="operator", port=22) is True
    ssh_argv = calls[0]
    assert "~/.ssh/omlx_cluster" in " ".join(ssh_argv)
    assert "BatchMode=yes" in " ".join(ssh_argv)
    assert ssh_argv[-1] == "true"


def test_provision_hosts_applies_same_key_to_all_hosts(monkeypatch, tmp_path):
    installed = []

    def fake_install(host, user, **kwargs):
        installed.append(host)
        return True

    monkeypatch.setattr(
        "omlx.cluster.provisioning.install_managed_key",
        fake_install,
    )
    monkeypatch.setattr(
        "omlx.cluster.provisioning.verify_managed_login",
        lambda **kwargs: True,
    )

    results = provision_hosts(
        hosts=[("192.168.1.5", "operator"), ("192.168.1.64", "operator")],
        admin_key_path="/tmp/nonexistent-key",
    )
    assert installed == ["192.168.1.5", "192.168.1.64"]
    assert results["ok"] == 2
    assert results["failed"] == 0


def test_provision_hosts_reports_failures_without_aborting(monkeypatch):
    def failing_install(host, user, **kwargs):
        raise ProvisioningError(f"cannot reach {host}")

    monkeypatch.setattr(
        "omlx.cluster.provisioning.install_managed_key",
        failing_install,
    )
    monkeypatch.setattr(
        "omlx.cluster.provisioning.verify_managed_login",
        lambda **kwargs: True,
    )

    results = provision_hosts(
        hosts=[("192.168.1.5", "operator"), ("192.168.1.64", "operator")],
        admin_key_path="/tmp/nonexistent-key",
    )
    assert results["ok"] == 0
    assert results["failed"] == 2
    assert "192.168.1.5" in results["errors"]


def test_provision_hosts_partial_failure_keeps_batch_going(monkeypatch):
    installed: list[str] = []

    def mixed_install(host, user, **kwargs):
        if host == "192.168.1.5":
            installed.append(host)
            return True
        raise ProvisioningError(f"cannot reach {host}")

    monkeypatch.setattr(
        "omlx.cluster.provisioning.install_managed_key",
        mixed_install,
    )
    monkeypatch.setattr(
        "omlx.cluster.provisioning.verify_managed_login",
        lambda **kwargs: True,
    )

    results = provision_hosts(
        hosts=[("192.168.1.5", "operator"), ("192.168.1.64", "operator")],
        admin_key_path="/tmp/nonexistent-key",
    )
    assert results["ok"] == 1
    assert results["failed"] == 1
    assert installed == ["192.168.1.5"]
    assert "192.168.1.64" in results["errors"]


def test_ask_pass_path_never_persists_password(monkeypatch, tmp_path):
    env_before = dict(os.environ)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    # sshpass is a brew package, not a macOS builtin: the CI runners do not
    # have it. Stub the lookup so this test proves the password handling, not
    # whichever binaries happen to be installed on the host running pytest.
    real_which = shutil.which

    def fake_which(name, *args, **kwargs):
        if name == "sshpass":
            return "/opt/homebrew/bin/sshpass"
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "omlx.cluster.provisioning.managed_public_key",
        lambda: "ssh-ed25519 AAAA test-pubkey",
    )

    install_managed_key(
        host="192.168.1.64",
        user="operator",
        password="SuperSecret123",
        port=22,
    )

    assert "SuperSecret123" not in calls[0][0]
    env_now = os.environ
    assert env_now == env_before
    assert "SSH_ASKPASS" not in env_now or "SuperSecret123" not in env_now.get(
        "SSH_ASKPASS", ""
    )


def test_provision_hosts_fails_a_host_whose_installed_key_cannot_log_in():
    """Writing authorized_keys is not the same as being able to use it."""

    verified: list[str] = []

    def ok_install(host, user, **kwargs):
        return True

    def refuse_login(*, host, user, **kwargs):
        verified.append(host)
        raise ProvisioningError(f"{host}: managed key was rejected")

    results = provision_hosts(
        hosts=[("192.168.1.5", "operator")],
        installer=ok_install,
        verifier=refuse_login,
    )

    assert verified == ["192.168.1.5"]
    assert results["ok"] == 0
    assert results["failed"] == 1
    assert "rejected" in results["errors"]["192.168.1.5"]


def test_provision_hosts_verifies_every_host_it_reports_ok():
    verified: list[str] = []

    results = provision_hosts(
        hosts=[("192.168.1.5", "operator"), ("192.168.1.64", "operator")],
        installer=lambda **kwargs: True,
        verifier=lambda *, host, user, **kwargs: verified.append(host) or True,
    )

    assert verified == ["192.168.1.5", "192.168.1.64"]
    assert results["ok"] == 2
