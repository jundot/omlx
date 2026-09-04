# SPDX-License-Identifier: Apache-2.0
"""Cluster automation must never stop to ask OpenSSH a question."""

from __future__ import annotations

import subprocess

import pytest

from omlx.cluster.ssh_policy import (
    apply_cluster_ssh_policy,
    cluster_ssh_options,
)
from omlx.cluster.transport import _mlx_config_ssh_policy


def test_new_aliases_are_accepted_without_weakening_changed_key_checks():
    options = cluster_ssh_options(connect_timeout=5, keepalive=True)

    assert "BatchMode=yes" in options
    assert "PasswordAuthentication=no" in options
    assert "KbdInteractiveAuthentication=no" in options
    assert "AddressFamily=inet" in options
    assert "StrictHostKeyChecking=accept-new" in options
    assert "CheckHostIP=no" in options
    assert "IdentityFile=~/.ssh/omlx_cluster" in options
    assert "LogLevel=ERROR" in options
    assert "StrictHostKeyChecking=no" not in options
    assert "ConnectTimeout=5" in options
    assert "ServerAliveInterval=15" in options


@pytest.mark.parametrize(
    "argv",
    (
        ["ssh", "peer-studio.local", "true"],
        ["/usr/bin/scp", "weights", "peer-studio.local:/models/weights"],
    ),
)
def test_policy_preserves_the_original_command_while_inserting_options(argv):
    original = list(argv)

    protected = apply_cluster_ssh_policy(argv, connect_timeout=10)

    assert argv == original
    assert protected[0] == original[0]
    assert protected[-2:] == original[-2:]
    assert "StrictHostKeyChecking=accept-new" in protected
    assert "AddressFamily=inet" in protected
    assert "CheckHostIP=no" in protected


def test_invalid_timeout_and_non_ssh_commands_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        cluster_ssh_options(connect_timeout=0)
    with pytest.raises(ValueError, match="ssh or scp"):
        apply_cluster_ssh_policy(["rsync", "source", "destination"])


def test_mlx_hard_coded_ssh_calls_are_wrapped_too():
    seen = []

    def run(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    class Config:
        pass

    Config.run = run
    with _mlx_config_ssh_policy(Config):
        Config.run(["ssh", "peer-studio.local", "true"], capture_output=True)

    argv, kwargs = seen[0]
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "CheckHostIP=no" in argv
    assert kwargs == {"capture_output": True}
    assert Config.run is run
