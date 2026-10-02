# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

from omlx.cluster import ssh_keys


def test_key_rotation_generates_a_complete_pair_before_replacing(monkeypatch, tmp_path):
    key_path = tmp_path / "omlx_cluster"
    public_key_path = Path(str(key_path) + ".pub")
    key_path.write_text("old private key")
    public_key_path.write_text("ssh-ed25519 AAAA old")

    def run(argv, **_kwargs):
        generated_path = Path(argv[argv.index("-f") + 1])
        assert generated_path != key_path
        generated_path.write_text("new private key")
        Path(str(generated_path) + ".pub").write_text("ssh-ed25519 AQID new")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(ssh_keys, "_ssh_keygen_executable", lambda: "ssh-keygen")
    monkeypatch.setattr(ssh_keys.subprocess, "run", run)

    rotated = ssh_keys.generate_ssh_key_pair(
        key_path=key_path,
        overwrite=True,
    )

    assert key_path.read_text() == "new private key"
    assert public_key_path.read_text() == "ssh-ed25519 AQID new"
    assert rotated.private_key_path == key_path
    assert rotated.public_key_path == public_key_path


def test_failed_key_rotation_keeps_the_live_pair(monkeypatch, tmp_path):
    key_path = tmp_path / "omlx_cluster"
    public_key_path = Path(str(key_path) + ".pub")
    key_path.write_text("old private key")
    public_key_path.write_text("ssh-ed25519 AAAA old")

    monkeypatch.setattr(ssh_keys, "_ssh_keygen_executable", lambda: "ssh-keygen")
    monkeypatch.setattr(
        ssh_keys.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            1,
            "",
            "generation failed",
        ),
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        ssh_keys.generate_ssh_key_pair(
            key_path=key_path,
            overwrite=True,
        )

    assert key_path.read_text() == "old private key"
    assert public_key_path.read_text() == "ssh-ed25519 AAAA old"
    assert list(tmp_path.glob(".omlx_cluster-*")) == []


def test_failed_private_key_replace_restores_the_live_public_key(monkeypatch, tmp_path):
    key_path = tmp_path / "omlx_cluster"
    public_key_path = Path(str(key_path) + ".pub")
    key_path.write_text("old private key")
    public_key_path.write_text("ssh-ed25519 AAAA old")

    def run(argv, **_kwargs):
        generated_path = Path(argv[argv.index("-f") + 1])
        generated_path.write_text("new private key")
        Path(str(generated_path) + ".pub").write_text("ssh-ed25519 AQID new")
        return subprocess.CompletedProcess(argv, 0, "", "")

    original_replace = ssh_keys.os.replace
    replace_calls = 0

    def fail_private_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("private key replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(ssh_keys, "_ssh_keygen_executable", lambda: "ssh-keygen")
    monkeypatch.setattr(ssh_keys.subprocess, "run", run)
    monkeypatch.setattr(ssh_keys.os, "replace", fail_private_replace)

    with pytest.raises(OSError, match="private key replace failed"):
        ssh_keys.generate_ssh_key_pair(
            key_path=key_path,
            overwrite=True,
        )

    assert key_path.read_text() == "old private key"
    assert public_key_path.read_text() == "ssh-ed25519 AAAA old"
    assert list(tmp_path.glob(".omlx_cluster-*")) == []
