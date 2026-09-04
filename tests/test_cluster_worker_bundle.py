# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import subprocess
import tarfile

import pytest

from omlx.cluster import cuda_worker_bootstrap
from omlx.cluster.worker_bundle import (
    build_cuda_join_command,
    cuda_bootstrap_digest,
    cuda_bootstrap_program,
    worker_source_bundle,
    worker_source_digest,
)


def test_copy_command_pins_the_program_and_keeps_secret_out_of_the_url():
    key = "one-time-secret-value-with-enough-entropy"
    command = build_cuda_join_command(
        controller_url="http://10.42.0.10:8000",
        join_key=key,
        controller_key_fingerprint="SHA256:" + "A" * 43,
        source_digest="b" * 64,
    )

    assert cuda_bootstrap_digest() in command
    assert "sha256sum -c -" in command
    assert "--max-filesize 1048576" in command
    assert "sudo apt-get install -y ca-certificates curl python3" in command
    assert "curl |" not in command
    assert "| sudo" not in command
    assert f"--join-key {key}" in command
    assert "--source-digest " + "b" * 64 in command
    assert key not in command.split("--join-key", 1)[0]
    assert "bootstrap.py?" not in command
    assert "; exit " not in command
    assert subprocess.run(
        ["bash", "-n", "-c", command], capture_output=True, check=False
    ).returncode == 0


def test_bootstrap_digest_covers_the_exact_served_bytes():
    assert hashlib.sha256(cuda_bootstrap_program()).hexdigest() == (
        cuda_bootstrap_digest()
    )


def test_worker_bundle_contains_runtime_source_but_no_native_controller_binary():
    bundle = worker_source_bundle()
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())

    assert "omlx/cluster/inference_worker.py" in names
    assert "omlx/adapter/output_parser.py" in names
    assert not any(name.endswith((".so", ".dylib", ".pyc")) for name in names)
    assert not any("tailwindcss-macos-arm64" in name for name in names)
    assert hashlib.sha256(bundle).hexdigest() == worker_source_digest()


def test_bootstrap_requires_a_literal_non_loopback_controller_ip():
    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="literal local IP"):
        cuda_worker_bootstrap._controller_url("http://studio.local:8000")
    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="reachable local IP"):
        cuda_worker_bootstrap._controller_url("http://127.0.0.1:8000")
    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="local IPv4"):
        cuda_worker_bootstrap._controller_url("http://8.8.8.8:8000")
    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="local IPv4"):
        cuda_worker_bootstrap._controller_url("http://[fd00::10]:8000")


def test_bootstrap_requires_the_command_pinned_source_digest():
    parser_source = cuda_bootstrap_program().decode()

    assert 'parser.add_argument("--source-digest", required=True)' in parser_source
    assert "source_digest != args.source_digest" in parser_source


def test_worker_requirements_pin_numpy_and_the_cuda_backend():
    requirements = cuda_worker_bootstrap.WORKER_REQUIREMENTS

    assert "numpy>=1.24.0,<2.4" in requirements
    assert "mlx==0.32.2" in requirements
    assert "mlx-cuda-13==0.32.2" in requirements
    assert "openai-harmony" in requirements
    assert "psutil>=5.9.0" in requirements


@pytest.mark.parametrize("unsafe_name", ("../escape.py", "/absolute.py"))
def test_bootstrap_rejects_unsafe_source_archive_paths(tmp_path, unsafe_name):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(unsafe_name)
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="unsafe path"):
        cuda_worker_bootstrap._safe_extract(archive, tmp_path / "output")


def test_bootstrap_rejects_source_archive_links(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("omlx/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        bundle.addfile(info)

    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="unsafe path"):
        cuda_worker_bootstrap._safe_extract(archive, tmp_path / "output")


def test_bootstrap_authorizes_only_the_pinned_controller_address(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cuda_worker_bootstrap.os, "chown", lambda *_args: None)
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAworker controller"

    for _ in range(2):
        cuda_worker_bootstrap._install_controller_key(
            public_key,
            controller_ip="10.42.0.10",
            home=tmp_path,
            uid=1000,
            gid=1000,
        )

    authorized = (tmp_path / ".ssh" / "authorized_keys").read_text()
    assert authorized == (
        'from="10.42.0.10",restrict '
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAworker controller\n"
    )


def test_reenrollment_moves_the_controller_key_source_restriction(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cuda_worker_bootstrap.os, "chown", lambda *_args: None)
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAworker controller"

    cuda_worker_bootstrap._install_controller_key(
        public_key,
        controller_ip="10.42.0.10",
        home=tmp_path,
        uid=1000,
        gid=1000,
    )
    authorized_keys = tmp_path / ".ssh" / "authorized_keys"
    authorized_keys.write_text(
        authorized_keys.read_text()
        + "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAunrelated operator\n"
    )
    cuda_worker_bootstrap._install_controller_key(
        public_key,
        controller_ip="10.42.0.11",
        home=tmp_path,
        uid=1000,
        gid=1000,
    )

    authorized = authorized_keys.read_text()
    assert authorized == (
        'from="10.42.0.11",restrict '
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAworker controller\n"
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAunrelated operator\n"
    )


def test_bootstrap_refuses_a_symlinked_authorized_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda_worker_bootstrap.os, "chown", lambda *_args: None)
    target = tmp_path / "unrelated"
    target.write_text("must stay unchanged\n")
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "authorized_keys").symlink_to(target)

    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="symbolic link"):
        cuda_worker_bootstrap._install_controller_key(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAworker controller",
            controller_ip="10.42.0.10",
            home=tmp_path,
            uid=1000,
            gid=1000,
        )

    assert target.read_text() == "must stay unchanged\n"


def test_bootstrap_refuses_a_symlinked_ssh_directory(tmp_path):
    target = tmp_path / "outside-ssh"
    target.mkdir()
    authorized_keys = target / "authorized_keys"
    authorized_keys.write_text("must stay unchanged\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".ssh").symlink_to(target, target_is_directory=True)

    with pytest.raises(cuda_worker_bootstrap.BootstrapError, match="SSH directory"):
        cuda_worker_bootstrap._install_controller_key(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAworker controller",
            controller_ip="10.42.0.10",
            home=home,
            uid=1000,
            gid=1000,
        )

    assert authorized_keys.read_text() == "must stay unchanged\n"


def test_bootstrap_claims_before_slow_system_provisioning(monkeypatch, tmp_path):
    events = []
    source = b"worker-source"
    source_digest = hashlib.sha256(source).hexdigest()
    fingerprint = "SHA256:" + "A" * 43
    controller_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAcontroller"

    monkeypatch.setattr(cuda_worker_bootstrap.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_controller_url",
        lambda _value: ("http://10.42.0.10:8000", "10.42.0.10", 8000),
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_ssh_user",
        lambda _value: ("worker", tmp_path, 1000, 1000),
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_ensure_system_packages",
        lambda: events.append("packages"),
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap.socket, "gethostname", lambda: "worker"
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_local_address",
        lambda *_args: "10.42.0.21",
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_machine_id",
        lambda _value: "machine",
    )

    def request(url, _payload, *, bearer, timeout=30):
        del bearer, timeout
        if url.endswith("/claim"):
            events.append("claim")
            return {
                "session_token": "session-token",
                "source_digest": source_digest,
                "controller_public_key": controller_key,
                "controller_key_fingerprint": fingerprint,
            }
        events.append("complete")
        return {"status": "joined"}

    monkeypatch.setattr(cuda_worker_bootstrap, "_request_json", request)
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_ssh_fingerprint",
        lambda _value: fingerprint,
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_install_controller_key",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(cuda_worker_bootstrap, "_start_sshd", lambda: None)

    def download(_url, target, *, bearer):
        del bearer
        target.write_bytes(source)

    monkeypatch.setattr(cuda_worker_bootstrap, "_download", download)
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_install_worker_source",
        lambda *_args: tmp_path / "source",
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_ensure_worker_venv",
        lambda _source: tmp_path / "venv" / "bin" / "python",
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_host_public_key",
        lambda: ("ssh-ed25519 worker", "SHA256:" + "B" * 43),
    )
    monkeypatch.setattr(
        cuda_worker_bootstrap,
        "_distribution_versions",
        lambda _python: {},
    )

    assert (
        cuda_worker_bootstrap.main(
            [
                "--controller",
                "http://10.42.0.10:8000",
                "--join-key",
                "join-key",
                "--controller-key-fingerprint",
                fingerprint,
                "--source-digest",
                source_digest,
            ]
        )
        == 0
    )
    assert events.index("claim") < events.index("packages")
