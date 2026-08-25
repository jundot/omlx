#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone Linux bootstrap used by the oMLX CUDA join command.

This file intentionally imports only the Python standard library.  The
controller serves these exact bytes and puts their SHA-256 in the copy/paste
command, so an HTTP intermediary cannot replace the program being run as root.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import platform
import pwd
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import suppress
from pathlib import Path

INSTALL_ROOT = Path("/opt/omlx-cluster-worker")
VENV = INSTALL_ROOT / "venv"
REQUIREMENTS_MARKER = INSTALL_ROOT / "worker-requirements-v1"
MAX_SOURCE_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_EXTRACTED_BYTES = 128 * 1024 * 1024
MAX_SOURCE_MEMBERS = 4096
WORKER_REQUIREMENTS = (
    "numpy>=1.24.0,<2.4",
    "mlx==0.32.0",
    "mlx-cuda-13==0.32.0",
    "mlx-lm @ git+https://github.com/ml-explore/mlx-lm@ab1806e8f5d6aa035973af194a1b9198ab4754dc",
    "transformers>=5.12.1,<5.13",
    "mistral-common>=1.10",
    "tokenizers>=0.19.0",
    "huggingface-hub>=1.19.0",
    "safetensors>=0.4.0",
    "sentencepiece",
    "protobuf",
    "pyyaml>=6.0",
    "jinja2>=3.0",
    "regex",
    "rich>=13.0",
    "tqdm>=4.66.0",
    "httpx>=0.27.0,<1",
    "requests>=2.28.0",
    "tiktoken",
    "psutil>=5.9.0",
    "jsonschema>=4.0.0",
    "openai-harmony",
    "cohere_melody>=0.9.0",
)


class BootstrapError(RuntimeError):
    pass


def _run(argv: list[str], *, timeout: float | None = None) -> None:
    completed = subprocess.run(argv, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise BootstrapError(f"command failed ({completed.returncode}): {argv[0]}")


def _controller_url(value: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BootstrapError("controller must be an http(s) URL with a local IP")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise BootstrapError("controller hostname must be a literal local IP") from exc
    if (
        address.version != 4
        or address.is_unspecified
        or address.is_multicast
        or address.is_loopback
        or not (address.is_private or address.is_link_local)
    ):
        raise BootstrapError("controller must use a reachable local IPv4 address")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    normalized_host = f"[{address}]" if address.version == 6 else str(address)
    normalized = f"{parsed.scheme}://{normalized_host}:{port}"
    return normalized, str(address), port


def _request_json(
    url: str,
    payload: dict,
    *,
    bearer: str,
    timeout: float = 30,
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("detail", "request refused")
        except Exception:
            detail = "request refused"
        raise BootstrapError(f"controller refused enrollment: {detail}") from None
    except (OSError, ValueError) as exc:
        raise BootstrapError(f"could not reach controller: {exc}") from exc
    if not isinstance(result, dict):
        raise BootstrapError("controller returned malformed enrollment data")
    return result


def _download(url: str, target: Path, *, bearer: str) -> None:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {bearer}"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            target.open("wb") as output,
        ):
            announced = response.headers.get("Content-Length")
            if announced:
                try:
                    announced_bytes = int(announced)
                except ValueError as exc:
                    raise BootstrapError(
                        "controller returned an invalid source size"
                    ) from exc
                if announced_bytes > MAX_SOURCE_BUNDLE_BYTES:
                    raise BootstrapError("worker source archive exceeds the size limit")
            downloaded = 0
            while True:
                chunk = response.read(min(1024 * 1024, MAX_SOURCE_BUNDLE_BYTES + 1))
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_SOURCE_BUNDLE_BYTES:
                    raise BootstrapError("worker source archive exceeds the size limit")
                output.write(chunk)
    except (OSError, urllib.error.HTTPError) as exc:
        raise BootstrapError(f"could not download worker source: {exc}") from exc


def _ssh_fingerprint(public_key: str) -> str:
    parts = public_key.strip().split()
    if len(parts) < 2 or parts[0] not in {"ssh-ed25519", "ssh-rsa"}:
        raise BootstrapError("controller returned an invalid SSH public key")
    try:
        key_data = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise BootstrapError("controller returned an invalid SSH public key") from exc
    digest = hashlib.sha256(key_data).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def _ensure_system_packages() -> None:
    if shutil.which("apt-get") is None:
        raise BootstrapError("the first worker bootstrap currently requires Ubuntu/Debian")
    _run(["apt-get", "update"], timeout=300)
    _run(
        [
            "apt-get",
            "install",
            "-y",
            "ca-certificates",
            "curl",
            "git",
            "openssh-server",
            "python3",
            "python3-venv",
        ],
        timeout=600,
    )


def _ssh_user(explicit: str | None) -> tuple[str, Path, int, int]:
    name = explicit or os.environ.get("SUDO_USER")
    if not name or name == "root":
        raise BootstrapError(
            "run the command with sudo from the Linux account oMLX should use, "
            "or pass --ssh-user"
        )
    try:
        account = pwd.getpwnam(name)
    except KeyError as exc:
        raise BootstrapError(f"SSH user does not exist: {name}") from exc
    return name, Path(account.pw_dir), account.pw_uid, account.pw_gid


def _local_address(controller_ip: str, controller_port: int) -> str:
    family = socket.AF_INET6 if ":" in controller_ip else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as stream:
        stream.connect((controller_ip, controller_port))
        return str(ipaddress.ip_address(stream.getsockname()[0]))


def _machine_id(hostname: str) -> str:
    try:
        raw = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    except OSError:
        raw = hostname
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _install_controller_key(
    public_key: str,
    *,
    controller_ip: str,
    home: Path,
    uid: int,
    gid: int,
) -> None:
    ssh_dir = home / ".ssh"
    authorized_keys = ssh_dir / "authorized_keys"
    if ssh_dir.is_symlink():
        raise BootstrapError("SSH directory must not be a symbolic link")
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ssh_dir.is_symlink():
        raise BootstrapError("SSH directory must not be a symbolic link")
    if authorized_keys.is_symlink():
        raise BootstrapError("authorized_keys must not be a symbolic link")
    existing = authorized_keys.read_text(encoding="utf-8") if authorized_keys.exists() else ""
    key_parts = public_key.strip().split()
    key_identity = " ".join(key_parts[:2])
    restricted_line = f'from="{controller_ip}",restrict {public_key.strip()}'
    updated: list[str] = []
    installed = False
    for line in existing.splitlines():
        fields = line.split()
        matches = any(
            " ".join(fields[index : index + 2]) == key_identity
            for index in range(max(0, len(fields) - 1))
        )
        if not matches:
            updated.append(line)
            continue
        if not installed:
            updated.append(restricted_line)
            installed = True
    if not installed:
        updated.append(restricted_line)
    rendered = "\n".join(updated) + "\n"
    if rendered != existing:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".authorized_keys.", dir=ssh_dir
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, authorized_keys)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
    os.chmod(ssh_dir, 0o700)
    os.chmod(authorized_keys, 0o600)
    os.chown(ssh_dir, uid, gid)
    os.chown(authorized_keys, uid, gid)


def _start_sshd() -> None:
    _run(["ssh-keygen", "-A"], timeout=30)
    if shutil.which("systemctl"):
        for service in ("ssh", "sshd"):
            completed = subprocess.run(
                ["systemctl", "enable", "--now", service], check=False, timeout=60
            )
            if completed.returncode == 0:
                return
    raise BootstrapError("OpenSSH was installed but its service could not be started")


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if not members:
            raise BootstrapError("worker source archive is empty")
        if len(members) > MAX_SOURCE_MEMBERS:
            raise BootstrapError("worker source archive contains too many files")
        extracted_bytes = 0
        for member in members:
            path = Path(member.name)
            extracted_bytes += max(0, int(member.size))
            if extracted_bytes > MAX_SOURCE_EXTRACTED_BYTES:
                raise BootstrapError("worker source archive expands beyond the size limit")
            if (
                path.is_absolute()
                or ".." in path.parts
                or not member.isfile()
            ):
                raise BootstrapError("worker source archive contains an unsafe path")
        # Python 3.11 is still common on supported Ubuntu releases and does
        # not consistently expose tarfile's newer extraction filters.  The
        # explicit path and link checks above provide the same property for
        # this controller-created archive.
        bundle.extractall(destination, members=members)


def _install_worker_source(archive: Path, source_digest: str) -> Path:
    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    releases = INSTALL_ROOT / "releases"
    releases.mkdir(exist_ok=True)
    release = releases / source_digest[:16]
    if not (release / "omlx" / "cluster" / "inference_worker.py").is_file():
        with tempfile.TemporaryDirectory(dir=releases, prefix=".incoming-") as temporary:
            incoming = Path(temporary) / "source"
            incoming.mkdir()
            _safe_extract(archive, incoming)
            if not (incoming / "omlx" / "cluster" / "inference_worker.py").is_file():
                raise BootstrapError("worker source archive is incomplete")
            if release.exists():
                shutil.rmtree(release)
            os.replace(incoming, release)
    return release


def _ensure_worker_venv(source: Path) -> Path:
    if not (VENV / "bin" / "python").is_file():
        _run(["python3", "-m", "venv", str(VENV)], timeout=120)
    python = VENV / "bin" / "python"
    expected_requirements = "\n".join(WORKER_REQUIREMENTS) + "\n"
    installed_requirements = (
        REQUIREMENTS_MARKER.read_text(encoding="utf-8")
        if REQUIREMENTS_MARKER.exists()
        else ""
    )
    requirements_changed = installed_requirements != expected_requirements
    if requirements_changed:
        _run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
            timeout=300,
        )
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *WORKER_REQUIREMENTS,
            ],
            timeout=1800,
        )
    # Always audit an existing environment too. A later manual pip operation
    # must not silently turn a previously valid worker into a launch-time
    # surprise. Re-applying the declarative set gives a GUI re-enrollment a
    # self-healing path without asking for terminal repair commands.
    pip_check = subprocess.run(
        [str(python), "-m", "pip", "check"], check=False, timeout=120
    )
    if pip_check.returncode != 0:
        if not requirements_changed:
            _run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    *WORKER_REQUIREMENTS,
                ],
                timeout=1800,
            )
        _run([str(python), "-m", "pip", "check"], timeout=120)
    if requirements_changed:
        REQUIREMENTS_MARKER.write_text(expected_requirements, encoding="utf-8")
    site_packages = subprocess.check_output(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
        timeout=30,
    ).strip()
    Path(site_packages, "omlx_cluster_worker.pth").write_text(str(source) + "\n")
    _run(
        [
            str(python),
            "-c",
            "import mlx.core, mlx_lm, omlx; "
            "import omlx.cluster.inference_worker",
        ],
        timeout=120,
    )
    return python


def _distribution_versions(python: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata,json; "
        "names=('mlx','mlx-cuda-13','mlx-lm'); "
        "print(json.dumps({name: importlib.metadata.version(name) for name in names}))"
    )
    try:
        output = subprocess.check_output(
            [str(python), "-c", script], text=True, timeout=30
        )
        result = json.loads(output)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    return {
        str(name): str(version)
        for name, version in result.items()
        if isinstance(name, str) and isinstance(version, str)
    }


def _host_public_key() -> tuple[str, str]:
    key_path = Path("/etc/ssh/ssh_host_ed25519_key.pub")
    if not key_path.is_file():
        raise BootstrapError("the SSH server has no Ed25519 host key")
    public_key = key_path.read_text(encoding="utf-8").strip()
    return public_key, _ssh_fingerprint(public_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join this CUDA machine to oMLX")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--join-key", required=True)
    parser.add_argument("--controller-key-fingerprint", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--ssh-user")
    args = parser.parse_args(argv)

    if os.geteuid() != 0:
        raise BootstrapError("the worker bootstrap must run through sudo")
    controller, controller_ip, controller_port = _controller_url(args.controller)
    ssh_user, home, uid, gid = _ssh_user(args.ssh_user)

    hostname = socket.gethostname().strip()[:255]
    local_ip = _local_address(controller_ip, controller_port)
    node_id = f"{hostname}-{_machine_id(hostname)}"
    claim = _request_json(
        controller + "/cluster/join/claim",
        {
            "node_id": node_id,
            "hostname": hostname,
            "ssh_user": ssh_user,
            "ssh_port": 22,
            "addresses": [local_ip],
            "accelerator": "cuda",
            "platform": platform.platform()[:255],
        },
        bearer=args.join_key,
    )
    session_token = str(claim.get("session_token") or "")
    source_digest = str(claim.get("source_digest") or "")
    controller_key = str(claim.get("controller_public_key") or "")
    returned_fingerprint = str(claim.get("controller_key_fingerprint") or "")
    if (
        not session_token
        or len(source_digest) != 64
        or source_digest != args.source_digest
        or returned_fingerprint != args.controller_key_fingerprint
        or _ssh_fingerprint(controller_key) != args.controller_key_fingerprint
    ):
        raise BootstrapError("controller identity or enrollment response did not verify")

    # Consume the short-lived join key before apt can use its full bounded
    # timeout. The longer claimed session covers provisioning and remains
    # revocable from the controller dashboard.
    _ensure_system_packages()
    _install_controller_key(
        controller_key,
        controller_ip=controller_ip,
        home=home,
        uid=uid,
        gid=gid,
    )
    _start_sshd()

    with tempfile.TemporaryDirectory(prefix="omlx-worker-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        _download(
            controller + "/cluster/join/source",
            archive,
            bearer=session_token,
        )
        observed_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if observed_digest != source_digest:
            raise BootstrapError("worker source digest did not match the join command")
        source = _install_worker_source(archive, source_digest)

    python = _ensure_worker_venv(source)
    host_public_key, host_fingerprint = _host_public_key()
    complete = _request_json(
        controller + "/cluster/join/complete",
        {
            "node_id": node_id,
            "hostname": hostname,
            "ssh_user": ssh_user,
            "ssh_port": 22,
            "addresses": [local_ip],
            "accelerator": "cuda",
            "platform": platform.platform()[:255],
            "python_executable": str(python),
            "source_digest": source_digest,
            "ssh_host_public_key": host_public_key,
            "ssh_host_fingerprint": host_fingerprint,
            "runtime": _distribution_versions(python),
        },
        bearer=session_token,
    )
    if complete.get("status") != "joined":
        raise BootstrapError("controller did not confirm the joined worker")
    print(f"oMLX worker joined as {ssh_user}@{local_ip}")
    print("Return to the Cluster dashboard; hardware verification will continue there.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"oMLX join failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
