# SPDX-License-Identifier: Apache-2.0
"""Automatic SSH key management for cluster peer pairing."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .deployment import validate_ssh_target

# Key management constants
_SSH_KEY_TYPE = "ed25519"
_SSH_KEY_COMMENT = "omlx-cluster"
_SSH_KEY_BITS = 256

# Default SSH directory
_SSH_DIR = Path.home() / ".ssh"
_SSH_KEY_PATH = _SSH_DIR / "omlx_cluster"


@dataclass(frozen=True)
class SSHKeyPair:
    """An SSH key pair with its public key in OpenSSH format."""

    private_key_path: Path
    public_key_path: Path
    public_key: str
    fingerprint: str
    key_type: str
    created_at: float


def _ssh_keygen_executable() -> str:
    """Find the ssh-keygen executable."""

    system_keygen = Path("/usr/bin/ssh-keygen")
    if system_keygen.is_file() and os.access(system_keygen, os.X_OK):
        return str(system_keygen)
    discovered = shutil.which("ssh-keygen")
    if discovered is None:
        raise RuntimeError("ssh-keygen is unavailable")
    return str(Path(discovered).resolve())


def _key_fingerprint(public_key: str) -> str:
    """Compute the SHA256 fingerprint of an SSH public key."""

    # Extract the base64-encoded key data
    parts = public_key.strip().split()
    if len(parts) < 2 or parts[0] not in {"ssh-ed25519", "ssh-rsa"}:
        raise ValueError("invalid SSH public key format")
    try:
        key_data = base64.b64decode(parts[1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid SSH public key format") from exc
    if not key_data:
        raise ValueError("invalid SSH public key format")
    digest = hashlib.sha256(key_data).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def ssh_public_key_fingerprint(public_key: str) -> str:
    """Return the OpenSSH SHA-256 fingerprint for a public key."""

    return _key_fingerprint(public_key)


def generate_ssh_key_pair(
    *,
    key_path: Path | None = None,
    comment: str = _SSH_KEY_COMMENT,
    overwrite: bool = False,
) -> SSHKeyPair:
    """Generate a new SSH key pair for cluster authentication."""

    if key_path is None:
        key_path = _SSH_KEY_PATH

    if key_path.exists() and not overwrite:
        # Load existing key
        pubkey_path = Path(str(key_path) + ".pub")
        if not pubkey_path.exists():
            raise RuntimeError(f"private key exists but public key missing: {key_path}")
        public_key = pubkey_path.read_text().strip()
        return SSHKeyPair(
            private_key_path=key_path,
            public_key_path=pubkey_path,
            public_key=public_key,
            fingerprint=_key_fingerprint(public_key),
            key_type=_SSH_KEY_TYPE,
            created_at=key_path.stat().st_mtime,
        )

    # Ensure SSH directory exists
    key_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(key_path.parent, 0o700)
    pubkey_path = Path(str(key_path) + ".pub")

    # ssh-keygen prompts before overwriting an existing path, which a server
    # process cannot answer. Build rotations beside the live key and replace
    # it only after both new files exist.
    rotation_dir = None
    generation_path = key_path
    if key_path.exists() and overwrite:
        rotation_dir = tempfile.TemporaryDirectory(
            prefix=f".{key_path.name}-",
            dir=key_path.parent,
        )
        generation_path = Path(rotation_dir.name) / key_path.name

    try:
        keygen = _ssh_keygen_executable()
        result = subprocess.run(
            [
                keygen,
                "-t",
                _SSH_KEY_TYPE,
                "-b",
                str(_SSH_KEY_BITS),
                "-f",
                str(generation_path),
                "-N",
                "",  # No passphrase
                "-C",
                comment,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ssh-keygen failed: {result.stderr.strip()}")

        generated_public_path = Path(str(generation_path) + ".pub")
        public_key = generated_public_path.read_text().strip()
        fingerprint = _key_fingerprint(public_key)
        if rotation_dir is not None:
            backup_public_path = Path(rotation_dir.name) / "previous.pub"
            public_key_existed = pubkey_path.exists()
            if public_key_existed:
                shutil.copy2(pubkey_path, backup_public_path)
            try:
                os.replace(generated_public_path, pubkey_path)
                os.replace(generation_path, key_path)
            except Exception:
                if public_key_existed:
                    os.replace(backup_public_path, pubkey_path)
                else:
                    with suppress(FileNotFoundError):
                        pubkey_path.unlink()
                raise

        os.chmod(key_path, 0o600)
        os.chmod(pubkey_path, 0o644)
    finally:
        if rotation_dir is not None:
            rotation_dir.cleanup()

    return SSHKeyPair(
        private_key_path=key_path,
        public_key_path=pubkey_path,
        public_key=public_key,
        fingerprint=fingerprint,
        key_type=_SSH_KEY_TYPE,
        created_at=time.time(),
    )


def get_or_create_ssh_key() -> SSHKeyPair:
    """Get the existing SSH key pair or create a new one."""

    return generate_ssh_key_pair(overwrite=False)


def install_authorized_key(
    *,
    public_key: str,
    authorized_keys_path: Path | None = None,
) -> bool:
    """Install a public key in the authorized_keys file."""

    if authorized_keys_path is None:
        authorized_keys_path = _SSH_DIR / "authorized_keys"

    # Ensure SSH directory exists
    authorized_keys_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(authorized_keys_path.parent, 0o700)

    # Read existing keys
    existing_keys: set[str] = set()
    if authorized_keys_path.exists():
        existing_keys = {
            line.strip()
            for line in authorized_keys_path.read_text().splitlines()
            if line.strip()
        }

    # Check if key already exists
    if public_key.strip() in existing_keys:
        return False  # Already installed

    # Append the new key
    with authorized_keys_path.open("a") as f:
        f.write(public_key.strip() + "\n")

    os.chmod(authorized_keys_path, 0o600)
    return True


def get_known_hosts_path() -> Path:
    """Get the path to the user's known_hosts file."""

    return _SSH_DIR / "known_hosts"


def pin_enrolled_host_key(
    *,
    hostname: str,
    public_key: str,
    known_hosts_path: Path | None = None,
) -> bool:
    """Install the host key reported by a one-time enrolled worker.

    Existing identities are compared rather than overwritten.  This preserves
    the same changed-host fail-closed rule used by normal cluster SSH.
    """

    hostname = validate_ssh_target(hostname)
    if "@" in hostname:
        hostname = hostname.rsplit("@", 1)[1]
    parts = public_key.strip().split()
    if len(parts) < 2 or parts[0] not in {"ssh-ed25519", "ssh-rsa"}:
        raise ValueError("invalid SSH host public key")
    normalized_key = " ".join(parts[:2])
    if known_hosts_path is None:
        known_hosts_path = get_known_hosts_path()

    existing_lines = (
        known_hosts_path.read_text(encoding="utf-8").splitlines()
        if known_hosts_path.exists()
        else []
    )
    matching_hosts = [
        line.strip()
        for line in existing_lines
        if line.strip() and not line.lstrip().startswith("#")
        and line.split(maxsplit=1)[0] == hostname
    ]
    if matching_hosts:
        if any(
            len(line.split()) >= 3
            and " ".join(line.split()[1:3]) == normalized_key
            for line in matching_hosts
        ):
            return False
        raise RuntimeError(
            f"refusing changed SSH host key for enrolled worker {hostname}"
        )

    known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(known_hosts_path.parent, 0o700)
    with known_hosts_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{hostname} {normalized_key}\n")
    os.chmod(known_hosts_path, 0o600)
    return True


