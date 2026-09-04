# SPDX-License-Identifier: Apache-2.0
"""Automatic SSH key management for cluster peer pairing."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deployment import validate_ssh_target
from .token_auth import sign_pairing_payload, verify_pairing_signature

# Key management constants
_SSH_KEY_TYPE = "ed25519"
_SSH_KEY_COMMENT = "omlx-cluster"
_SSH_KEY_BITS = 256
_KEY_EXCHANGE_TTL = 300  # 5 minutes

# Default SSH directory
_SSH_DIR = Path.home() / ".ssh"
_SSH_KEY_PATH = _SSH_DIR / "omlx_cluster"
_SSH_PUBKEY_PATH = _SSH_DIR / "omlx_cluster.pub"


@dataclass(frozen=True)
class SSHKeyPair:
    """An SSH key pair with its public key in OpenSSH format."""

    private_key_path: Path
    public_key_path: Path
    public_key: str
    fingerprint: str
    key_type: str
    created_at: float


@dataclass(frozen=True)
class KeyExchangeToken:
    """A signed token for exchanging SSH public keys between peers."""

    token: str
    public_key: str
    fingerprint: str
    node_id: str
    created_at: float
    expires_at: float


def _ssh_executable() -> str:
    """Find the OpenSSH executable."""

    system_ssh = Path("/usr/bin/ssh")
    if system_ssh.is_file() and os.access(system_ssh, os.X_OK):
        return str(system_ssh)
    discovered = shutil.which("ssh")
    if discovered is None:
        raise RuntimeError("OpenSSH client is unavailable")
    return str(Path(discovered).resolve())


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


def create_key_exchange_token(
    *,
    public_key: str,
    node_id: str,
    shared_secret: str,
    ttl: int = _KEY_EXCHANGE_TTL,
) -> str:
    """Create an authenticated token containing an SSH public key."""

    if not 1 <= ttl <= _KEY_EXCHANGE_TTL:
        raise ValueError(f"key exchange TTL must be between 1 and {_KEY_EXCHANGE_TTL}")
    node_id = validate_ssh_target(node_id)

    fingerprint = _key_fingerprint(public_key)
    token = secrets.token_urlsafe(32)
    created_at = time.time()
    expires_at = created_at + ttl

    payload = {
        "token": token,
        "public_key": public_key,
        "fingerprint": fingerprint,
        "node_id": node_id,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    payload_json = json.dumps(payload, sort_keys=True)
    signature = sign_pairing_payload(payload_json, shared_secret=shared_secret)

    exchange_data = {
        "token": token,
        "signature": signature,
        "public_key": public_key,
        "fingerprint": fingerprint,
        "node_id": node_id,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    return base64.urlsafe_b64encode(
        json.dumps(exchange_data).encode()
    ).decode()


def verify_key_exchange_token(
    encoded_token: str,
    *,
    shared_secret: str,
) -> KeyExchangeToken | None:
    """Verify a key exchange token and return the key data."""

    try:
        data = json.loads(base64.urlsafe_b64decode(encoded_token))
        token = data["token"]
        signature = data["signature"]
        public_key = data["public_key"]
        fingerprint = data["fingerprint"]
        node_id = data["node_id"]
        created_at = data["created_at"]
        expires_at = data["expires_at"]
    except (binascii.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    if (
        not isinstance(created_at, (int, float))
        or isinstance(created_at, bool)
        or not isinstance(expires_at, (int, float))
        or isinstance(expires_at, bool)
        or expires_at <= created_at
        or expires_at - created_at > _KEY_EXCHANGE_TTL
    ):
        return None
    try:
        node_id = validate_ssh_target(node_id)
    except ValueError:
        return None

    # Check TTL
    if time.time() > expires_at:
        return None

    # Verify signature
    payload = {
        "token": token,
        "public_key": public_key,
        "fingerprint": fingerprint,
        "node_id": node_id,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    payload_json = json.dumps(payload, sort_keys=True)
    if not verify_pairing_signature(
        payload_json,
        signature,
        shared_secret=shared_secret,
    ):
        return None

    # Verify fingerprint matches
    if _key_fingerprint(public_key) != fingerprint:
        return None

    return KeyExchangeToken(
        token=token,
        public_key=public_key,
        fingerprint=fingerprint,
        node_id=node_id,
        created_at=created_at,
        expires_at=expires_at,
    )


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


def is_host_known(hostname: str) -> bool:
    """Check if a host is already in known_hosts."""

    known_hosts = get_known_hosts_path()
    if not known_hosts.exists():
        return False

    ssh_keygen = _ssh_keygen_executable()
    result = subprocess.run(
        [ssh_keygen, "-F", hostname, "-f", str(known_hosts)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def add_host_key(
    *,
    hostname: str,
    host_key: str,
    known_hosts_path: Path | None = None,
) -> bool:
    """Add a host key to known_hosts."""

    if known_hosts_path is None:
        known_hosts_path = get_known_hosts_path()

    if is_host_known(hostname):
        return False  # Already known

    known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(known_hosts_path.parent, 0o700)

    # Append the host key
    with known_hosts_path.open("a") as f:
        f.write(f"{hostname} {host_key}\n")

    if known_hosts_path.exists():
        os.chmod(known_hosts_path, 0o600)

    return True


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


def get_ssh_key_info() -> dict[str, Any]:
    """Get information about the current SSH key pair."""

    try:
        key_pair = get_or_create_ssh_key()
    except RuntimeError:
        return {
            "available": False,
            "error": "SSH key generation failed",
        }

    return {
        "available": True,
        "key_type": key_pair.key_type,
        "fingerprint": key_pair.fingerprint,
        "private_key_path": str(key_pair.private_key_path),
        "public_key_path": str(key_pair.public_key_path),
        "created_at": key_pair.created_at,
    }


def generate_key_exchange_for_peer(
    *,
    node_id: str,
    shared_secret: str,
    key_pair: SSHKeyPair | None = None,
) -> str:
    """Generate a key exchange token for a specific peer."""

    if key_pair is None:
        key_pair = get_or_create_ssh_key()

    return create_key_exchange_token(
        public_key=key_pair.public_key,
        node_id=node_id,
        shared_secret=shared_secret,
    )


def exchange_keys_with_peer(
    *,
    peer_token: str,
    shared_secret: str,
    known_hosts_path: Path | None = None,
) -> dict[str, Any]:
    """Exchange keys with a peer using their key exchange token."""

    verified = verify_key_exchange_token(
        peer_token,
        shared_secret=shared_secret,
    )
    if verified is None:
        return {
            "success": False,
            "error": "invalid or expired key exchange token",
        }

    # Install the peer's public key in our authorized_keys
    key_installed = install_authorized_key(public_key=verified.public_key)

    # Fetch and add the peer's SSH host key to known_hosts
    host_key_added = False
    with suppress(Exception):
        host_key_added = add_verified_peer_host_key(hostname=verified.node_id)

    return {
        "success": True,
        "peer_node_id": verified.node_id,
        "peer_fingerprint": verified.fingerprint,
        "key_installed": key_installed,
        "host_key_added": host_key_added,
        "expires_at": verified.expires_at,
    }


def add_verified_peer_host_key(*, hostname: str) -> bool:
    """Fetch and add a peer's SSH host key to known_hosts.

    Uses ssh-keyscan to retrieve the host key, which is then added to
    known_hosts. The pairing token authenticates the peer's *user* key, not
    its host key, so this is trust-on-first-use over the current network
    path, same as the accept-new policy every cluster SSH call already uses.
    It only saves the first real connection from doing that registration.
    """

    if is_host_known(hostname):
        return False  # Already known

    ssh_keyscan = shutil.which("ssh-keyscan")
    if ssh_keyscan is None:
        raise RuntimeError("ssh-keyscan is unavailable")

    result = subprocess.run(
        [ssh_keyscan, "-t", "ed25519,rsa", hostname],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ssh-keyscan failed for {hostname}")

    known_hosts_path = get_known_hosts_path()
    known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(known_hosts_path.parent, 0o700)

    with known_hosts_path.open("a") as f:
        f.write(result.stdout)

    if known_hosts_path.exists():
        os.chmod(known_hosts_path, 0o600)

    return True


def store_key_in_keychain(*, service: str = "omlx-cluster", account: str = "ssh-key") -> bool:
    """Store the SSH key fingerprint in macOS Keychain.

    This provides an additional layer of security by storing the key
    metadata in the system keychain rather than relying solely on
    filesystem permissions.
    """

    try:
        key_pair = get_or_create_ssh_key()
    except RuntimeError:
        return False

    # Use security command to store in keychain
    security = shutil.which("security")
    if security is None:
        return False  # Not on macOS or security command unavailable

    try:
        # Delete any existing entry first
        subprocess.run(
            [security, "delete-generic-password", "-s", service, "-a", account],
            capture_output=True,
            check=False,
        )

        # Store the fingerprint as the keychain item
        subprocess.run(
            [
                security,
                "add-generic-password",
                "-s", service,
                "-a", account,
                "-w",
                key_pair.fingerprint,
            ],
            capture_output=True,
            check=True,
        )

        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
