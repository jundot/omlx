# SPDX-License-Identifier: Apache-2.0
"""SSH key provisioning for cluster hosts.

The runtime cluster transport stays keys-only (``PasswordAuthentication=no``).
Provisioning has two transports:

* **Admin key (default):** an operator-supplied private key authenticates to
  each host so the managed ``~/.ssh/omlx_cluster.pub`` can be installed.
* **One-shot password (``--ask-pass``):** a password is used exactly once, via
  the ``sshpass`` transport with ``SSHPASS`` in the subprocess environment, to
  install the managed key.  The password is never persisted, logged, or placed
  on the command line, and after provisioning the host still only accepts keys.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from .deployment import validate_ssh_target
from .ssh_identity import _MANAGED_IDENTITY, _SSH_PORT, default_ssh_user
from .ssh_keys import generate_ssh_key_pair
from .ssh_policy import cluster_ssh_options

_SSH_TRANSPORT_TIMEOUT = 120  # seconds per ssh subprocess

# A canonical OpenSSH authorized-keys line: key type, base64 blob, optional
# comment.  The value is interpolated inside single quotes in the remote
# install command, so the only characters that could break out are `'`, CR and
# LF.  The comment charset is an allowlist that excludes those (and every other
# shell metacharacter) while still accepting the `user@host.example` form
# ssh-keygen emits by default -- a dot in a comment is not an injection vector.
_AUTHORIZED_KEY_RE = re.compile(
    r"^ssh-(?:ed25519|rsa) [A-Za-z0-9+/]+={0,3}(?: [A-Za-z0-9@._+:/ -]{1,4096})?$"
)


class ProvisioningError(RuntimeError):
    """A host could not be provisioned."""


def managed_public_key() -> str:
    """Return the managed cluster public key, generating the pair if needed."""

    return generate_ssh_key_pair().public_key


def build_authorized_keys_command(public_key: str) -> str:
    """Build the remote command that installs a key idempotently.

    Mirrors ``ssh-copy-id`` semantics: the key is appended only when absent,
    and the remote ``authorized_keys`` permissions are corrected afterward.
    The public key is validated against the canonical OpenSSH line format so
    that it cannot inject shell metacharacters into the remote command.
    """

    public_key = public_key.strip()
    if not _AUTHORIZED_KEY_RE.fullmatch(public_key):
        raise ValueError("invalid SSH public key format")
    return (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"grep -qF '{public_key}' ~/.ssh/authorized_keys 2>/dev/null || "
        f"echo '{public_key}' >> ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys"
    )


def _ssh_argv(
    host: str,
    user: str,
    *,
    port: int,
    identity: str,
    connect_timeout: float | None,
    command: Sequence[str],
) -> list[str]:
    """Build a non-interactive ssh argv using a specific identity."""

    if not (1 <= int(port) <= 65535):
        raise ValueError("SSH port must be between 1 and 65535")
    target = f"{user}@{host}"
    # One policy for every cluster subprocess: a second hand-rolled option list
    # is how an interactive prompt gets back in. Only the identity differs here,
    # because the managed key is not on the host yet.
    return [
        "ssh",
        *cluster_ssh_options(connect_timeout=connect_timeout, identity=identity),
        "-p",
        str(port),
        target,
        *command,
    ]


def _run_checked(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SSH_TRANSPORT_TIMEOUT,
            check=False,
            **kwargs,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvisioningError(f"ssh transport failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ProvisioningError(detail or f"ssh exited with {result.returncode}")
    return result


def install_managed_key(
    *,
    host: str,
    user: str | None = None,
    port: int = _SSH_PORT,
    admin_key_path: str | None = None,
    password: str | None = None,
    connect_timeout: float | None = None,
) -> bool:
    """Install the managed public key on one host.

    ``admin_key_path`` authenticates with an existing private key.  When
    ``password`` is supplied the ``sshpass`` binary (via ``SSHPASS`` in the
    subprocess environment) authenticates exactly once; the password is never
    written to disk, the process list, or the parent environment.
    """

    host = validate_ssh_target(host)
    if not (1 <= port <= 65535):
        raise ValueError("SSH port must be between 1 and 65535")
    resolved_user = user or default_ssh_user()
    if password is None and admin_key_path is None:
        raise ProvisioningError(
            "provisioning needs an admin key path or a one-shot password"
        )
    if password is not None and admin_key_path is not None:
        raise ProvisioningError("use either --key or --ask-pass, not both")

    command = [build_authorized_keys_command(managed_public_key())]

    if password is not None:
        sshpass = shutil.which("sshpass")
        if sshpass is None:
            raise ProvisioningError(
                "--ask-pass requires the sshpass binary (brew install sshpass)"
            )
        argv = _ssh_argv(
            host,
            resolved_user,
            port=port,
            identity=_MANAGED_IDENTITY,
            connect_timeout=connect_timeout,
            command=command,
        )
        env = dict(os.environ)
        env["SSHPASS"] = password
        argv = [sshpass, "-e", *argv]
        # Password reaches ssh only through the inherited environment, which
        # dies with the subprocess; nothing is persisted or echoed.
        _run_checked(argv, env=env)
        return True

    argv = _ssh_argv(
        host,
        resolved_user,
        port=port,
        identity=admin_key_path or _MANAGED_IDENTITY,
        connect_timeout=connect_timeout,
        command=command,
    )
    _run_checked(argv)
    return True


def verify_managed_login(
    *,
    host: str,
    user: str | None = None,
    port: int = _SSH_PORT,
    connect_timeout: float | None = None,
) -> bool:
    """Confirm the managed identity can log in without prompting."""

    host = validate_ssh_target(host)
    resolved_user = user or default_ssh_user()
    argv = _ssh_argv(
        host,
        resolved_user,
        port=port,
        identity=_MANAGED_IDENTITY,
        connect_timeout=connect_timeout,
        command=["true"],
    )
    _run_checked(argv)
    return True


def provision_hosts(
    hosts: Sequence[tuple[str, str]],
    *,
    admin_key_path: str | None = None,
    password: str | None = None,
    installer: Callable[..., bool] | None = None,
    verifier: Callable[..., bool] | None = None,
) -> dict:
    """Provision one or more hosts; per-host failures never abort the batch.

    ``hosts`` is a sequence of ``(host, user)`` pairs.  Returns
    ``{"ok": n, "failed": n, "errors": {host: message}}``.

    Writing the key is not the same as being able to use it: a wrong user, a
    home directory the server will not trust, or an ``authorized_keys`` mode
    OpenSSH rejects all leave the install looking successful. Each host is
    therefore logged into with the managed identity before it is counted, so a
    host reported ``ok`` is one the cluster can actually reach.
    """

    installer = installer or install_managed_key
    verifier = verifier or verify_managed_login
    ok = 0
    failed = 0
    errors: dict[str, str] = {}
    for host, user in hosts:
        try:
            installer(
                host=host,
                user=user,
                admin_key_path=admin_key_path,
                password=password,
            )
            verifier(host=host, user=user)
        except (ProvisioningError, ValueError, OSError) as exc:
            failed += 1
            errors[host] = str(exc)
        else:
            ok += 1
    return {"ok": ok, "failed": failed, "errors": errors}
