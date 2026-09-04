# SPDX-License-Identifier: Apache-2.0

import json
import re
import stat

import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.registry import ClusterRegistry


def _deployment(model: str, deployment_id: str = "test-cluster") -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id=deployment_id,
        model=model,
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 2, 4, 64),
            PipelineAssignment("peer", 1, 0, 2, 10, 2, 4, 32),
        ),
        plan_hash="e" * 64,
    )


def test_registry_persists_atomically_with_private_permissions(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    deployment = _deployment(str(model))
    registry = ClusterRegistry(tmp_path)

    registry.upsert(deployment)
    restored = ClusterRegistry(tmp_path)

    assert restored.get_for_model(str(model)) == deployment
    assert restored.get("test-cluster") == deployment
    assert stat.S_IMODE(registry.path.stat().st_mode) == 0o600


def test_registry_corruption_fails_closed_without_blocking_server(tmp_path):
    path = tmp_path / "cluster" / "deployments.json"
    path.parent.mkdir()
    path.write_text("{broken")

    registry = ClusterRegistry(tmp_path)

    assert registry.list() == ()
    assert "could not read" in registry.load_error
    assert registry.to_dict()["load_error"]


def test_registry_rejects_duplicate_id_for_another_model(tmp_path):
    registry = ClusterRegistry(tmp_path)
    registry.upsert(_deployment("model-a"))

    with pytest.raises(ValueError, match="already in use"):
        registry.upsert(_deployment("model-b"))


def test_registry_rolls_back_memory_if_atomic_save_fails(tmp_path, monkeypatch):
    registry = ClusterRegistry(tmp_path)
    first = _deployment("model-a")
    registry.upsert(first)

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(registry, "_save", fail)
    with pytest.raises(OSError, match="disk full"):
        registry.upsert(_deployment("model-a", "replacement"))

    assert registry.get_for_model("model-a") == first


_CREDENTIAL_KEYS = {
    "token", "access_token", "auth_token", "api_token", "pairing_token",
    "bearer", "private_key", "api_key", "secret", "secret_key",
    "password", "passphrase", "credential", "credentials",
}


# Keys that legitimately hold an opaque blob. ``plan_hash`` is a content hash
# of the plan — public by design, and indistinguishable from a token by shape
# alone, so it is named here rather than weakening the check for everything.
_OPAQUE_BY_DESIGN = {"plan_hash"}


def _keys_and_strings(node, keys, strings, parent=""):
    """Every key name, and every string value not opaque by design."""

    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(str(key).lower())
            _keys_and_strings(value, keys, strings, str(key).lower())
    elif isinstance(node, list):
        for item in node:
            _keys_and_strings(item, keys, strings, parent)
    elif isinstance(node, str) and parent not in _OPAQUE_BY_DESIGN:
        strings.append(node)


def test_registry_file_contains_no_credential_fields(tmp_path):
    """No secret may be written to the registry file.

    Checks credential-bearing *key names* plus high-entropy values, rather
    than the substring "token": the plan legitimately reports context lengths
    in fields like ``max_context_tokens`` and ``kv_bytes_per_token``, and a
    substring check cannot tell a context token from an auth token. Matching
    on key names and on what a secret actually looks like is the stricter
    test, not the looser one.
    """

    registry = ClusterRegistry(tmp_path)
    registry.upsert(_deployment("model-a"))

    payload = json.loads(registry.path.read_text())
    keys: list[str] = []
    strings: list[str] = []
    _keys_and_strings(payload, keys, strings)

    leaked = sorted(set(keys) & _CREDENTIAL_KEYS)
    assert not leaked, f"credential-bearing keys in the registry: {leaked}"

    # Substring checks still hold for names no legitimate field uses.
    serialized = json.dumps(payload).lower()
    assert "private_key" not in serialized
    assert "password" not in serialized
    assert "passphrase" not in serialized

    # And no value that merely looks like a secret: a long opaque blob of
    # base64/hex with no structure, which is what a key or token would be.
    for value in strings:
        assert not (
            len(value) >= 32
            and re.fullmatch(r"[A-Za-z0-9+/=_-]+", value)
            and not value.startswith(("/", "~", "http"))
            and "." not in value
            and "-" not in value
        ), f"value looks like a credential: {value[:16]}…"
