# SPDX-License-Identifier: Apache-2.0

import json
import stat

import pytest

from omlx.cluster.identity import (
    NodeIdentity,
    configure_node_identity,
    get_node_identity,
    load_or_create,
    reset_configured_identity,
)


def test_creates_and_persists_identity_with_private_permissions(tmp_path):
    path = tmp_path / "cluster" / "identity.json"

    identity = load_or_create(path)

    assert identity.node_id
    assert identity.friendly_name
    assert identity.schema_version == 1
    assert identity.created_at > 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_load_or_create_is_stable_across_reloads(tmp_path):
    path = tmp_path / "identity.json"

    first = load_or_create(path)
    second = load_or_create(path)

    assert second.node_id == first.node_id
    assert second.friendly_name == first.friendly_name
    assert second.created_at == first.created_at


def test_collision_repair_suffixes_name_but_keeps_node_id(tmp_path):
    path = tmp_path / "identity.json"
    identity = load_or_create(path)
    original_id = identity.node_id
    original_name = identity.friendly_name

    repaired = load_or_create(path, taken_names={original_name})

    assert repaired.node_id == original_id
    assert repaired.friendly_name == f"{original_name}-2"
    # Repair is persisted.
    assert load_or_create(path).friendly_name == f"{original_name}-2"


def test_collision_repair_escalates_past_taken_suffixes(tmp_path):
    path = tmp_path / "identity.json"
    identity = load_or_create(path)
    name = identity.friendly_name

    repaired = load_or_create(path, taken_names={name, f"{name}-2"})

    assert repaired.friendly_name == f"{name}-3"


def test_rename_persists_and_never_touches_node_id(tmp_path):
    path = tmp_path / "identity.json"
    identity = load_or_create(path)
    original_id = identity.node_id

    identity.rename("  My   Studio Mac  ")

    assert identity.node_id == original_id
    assert identity.friendly_name == "My-Studio-Mac"
    assert load_or_create(path).friendly_name == "My-Studio-Mac"


def test_rename_rejects_empty_name(tmp_path):
    identity = load_or_create(tmp_path / "identity.json")
    with pytest.raises(ValueError, match="must not be empty"):
        identity.rename("   ")


def test_corrupt_file_fails_safe_without_rotating_node_id(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text("{broken", encoding="utf-8")

    identity = load_or_create(path)

    # In-memory identity works, the error is surfaced, and the corrupt file
    # is preserved rather than overwritten with a fresh node_id.
    assert identity.node_id
    assert identity.load_error and "could not read" in identity.load_error
    assert path.read_text(encoding="utf-8") == "{broken"


def test_wrong_schema_version_is_rejected_as_corrupt(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "node_id": "abc",
                "friendly_name": "x",
                "created_at": 1.0,
            }
        ),
        encoding="utf-8",
    )

    identity = load_or_create(path)

    assert identity.load_error is not None


def test_roundtrip_dict():
    identity = NodeIdentity(node_id="n-1", friendly_name="studio", created_at=7.5)
    restored = NodeIdentity.from_dict(identity.to_dict())
    assert restored == identity


def test_configure_and_get_process_wide_identity(tmp_path):
    reset_configured_identity()
    try:
        identity = configure_node_identity(tmp_path)
        assert get_node_identity() is identity
        assert (tmp_path / "cluster" / "identity.json").exists()
    finally:
        reset_configured_identity()


def test_get_node_identity_requires_configuration():
    reset_configured_identity()
    with pytest.raises(RuntimeError, match="not configured"):
        get_node_identity()
