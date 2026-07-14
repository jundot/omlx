# SPDX-License-Identifier: Apache-2.0
"""Lifecycle tests for filesystem-driven model-state reconciliation."""

import json
import shutil
from pathlib import Path

from omlx.model_discovery import discover_models_from_dirs_with_status
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import reconcile_discovered_model_state
from omlx.settings import ModelSettings as GlobalModelSettings


def _make_model(root: Path, model_id: str) -> Path:
    model_dir = root / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )
    (model_dir / "model.safetensors").write_bytes(b"model")
    return model_dir


def _make_hf_cache_model(root: Path, model_id: str) -> tuple[Path, Path]:
    entry = root / f"models--{model_id}"
    snapshot = entry / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "llama"}),
        encoding="utf-8",
    )
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"model")
    return entry, weights


def _scan(*roots: Path):
    return discover_models_from_dirs_with_status(list(roots))


def _reconcile(
    manager: ModelSettingsManager,
    configured_roots: list[Path],
    *,
    scanned_roots: list[Path] | None = None,
    hf_cache_dir: Path | None = None,
    hf_cache_enabled: bool = False,
    retire_unobserved_configured: bool = False,
    prune_missing: bool = True,
) -> list[str]:
    discovery = _scan(*(scanned_roots or configured_roots))
    return reconcile_discovered_model_state(
        manager,
        discovery,
        configured_roots,
        hf_cache_dir=hf_cache_dir,
        hf_cache_enabled=hf_cache_enabled,
        retire_unobserved_configured=retire_unobserved_configured,
        prune_missing=prune_missing,
    )


def test_first_inventory_releases_legacy_alias_with_empty_hf_cache(tmp_path):
    """Upgrading users do not need a pre-existing root inventory."""
    model_root = tmp_path / "models"
    hf_cache = tmp_path / "hf-cache"
    _make_model(model_root, "model-a")
    hf_cache.mkdir()

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("deleted-model", ModelSettings(model_alias="shared"))
    manager.save_profile("deleted-model", "fast", "Fast", None, {"temperature": 0.1})

    removed = _reconcile(
        manager,
        [model_root],
        scanned_roots=[model_root, hf_cache],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=True,
    )

    assert removed == ["deleted-model"]
    assert "deleted-model" not in manager.get_all_settings()
    assert manager.list_profiles("deleted-model") == []


def test_cold_start_releases_alias_after_external_folder_removal(tmp_path):
    """Deleting a model while oMLX is stopped releases its state on startup."""
    model_root = tmp_path / "models"
    _make_model(model_root, "model-a")
    deleted_dir = _make_model(model_root, "model-b")
    settings_path = tmp_path / "settings"

    manager = ModelSettingsManager(settings_path)
    manager.set_settings("model-b", ModelSettings(model_alias="shared"))
    manager.save_profile("model-b", "fast", "Fast", None, {"top_k": 10})
    assert _reconcile(manager, [model_root]) == []

    shutil.rmtree(deleted_dir)
    restarted = ModelSettingsManager(settings_path)
    assert _reconcile(restarted, [model_root]) == ["model-b"]
    assert "model-b" not in restarted.get_all_settings()
    assert restarted.list_profiles("model-b") == []

    restarted.set_settings("model-a", ModelSettings(model_alias="shared"))
    assert restarted.get_settings("model-a").model_alias == "shared"

    # Reappearance never restores state that reconciliation already deleted.
    _make_model(model_root, "model-b")
    assert _reconcile(restarted, [model_root]) == []
    assert "model-b" not in restarted.get_all_settings()
    assert restarted.get_settings("model-b").model_alias is None


def test_explicit_reload_releases_alias_after_live_folder_removal(tmp_path):
    """A manual reload reconciles folders removed while oMLX is running."""
    model_root = tmp_path / "models"
    _make_model(model_root, "model-a")
    deleted_dir = _make_model(model_root, "model-b")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("model-b", ModelSettings(model_alias="shared"))
    assert _reconcile(manager, [model_root]) == []

    shutil.rmtree(deleted_dir)

    assert _reconcile(manager, [model_root]) == ["model-b"]
    assert "model-b" not in manager.get_all_settings()


def test_observation_only_refresh_seeds_inventory_after_first_download(tmp_path):
    """A download into a new empty root enables later offline cleanup."""
    model_root = tmp_path / "models"
    model_root.mkdir()
    settings_path = tmp_path / "settings"
    manager = ModelSettingsManager(settings_path)

    assert _reconcile(manager, [model_root]) == []

    downloaded_dir = _make_model(model_root, "only-model")
    assert _reconcile(manager, [model_root], prune_missing=False) == []
    manager.set_settings("only-model", ModelSettings(model_alias="shared"))

    shutil.rmtree(downloaded_dir)
    restarted = ModelSettingsManager(settings_path)
    assert _reconcile(restarted, [model_root]) == ["only-model"]
    assert "only-model" not in restarted.get_all_settings()


def test_observation_only_refresh_never_prunes_missing_state(tmp_path):
    """A download/visibility refresh records additions without acting as reload."""
    model_root = tmp_path / "models"
    removed_dir = _make_model(model_root, "model-a")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("model-a", ModelSettings(model_alias="protected"))
    assert _reconcile(manager, [model_root]) == []

    shutil.rmtree(removed_dir)
    _make_model(model_root, "model-b")
    assert _reconcile(manager, [model_root], prune_missing=False) == []
    assert manager.get_settings("model-a").model_alias == "protected"

    assert _reconcile(manager, [model_root]) == ["model-a"]


def test_existing_incomplete_local_folder_does_not_release_alias(tmp_path):
    """An incomplete folder protects itself without blocking a real removal."""
    model_root = tmp_path / "models"
    model_dir = _make_model(model_root, "model-a")
    removed_dir = _make_model(model_root, "model-b")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("model-a", ModelSettings(model_alias="protected"))
    manager.set_settings("model-b", ModelSettings(model_alias="released"))
    assert _reconcile(manager, [model_root]) == []

    (model_dir / "config.json").unlink()
    shutil.rmtree(removed_dir)
    assert _reconcile(manager, [model_root]) == ["model-b"]
    assert manager.get_settings("model-a").model_alias == "protected"

    shutil.rmtree(model_dir)
    assert _reconcile(manager, [model_root]) == ["model-a"]


def test_unavailable_nfs_root_only_protects_its_own_models(tmp_path):
    """A missing NFS/external root cannot block a healthy root's cleanup."""
    local_root = tmp_path / "local"
    nfs_root = tmp_path / "nfs"
    _make_model(local_root, "local-active")
    local_deleted = _make_model(local_root, "local-deleted")
    _make_model(nfs_root, "nfs-model")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("local-deleted", ModelSettings(model_alias="local-alias"))
    manager.set_settings("nfs-model", ModelSettings(model_alias="nfs-alias"))
    assert _reconcile(manager, [local_root, nfs_root]) == []

    shutil.rmtree(local_deleted)
    shutil.rmtree(nfs_root)
    removed = _reconcile(
        manager,
        [local_root, nfs_root],
        scanned_roots=[local_root],
    )

    assert removed == ["local-deleted"]
    assert "local-deleted" not in manager.get_all_settings()
    assert manager.get_settings("nfs-model").model_alias == "nfs-alias"


def test_same_model_id_found_in_another_root_keeps_state(tmp_path):
    """Moving a model between still-configured roots is not a deletion."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    moved_dir = _make_model(first_root, "model-a")
    _make_model(second_root, "model-b")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("model-a", ModelSettings(model_alias="moved"))
    assert _reconcile(manager, [first_root, second_root]) == []

    shutil.move(str(moved_dir), second_root / "model-a")
    assert _reconcile(manager, [first_root, second_root]) == []
    assert manager.get_settings("model-a").model_alias == "moved"


def test_explicit_model_root_change_retires_absent_state(tmp_path):
    """A trusted replacement root releases IDs absent from the new config."""
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _make_model(old_root, "model-a")
    _make_model(old_root, "model-b")
    _make_model(new_root, "model-a")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("model-a", ModelSettings(model_alias="kept"))
    manager.set_settings("model-b", ModelSettings(model_alias="released"))
    assert _reconcile(manager, [old_root]) == []

    removed = _reconcile(
        manager,
        [new_root],
        retire_unobserved_configured=True,
    )

    assert removed == ["model-b"]
    assert manager.get_settings("model-a").model_alias == "kept"
    assert "model-b" not in manager.get_all_settings()


def test_new_empty_replacement_root_defers_retirement(tmp_path):
    """A newly created mount point is not trusted until it yields models."""
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _make_model(old_root, "old-model")
    new_root.mkdir()

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("old-model", ModelSettings(model_alias="protected"))
    assert _reconcile(manager, [old_root]) == []

    assert (
        _reconcile(
            manager,
            [new_root],
            retire_unobserved_configured=True,
        )
        == []
    )
    assert manager.get_settings("old-model").model_alias == "protected"

    _make_model(new_root, "new-model")
    assert _reconcile(manager, [new_root]) == ["old-model"]


def test_mount_identity_change_requires_a_stable_followup_scan(tmp_path):
    """A changed mount target gets one non-destructive confirmation scan."""
    volume_a = tmp_path / "volume-a"
    volume_b = tmp_path / "volume-b"
    _make_model(volume_a, "old-model")
    _make_model(volume_b, "new-model")
    logical_root = tmp_path / "models"
    logical_root.symlink_to(volume_a, target_is_directory=True)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("old-model", ModelSettings(model_alias="protected"))
    assert _reconcile(manager, [logical_root]) == []

    logical_root.unlink()
    logical_root.symlink_to(volume_b, target_is_directory=True)
    assert _reconcile(manager, [logical_root]) == []
    assert manager.get_settings("old-model").model_alias == "protected"

    assert _reconcile(manager, [logical_root]) == ["old-model"]


def test_empty_replacement_mount_never_confirms_deletion(tmp_path):
    """An empty replacement can be an unmounted disk or network filesystem."""
    volume_a = tmp_path / "volume-a"
    volume_b = tmp_path / "volume-b"
    _make_model(volume_a, "old-model")
    volume_b.mkdir()
    logical_root = tmp_path / "models"
    logical_root.symlink_to(volume_a, target_is_directory=True)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("old-model", ModelSettings(model_alias="protected"))
    assert _reconcile(manager, [logical_root]) == []

    logical_root.unlink()
    logical_root.symlink_to(volume_b, target_is_directory=True)
    for _ in range(2):
        assert _reconcile(manager, [logical_root]) == []
    assert manager.get_settings("old-model").model_alias == "protected"


def test_unavailable_known_hf_cache_does_not_block_local_cleanup(tmp_path):
    """Known HF state is protected independently from a healthy local root."""
    local_root = tmp_path / "local"
    hf_cache = tmp_path / "hf-cache"
    _make_model(local_root, "local-active")
    local_deleted = _make_model(local_root, "local-deleted")
    _make_model(hf_cache, "hf-model")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("local-deleted", ModelSettings(model_alias="local-alias"))
    manager.set_settings("hf-model", ModelSettings(model_alias="hf-alias"))
    assert (
        _reconcile(
            manager,
            [local_root],
            scanned_roots=[local_root, hf_cache],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )

    shutil.rmtree(local_deleted)
    shutil.rmtree(hf_cache)
    removed = _reconcile(
        manager,
        [local_root],
        scanned_roots=[local_root],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=True,
    )

    assert removed == ["local-deleted"]
    assert manager.get_settings("hf-model").model_alias == "hf-alias"


def test_incomplete_hf_snapshot_preserves_state_until_entry_is_deleted(tmp_path):
    """Structural HF cache presence protects state without exposing the model."""
    local_root = tmp_path / "local"
    local_root.mkdir()
    hf_cache = tmp_path / "hf-cache"
    model_id = "mlx-community--demo"
    cache_entry, weights = _make_hf_cache_model(hf_cache, model_id)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(model_id, ModelSettings(model_alias="shared"))
    assert (
        _reconcile(
            manager,
            [local_root],
            scanned_roots=[local_root, hf_cache],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )

    weights.unlink()
    incomplete = _scan(local_root, hf_cache)
    hf_key = hf_cache.resolve()
    assert incomplete.models == {}
    assert incomplete.root_complete[hf_key] is True
    assert incomplete.root_models[hf_key] == frozenset()
    assert incomplete.root_model_entries[hf_key] == frozenset({model_id})
    assert (
        reconcile_discovered_model_state(
            manager,
            incomplete,
            [local_root],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )
    assert manager.get_settings(model_id).model_alias == "shared"

    shutil.rmtree(cache_entry)
    assert _reconcile(
        manager,
        [local_root],
        scanned_roots=[local_root, hf_cache],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=True,
    ) == [model_id]
    assert model_id not in manager.get_all_settings()


def test_empty_hf_snapshot_tree_preserves_state_until_entry_is_deleted(tmp_path):
    """Removing snapshots alone is not equivalent to removing the cache entry."""
    local_root = tmp_path / "local"
    local_root.mkdir()
    hf_cache = tmp_path / "hf-cache"
    model_id = "mlx-community--demo"
    cache_entry, _ = _make_hf_cache_model(hf_cache, model_id)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(model_id, ModelSettings(model_alias="shared"))
    assert (
        _reconcile(
            manager,
            [local_root],
            scanned_roots=[local_root, hf_cache],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )

    shutil.rmtree(cache_entry / "snapshots" / "revision")
    assert (
        _reconcile(
            manager,
            [local_root],
            scanned_roots=[local_root, hf_cache],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )
    assert manager.get_settings(model_id).model_alias == "shared"

    shutil.rmtree(cache_entry)
    assert _reconcile(
        manager,
        [local_root],
        scanned_roots=[local_root, hf_cache],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=True,
    ) == [model_id]


def test_incomplete_hf_snapshot_protects_legacy_state_before_inventory(tmp_path):
    """An upgrade cannot prune a legacy alias while its HF entry still exists."""
    local_root = tmp_path / "local"
    _make_model(local_root, "local-active")
    hf_cache = tmp_path / "hf-cache"
    model_id = "mlx-community--demo"
    _, weights = _make_hf_cache_model(hf_cache, model_id)
    weights.unlink()

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(model_id, ModelSettings(model_alias="legacy"))

    assert (
        _reconcile(
            manager,
            [local_root],
            scanned_roots=[local_root, hf_cache],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )
    assert manager.get_settings(model_id).model_alias == "legacy"


def test_incomplete_untracked_hf_root_defers_legacy_cleanup(tmp_path, monkeypatch):
    """An unreadable first HF scan cannot prove that legacy state is stale."""
    local_root = tmp_path / "local"
    _make_model(local_root, "local-active")
    hf_cache = tmp_path / "hf-cache"
    hf_cache.mkdir()
    original_iterdir = Path.iterdir

    def fake_iterdir(path):
        if path == hf_cache:
            raise PermissionError("Operation not permitted")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    discovery = _scan(local_root, hf_cache)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(
        "mlx-community--unreadable",
        ModelSettings(model_alias="protected"),
    )
    removed = reconcile_discovered_model_state(
        manager,
        discovery,
        [local_root],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=True,
    )

    assert removed == []
    assert manager.get_settings("mlx-community--unreadable").model_alias == "protected"


def test_disabled_hf_cache_is_scanned_for_reconciliation_evidence(tmp_path):
    """Disabling HF visibility does not delete first-inventory model state."""
    local_root = tmp_path / "local"
    _make_model(local_root, "local-active")
    hf_cache = tmp_path / "hf-cache"
    model_id = "mlx-community--demo"
    _make_hf_cache_model(hf_cache, model_id)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(model_id, ModelSettings(model_alias="protected"))
    discovery = _scan(local_root)
    assert model_id not in discovery.models

    removed = reconcile_discovered_model_state(
        manager,
        discovery,
        [local_root],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=False,
    )

    assert removed == []
    assert manager.get_settings(model_id).model_alias == "protected"


def test_disabled_hf_cache_mount_change_converges_without_losing_state(tmp_path):
    """Structural evidence can confirm a changed disabled-cache identity."""
    local_root = tmp_path / "local"
    local_root.mkdir()
    hf_cache = tmp_path / "hf-cache"
    model_id = "mlx-community--demo"
    _make_hf_cache_model(hf_cache, model_id)

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(model_id, ModelSettings(model_alias="protected"))
    assert (
        _reconcile(
            manager,
            [local_root],
            scanned_roots=[local_root, hf_cache],
            hf_cache_dir=hf_cache,
            hf_cache_enabled=True,
        )
        == []
    )

    hf_cache.rename(tmp_path / "old-hf-cache")
    cache_entry, _ = _make_hf_cache_model(hf_cache, model_id)
    discovery = _scan(local_root)

    for _ in range(2):
        assert (
            reconcile_discovered_model_state(
                manager,
                discovery,
                [local_root],
                hf_cache_dir=hf_cache,
                hf_cache_enabled=False,
            )
            == []
        )
    assert manager.get_settings(model_id).model_alias == "protected"

    shutil.rmtree(cache_entry)
    assert reconcile_discovered_model_state(
        manager,
        discovery,
        [local_root],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=False,
    ) == [model_id]


def test_missing_unused_hf_cache_does_not_block_legacy_cleanup(tmp_path):
    """A never-created optional cache does not stall configured-root migration."""
    local_root = tmp_path / "local"
    _make_model(local_root, "local-active")
    hf_cache = tmp_path / "missing-hf-cache"

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("deleted-model", ModelSettings(model_alias="released"))
    discovery = _scan(local_root)

    removed = reconcile_discovered_model_state(
        manager,
        discovery,
        [local_root],
        hf_cache_dir=hf_cache,
        hf_cache_enabled=True,
    )

    assert removed == ["deleted-model"]
    assert "deleted-model" not in manager.get_all_settings()


def test_mount_change_after_scan_cannot_poison_root_fingerprint(tmp_path):
    """Discovery IDs remain paired with the identity that produced them."""
    model_root = tmp_path / "models"
    _make_model(model_root, "model-a")

    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings("model-a", ModelSettings(model_alias="protected"))
    assert _reconcile(manager, [model_root]) == []

    stale_discovery = _scan(model_root)
    detached_root = tmp_path / "detached-models"
    model_root.rename(detached_root)
    model_root.mkdir()

    assert (
        reconcile_discovered_model_state(
            manager,
            stale_discovery,
            [model_root],
        )
        == []
    )
    assert _reconcile(manager, [model_root]) == []
    assert manager.get_settings("model-a").model_alias == "protected"


def test_configured_paths_preserve_logical_mount_identity(tmp_path):
    """Configuration retains a symlink while scanning resolves its target."""
    physical_root = tmp_path / "volume"
    physical_root.mkdir()
    logical_root = tmp_path / "models"
    logical_root.symlink_to(physical_root, target_is_directory=True)
    settings = GlobalModelSettings(model_dirs=[str(logical_root)])

    assert settings.get_configured_model_dirs(tmp_path) == [logical_root.absolute()]
    assert settings.get_model_dirs(tmp_path) == [physical_root.resolve()]
