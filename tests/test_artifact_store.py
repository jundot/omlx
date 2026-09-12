# SPDX-License-Identifier: Apache-2.0
"""Tests for paged-SSD cache artifact export/import (omlx/cache/artifact_store.py)."""

import hashlib
import json
import time
import zipfile
from pathlib import Path

import pytest

from omlx.cache import artifact_store
from omlx.cache.artifact_store import ArtifactError, export_artifact, import_artifact
from omlx.cache.paged_cache import compute_block_hash
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

MODEL = "artifact-test-model"
OTHER_MODEL = "artifact-other-model"
BLOCK_SIZE = 4
LAYERS = 2
TOKENS = list(range(1, 13))  # 12 tokens -> 3 full blocks


def _wait_for_file(path: Path, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def _make_manager(root: Path, *, model_name: str = MODEL) -> PagedSSDCacheManager:
    return PagedSSDCacheManager(
        cache_dir=root / "ssd",
        max_size_bytes=1 << 30,
        expected_model_name=model_name,
        expected_block_size=BLOCK_SIZE,
    )


def _save_chain(manager: PagedSSDCacheManager, tokens: list[int]) -> list[bytes]:
    """Save one block per chain segment and wait for the background writer."""
    parent = b""
    saved: list[bytes] = []
    cache_data = [
        (mx.zeros((1, 2, BLOCK_SIZE, 8)), mx.zeros((1, 2, BLOCK_SIZE, 8)))
        for _ in range(LAYERS)
    ]
    for start in range(0, len(tokens), BLOCK_SIZE):
        chunk = list(tokens[start : start + BLOCK_SIZE])
        parent = compute_block_hash(
            parent, chunk, extra_keys=None, model_name=MODEL
        )
        assert (
            manager.save_block(
                block_hash=parent,
                cache_data=cache_data,
                token_count=len(chunk),
                model_name=MODEL,
                layer_cache_types=["KVCache"] * LAYERS,
            )
            is True
        )
        assert _wait_for_file(manager._get_file_path(parent))
        saved.append(parent)
    return saved


def test_export_import_round_trip(tmp_path: Path):
    """Export after prefill, import into an empty cache -> full restore."""
    manager_a = _make_manager(tmp_path / "a")
    manager_b = _make_manager(tmp_path / "b")
    try:
        saved = _save_chain(manager_a, TOKENS)

        result = export_artifact(
            manager_a,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        artifact = Path(result["artifact_path"])
        assert artifact.is_file()
        manifest = result["manifest"]
        assert manifest["model_name"] == MODEL
        assert manifest["block_size"] == BLOCK_SIZE
        assert manifest["restorable_tokens"] == len(TOKENS)
        assert len(manifest["blocks"]) == 3
        assert manifest["token_ids"] == TOKENS
        assert len(manifest["gdn_sidecars"]) == 0

        # Every manifest block file is a byte-identical copy of the cache file.
        with zipfile.ZipFile(artifact) as archive:
            for entry in manifest["blocks"]:
                assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == (
                    entry["sha256"]
                )
                source = manager_a._get_file_path(bytes.fromhex(entry["hash"]))
                assert archive.read(entry["path"]) == source.read_bytes()

        outcome = import_artifact(
            manager_b, artifact_path=artifact, model_name=MODEL
        )
        assert outcome["restorable_tokens"] == len(TOKENS)
        assert outcome["imported_blocks"] == 3
        assert outcome["existing_blocks"] == 0
        assert outcome["imported_gdn_sidecars"] == 0

        for block_hash in saved:
            assert manager_b.has_block(block_hash)
            loaded = manager_b.load_block(block_hash)
            assert loaded is not None and len(loaded) == LAYERS
    finally:
        manager_a.close()
        manager_b.close()


def test_export_stops_at_first_hole(tmp_path: Path):
    """A missing middle block truncates the export to the intact prefix."""
    manager = _make_manager(tmp_path / "a")
    try:
        saved = _save_chain(manager, TOKENS)
        # Drop the middle block file; the index entry may remain.
        manager._get_file_path(saved[1]).unlink()

        result = export_artifact(
            manager,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        manifest = result["manifest"]
        assert manifest["restorable_tokens"] == BLOCK_SIZE
        assert len(manifest["blocks"]) == 1
        assert manifest["blocks"][0]["hash"] == saved[0].hex()

        # Importing the truncated artifact restores only that prefix.
        manager_b = _make_manager(tmp_path / "b")
        try:
            outcome = import_artifact(
                manager_b,
                artifact_path=Path(result["artifact_path"]),
                model_name=MODEL,
            )
            assert outcome["restorable_tokens"] == BLOCK_SIZE
            assert manager_b.has_block(saved[0])
            assert not manager_b.has_block(saved[1])
            assert not manager_b.has_block(saved[2])
        finally:
            manager_b.close()
    finally:
        manager.close()


def test_import_is_idempotent(tmp_path: Path):
    """Importing twice counts the second pass as existing, not re-imported."""
    manager_a = _make_manager(tmp_path / "a")
    manager_b = _make_manager(tmp_path / "b")
    try:
        _save_chain(manager_a, TOKENS)
        result = export_artifact(
            manager_a,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        artifact = Path(result["artifact_path"])

        first = import_artifact(manager_b, artifact_path=artifact, model_name=MODEL)
        assert first["imported_blocks"] == 3
        second = import_artifact(manager_b, artifact_path=artifact, model_name=MODEL)
        assert second["imported_blocks"] == 0
        assert second["existing_blocks"] == 3
        assert second["restorable_tokens"] == len(TOKENS)
    finally:
        manager_a.close()
        manager_b.close()


def _rewrite_zip(artifact: Path, out_path: Path, mangle: str | None = None):
    """Repackage an artifact, optionally corrupting one member's bytes."""
    with zipfile.ZipFile(artifact) as src, zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_STORED
    ) as dst:
        for name in src.namelist():
            data = src.read(name)
            if mangle == name:
                data = data[:-4] + b"\x00\xff\x00\x01"
            dst.writestr(name, data)


def test_import_rejects_tampered_payload(tmp_path: Path):
    """A single flipped byte in a block file fails the import; nothing lands."""
    manager_a = _make_manager(tmp_path / "a")
    manager_b = _make_manager(tmp_path / "b")
    try:
        _save_chain(manager_a, TOKENS)
        result = export_artifact(
            manager_a,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        artifact = Path(result["artifact_path"])
        block_member = result["manifest"]["blocks"][0]["path"]

        tampered = tmp_path / "tampered.omlxzip"
        _rewrite_zip(artifact, tampered, mangle=block_member)

        with pytest.raises(ArtifactError, match="sha256"):
            import_artifact(manager_b, artifact_path=tampered, model_name=MODEL)
        # Fail closed: nothing was registered or written.
        assert manager_b._index._total_size == 0
        assert list((tmp_path / "b" / "ssd").rglob("*.safetensors")) == []
    finally:
        manager_a.close()
        manager_b.close()


def test_import_rejects_wrong_model(tmp_path: Path):
    manager_a = _make_manager(tmp_path / "a")
    manager_b = _make_manager(tmp_path / "b", model_name=OTHER_MODEL)
    try:
        _save_chain(manager_a, TOKENS)
        result = export_artifact(
            manager_a,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        with pytest.raises(ArtifactError, match="for model"):
            import_artifact(
                manager_b,
                artifact_path=Path(result["artifact_path"]),
                model_name=OTHER_MODEL,
            )
        assert manager_b._index._total_size == 0
    finally:
        manager_a.close()
        manager_b.close()


def test_import_rejects_inconsistent_chain(tmp_path: Path):
    """A manifest whose block hashes do not match its token ids is rejected."""
    manager_a = _make_manager(tmp_path / "a")
    manager_b = _make_manager(tmp_path / "b")
    try:
        _save_chain(manager_a, TOKENS)
        result = export_artifact(
            manager_a,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        artifact = Path(result["artifact_path"])

        # Rewrite the manifest with a shifted token chain (same length).
        with zipfile.ZipFile(artifact) as src:
            names = src.namelist()
            members = {name: src.read(name) for name in names}
        manifest = json.loads(members[artifact_store.MANIFEST_NAME])
        manifest["token_ids"] = [t + 1000 for t in TOKENS]
        members[artifact_store.MANIFEST_NAME] = json.dumps(manifest).encode()

        shifted = tmp_path / "shifted.omlxzip"
        with zipfile.ZipFile(shifted, "w", compression=zipfile.ZIP_STORED) as dst:
            for name in names:
                dst.writestr(name, members[name])

        with pytest.raises(ArtifactError, match="token chain"):
            import_artifact(manager_b, artifact_path=shifted, model_name=MODEL)
    finally:
        manager_a.close()
        manager_b.close()


def test_import_rejects_corrupt_zip(tmp_path: Path):
    manager = _make_manager(tmp_path / "a")
    bad = tmp_path / "bad.omlxzip"
    bad.write_bytes(b"this is not a zip at all")
    with pytest.raises(ArtifactError, match="not a valid artifact"):
        import_artifact(manager, artifact_path=bad, model_name=MODEL)
    manager.close()


def test_gdn_sidecar_round_trip(tmp_path: Path):
    """GDN sidecars travel with the artifact and re-register on import."""
    manager_a = _make_manager(tmp_path / "a")
    manager_b = _make_manager(tmp_path / "b")
    try:
        saved = _save_chain(manager_a, TOKENS)
        # Sidecars are keyed by the live cache signature, the same one the
        # blocks carry in their metadata (that pairing is what import
        # validates).
        signature = manager_a.get_block_metadata(saved[0]).cache_signature or ""
        staged = manager_a._cache_dir / "staged_sidecar.safetensors"
        staged.write_bytes(b"fake-gdn-checkpoint-bytes")
        committed = manager_a.commit_gdn_checkpoint_file(
            saved[0],
            staged,
            token_count=BLOCK_SIZE,
            model_name=MODEL,
            cache_signature=signature,
            block_size=BLOCK_SIZE,
        )
        assert committed is not None

        result = export_artifact(
            manager_a,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        sidecars = result["manifest"]["gdn_sidecars"]
        assert len(sidecars) == 1
        assert sidecars[0]["source_hash"] == saved[0].hex()

        outcome = import_artifact(
            manager_b, artifact_path=Path(result["artifact_path"]), model_name=MODEL
        )
        assert outcome["imported_gdn_sidecars"] == 1
        assert manager_b.has_gdn_checkpoint(saved[0], signature)
    finally:
        manager_a.close()
        manager_b.close()


def test_export_empty_cache_yields_zero_artifact(tmp_path: Path):
    """Nothing cached -> zero-block manifest, caller treats it as a miss."""
    manager = _make_manager(tmp_path / "a")
    try:
        result = export_artifact(
            manager,
            model_name=MODEL,
            token_ids=TOKENS,
            block_size=BLOCK_SIZE,
            output_dir=tmp_path / "artifacts",
        )
        assert result["manifest"]["restorable_tokens"] == 0
        assert result["manifest"]["blocks"] == []
    finally:
        manager.close()


def test_hot_cache_only_property(tmp_path: Path):
    """The admin layer uses this flag to refuse artifact operations."""
    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd",
        max_size_bytes=1 << 30,
        hot_cache_only=True,
    )
    assert manager.hot_cache_only is True
    manager.close()

    manager_default = _make_manager(tmp_path / "b")
    assert manager_default.hot_cache_only is False
    manager_default.close()
