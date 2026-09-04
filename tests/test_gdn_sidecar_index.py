# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the durable GDN sidecar file/index layer."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from omlx.cache.paged_ssd_cache import (
    _READABLE_CACHE_FORMAT_VERSIONS,
    PagedSSDBlockMetadata,
    PagedSSDCacheManager,
    cache_signature_for,
)


def _make_manager(cache_dir: Path, *, max_size: int = 1024 * 1024, **kwargs):
    return PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=max_size,
        **kwargs,
    )


def test_commit_uses_opaque_atomic_sidecar_path_and_api(tmp_path):
    cache_dir = tmp_path / "cache"
    staged = tmp_path / "staged.safetensors"
    payload = b"opaque-safetensors-bytes\x00\x01"
    staged.write_bytes(payload)
    source_hash = b"source-block"
    signature = "signature-v1"

    manager = _make_manager(cache_dir)
    try:
        final_path = manager.commit_gdn_checkpoint_file(
            source_hash,
            staged,
            token_count=2048,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        )

        digest = hashlib.sha256(signature.encode()).hexdigest()
        expected = (
            cache_dir
            / "_gdn_sidecars"
            / digest
            / f"{source_hash.hex()}.safetensors"
        )
        assert final_path == expected
        assert final_path.exists()
        assert not staged.exists()
        assert final_path.read_bytes() == payload
        assert manager.has_gdn_checkpoint(source_hash, signature)
        assert manager.get_gdn_checkpoint_file(source_hash, signature) == expected
        assert manager._gdn_sidecar_index.total_size == len(payload)
        assert not manager._hot_cache

        assert manager.forget_gdn_checkpoint(source_hash, signature)
        assert not expected.exists()
        assert not manager.has_gdn_checkpoint(source_hash, signature)
        assert manager._gdn_sidecar_index.total_size == 0
    finally:
        manager.close()


def test_failed_replacement_preserves_existing_sidecar(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    source_hash = b"source-block"
    signature = "signature-v1"
    manager = _make_manager(cache_dir, max_size=25)
    try:
        first_stage = tmp_path / "first.stage"
        first_stage.write_bytes(b"o" * 10)
        final_path = manager.commit_gdn_checkpoint_file(
            source_hash,
            first_stage,
            token_count=2048,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        )
        assert final_path is not None

        # Make replacement pressure evict one LRU entry. The destination is
        # oldest, so an unprotected replacement would unlink it before the
        # promotion attempt and have nothing to restore when os.replace fails.
        other_stage = tmp_path / "other.stage"
        other_stage.write_bytes(b"d" * 11)
        assert manager.commit_gdn_checkpoint_file(
            b"other-block",
            other_stage,
            token_count=2048,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        ) is not None

        replacement = tmp_path / "replacement.stage"
        replacement.write_bytes(b"n" * 15)

        def fail_replace(_source, _destination):
            raise OSError("simulated promotion failure")

        monkeypatch.setattr(os, "replace", fail_replace)
        assert (
            manager.commit_gdn_checkpoint_file(
                source_hash,
                replacement,
                token_count=4096,
                model_name="model",
                cache_signature=signature,
                block_size=2048,
            )
            is None
        )

        assert final_path.read_bytes() == b"o" * 10
        assert replacement.exists()
        assert manager.has_gdn_checkpoint(source_hash, signature)
        assert manager._gdn_sidecar_index.total_size == 10
    finally:
        manager.close()


def test_commit_refreshes_mtime_for_restart_lru(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    source_hash = b"source-block"
    signature = "signature-v1"
    staged = tmp_path / "old.stage"
    staged.write_bytes(b"checkpoint")
    committed_at = 1_700_000_000.0
    os.utime(staged, (committed_at - 86_400, committed_at - 86_400))

    manager = _make_manager(cache_dir)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(time, "time", lambda: committed_at)
            final_path = manager.commit_gdn_checkpoint_file(
                source_hash,
                staged,
                token_count=2048,
                model_name="model",
                cache_signature=signature,
                block_size=2048,
            )
        assert final_path is not None
        assert final_path.stat().st_mtime == pytest.approx(committed_at)
    finally:
        manager.close()

    restarted = _make_manager(cache_dir)
    try:
        digest = hashlib.sha256(signature.encode()).hexdigest()
        metadata = restarted._gdn_sidecar_index.get(source_hash, digest)
        assert metadata is not None
        assert metadata.last_access == pytest.approx(committed_at)
    finally:
        restarted.close()


def test_commit_rejects_symlink_source_and_destination(tmp_path):
    cache_dir = tmp_path / "cache"
    manager = _make_manager(cache_dir)
    signature = "signature-v1"
    try:
        real_stage = tmp_path / "real.stage"
        real_stage.write_bytes(b"checkpoint")
        linked_stage = tmp_path / "linked.stage"
        linked_stage.symlink_to(real_stage)
        assert (
            manager.commit_gdn_checkpoint_file(
                b"source",
                linked_stage,
                token_count=2048,
                model_name="model",
                cache_signature=signature,
                block_size=2048,
            )
            is None
        )
        assert real_stage.read_bytes() == b"checkpoint"

        sidecar_root = cache_dir / "_gdn_sidecars"
        if sidecar_root.exists():
            sidecar_root.rmdir()
        external = tmp_path / "external"
        external.mkdir()
        sidecar_root.symlink_to(external, target_is_directory=True)
        direct_stage = tmp_path / "direct.stage"
        direct_stage.write_bytes(b"checkpoint")
        assert (
            manager.commit_gdn_checkpoint_file(
                b"source",
                direct_stage,
                token_count=2048,
                model_name="model",
                cache_signature=signature,
                block_size=2048,
            )
            is None
        )
        assert direct_stage.exists()
        assert not list(external.rglob("*.safetensors"))
    finally:
        manager.close()


def test_startup_scan_ignores_symlinked_sidecars(tmp_path):
    cache_dir = tmp_path / "cache"
    signature = "signature-v1"
    digest = hashlib.sha256(signature.encode()).hexdigest()
    signature_dir = cache_dir / "_gdn_sidecars" / digest
    signature_dir.mkdir(parents=True)
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"not-a-cache-sidecar")
    linked = signature_dir / f"{b'source'.hex()}.safetensors"
    linked.symlink_to(external)

    manager = _make_manager(cache_dir)
    try:
        assert manager._gdn_sidecar_index.count == 0
        assert not manager.has_gdn_checkpoint(b"source", signature)
        assert external.exists()
    finally:
        manager.close()


def test_lookup_rejects_sidecar_root_replaced_by_symlink(tmp_path):
    cache_dir = tmp_path / "cache"
    signature = "signature-v1"
    source_hash = b"source"
    manager = _make_manager(cache_dir)
    try:
        staged = tmp_path / "staged"
        staged.write_bytes(b"valid")
        final_path = manager.commit_gdn_checkpoint_file(
            source_hash,
            staged,
            token_count=2048,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        )
        assert final_path is not None

        sidecar_root = cache_dir / "_gdn_sidecars"
        saved_root = tmp_path / "saved-sidecars"
        sidecar_root.rename(saved_root)
        external_root = tmp_path / "external-sidecars"
        external_file = external_root / final_path.parent.name / final_path.name
        external_file.parent.mkdir(parents=True)
        external_file.write_bytes(b"external")
        sidecar_root.symlink_to(external_root, target_is_directory=True)

        assert manager.get_gdn_checkpoint_file(source_hash, signature) is None
        assert external_file.read_bytes() == b"external"
    finally:
        manager.close()


@pytest.mark.parametrize("operation", ["forget", "lru", "clear"])
def test_sidecar_deletion_rejects_swapped_root_symlink(tmp_path, operation):
    cache_dir = tmp_path / "cache"
    signature = "signature-v1"
    source_hash = b"source"
    payload = b"owned-checkpoint"
    manager = _make_manager(cache_dir)
    try:
        staged = tmp_path / "staged"
        staged.write_bytes(payload)
        final_path = manager.commit_gdn_checkpoint_file(
            source_hash,
            staged,
            token_count=2048,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        )
        assert final_path is not None

        sidecar_root = cache_dir / "_gdn_sidecars"
        saved_root = tmp_path / "saved-sidecars"
        sidecar_root.rename(saved_root)
        original_file = saved_root / final_path.parent.name / final_path.name
        external_root = tmp_path / "external-sidecars"
        external_file = external_root / final_path.parent.name / final_path.name
        external_file.parent.mkdir(parents=True)
        external_file.write_bytes(b"external-file")
        sidecar_root.symlink_to(external_root, target_is_directory=True)

        if operation == "forget":
            assert not manager.forget_gdn_checkpoint(source_hash, signature)
        elif operation == "lru":
            manager._max_size = 0
            manager.enforce_size_limit()
        else:
            assert manager.clear() == 0

        assert external_file.read_bytes() == b"external-file"
        assert original_file.read_bytes() == payload
        # Unsafe deletion is a failed deletion. Keep conservative accounting
        # until the original cache namespace is restored or restarted.
        assert manager._gdn_sidecar_index.count == 1
        assert manager._gdn_sidecar_index.total_size == len(payload)
    finally:
        manager.close()


def test_sidecars_are_indexed_from_stat_and_lru_survives_restart(tmp_path):
    cache_dir = tmp_path / "cache"
    signature = "signature-v1"
    source_a = b"a"
    source_b = b"b"
    manager = _make_manager(cache_dir)
    try:
        paths = []
        for source_hash, content in ((source_a, b"a" * 7), (source_b, b"b" * 11)):
            staged = tmp_path / f"{source_hash.decode()}.stage"
            staged.write_bytes(content)
            paths.append(
                manager.commit_gdn_checkpoint_file(
                    source_hash,
                    staged,
                    token_count=2048,
                    model_name="model",
                    cache_signature=signature,
                    block_size=2048,
                )
            )

        old = time.time() - 10
        new = time.time() - 5
        os.utime(paths[0], (old, old))
        os.utime(paths[1], (new, new))
    finally:
        manager.close()

    restarted = _make_manager(cache_dir)
    try:
        assert restarted._gdn_sidecar_index.count == 2
        assert restarted._gdn_sidecar_index.total_size == 18
        oldest = restarted._gdn_sidecar_index.get_lru_entries(1)[0]
        assert oldest.source_block_hash == source_a

        assert restarted.get_gdn_checkpoint_file(source_a, signature) == paths[0]
        newest_first = restarted._gdn_sidecar_index.get_lru_entries(1)[0]
        assert newest_first.source_block_hash == source_b
    finally:
        restarted.close()


def test_sidecar_and_main_block_share_global_lru_budget(tmp_path):
    cache_dir = tmp_path / "cache"
    manager = _make_manager(cache_dir, max_size=12)
    signature = "signature-v1"
    first_source = b"first"
    second_source = b"second"
    main_hash = b"main-block"
    try:
        first_stage = tmp_path / "first.stage"
        first_stage.write_bytes(b"1" * 6)
        first_path = manager.commit_gdn_checkpoint_file(
            first_source,
            first_stage,
            token_count=2048,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        )
        assert first_path is not None

        main_path = manager._get_file_path(main_hash)
        main_path.write_bytes(b"m" * 6)
        now = time.time()
        manager._index.add(
            PagedSSDBlockMetadata(
                block_hash=main_hash,
                file_path=main_path,
                file_size=6,
                token_count=2048,
                created_at=now,
                last_access=now + 1,
                num_layers=1,
            )
        )

        second_stage = tmp_path / "second.stage"
        second_stage.write_bytes(b"2" * 6)
        second_path = manager.commit_gdn_checkpoint_file(
            second_source,
            second_stage,
            token_count=4096,
            model_name="model",
            cache_signature=signature,
            block_size=2048,
        )

        assert second_path is not None
        assert not first_path.exists()
        assert main_path.exists()
        assert manager.has_gdn_checkpoint(second_source, signature)
        assert manager._tracked_ssd_size() == 12
    finally:
        manager.close()


def test_hot_cache_only_rejects_all_sidecar_operations(tmp_path):
    manager = _make_manager(tmp_path / "cache", hot_cache_only=True)
    staged = tmp_path / "staged.safetensors"
    staged.write_bytes(b"opaque")
    try:
        assert (
            manager.commit_gdn_checkpoint_file(
                b"source",
                staged,
                token_count=1,
                model_name="model",
                cache_signature="signature",
                block_size=1,
            )
            is None
        )
        assert manager.get_gdn_checkpoint_file(b"source", "signature") is None
        assert not manager.has_gdn_checkpoint(b"source", "signature")
        assert not manager.forget_gdn_checkpoint(b"source", "signature")
        assert staged.exists()
    finally:
        manager.close()


def test_public_and_manager_signatures_stamp_expected_layout_settings(tmp_path):
    layer_types = ["KVCache", "ArraysCache"]
    stateless = cache_signature_for(
        model_name="model",
        num_layers=2,
        block_size=2048,
        layer_cache_types=layer_types,
        turboquant_kv_bits=6,
        cachelist_subtypes={"1": ["ArraysCache:1"]},
    )
    stateless_payload = json.loads(stateless)
    assert stateless_payload["turboquant_kv_bits"] == 6.0
    assert "payload_layout" not in stateless_payload

    manager = _make_manager(
        tmp_path / "cache",
        expected_layer_cache_types=layer_types,
        gdn_ssd_split_enabled=True,
    )
    try:
        manager.set_expected_layer_signature(
            layer_types,
            turboquant_kv_bits=6,
            cachelist_subtypes={"1": ["ArraysCache:1"]},
        )
        split_signature = manager.cache_signature_for(
            model_name="model",
            num_layers=2,
            block_size=2048,
            layer_cache_types=layer_types,
        )
        split_payload = json.loads(split_signature)
        assert split_payload["payload_layout"] == "split_recurrent_v1"
        assert split_payload["turboquant_kv_bits"] == 6.0
        assert split_payload["cachelist_subtypes"] == {"1": ["ArraysCache:1"]}

        embedded = _make_manager(
            tmp_path / "embedded-cache",
            expected_layer_cache_types=layer_types,
        )
        try:
            embedded_signature = embedded.cache_signature_for(
                model_name="model",
                num_layers=2,
                block_size=2048,
                layer_cache_types=layer_types,
            )
        finally:
            embedded.close()
        assert json.loads(embedded_signature)["payload_layout"] == "embedded"
        assert embedded_signature != split_signature

        embedded_probe = _make_manager(
            tmp_path / "embedded-probe",
            expected_model_name="model",
            expected_num_layers=2,
            expected_block_size=2048,
            expected_layer_cache_types=layer_types,
        )
        try:
            assert not embedded_probe._is_compatible_block(
                PagedSSDBlockMetadata(
                    block_hash=b"probe",
                    file_path=tmp_path / "probe.safetensors",
                    file_size=1,
                    token_count=1,
                    created_at=1.0,
                    last_access=1.0,
                    num_layers=2,
                    model_name="model",
                    block_size=2048,
                    cache_signature=split_signature,
                    layer_cache_types=layer_types,
                )
            )
        finally:
            embedded_probe.close()
    finally:
        manager.close()


def test_split_save_writes_format_five_and_payload_layout_metadata(tmp_path):
    import mlx.core as mx

    manager = _make_manager(
        tmp_path / "cache",
        expected_model_name="model",
        expected_layer_cache_types=["KVCache"],
        gdn_ssd_split_enabled=True,
    )
    block_hash = b"block-hash"
    try:
        assert manager.save_block(
            block_hash,
            [(mx.zeros((1, 1)), mx.zeros((1, 1)))],
            token_count=1,
            model_name="model",
            layer_cache_types=["KVCache"],
            layer_meta_states=[(0,)],
        )
        file_path = manager._get_file_path(block_hash)
        deadline = time.time() + 5
        while not file_path.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert file_path.exists()
        _, metadata = mx.load(str(file_path), return_metadata=True)
        assert metadata["omlx_cache_format_version"] == "5"
        assert metadata["payload_layout"] == "split_recurrent_v1"
        signature_payload = json.loads(metadata["cache_signature"])
        assert signature_payload["payload_layout"] == "split_recurrent_v1"
        assert {"2", "3", "4", "5"}.issubset(_READABLE_CACHE_FORMAT_VERSIONS)
        assert manager.load_block(block_hash) is not None
    finally:
        manager.close()


def test_sidecar_signature_canonicalizes_wrapper_class_names(tmp_path):
    """Warm-restored requests extract SizedArraysCache; cold stores and block
    metadata say ArraysCache. Both spellings must address the same sidecar
    directory or commits from resumed requests become unrestorable."""
    cold_types = ["ArraysCache", "ArraysCache", "ArraysCache", "KVCache"]
    resumed_types = [
        "SizedArraysCache",
        "SizedArraysCache",
        "SizedArraysCache",
        "KVCache",
    ]

    manager = _make_manager(tmp_path / "cache", gdn_ssd_split_enabled=True)
    try:
        cold_signature = manager.cache_signature_for(
            model_name="model",
            num_layers=4,
            block_size=2048,
            layer_cache_types=cold_types,
        )
        resumed_signature = manager.cache_signature_for(
            model_name="model",
            num_layers=4,
            block_size=2048,
            layer_cache_types=resumed_types,
        )
        assert cold_signature == resumed_signature

        staged = tmp_path / "staged.safetensors"
        staged.write_bytes(b"checkpoint")
        assert (
            manager.commit_gdn_checkpoint_file(
                b"source",
                staged,
                token_count=2048,
                model_name="model",
                cache_signature=resumed_signature,
                block_size=2048,
            )
            is not None
        )
        assert manager.has_gdn_checkpoint(b"source", cold_signature)
    finally:
        manager.close()
