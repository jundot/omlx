# SPDX-License-Identifier: Apache-2.0
"""Token provenance and opt-in plumbing through the existing cache pipeline."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.cache.inspection import InspectionRenderer
from omlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.settings import CacheSettings, GlobalSettings


def test_prefix_capture_has_exact_offsets_parents_and_cached_tokens(tmp_path):
    paged = PagedCacheManager(
        block_size=2, max_blocks=20, initial_blocks=20, model_name="test"
    )
    ssd = PagedSSDCacheManager(
        tmp_path / "cache", 10**7, inspection_renderer=InspectionRenderer(None, "test")
    )
    model = SimpleNamespace(layers=[None])
    prefix = BlockAwarePrefixCache(model, paged, ssd)
    try:
        tokens = [1, 2, 3, 4, 5]  # The trailing partial block isn't persisted.
        data = [
            {
                "state": (mx.zeros((1, 1, 5, 4)), mx.ones((1, 1, 5, 4))),
                "cache_type": "KVCache",
                "meta_state": (5,),
            }
        ]
        table = prefix.store_cache("first", tokens, data)
        assert table.num_tokens == 4
        ssd.close()
        first_hash = compute_block_hash(None, [1, 2], model_name="test")
        second_hash = compute_block_hash(first_hash, [3, 4], model_name="test")
        first = json.loads(
            ssd._get_file_path(first_hash).with_suffix(".tokens").read_text()
        )
        second = json.loads(
            ssd._get_file_path(second_hash).with_suffix(".tokens").read_text()
        )
        assert first["token_ids"] == [1, 2] and first["parent_hash"] is None
        assert (
            second["token_ids"] == [3, 4] and second["parent_hash"] == first_hash.hex()
        )
        assert second["token_start"] == 2
        assert len(list((tmp_path / "cache").glob("*/*.tokens"))) == 2
    finally:
        ssd.close()


def test_settings_opt_in_round_trip_and_environment(monkeypatch):
    assert not CacheSettings().cache_inspection
    assert not CacheSettings.from_dict({"cache_inspection": "false"}).cache_inspection
    cache = CacheSettings(cache_inspection=True)
    assert CacheSettings.from_dict(cache.to_dict()).cache_inspection
    settings = GlobalSettings()
    monkeypatch.setenv("OMLX_CACHE_INSPECTION", "true")
    settings._apply_env_overrides()
    assert settings.to_scheduler_config().cache_inspection
    monkeypatch.setenv("OMLX_CACHE_INSPECTION", "false")
    settings._apply_env_overrides()
    assert not settings.to_scheduler_config().cache_inspection


def test_cached_prefix_revisit_backfills_without_tensor_rewrite(tmp_path):
    paged = PagedCacheManager(
        block_size=2, max_blocks=20, initial_blocks=20, model_name="test"
    )
    ssd = PagedSSDCacheManager(tmp_path / "cache", 10**7)
    prefix = BlockAwarePrefixCache(SimpleNamespace(layers=[None]), paged, ssd)
    tokens = [1, 2, 3, 4]
    data = [
        {
            "state": (mx.zeros((1, 1, 4, 4)), mx.ones((1, 1, 4, 4))),
            "cache_type": "KVCache",
            "meta_state": (4,),
        }
    ]
    prefix.store_cache("first", tokens, data)
    ssd.close()
    original = {
        path: path.stat().st_mtime_ns
        for path in (tmp_path / "cache").glob("*/*.safetensors")
    }
    ssd = PagedSSDCacheManager(
        tmp_path / "cache", 10**7, inspection_renderer=InspectionRenderer(None, "test")
    )
    prefix.set_paged_ssd_cache_manager(ssd)
    try:
        # The existing request block table takes the all-tokens-cached branch.
        prefix.store_cache("first", tokens, data)
        ssd.close()
        assert len(original) == 2
        assert all(
            path.stat().st_mtime_ns == timestamp for path, timestamp in original.items()
        )
        assert all(path.with_suffix(".tokens").exists() for path in original)
    finally:
        ssd.close()


def test_scheduler_creates_renderer_only_when_enabled(
    tmp_path, mock_model, mock_tokenizer
):
    from omlx.scheduler import Scheduler, SchedulerConfig

    for enabled in (False, True):
        config = SchedulerConfig(
            paged_ssd_cache_dir=str(tmp_path / str(enabled)),
            cache_inspection=enabled,
            model_name="test",
        )
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        try:
            assert scheduler.paged_ssd_cache_manager.inspection_enabled is enabled
        finally:
            scheduler.paged_ssd_cache_manager.close()


@pytest.mark.asyncio
async def test_engine_removes_transport_metadata_from_model_kwargs(
    mock_model, mock_tokenizer
):
    from omlx.engine_core import EngineCore

    with patch("omlx.engine_core.get_registry") as registry:
        registry.return_value.acquire.return_value = True
        engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
        try:
            engine.scheduler.add_request = MagicMock()
            media = ({"kind": "image", "key_start": 0},)
            original = {"_cache_inspection_media": media, "position_ids": "retained"}
            # Preserve the pre-existing positional arguments after the VLM
            # fields when adding optional inspection metadata to this API.
            await engine.add_request(
                [1, 2],
                None,  # sampling_params
                None,  # request_id
                None,  # images
                None,  # videos
                None,  # vlm_inputs_embeds
                original,
                None,  # vlm_image_hash
                0,  # vlm_cache_key_start
                None,  # vlm_cache_key_ranges
                True,  # specprefill
                0.75,  # specprefill_keep_pct
                0.5,  # specprefill_threshold
                1,  # specprefill_system_end
                True,  # skip_cache_store
            )
            request = engine.scheduler.add_request.call_args.args[0]
            assert request.cache_inspection_media == media
            assert request.vlm_extra_kwargs == {"position_ids": "retained"}
            assert "_cache_inspection_media" in original
            assert request._specprefill_enabled is True
            assert request._specprefill_keep_pct == 0.75
            assert request._specprefill_threshold == 0.5
            assert request.specprefill_system_end == 1
            assert request.skip_cache_store is True
        finally:
            engine.close()


def test_admin_settings_exposes_opt_in():
    from omlx.admin.routes import GlobalSettingsRequest

    assert GlobalSettingsRequest().cache_inspection is None
    assert GlobalSettingsRequest(cache_inspection=True).cache_inspection is True
