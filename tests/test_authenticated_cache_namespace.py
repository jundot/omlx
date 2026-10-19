# SPDX-License-Identifier: Apache-2.0
"""Authentication-to-cache ownership, including cold and exact prefix reuse."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import mlx.core as mx
import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from omlx.cache_namespace import (
    current_cache_namespace,
    namespace_for_api_key,
    namespace_keys,
)
from omlx.cache.paged_cache import PagedCacheManager, resolve_block_extra_keys
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.settings import AuthSettings, GlobalSettings, SubKeyEntry


@pytest.fixture(autouse=True)
def namespace_context():
    token = current_cache_namespace.set(None)
    yield
    current_cache_namespace.reset(token)


def request(scope, **kwargs):
    return Request(
        "request", [1, 2, 3], SamplingParams(), cache_namespace=scope, **kwargs
    )


def test_namespace_survives_context_change_and_worker_handoff():
    current_cache_namespace.set(namespace_for_api_key("credential-a"))
    owned = Request(
        "a", [1], SamplingParams(), cache_namespace=current_cache_namespace.get()
    )
    current_cache_namespace.set(namespace_for_api_key("credential-b"))
    with ThreadPoolExecutor(max_workers=1) as worker:
        keys = worker.submit(lambda: owned.vlm_extra_keys_for_cache).result()
    assert keys == namespace_keys(namespace_for_api_key("credential-a"))
    assert "credential-a" not in repr(keys)


@pytest.mark.parametrize("segmented", [False, True])
def test_media_key_never_leaves_pre_image_text_shared(segmented):
    a = request(
        "a",
        vlm_image_hash="image",
        vlm_cache_key_start=8,
        vlm_cache_key_ranges=[(8, "image"), (16, "image-two")] if segmented else None,
    )
    b = request(
        "b",
        vlm_image_hash="image",
        vlm_cache_key_start=8,
        vlm_cache_key_ranges=[(8, "image"), (16, "image-two")] if segmented else None,
    )
    for end in (4, 8, 12, 16, 20):

        def keys(r):
            return resolve_block_extra_keys(
                end,
                r.vlm_extra_keys_for_cache,
                r.vlm_extra_key_token_start_for_cache,
                r.vlm_extra_key_ranges_for_cache,
            )

        assert keys(a) != keys(b)
        assert keys(a)[0] == namespace_keys("a")[0]
    assert keys(a)[-1] == ("image-two" if segmented else "image")


def test_shared_media_keys_unchanged():
    r = request(None, vlm_image_hash="image", vlm_cache_key_start=8)
    assert r.vlm_extra_keys_for_cache == ("image",)
    assert r.vlm_extra_key_token_start_for_cache == 8
    assert r.vlm_extra_key_ranges_for_cache is None


class Model:
    layers = [object()]


def prefix(tmp_path):
    paged = PagedCacheManager(
        block_size=4, max_blocks=64, initial_blocks=64, model_name="namespace-model"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path,
        max_size_bytes=64 * 1024**2,
        expected_model_name="namespace-model",
        expected_num_layers=1,
        expected_block_size=4,
        expected_layer_cache_types=["KVCache"],
    )
    paged.set_paged_ssd_cache_manager(ssd)
    return BlockAwarePrefixCache(Model(), paged, ssd), ssd


def state(n, value):
    return [
        {
            "state": (mx.full((1, 1, n, 8), value), mx.full((1, 1, n, 8), value + 1)),
            "class_name": "KVCache",
            "cache_type": "KVCache",
            "meta_state": (str(n),),
        }
    ]


@pytest.mark.parametrize("exact", [False, True])
def test_hot_and_cold_prefix_reuse_is_credential_owned(tmp_path, exact):
    tokens = list(range(9 if exact else 8))
    cache, ssd = prefix(tmp_path)
    try:
        for scope, value in (("a", 10.0), ("b", 20.0)):
            if exact:
                table = cache.store_exact_prefix(
                    scope,
                    tokens,
                    state(len(tokens), value),
                    extra_keys=namespace_keys(scope),
                )
            else:
                table = cache.store_cache(
                    scope,
                    tokens,
                    state(len(tokens), value),
                    extra_keys=namespace_keys(scope),
                )
                cache.release_cache(scope)
            assert table is not None
        assert ssd.wait_for_pending_writes(timeout=5)
        for cold in (False, True):
            if cold:
                ssd.close()
                cache, ssd = prefix(tmp_path)
            for scope, value in (("a", 10.0), ("b", 20.0)):
                if exact:
                    restored = cache.restore_exact_prefix(
                        "read",
                        tokens,
                        promote_to_hot_cache=True,
                        extra_keys=namespace_keys(scope),
                    )
                else:
                    table, remaining = cache.fetch_cache(
                        "read", tokens, extra_keys=namespace_keys(scope)
                    )
                    assert remaining == []
                    restored = cache.reconstruct_cache(table)
                    cache.release_cache("read")
                assert restored is not None
                assert float(restored[0].state[0][0, 0, 0, 0]) == value
            if exact:
                assert (
                    cache.fetch_exact_prefix(
                        "foreign", tokens, extra_keys=namespace_keys("c")
                    )
                    is None
                )
            else:
                assert (
                    cache.fetch_cache(
                        "foreign", tokens, extra_keys=namespace_keys("c")
                    )[0]
                    is None
                )
    finally:
        ssd.close()


def test_settings_round_trip_and_reject_auth_bypasses():
    assert AuthSettings.from_dict({}).cache_isolation == "shared"
    assert (
        AuthSettings.from_dict(
            AuthSettings(cache_isolation="api_key").to_dict()
        ).cache_isolation
        == "api_key"
    )
    settings = GlobalSettings()
    settings.auth.api_key = "main"
    settings.auth.cache_isolation = "api_key"
    assert not [e for e in settings.validate() if "isolation" in e]
    for option in ("skip_api_key_verification", "allow_unauthenticated_inference"):
        setattr(settings.auth, option, True)
        assert any("isolation" in e for e in settings.validate())
        setattr(settings.auth, option, False)


def test_concurrent_auth_and_streaming_capture_verified_identity(monkeypatch):
    from omlx import server

    settings = GlobalSettings()
    settings.auth.api_key = "main"
    settings.auth.sub_keys = [SubKeyEntry(key="sub", name="sub")]
    settings.auth.cache_isolation = "api_key"
    monkeypatch.setattr(server._server_state, "global_settings", settings)
    monkeypatch.setattr(server._server_state, "api_key", "main")
    monkeypatch.setattr(server._server_state, "bind_host", "127.0.0.1")
    monkeypatch.setattr(server._server_state, "distributed_inference_enabled", False)
    app = FastAPI()
    from omlx.engine_core import EngineCore
    from unittest.mock import MagicMock

    owned_requests = {}
    core = EngineCore.__new__(EngineCore)
    core.config = SimpleNamespace(stream_interval=1)
    core.scheduler = MagicMock()
    core.scheduler._specprefill_draft_model = None
    core.scheduler.add_request.side_effect = lambda owned: owned_requests.update(
        {owned.request_id: owned}
    )
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core._wake_engine_loop = lambda: None
    worker = ThreadPoolExecutor(max_workers=1)
    core._mlx_executor = worker

    @app.get("/capture")
    async def capture(_=Depends(server.verify_inference_api_key)):
        async def stream():
            await asyncio.sleep(0)
            request_id = await core.add_request([1])
            yield owned_requests[request_id].cache_namespace

        return StreamingResponse(stream())

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            a, b, rejected = await asyncio.gather(
                client.get(
                    "/capture",
                    headers={"Authorization": "Bearer main", "X-Tenant-ID": "sub"},
                ),
                client.get(
                    "/capture", headers={"x-api-key": "sub", "X-Tenant-ID": "main"}
                ),
                client.get("/capture", headers={"Authorization": "Bearer invalid"}),
            )
        assert a.text == namespace_for_api_key("main")
        assert b.text == namespace_for_api_key("sub")
        assert rejected.status_code == 401

    try:
        asyncio.run(exercise())
    finally:
        worker.shutdown()


def test_unsupported_engine_fails_closed():
    from fastapi import HTTPException
    from omlx.server import _require_cache_namespace_support

    current_cache_namespace.set("a")
    with pytest.raises(HTTPException) as error:
        _require_cache_namespace_support(SimpleNamespace())
    assert error.value.status_code == 400
    _require_cache_namespace_support(SimpleNamespace(supports_cache_namespace=True))


def test_mtp_sidecars_use_the_same_credential_owned_backbone_tip(tmp_path):
    cache, ssd = prefix(tmp_path)
    tokens = list(range(8))
    snapshots = {"a": object(), "b": object()}
    try:
        for scope in snapshots:
            keys = namespace_keys(scope)
            assert cache.store_cache(scope, tokens, state(8, 10), extra_keys=keys)
            assert cache.store_mtp_prefix_snapshot(
                tokens, 8, snapshots[scope], extra_keys=keys
            )
        for scope in snapshots:
            assert (
                cache.restore_mtp_prefix_snapshot(
                    tokens, 8, extra_keys=namespace_keys(scope)
                )
                is snapshots[scope]
            )
        assert (
            cache.restore_mtp_prefix_snapshot(tokens, 8, extra_keys=namespace_keys("c"))
            is None
        )
    finally:
        ssd.close()


def test_specprefill_draft_passes_verified_scope_on_read_and_write():
    from test_specprefill_draft import _DraftCache, _request_and_plan, _run

    seen = []

    class ScopedDraftCache(_DraftCache):
        def fetch_cache(self, *args, extra_keys=None):
            seen.append(("read", extra_keys))
            return super().fetch_cache(*args)

        def store_cache(self, *args, extra_keys=None, **kwargs):
            seen.append(("write", extra_keys))
            return super().store_cache(*args, **kwargs)

    owned, plan = _request_and_plan()
    owned.cache_namespace = "a"
    _run(
        owned,
        plan,
        draft_cache=ScopedDraftCache(),
        extract_cache_states=lambda cache: ([{"state": "value"}], None),
    )
    assert seen == [("read", namespace_keys("a")), ("write", namespace_keys("a"))]


def test_specprefill_static_prefix_passes_scope_on_restore_and_store(monkeypatch):
    import test_specprefill_target as helper

    seen = []
    original_namespace = helper.SimpleNamespace

    def owned_namespace(**kwargs):
        if "request_id" in kwargs:
            kwargs["cache_namespace"] = "a"
        return original_namespace(**kwargs)

    class ScopedExactCache(helper._TieredExactPrefixCache):
        def restore_exact_prefix(self, *args, extra_keys=None, **kwargs):
            seen.append(("read", extra_keys))
            return super().restore_exact_prefix(*args, **kwargs)

        def store_exact_prefix(self, *args, extra_keys=None, **kwargs):
            seen.append(("write", extra_keys))
            return super().store_exact_prefix(*args, **kwargs)

    monkeypatch.setattr(helper, "SimpleNamespace", owned_namespace)
    helper._run(
        system_token_count=8,
        conversation_token_count=8,
        selected_indices=[0, 2, 6],
        exact_prefix_cache=ScopedExactCache(),
        static_prefix_tokens=list(range(8)),
        extract_cache_states=helper._extract_cache_states,
    )
    assert seen == [("read", namespace_keys("a")), ("write", namespace_keys("a"))]


def test_exact_split_gdn_sidecars_are_credential_owned(tmp_path):
    from test_prefix_cache_gdn_split import _HybridModel, _hybrid_extracted, LAYER_TYPES
    from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
    from omlx.scheduler import _BoundarySnapshotProvider

    paged = PagedCacheManager(
        block_size=4, max_blocks=64, initial_blocks=64, model_name="hybrid-model"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path,
        max_size_bytes=64 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=4,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path, pending_max_bytes=1024**2)
    cache = BlockAwarePrefixCache(
        _HybridModel(), paged, ssd, gdn_ssd_split_enabled=True
    )
    cache.set_gdn_checkpoint_loader(boundary.load_file)
    cache.set_exact_gdn_checkpoint_writer(
        lambda **kwargs: _BoundarySnapshotProvider.stage_and_commit(
            store=boundary, paged_ssd_manager=ssd, **kwargs
        )
    )
    tokens = list(range(5))
    try:
        for scope, value in (("a", 10.0), ("b", 20.0)):
            assert cache.store_exact_prefix(
                scope,
                tokens,
                _hybrid_extracted(5, value),
                extra_keys=namespace_keys(scope),
            )
        for scope, value in (("a", 10.0), ("b", 20.0)):
            restored = cache.restore_exact_prefix(
                "read",
                tokens,
                promote_to_hot_cache=False,
                extra_keys=namespace_keys(scope),
            )
            assert float(restored[1].cache[0][0, 0, 0]) == value
        assert (
            cache.restore_exact_prefix(
                "foreign",
                tokens,
                promote_to_hot_cache=False,
                extra_keys=namespace_keys("c"),
            )
            is None
        )
    finally:
        boundary.shutdown()
        ssd.close()


@pytest.mark.parametrize(
    "option",
    ["skip_api_key_verification", "allow_unauthenticated_inference", "distributed"],
)
def test_isolation_rejects_runtime_auth_bypasses(monkeypatch, option):
    from fastapi import HTTPException
    from starlette.requests import Request as HttpRequest
    from fastapi.security import HTTPAuthorizationCredentials
    from omlx import server

    settings = GlobalSettings()
    settings.auth.api_key = "main"
    settings.auth.cache_isolation = "api_key"
    if option != "distributed":
        setattr(settings.auth, option, True)
    monkeypatch.setattr(server._server_state, "global_settings", settings)
    monkeypatch.setattr(server._server_state, "api_key", "main")
    monkeypatch.setattr(server._server_state, "bind_host", "127.0.0.1")
    monkeypatch.setattr(
        server._server_state, "distributed_inference_enabled", option == "distributed"
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            server.verify_inference_api_key(
                HttpRequest({"type": "http", "headers": []}),
                HTTPAuthorizationCredentials(scheme="Bearer", credentials="main"),
            )
        )
    assert error.value.status_code == 503
