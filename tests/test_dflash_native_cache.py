from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_vlm.models.cache import CacheList
from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.engine.spec_epoch import (
    SpeculativeSession,
    _SessionRequest,
    _RequestState,
    _YieldPauseTracker,
)
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.patches.deepseek_v4.cache_extras import PoolingCache
from omlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler
from omlx.patches.dflash_glm5 import _Glm5RecurrentRollbackCache
from omlx.cache import dflash as bridge


class Ops:
    backend_name = "test_glm_layout"

    def capabilities_for(self, model):
        return TargetCapabilities(True, True, True, False, False, False, True)

    def make_cache(self, model, **kwargs):
        recurrent = _Glm5RecurrentRollbackCache(2, conv_kernel_size=3)
        return [recurrent, CacheList(KVCache(), PoolingCache(4))]

    def forward_with_hidden_capture(
        self, model, *, input_ids, cache, capture_layer_ids, logits_last_only
    ):
        recurrent, composite = cache
        kv, pool = composite.caches
        n = kv.offset + input_ids.shape[1]
        previous = recurrent.cache[1]
        total = input_ids.astype(mx.float32).sum().reshape(1, 1, 1)
        recurrent.cache = [
            mx.full((1, 2, 1), n),
            total if previous is None else previous + total,
        ]
        keys = input_ids.astype(mx.float32).reshape(1, 1, -1, 1)
        kv.update_and_fetch(keys, keys + 10)
        pool.state = (
            mx.ones((1, n % 4, 1)) if n % 4 else None,
            mx.ones((1, n % 4, 1)) if n % 4 else None,
            mx.ones((1, n // 4, 1)),
            None,
            None,
        )
        hidden = input_ids.astype(mx.float32)[:, :, None]
        logits = mx.broadcast_to(recurrent.cache[1], (1, 1, 8)) + mx.arange(8)
        return logits, {1: hidden}

    def extract_context_feature(self, captured, layer_ids):
        return captured[1]

    def embed_tokens(self, model):
        # The test model has no embeddings; draft tokens only flow through
        # forward_with_hidden_capture, so a constant table is enough.
        return lambda ids: mx.zeros((ids.shape[0], ids.shape[1], 4))

    def logits_from_hidden(self, model, hidden):
        return mx.zeros((1, hidden.shape[1], 8)) + mx.arange(8)

    def arm_rollback(self, cache_entries, *, prefix_len):
        for entry in cache_entries:
            if hasattr(entry, "arm_rollback"):
                entry.arm_rollback(prefix_len=prefix_len)

    def verify_block(self, *, target_model, verify_ids, target_cache, capture_layer_ids):
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
            logits_last_only=False,
        )

    def restore_after_acceptance(
        self, cache_entries, *, target_len, acceptance_length, drafted_tokens=0
    ):
        for entry in cache_entries:
            if hasattr(entry, "rollback"):
                entry.rollback(max(0, int(acceptance_length)))
        return 0

    def cleanup_generation_caches(self, target, draft):
        for entry in target:
            clear = getattr(entry, "clear", None)
            if callable(clear):
                clear()
        for entry in draft:
            clear = getattr(entry, "clear", None)
            if callable(clear):
                clear()


def _fake_draft_hidden(noise_embedding, draft_context):
    """Drafter stand-in: seq_len = 1 staged token + (block_len - 1) noise rows."""
    import mlx.core as mx

    if isinstance(draft_context, bridge.TargetHiddenChunks):
        context = draft_context.slice(
            max(0, draft_context.total_len - 4), draft_context.total_len
        )
    else:
        context = draft_context
    rows = max(1, int(noise_embedding.shape[1]))
    return mx.concatenate(
        [context[:, -1:, :], mx.zeros((1, rows - 1, context.shape[-1]))], axis=1
    )


@contextmanager
def install_fixtures(tmp_path, monkeypatch):
    import mlx_lm.models.cache as lm_cache

    monkeypatch.setattr(lm_cache, "PoolingCache", PoolingCache, raising=False)
    CacheTypeRegistry.register(PoolingCacheHandler())
    model = SimpleNamespace(layers=[None, None])
    ops = Ops()
    cache = bridge.DFlashNativeCache(
        model=model,
        target_ops=ops,
        model_name="proof",
        cache_dir=tmp_path,
        config=SimpleNamespace(paged_cache_block_size=2048),
        max_size_bytes=2**26,
    )
    draft = SimpleNamespace(
        target_layer_ids=[0],
        args=SimpleNamespace(sliding_window=4, layer_types=["sliding_attention"]),
        project_target_hidden=lambda x: x * 2,
        block_size=2,
        mask_token_id=0,
        layers=[None],
        embed_scale=1.0,
        norm=lambda x: x,
        forward_projected_context=lambda *, noise_embedding, draft_context, cache=None: (
            _fake_draft_hidden(noise_embedding, draft_context)
        ),
    )
    draft_backend = SimpleNamespace(make_cache=lambda **kwargs: [])
    tokenizer = SimpleNamespace(chat_template="")
    provider = SimpleNamespace(
        model_key=("proof", None, "draft"),
        model=model,
        target_ops=ops,
        tokenizer=tokenizer,
        cli_args=None,
    )
    context = build_runtime_context(
        runtime_config_from_defaults(
            prefix_cache=False,
            prefix_cache_l2=False,
            draft_sink_size=2,
            draft_window_size=4,
            clear_cache_boundaries=False,
            copyspec_mode="off",
        )
    )
    bridge.install_native_cache_hooks()
    try:
        yield cache, draft, draft_backend, provider, context
    finally:
        cache.close()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    with install_fixtures(tmp_path, monkeypatch) as fixtures:
        yield fixtures


def prefill(setup, tokens):
    cache, draft, backend, provider, context = setup
    flow = cache.for_request(
        model_provider=provider,
        draft_model=draft,
        tokenizer=provider.tokenizer,
        prompt=tokens,
        max_new_tokens=3,
        runtime_context=context,
    )
    ops = bridge.NativeCacheTargetOps(provider.target_ops)
    session = SpeculativeSession.open(
        target_model=provider.model,
        draft_model=draft,
        draft_backend=backend,
        target_ops=ops,
        supports_prefix_snapshot=True,
        allow_full_context_draft_layers=False,
        prompt_tokens=tokens,
        max_new_tokens=3,
        prefix_snapshot=flow.snapshot,
        quantize_kv_cache=False,
        target_fa_window=0,
        runtime_context=context,
    )
    request = _SessionRequest.from_tokens(
        prompt_tokens=tokens,
        max_new_tokens=3,
        block_tokens=2,
        stop_token_ids=[],
        suppress_token_ids=None,
        prefix_snapshot=flow.snapshot,
        snapshot_service=flow.snapshot_service,
        stable_prefix_len=len(tokens),
        prefix_cache_active=True,
        temperature=0,
        top_p=1,
        top_k=0,
        min_p=0,
    )
    state = _RequestState()
    it = session._run_prefill_events(
        request=request, state=state, yield_pause=_YieldPauseTracker(False)
    )
    events = []
    while True:
        try:
            events.append(next(it))
        except StopIteration as done:
            result = done.value
            break
    return flow, session, state, result, events


@pytest.mark.parametrize("n", [17, 4101])
def test_repeat_and_restart(setup, n):
    tokens = [i % 7 + 1 for i in range(n)]
    cold, session, state, result, _ = prefill(setup, tokens)
    assert cold.hit_tokens == 0
    expected = state.prefill_logits.tolist()
    warm, session2, state2, result2, _ = prefill(setup, tokens)
    assert warm.hit_tokens == n
    assert state2.prefill_logits.tolist() == expected
    assert session2.target_cache[1].caches[0].offset == n
    cache, draft, backend, provider, context = setup
    cache.ssd.close()
    other = bridge.DFlashNativeCache(
        model=provider.model,
        target_ops=provider.target_ops,
        model_name="proof",
        cache_dir=cache.ssd._cache_dir,
        config=SimpleNamespace(paged_cache_block_size=2048),
        max_size_bytes=2**26,
    )
    try:
        flow = other.for_request(
            model_provider=provider,
            draft_model=draft,
            tokenizer=provider.tokenizer,
            prompt=tokens,
            max_new_tokens=3,
            runtime_context=context,
        )
        assert flow.hit_tokens == n
    finally:
        other.close()


@pytest.mark.parametrize("sink", [0, 2])
def test_native_entry_without_features_replays_window(setup, sink, monkeypatch):
    cache, draft, backend, provider, context = setup
    from dataclasses import replace

    context = replace(context, runtime=replace(context.runtime, draft_sink_size=sink))
    setup = (cache, draft, backend, provider, context)
    tokens = [i % 7 + 1 for i in range(4101)]
    service = bridge.NativeSnapshotService(cache, draft, "unused", context.runtime)
    live = provider.target_ops.make_cache(provider.model)
    for end in (2048, 4096):
        start = 0 if end == 2048 else 2048
        provider.target_ops.forward_with_hidden_capture(
            provider.model,
            input_ids=mx.array(tokens[start:end])[None],
            cache=live,
            capture_layer_ids={1},
            logits_last_only=True,
        )
        assert service.store_target(tokens[:end], live) is not None
    reconstructions = []
    reconstruct = cache.prefix.reconstruct_cache

    def record_reconstruction(table, **kwargs):
        reconstructions.append(table.num_tokens)
        return reconstruct(table, **kwargs)

    monkeypatch.setattr(cache.prefix, "reconstruct_cache", record_reconstruction)
    flow, session, state, result, _ = prefill(setup, tokens)
    assert reconstructions == [4096]
    assert flow.hit_tokens == 4096 - sink
    hidden = result.feature_store.current_hidden
    expected = mx.array(tokens, dtype=mx.float32)[None, :, None] * 2
    if sink:
        assert mx.array_equal(hidden.slice(0, sink), expected[:, :sink]).item()
    assert mx.array_equal(hidden.slice(len(tokens) - 4, len(tokens)), expected[:, -4:]).item()
    assert session.target_cache[1].caches[0].offset == len(tokens)
    assert state.prefill_logits[0, 0, 0].item() == sum(tokens)


def test_cache_write_failure_keeps_generation_working(setup, monkeypatch):
    cache, *_ = setup

    def fail(request_id, *args, **kwargs):
        cache.paged.create_block_table(request_id)
        raise OSError("unavailable cache drive")

    monkeypatch.setattr(cache.prefix, "store_cache", fail)
    flow, session, state, _, _ = prefill(setup, [1, 2, 3, 4, 5])
    assert flow.hit_tokens == 0
    assert state.prefill_logits[0, 0, 0].item() == 15
    assert not cache.prefix._request_tables
    assert not cache.paged.request_tables


@pytest.mark.parametrize("restored", [False, True])
def test_prefill_extends_one_block_table_without_repeated_lookup(setup, monkeypatch, restored):
    cache, *_ = setup
    tokens = [i % 7 + 1 for i in range(3 * cache.block_size + 5)]
    prefix_len = cache.block_size + 5 if restored else 0
    if restored:
        prefill(setup, tokens[:prefix_len])
    lookups = []
    stores = []
    fetch = cache.prefix.fetch_cache
    store = cache.prefix.store_cache

    def record_fetch(request_id, token_ids, **kwargs):
        lookups.append(request_id)
        return fetch(request_id, token_ids, **kwargs)

    def record_store(request_id, token_ids, *args, **kwargs):
        table = cache.paged.get_block_table(request_id)
        stores.append((request_id, table, table.num_tokens if table else 0))
        return store(request_id, token_ids, *args, **kwargs)

    monkeypatch.setattr(cache.prefix, "fetch_cache", record_fetch)
    monkeypatch.setattr(cache.prefix, "store_cache", record_store)
    flow, _, state, _, _ = prefill(setup, tokens)
    request_ids = {request_id for request_id, _, _ in stores}
    assert len(request_ids) == 1
    assert sum(request_id in request_ids for request_id in lookups) == 1
    assert [n for _, _, n in stores] == (
        [prefix_len, 2 * cache.block_size, 3 * cache.block_size]
        if restored else [0, cache.block_size, 2 * cache.block_size, 3 * cache.block_size]
    )
    assert all(table is stores[1][1] for _, table, _ in stores[1:])
    assert state.prefill_logits[0, 0, 0].item() == sum(tokens)
    assert flow.snapshot_service._prefill_request_id is None
    assert not cache.prefix._request_tables
    assert not cache.paged.request_tables
    assert prefill(setup, tokens)[0].hit_tokens == len(tokens)


def test_changed_drafter_reuses_target_and_replays_features(setup):
    cache, draft, backend, provider, context = setup
    tokens = [i % 7 + 1 for i in range(4101)]
    _, _, cold, _, _ = prefill(setup, tokens)
    provider.draft_cache_identity = (8, 16, 64)
    flow, _, warm, result, _ = prefill(setup, tokens)
    assert flow.hit_tokens == 4094
    assert warm.prefill_logits.tolist() == cold.prefill_logits.tolist()
    expected = mx.array(tokens[-4:], dtype=mx.float32)[None, :, None] * 2
    assert mx.array_equal(result.feature_store.current_hidden.slice(len(tokens) - 4, len(tokens)), expected).item()


@pytest.mark.parametrize("hot_only", [False, True])
def test_native_tiers_share_budget_and_clear(setup, tmp_path, hot_only):
    from omlx.cache.paged_ssd_cache import SharedHotCacheBudget

    cache, draft, backend, provider, context = setup
    budget = SharedHotCacheBudget(2**20)
    other = bridge.DFlashNativeCache(
        model=provider.model,
        target_ops=provider.target_ops,
        model_name="proof",
        cache_dir=None if hot_only else tmp_path / "tiers",
        config=SimpleNamespace(hot_cache_budget=budget),
        hot_cache_only=hot_only,
        hot_cache_max_bytes=2**20,
        max_size_bytes=2**26,
    )
    scenario = (other, draft, backend, provider, context)
    tokens = [1, 2, 3, 4, 5, 6, 7]
    try:
        prefill(scenario, tokens)
        assert prefill(scenario, tokens)[0].hit_tokens == len(tokens)
        assert 0 < budget.total_bytes <= budget.max_bytes
        report = other.clear(hot=True)
        assert report["hot_cleared"] > 0
        if hot_only:
            assert prefill(scenario, tokens)[0].hit_tokens == 0
        else:
            # Cold replay still works after releasing all hot buffers.
            other.ssd.close()
            other.ssd = bridge.PagedSSDCacheManager(**other._ssd_kwargs)
            other.ssd.set_expected_layer_signature(
                other.layer_types,
                cachelist_subtypes=bridge.cachelist_subtypes_from_cache_list(
                    other.templates
                ),
                numerics=bridge.numerics_revision_for_model(other.model),
            )
            other.paged.set_paged_ssd_cache_manager(other.ssd)
            other.prefix.paged_ssd_cache = other.ssd
            assert prefill(scenario, tokens)[0].hit_tokens == len(tokens)
        other.clear(ssd=True)
        assert other.ssd.get_stats_dict()["num_files"] == 0
        assert budget.total_bytes == 0
        assert prefill(scenario, tokens)[0].hit_tokens == 0
    finally:
        other.close()
    assert budget.total_bytes == 0


def test_native_snapshot_binds_to_baseline_cache(setup):
    from mlx_lm.models.cache import ArraysCache
    from omlx.cache.state import restore_cache

    cache, *_ = setup
    tokens = list(range(17))
    prefill(setup, tokens)
    n, _, native = cache._lookup(tokens)
    baseline = restore_cache(
        native, [ArraysCache(2), CacheList(KVCache(), PoolingCache(4))]
    )
    assert n == len(tokens)
    assert type(baseline[0]) is ArraysCache
    assert baseline[0].cache[1].item() == sum(tokens)
    assert baseline[1].caches[0].offset == n
    assert baseline[1].caches[1].state[2].shape[1] == n // 4


def test_full_context_drafter_requires_features(setup, monkeypatch):
    from dataclasses import replace

    cache, draft, backend, provider, context = setup
    tokens = [1] * 4101
    prefill(setup, tokens)
    # A different full-context drafter cannot reconstruct historical features
    # from target KV/recurrent state alone.
    monkeypatch.setattr(
        provider.target_ops,
        "capabilities_for",
        lambda model: replace(
            Ops().capabilities_for(model), supports_full_context_draft_layers=True
        ),
    )
    provider.draft_cache_identity = "full-context"
    draft.args.layer_types = ["full_attention"]
    context = replace(
        context, runtime=replace(context.runtime, draft_full_context_min_ctx=0)
    )
    flow = cache.for_request(
        model_provider=provider,
        draft_model=draft,
        tokenizer=provider.tokenizer,
        prompt=tokens,
        max_new_tokens=3,
        runtime_context=context,
    )
    assert flow.hit_tokens == 0 and flow.snapshot is None


def test_cancel_prefill_clears_boundary_capture(setup):
    cache, draft, backend, provider, context = setup
    tokens = [1] * 4101
    flow = cache.for_request(
        model_provider=provider,
        draft_model=draft,
        tokenizer=provider.tokenizer,
        prompt=tokens,
        max_new_tokens=3,
        runtime_context=context,
    )
    ops = bridge.NativeCacheTargetOps(provider.target_ops)
    session = SpeculativeSession.open(
        target_model=provider.model,
        draft_model=draft,
        draft_backend=backend,
        target_ops=ops,
        supports_prefix_snapshot=True,
        allow_full_context_draft_layers=False,
        prompt_tokens=tokens,
        max_new_tokens=3,
        prefix_snapshot=flow.snapshot,
        quantize_kv_cache=False,
        target_fa_window=0,
        runtime_context=context,
    )
    request = _SessionRequest.from_tokens(
        prompt_tokens=tokens,
        max_new_tokens=3,
        block_tokens=2,
        stop_token_ids=[],
        suppress_token_ids=None,
        prefix_snapshot=flow.snapshot,
        snapshot_service=flow.snapshot_service,
        stable_prefix_len=len(tokens),
        prefix_cache_active=True,
        temperature=0,
        top_p=1,
        top_k=0,
        min_p=0,
    )
    iterator = session._run_prefill_events(
        request=request, state=_RequestState(), yield_pause=_YieldPauseTracker(False)
    )
    next(iterator)
    assert cache.paged.get_block_table(flow.snapshot_service._prefill_request_id) is not None
    iterator.close()
    assert ops.prefill_service is None and ops.prompt_tokens == ()
    assert not cache.prefix._request_tables
    assert not cache.paged.request_tables
    assert flow.snapshot_service._prefill_request_id is None


def test_split_recurrent_blocks_restore_through_native_handlers(setup, tmp_path):
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

    class HybridOps(Ops):
        def make_cache(self, model, **kwargs):
            return [RecurrentRollbackCache(2), KVCache()]

        def forward_with_hidden_capture(self, model, *, input_ids, cache, **kwargs):
            rec, kv = cache
            previous = rec.cache[1]
            total = input_ids.astype(mx.float32).sum().reshape(1, 1, 1)
            rec.cache = [
                mx.ones((1, 2, 1)),
                total if previous is None else previous + total,
            ]
            keys = input_ids.astype(mx.float32).reshape(1, 1, -1, 1)
            kv.update_and_fetch(keys, keys)
            return mx.broadcast_to(rec.cache[1], (1, 1, 8)), {
                1: input_ids.astype(mx.float32)[:, :, None]
            }

    _, draft, backend, provider, context = setup
    ops = HybridOps()
    provider = SimpleNamespace(**{**vars(provider), "target_ops": ops})
    cache = bridge.DFlashNativeCache(
        model=provider.model,
        target_ops=ops,
        model_name="proof",
        cache_dir=tmp_path / "split",
        config=SimpleNamespace(gdn_ssd_split_enabled=True),
        max_size_bytes=2**26,
    )
    scenario = (cache, draft, backend, provider, context)
    tokens = [1] * 4101
    try:
        _, _, cold, _, _ = prefill(scenario, tokens)
        warm, session, state, _, _ = prefill(scenario, tokens)
        assert warm.hit_tokens == len(tokens)
        assert state.prefill_logits.tolist() == cold.prefill_logits.tolist()
        assert session.target_cache[1].offset == len(tokens)
        assert cache.ssd.get_stats_dict()["gdn_sidecar_count"] > 1
    finally:
        cache.close()


@pytest.mark.parametrize("extra", [2, 20])
def test_generation_publication_preserves_valid_checkpoint(setup, extra):
    cache, draft, _, provider, context = setup
    tokens = [i % 7 + 1 for i in range(4090)]
    flow, session, _, _, _ = prefill(setup, tokens)
    extended = tokens + [1] * extra
    logits, _ = provider.target_ops.forward_with_hidden_capture(
        provider.model,
        input_ids=mx.array(extended[len(tokens) :])[None],
        cache=session.target_cache,
        capture_layer_ids={1},
        logits_last_only=True,
    )
    publication = flow.snapshot_service.publish(
        token_ids=extended,
        target_cache=session.target_cache,
        target_hidden=mx.array(extended, dtype=mx.float32)[None, :, None] * 2,
        last_logits=logits[:, -1, :],
        kind="generation",
        snapshot_boundary=len(extended),
        allow_full_attention_context=False,
    )
    assert publication.admitted == (extra == 2)
    assert prefill(setup, tokens)[0].hit_tokens == len(tokens)
    warm, _, state, _, _ = prefill(setup, extended)
    assert warm.hit_tokens == (len(extended) if extra == 2 else len(tokens))
    assert state.prefill_logits[0, 0, 0].item() == sum(extended)


@pytest.mark.asyncio
async def test_engine_uses_global_cache_identity_and_budgets(
    setup, tmp_path, monkeypatch
):
    from omlx.engine import dflash as engine_module
    from omlx.engine.dflash import DFlashEngine
    from omlx.scheduler import SchedulerConfig
    import omlx.scheduler as scheduler_module
    from omlx.cache.paged_ssd_cache import SharedHotCacheBudget
    from dflash_mlx.runtime import loading
    import dflash_mlx.engine.target_ops as ops_module

    _, draft, _, provider, _ = setup
    monkeypatch.setattr(
        loading,
        "load_target_bundle",
        lambda *a, **kw: SimpleNamespace(
            model=provider.model,
            tokenizer=provider.tokenizer,
            target_ops=provider.target_ops,
            meta={"config": {"model_type": "test"}},
        ),
    )
    monkeypatch.setattr(
        loading,
        "load_draft_bundle",
        lambda *a, **kw: (draft, {"config": {"sliding_window": 4}}),
    )
    monkeypatch.setattr(ops_module, "bind_draft_to_target", lambda *a, **kw: None)
    monkeypatch.setattr(
        engine_module, "maybe_apply_pre_load_patches", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        engine_module, "load_generation_config_token_ids", lambda *a, **kw: set()
    )
    monkeypatch.setattr(engine_module, "detect_output_parser", lambda *a, **kw: None)
    monkeypatch.setattr(
        engine_module, "set_model_info_from_model", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        scheduler_module, "_detect_qwen35_prefill_floor", lambda model: 4096
    )
    budget = SharedHotCacheBudget(2**20)
    cfg = SchedulerConfig(
        model_name="registered-id",
        paged_ssd_cache_dir=str(tmp_path / "global"),
        hot_cache_max_size=0,
        hot_cache_budget=budget,
        paged_ssd_cache_max_size=2**26,
    )
    engine = DFlashEngine(
        "/model/path",
        "/draft/path",
        scheduler_config=cfg,
        model_settings=SimpleNamespace(
            dflash_in_memory_cache=False, dflash_ssd_cache=False
        ),
    )
    await engine.start()
    try:
        assert engine._native_cache.model_name == "registered-id"
        assert engine._native_cache.ssd._hot_cache_budget is budget
        assert engine._native_cache.ssd._cache_dir == tmp_path / "global"
        assert engine.prefix_cache_enabled
        assert engine._runtime_context.runtime.prefill_step_size == 4096
        assert engine._prefill_guard._prefill_step_size == 4096
        assert not engine._runtime_context.runtime.prefix_cache
        assert not engine._runtime_context.runtime.prefix_cache_l2
        assert await engine.clear_prompt_caches(ssd=True) == {
            "hot_cleared": 0,
            "ssd_deleted": 0,
            "ranks": [],
        }
    finally:
        await engine.stop()
    assert engine._native_cache is None and budget.total_bytes == 0


def test_full_runtime_repeats_prompt_after_generation(setup, monkeypatch):
    from omlx.engine.dflash import DFlashEngine
    from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent, TokenEvent

    cache, draft, backend, provider, context = setup
    ops = provider.target_ops
    monkeypatch.setattr(ops, "arm_rollback", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(
        ops, "restore_after_acceptance", lambda *a, **kw: 0, raising=False
    )

    def verify(*, target_model, verify_ids, target_cache, capture_layer_ids):
        return ops.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
            logits_last_only=False,
        )

    monkeypatch.setattr(ops, "verify_block", verify, raising=False)
    monkeypatch.setattr(ops, "text_model", lambda model: model, raising=False)
    draft.mask_token_id = 0
    draft.block_size = 1
    engine = DFlashEngine("proof", "draft")
    engine._target_model = provider.model
    engine._target_ops = ops
    engine._draft_model = draft
    engine._draft_backend = backend
    engine._executor_tokenizer = provider.tokenizer
    engine._runtime_context = context
    engine._native_cache = cache
    engine._block_size = 1
    tokens = list(range(17))
    for expected in (0, len(tokens)):
        iterator, flow, _ = engine._stream_dflash_events(tokens, max_tokens=1)
        events = list(iterator)
        assert any(isinstance(e, SummaryEvent) for e in events)
        assert any(isinstance(e, TokenEvent) for e in events)
        prefill = next(e for e in events if isinstance(e, PrefillCompleteEvent))
        assert prefill.prefill_tokens_restored == expected
        assert flow.hit_tokens == expected


@pytest.mark.parametrize("cache_hit", ["miss", "exact", "target_only"])
def test_skip_cache_store_preserves_lookup_without_publication(setup, monkeypatch, cache_hit):
    from omlx.engine.dflash import DFlashEngine
    from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent

    cache, draft, backend, provider, context = setup
    tokens = [i % 7 + 1 for i in range(4101)]
    engine = DFlashEngine("proof", "draft")
    provider.draft_cache_identity = (
        engine._draft_quant_enabled, engine._draft_quant_weight_bits,
        engine._draft_quant_activation_bits, engine._draft_quant_group_size,
    )
    if cache_hit != "miss":
        prefill(setup, tokens)
    if cache_hit == "target_only":
        monkeypatch.setattr(cache.ssd, "load_prefix_context", lambda *a: None)

    def fail(*args, **kwargs):
        pytest.fail("skip_cache_store must bypass cache publication and preparation")

    monkeypatch.setattr(bridge.NativeSnapshotService, "store_target", fail)
    monkeypatch.setattr(cache.prefix, "store_cache", fail)
    monkeypatch.setattr(cache.ssd, "save_prefix_context", fail)
    monkeypatch.setattr(cache, "prune_context_tips", fail)
    if cache.boundary_store is not None:
        monkeypatch.setattr(cache.boundary_store, "save", fail)
    before = cache.ssd.get_stats_dict()["prefix_context_count"]
    engine._target_model = provider.model
    engine._target_ops = provider.target_ops
    engine._draft_model = draft
    engine._draft_backend = backend
    engine._executor_tokenizer = provider.tokenizer
    engine._runtime_context = context
    engine._native_cache = cache
    engine._block_size = 1
    iterator, flow, _ = engine._stream_dflash_events(
        tokens, max_tokens=1, skip_cache_store=True
    )
    events = list(iterator)
    complete = next(e for e in events if isinstance(e, PrefillCompleteEvent))
    summary = next(e for e in events if isinstance(e, SummaryEvent))
    assert complete.prefill_tokens_restored == {
        "miss": 0, "exact": len(tokens), "target_only": 4094,
    }[cache_hit]
    assert summary.generated_token_ids == (7,)
    assert flow.hit_tokens == complete.prefill_tokens_restored
    assert not flow.snapshot_service.active
    assert flow.insert_ms == 0
    assert cache.ssd.get_stats_dict()["prefix_context_count"] == before
    assert not cache.prefix._request_tables
    assert not cache.paged.request_tables


@pytest.mark.parametrize("as_list", [False, True])
def test_boundary_capture_preserves_backend_hidden_layout(setup, monkeypatch, as_list):
    _, _, _, provider, _ = setup
    forward = provider.target_ops.forward_with_hidden_capture

    def capture(model, **kwargs):
        logits, hidden = forward(model, **kwargs)
        last = mx.full((1, 1, 1), kwargs["cache"][1].caches[0].offset)
        return logits, [last, hidden[1]] if as_list else {1: hidden[1], -1: last}

    monkeypatch.setattr(provider.target_ops, "forward_with_hidden_capture", capture)
    ops = bridge.NativeCacheTargetOps(provider.target_ops)
    boundaries = []
    ops.prefill_service = SimpleNamespace(
        cache=SimpleNamespace(block_size=4),
        store_target=lambda tokens, caches: boundaries.append(len(tokens)),
    )
    ops.prompt_tokens = tuple(range(9))
    cache = provider.target_ops.make_cache(provider.model)
    _, hidden = ops.forward_with_hidden_capture(
        provider.model,
        input_ids=mx.arange(9)[None],
        cache=cache,
        capture_layer_ids={1},
        logits_last_only=True,
    )
    assert boundaries == [4, 8]
    assert mx.array_equal(
        hidden[1], mx.arange(9, dtype=mx.float32)[None, :, None]
    ).item()
    if as_list:
        assert isinstance(hidden, list) and hidden[0].shape[1] == 3
    else:
        assert (
            isinstance(hidden, dict)
            and hidden[-1].shape[1] == 1
            and hidden[-1].item() == 9
        )


@pytest.mark.parametrize("broken", ["rank", "coverage"])
def test_invalid_context_is_a_safe_cache_miss(setup, monkeypatch, broken):
    cache, draft, backend, provider, context = setup
    tokens = list(range(17))
    _, _, cold, _, _ = prefill(setup, tokens)
    value = mx.ones((2,)) if broken == "rank" else mx.ones((1, 2, 1))
    monkeypatch.setattr(
        cache.ssd,
        "load_prefix_context",
        lambda *a: (
            {"hidden_0": value, "logits": mx.ones((1, 8))},
            {"spans": "[[0,2]]", "prefix_len": "17"},
        ),
    )
    flow, _, warm, _, _ = prefill(setup, tokens)
    assert flow.hit_tokens == 0
    assert warm.prefill_logits.tolist() == cold.prefill_logits.tolist()


@pytest.mark.parametrize("hot_only", [False, True])
def test_long_prefix_stores_pool_deltas_and_one_draft_window(setup, tmp_path, monkeypatch, hot_only):
    cache, draft, backend, provider, context = setup
    cache.close()
    cache = bridge.DFlashNativeCache(
        model=provider.model,
        target_ops=provider.target_ops,
        model_name="proof",
        cache_dir=tmp_path / "bounded",
        config=SimpleNamespace(paged_cache_block_size=2048),
        hot_cache_max_bytes=2**26,
        hot_cache_only=hot_only,
        max_size_bytes=2**26,
    )
    setup = (cache, draft, backend, provider, context)
    saved_contexts = []
    save = cache.ssd.save_prefix_context

    def record_context(*args, **kwargs):
        saved_contexts.append(kwargs["token_count"])
        return save(*args, **kwargs)

    monkeypatch.setattr(cache.ssd, "save_prefix_context", record_context)
    tokens = [i % 7 + 1 for i in range(8 * cache.block_size + 5)]
    try:
        _, session, cold, result, _ = prefill(setup, tokens)
        assert saved_contexts == [len(tokens)]
        features = result.feature_store.current_hidden
        assert isinstance(features, bridge.TargetHiddenChunks)
        assert sum(c.shape[1] for c in features.chunks) == 6  # sink=2, window=4
        assert features.total_len == len(tokens)
        assert features.slice(len(tokens) - 4, len(tokens)).tolist() == [
            [[2 * t] for t in tokens[-4:]]
        ]
        table, remaining = cache.prefix.fetch_cache("inspect", tokens)
        assert not remaining
        pooled_rows = 0
        for block_id in table.block_ids:
            block = cache.paged.allocated_blocks[block_id]
            data, _ = cache.ssd.load_block_with_metadata(block.block_hash)
            pooled = data[1][1][1]
            assert pooled[0] == "__nstate__"
            assert pooled[1] == "PoolingCacheDelta"
            rows = pooled[2][2].shape[1]
            assert rows <= cache.block_size // 4
            pooled_rows += rows
        assert pooled_rows == len(tokens) // 4
        cache.prefix.release_cache("inspect")
        warm, restored, state, _, _ = prefill(setup, tokens)
        assert warm.hit_tokens == len(tokens)
        assert state.prefill_logits.tolist() == cold.prefill_logits.tolist()
        assert restored.target_cache[1].caches[1].pooled.shape[1] == pooled_rows
        # A partial hit has no feature blob; the native target plus window replay
        # must still produce exactly the cold state after an edited suffix.
        edited = tokens[:6 * cache.block_size] + [3] * 9
        flow, _, state, _, _ = prefill(setup, edited)
        assert flow.hit_tokens >= 5 * cache.block_size - 2
        assert state.prefill_logits.tolist() == [
            [[sum(edited) + i for i in range(8)]]
        ]
    finally:
        cache.close()



def test_native_memory_waterfall_preserves_cache_byte_counts(setup, monkeypatch):
    from dflash_mlx.engine.memory_waterfall import prefix_cache_memory_fields
    cache, *_ = setup
    monkeypatch.setattr(cache.ssd, "get_stats_dict", lambda: {
        "hot_cache_size_bytes": 123, "total_size": 456,
        "prefix_context_count": 2, "prefix_context_size_bytes": 654,
    })
    fields = prefix_cache_memory_fields(cache.memory_waterfall_bytes())
    assert fields["l1_snapshot_bytes"] == 123
    assert fields["l2_disk_bytes"] == 456
    assert fields["l1_snapshot_draft_context_bytes"] == 654
    assert fields["l1_snapshot_target_hidden_bytes"] == 654
    # Raw counts survive only through collect_memory_waterfall's extra dict;
    # the whitelisted field filter drops them here. The whitelisted byte
    # buckets are the dashboard-visible signal.
    assert cache.memory_waterfall_bytes()["prefix_context_files"] == 2
    assert cache.memory_waterfall_bytes()["prefix_context_bytes"] == 654


def test_required_snapshot_logits_fail_before_publication(setup):
    cache, draft, *_rest, context = setup
    service = bridge.NativeSnapshotService(cache, draft, "unused", context.runtime)
    with pytest.raises(ValueError, match="requires last_logits"):
        service.publish(
            token_ids=[1], target_cache=[], target_hidden=mx.ones((1, 1, 1)),
            last_logits=None, kind="prefill", snapshot_boundary=1,
            allow_full_attention_context=False, require_logits=True,
        )
    assert service.insert_ms == 0
