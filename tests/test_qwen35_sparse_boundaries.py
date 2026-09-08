# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in Qwen embedded-GDN sparse-boundary experiment."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache as MLXArraysCache
from mlx_lm.models.cache import KVCache as MLXKVCache

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


class ArraysCache:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self.cache = [mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))]

    @property
    def state(self):
        return self.cache

    def size(self):
        return self.offset


class KVCache:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self._state = (mx.zeros((1, 1, 1, 1)), mx.zeros((1, 1, 1, 1)))

    @property
    def state(self):
        return self._state

    def size(self):
        return self.offset


class CacheList:
    def __init__(self):
        self.caches = [KVCache(), ArraysCache()]


class _ChunkModel:
    model_type = "qwen3_5"

    def __init__(self, cache_names=("ArraysCache", "KVCache")):
        self.config = SimpleNamespace(
            model_type="qwen3_5",
            num_hidden_layers=2,
            hidden_size=8,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        self.cache_names = cache_names
        self.prefill_calls = []

    def make_cache(self):
        constructors = {
            "ArraysCache": ArraysCache,
            "KVCache": KVCache,
            "CacheList": CacheList,
        }
        return [constructors[name]() for name in self.cache_names]

    def __call__(self, tokens, cache=None, **kwargs):
        count = int(tokens.shape[1])
        self.prefill_calls.append(count)
        for layer in cache or []:
            if hasattr(layer, "offset"):
                layer.offset += count
        return mx.zeros((1, count, 1))


def _scheduler(
    monkeypatch,
    tmp_path,
    model=None,
    *,
    split=False,
    explicit=512,
    enabled=True,
):
    if enabled:
        monkeypatch.setenv("OMLX_QWEN35_SPARSE_BOUNDARIES", "1")
    else:
        monkeypatch.delenv("OMLX_QWEN35_SPARSE_BOUNDARIES", raising=False)
    model = model or _ChunkModel()
    with (
        patch("omlx.settings.get_system_memory", return_value=32 * 1024**3),
        patch("omlx.custom_kernels.nax.is_nax_available", return_value=True),
    ):
        scheduler = Scheduler(
            model=model,
            tokenizer=MagicMock(),
            config=SchedulerConfig(
                prefill_step_size=2048,
                paged_ssd_cache_dir=str(tmp_path / "cache"),
                paged_ssd_cache_max_size=32 * 1024**2,
                paged_cache_block_size=256,
                arrays_cache_block_size=explicit,
                gdn_ssd_split_enabled=split,
            ),
        )
    scheduler._adaptive_chunk_size = lambda n, **kwargs: n
    scheduler._guard_prefill_chunk = lambda n, **kwargs: n
    scheduler._record_chunk_transient = lambda *args, **kwargs: None
    scheduler._maybe_record_fixed_state_bytes = lambda cache: None
    scheduler._consume_pressure_clear = lambda: False
    scheduler._check_pending_aborts_for_uids = lambda uids: set()
    scheduler._memory_limit_bytes = 0
    return scheduler


def _run_external_prefill(scheduler, token_count, *, base=0):
    request = Request(
        request_id=f"prefill-{base}-{token_count}",
        prompt="",
        sampling_params=SamplingParams(),
    )
    request.cached_tokens = base
    cache = scheduler.model.make_cache()
    for layer in cache:
        if hasattr(layer, "offset"):
            layer.offset = base
    captured = []
    scheduler._emit_prefill_boundary_snapshot = (
        lambda request, prompt_cache, total: captured.append(total)
    )
    scheduler.model.prefill_calls.clear()
    scheduler._do_external_prefill(
        request,
        list(range(token_count)),
        cache if base else None,
    )
    return scheduler.model.prefill_calls, captured


def test_sparse_fresh_prefill_keeps_wide_chunks_and_one_final_split(
    monkeypatch, tmp_path
):
    scheduler = _scheduler(monkeypatch, tmp_path)
    try:
        chunks, captures = _run_external_prefill(scheduler, 5000)
        assert chunks == [2048, 2048, 512, 391]
        assert captures == [2048, 4096, 4608]
    finally:
        scheduler.shutdown()


def test_sparse_restored_prefill_honours_absolute_boundaries(monkeypatch, tmp_path):
    scheduler = _scheduler(monkeypatch, tmp_path)
    try:
        chunks, captures = _run_external_prefill(scheduler, 2441, base=2560)
        assert chunks == [1536, 512, 392]
        assert captures == [4096, 4608]
    finally:
        scheduler.shutdown()


def test_sparse_exact_multiple_prompt_keeps_last_token_for_insert(
    monkeypatch, tmp_path
):
    scheduler = _scheduler(monkeypatch, tmp_path)
    try:
        chunks, captures = _run_external_prefill(scheduler, 5120)
        assert chunks == [2048, 2048, 512, 511]
        assert captures == [2048, 4096, 4608]
        assert 5120 not in captures
    finally:
        scheduler.shutdown()


def test_sparse_gate_requires_exact_embedded_flat_qwen_layout(monkeypatch, tmp_path):
    valid = _scheduler(monkeypatch, tmp_path / "valid")
    try:
        assert valid._qwen35_sparse_boundaries is True
    finally:
        valid.shutdown()

    for model, split, explicit in (
        (_ChunkModel(("ArraysCache", "KVCache")), True, 512),
        (_ChunkModel(("ArraysCache", "KVCache")), False, 0),
        (_ChunkModel(("CacheList",)), False, 512),
        (_ChunkModel(("ArraysCache",)), False, 512),
    ):
        scheduler = _scheduler(
            monkeypatch,
            tmp_path / f"invalid-{split}-{explicit}-{model.cache_names[0]}",
            model,
            split=split,
            explicit=explicit,
        )
        try:
            assert scheduler._qwen35_sparse_boundaries is False
        finally:
            scheduler.shutdown()

    disabled = _scheduler(monkeypatch, tmp_path / "disabled", enabled=False)
    try:
        assert disabled._qwen35_sparse_boundaries is False
    finally:
        disabled.shutdown()


def test_split_gdn_explicit_512_keeps_every_block_checkpoint(monkeypatch, tmp_path):
    scheduler = _scheduler(monkeypatch, tmp_path, split=True, explicit=512)
    try:
        assert scheduler.config.paged_cache_block_size == 512
        assert scheduler._qwen35_sparse_boundaries is False

        final_forwarded_total = 2048
        targets = []
        current = 0
        while current < final_forwarded_total:
            target = scheduler._next_prefill_snapshot_boundary(
                current,
                final_forwarded_total,
                512,
            )
            assert target is not None
            targets.append(target)
            current = target

        assert targets == [512, 1024, 1536, 2048]
        assert [
            target
            for target in targets
            if scheduler._should_capture_prefill_boundary(
                target,
                final_forwarded_total,
                512,
            )
        ] == targets
    finally:
        scheduler.shutdown()


def _embedded_extracted(token_count: int, recurrent_value: float):
    return [
        {
            "state": (
                mx.full((1, 1, token_count, 2), recurrent_value),
                mx.full((1, 1, token_count, 2), recurrent_value + 1),
            ),
            "class_name": "KVCache",
            "cache_type": "KVCache",
            "meta_state": (token_count,),
        },
        {
            "state": (
                mx.full((1, 1, 2), recurrent_value),
                mx.full((1, 1, 1, 2), recurrent_value),
            ),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
            "meta_state": (),
        },
    ]


def _block_hashes(prefix, table):
    return [
        prefix.paged_cache.allocated_blocks[block_id].block_hash
        for block_id in table.block_ids
    ]


def test_embedded_sparse_snapshots_walk_back_without_skipping_suffix_and_dedup(
    tmp_path,
):
    block_size = 4
    tokens = list(range(16))
    paged = PagedCacheManager(
        block_size=block_size,
        max_blocks=32,
        initial_blocks=32,
        model_name="sparse-embedded",
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "embedded",
        max_size_bytes=32 * 1024**2,
        expected_model_name="sparse-embedded",
        expected_num_layers=2,
        expected_block_size=block_size,
        expected_layer_cache_types=["KVCache", "ArraysCache"],
        gdn_ssd_split_enabled=False,
    )
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(
            make_cache=lambda: [
                MLXKVCache(),
                MLXArraysCache(size=2),
            ]
        ),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=False,
    )
    sparse = {8: _embedded_extracted(8, 8.0)}

    try:
        first = prefix.store_cache(
            "first",
            tokens,
            _embedded_extracted(16, 16.0),
            boundary_snapshots=sparse,
        )
        assert first is not None and first.num_tokens == 16
        hashes = _block_hashes(prefix, first)

        second = prefix.store_cache(
            "dedup",
            tokens,
            _embedded_extracted(16, 16.0),
            boundary_snapshots=sparse,
        )
        assert second is not None and second.num_tokens == 16
        assert _block_hashes(prefix, second) == hashes

        mid_prompt = tokens[:12]
        table, initial_remaining = prefix.fetch_cache("restore-mid", mid_prompt)
        assert table is not None and initial_remaining == []
        restored = prefix.reconstruct_cache(table)
        assert restored is not None
        assert table.num_tokens == 8
        corrected_remaining = mid_prompt[table.num_tokens :]
        assert corrected_remaining == tokens[8:12]
        assert restored[0].state[0].shape[2] == 8
        assert restored[1].size() == 8
        prefix.release_cache("restore-mid")

        extended = tokens + [16, 17]
        table, remaining = prefix.fetch_cache("restore-final", extended)
        assert table is not None and remaining == [16, 17]
        restored = prefix.reconstruct_cache(table)
        assert restored is not None and table.num_tokens == 16
        assert restored[1].size() == 16
        prefix.release_cache("restore-final")
    finally:
        ssd.close()
