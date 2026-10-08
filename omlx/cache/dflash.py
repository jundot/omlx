# SPDX-License-Identifier: Apache-2.0
"""DFlash publication and restoration through oMLX's ordinary prefix cache."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any
from pathlib import Path

import mlx.core as mx
from dflash_mlx.cache.codecs import (
    _build_target_hidden_chunks,
    _resolve_effective_trim_window,
)
from dflash_mlx.cache.snapshot import TargetHiddenChunks, validate_prefix_snapshot
from dflash_mlx.cache.snapshot_service import SnapshotPublication
from dflash_mlx.engine.prefill import snapshot_covers_prefix, spans_cover_prefix
from dflash_mlx.engine.spec_epoch import resolve_full_context_draft_layers
from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow
from dflash_mlx.server.prefix_cache_manager import build_prefix_key

from omlx.cache.deepseek_v41_delta import compact_snapshot as compact_deepseek_v41_snapshot
from omlx.cache.pooling_delta import compact_pooling_cache_snapshot
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import (
    PagedSSDCacheManager,
    cachelist_subtypes_from_cache_list,
    numerics_revision_for_model,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache, _TIP_LINEAGE_MAX_ENTRIES
from omlx.cache.state import cache_block_size, extract_cache_states, restore_cache
from omlx.cache.type_registry import CacheTypeRegistry

logger = logging.getLogger(__name__)


def _offset(cache):
    children = getattr(cache, "caches", None)
    if children is not None:
        return next((n for c in children if (n := _offset(c)) is not None), None)
    value = getattr(cache, "offset", None)
    return int(value) if isinstance(value, int) else None


@dataclass
class NativePrefixSnapshot:
    token_ids: tuple[int, ...]
    target_cache: list[Any]
    target_hidden_chunks: tuple[Any, ...]
    target_hidden_chunk_spans: tuple[tuple[int, int], ...]
    target_hidden_total_len: int
    last_logits: Any = None
    replayed_tokens: int = 0

    @property
    def prefix_len(self):
        return len(self.token_ids)


class NativeCacheTargetOps:
    """Use the native registry's prefix support and capture committed prefill boundaries."""

    def __init__(self, target_ops):
        self._target_ops = target_ops
        self.prefill_service = None
        self.prompt_tokens = ()
        self.position = 0

    def __getattr__(self, name):
        return getattr(self._target_ops, name)

    def capabilities_for(self, model):
        return replace(
            self._target_ops.capabilities_for(model), supports_prefix_snapshot=True
        )

    def forward_with_hidden_capture(self, model, *, input_ids, **kwargs):
        service = self.prefill_service
        if service is None:
            return self._target_ops.forward_with_hidden_capture(
                model, input_ids=input_ids, **kwargs
            )
        captures = []
        logits = None
        start = 0
        while start < input_ids.shape[1]:
            length = min(
                input_ids.shape[1] - start,
                service.cache.block_size - self.position % service.cache.block_size,
            )
            end = start + length
            logits, captured = self._target_ops.forward_with_hidden_capture(
                model, input_ids=input_ids[:, start:end], **kwargs
            )
            captures.append(captured)
            self.position += length
            if self.position % service.cache.block_size == 0:
                service.store_target(
                    list(self.prompt_tokens[: self.position]), kwargs["cache"]
                )
            start = end
        if len(captures) == 1:
            return logits, captures[0]
        if isinstance(captures[-1], dict):
            return logits, {
                layer: (
                    captures[-1][layer]
                    if layer == -1
                    else mx.concatenate([part[layer] for part in captures], axis=1)
                )
                for layer in captures[-1]
            }
        return logits, [
            mx.concatenate(columns, axis=1) for columns in zip(*captures, strict=True)
        ]


def install_native_cache_hooks():
    """Install idempotent process-lifetime hooks scoped to native sessions.

    They stay installed across model unloads; non-native sessions delegate
    to the pinned runtime unchanged.
    """
    from dflash_mlx.engine.spec_epoch import SpeculativeSession

    if getattr(SpeculativeSession, "_omlx_native_cache", False):
        return
    original_open = SpeculativeSession.open
    original_prefill = SpeculativeSession._run_prefill_events
    from dflash_mlx.engine.target_features import TargetFeatureStore
    from dflash_mlx.engine.events import PrefillCompleteEvent

    original_features = TargetFeatureStore.hydrate_from_snapshot

    def hydrate_features(self, snapshot, *, snap_prefix_len):
        if (
            isinstance(snapshot, NativePrefixSnapshot)
            and not snapshot.target_hidden_chunks
        ):
            # The replayed suffix supplies every feature the windowed drafter reads.
            return None
        return original_features(self, snapshot, snap_prefix_len=snap_prefix_len)

    def open_session(cls, **kwargs):
        snapshot = kwargs.get("prefix_snapshot")
        if not isinstance(snapshot, NativePrefixSnapshot):
            return original_open(**kwargs)
        n = validate_prefix_snapshot(snapshot, kwargs["prompt_tokens"])
        session = original_open(**{**kwargs, "prefix_snapshot": None})
        if n:
            session.target_cache = restore_cache(
                snapshot.target_cache, session.target_cache
            )
            snapshot.target_cache = []
            session.snap_prefix_len = n
        return session

    def run_prefill(self, *, request, **kwargs):
        service = request.snapshot_service
        if not isinstance(service, NativeSnapshotService):
            return (yield from original_prefill(self, request=request, **kwargs))
        # Stock generation snapshots suppress prompt publication. Native blocks
        # need prompt and accepted-generation boundaries independently.
        ops = self.target_ops
        ops.prefill_service = service if service.active else None
        ops.prompt_tokens = request.prompt_tokens
        ops.position = self.snap_prefix_len
        service._prefill_request_id = uuid.uuid4().hex
        try:
            replayed = getattr(request.prefix_snapshot, "replayed_tokens", 0)
            iterator = original_prefill(
                self,
                request=replace(request, publish_generation_snapshot=False),
                **kwargs,
            )
            try:
                while True:
                    try:
                        event = next(iterator)
                    except StopIteration as done:
                        result = done.value
                        break
                    if replayed and isinstance(event, PrefillCompleteEvent):
                        event = replace(
                            event,
                            prefill_tokens_restored=event.prefill_tokens_restored
                            - replayed,
                            prefill_tokens_computed=event.prefill_tokens_computed
                            + replayed,
                            physical_prefill_tokens=event.physical_prefill_tokens
                            + replayed,
                        )
                    yield event
            finally:
                iterator.close()
            features = result.feature_store
            chunks, spans, total_len = _build_target_hidden_chunks(
                features.require_current_hidden(),
                draft_model=self.draft_model,
                draft_sink_size=self.draft_sink_size,
                draft_window_size=self.draft_window_size,
                allow_full_attention_context=self.allow_full_context_draft_layers,
                clone=False,
            )
            if spans != ((0, total_len),):
                # Copies release the full prompt allocation during decode; slices pin it.
                chunks = tuple(
                    mx.take(chunk, mx.arange(chunk.shape[1]), axis=1) for chunk in chunks
                )
                mx.eval(*chunks)
                features._current_hidden = TargetHiddenChunks(total_len, chunks, spans)
            features.freeze_prefill_for_snapshot(
                enabled=request.should_collect_generation_snapshot_hidden(
                    self.supports_prefix_snapshot
                )
            )
            return result
        finally:
            ops.prefill_service = None
            ops.prompt_tokens = ()
            object.__setattr__(request, "prefix_snapshot", None)
            service.cache.prefix.release_cache(service._prefill_request_id)
            service.cache.paged.delete_block_table(service._prefill_request_id)
            service._prefill_request_id = None

    TargetFeatureStore.hydrate_from_snapshot = hydrate_features
    SpeculativeSession.open = classmethod(open_session)
    SpeculativeSession._run_prefill_events = run_prefill
    SpeculativeSession._omlx_native_cache = True


class DFlashNativeCache:
    def __init__(
        self,
        *,
        model,
        target_ops,
        model_name,
        cache_dir,
        config,
        hot_cache_max_bytes=0,
        hot_cache_only=False,
        max_size_bytes=0,
        prefill_step_size=2048,
    ):
        self.model = model
        self.target_ops = target_ops
        self.model_name = model_name
        self._hits = self._misses = self._tokens_saved = 0
        self.templates = target_ops.make_cache(
            model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=False,
            target_fa_window=0,
        )
        if not isinstance(self.templates, list) or not self.templates:
            raise ValueError("Target backend did not provide native cache templates")
        self.layer_types = [CacheTypeRegistry.canonical_name(c) for c in self.templates]
        self.block_size = int(getattr(config, "paged_cache_block_size", 256))
        # Publication hash -> newest/previous successful contexts. Keying by
        # the matched tip keeps unrelated conversations independent, including
        # target-only publications whose context write failed.
        self._context_tips: dict[tuple[str, bytes], tuple[bytes, ...]] = {}
        leaves = []

        def visit(c):
            children = getattr(c, "caches", None)
            if children is not None:
                for child in children:
                    visit(child)
            else:
                leaves.append(c)

        for c in self.templates:
            visit(c)
        windows = {
            int(c.max_size)
            for c in leaves
            if hasattr(c, "_idx") and int(c.max_size) > 0
        }
        from omlx.scheduler import (
            _detect_qwen35_prefill_floor,
            _detect_qwen4_wide_prefill_step,
            _is_mimo_hybrid,
        )

        self.block_size = cache_block_size(
            self.block_size,
            window_sizes=windows,
            has_pooling=any(type(c).__name__ == "PoolingCache" for c in leaves),
            is_mimo=_is_mimo_hybrid(model),
            has_arrays=any(
                CacheTypeRegistry.detect_cache_type(c).value == "ArraysCache"
                for c in leaves
            ),
            prefill_step_size=prefill_step_size,
            prefill_floor=_detect_qwen35_prefill_floor(model),
            wide_prefill_step=_detect_qwen4_wide_prefill_step(model),
        )
        split = bool(getattr(config, "gdn_ssd_split_enabled", False))
        self.paged = PagedCacheManager(
            block_size=self.block_size,
            max_blocks=getattr(config, "max_cache_blocks", None) or 100000,
            model_name=model_name,
            initial_blocks=getattr(config, "initial_cache_blocks", 256),
        )
        self._ssd_kwargs = dict(
            # Hot-only storage still computes logical block paths; no files
            # are opened when hot_cache_only is set.
            cache_dir=cache_dir if cache_dir is not None else Path("."),
            max_size_bytes=max_size_bytes,
            hot_cache_max_bytes=hot_cache_max_bytes,
            hot_cache_only=hot_cache_only,
            hot_cache_write_through=bool(
                getattr(config, "hot_cache_write_through", False)
            ),
            hot_cache_budget=getattr(config, "hot_cache_budget", None),
            auto_size=bool(getattr(config, "paged_ssd_cache_auto_size", False)),
            expected_model_name=model_name,
            expected_num_layers=len(self.templates),
            expected_block_size=self.block_size,
            expected_block_size_tokens=self.block_size,
            gdn_ssd_split_enabled=split,
            gdn_sidecar_state_dtype=getattr(config, "gdn_sidecar_state_dtype", "fp32"),
        )
        self.ssd = PagedSSDCacheManager(**self._ssd_kwargs)
        self.paged.set_paged_ssd_cache_manager(self.ssd)
        try:
            self.prefix = BlockAwarePrefixCache(
                model, self.paged, self.ssd, gdn_ssd_split_enabled=split
            )
            self.prefix.expected_num_layers = len(self.templates)
            self.ssd.set_expected_layer_signature(
                self.layer_types,
                cachelist_subtypes=cachelist_subtypes_from_cache_list(self.templates),
                numerics=numerics_revision_for_model(model),
            )
            self.ssd.invalidate_stale_layer_signature()
            self.boundary_store = None
            if split and not hot_cache_only:
                from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
                from omlx.scheduler import _BoundarySnapshotProvider

                self.boundary_store = BoundarySnapshotSSDStore(
                    cache_dir,
                    pending_max_bytes=getattr(
                        config, "gdn_ssd_pending_max_bytes", 512 * 1024**2
                    ),
                    gdn_sidecar_state_dtype=getattr(
                        config, "gdn_sidecar_state_dtype", "fp32"
                    ),
                )
                self.prefix.set_gdn_checkpoint_loader(self.boundary_store.load_file)
                self.prefix.set_exact_gdn_checkpoint_writer(
                    lambda **kw: _BoundarySnapshotProvider.stage_and_commit(
                        store=self.boundary_store, paged_ssd_manager=self.ssd, **kw
                    )
                )
        except BaseException:
            self.ssd.close()
            raise

    def _lookup(self, tokens, *, reconstruct=True):
        request_id = uuid.uuid4().hex
        try:
            table, _ = self.prefix.fetch_cache(request_id, tokens)
            if table is None or not table.num_tokens or not table.block_ids:
                return None
            if not reconstruct:
                block = self.paged.allocated_blocks[table.block_ids[-1]]
                return table.num_tokens, block.block_hash, None
            caches = self.prefix.reconstruct_cache(table, promote_to_hot_cache=False)
            if caches:
                caches = restore_cache(
                    caches,
                    self.target_ops.make_cache(
                        self.model,
                        enable_speculative_linear_cache=True,
                        quantize_kv_cache=False,
                        target_fa_window=0,
                    ),
                )
            if not caches or not table.num_tokens or not table.block_ids:
                return None
            block = self.paged.allocated_blocks[table.block_ids[-1]]
            return table.num_tokens, block.block_hash, caches
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Cannot restore DFlash native prefix: %s", exc)
            return None
        finally:
            self.prefix.release_cache(request_id)

    def for_request(
        self,
        *,
        model_provider,
        draft_model,
        tokenizer,
        prompt,
        max_new_tokens,
        runtime_context,
    ):
        started = time.perf_counter()
        key = build_prefix_key(model_provider, draft_model, runtime_context)
        projection = getattr(draft_model, "fc", None)
        signature = json.dumps(
            {
                "dflash_context": 1,
                **asdict(key),
                "draft_quantization": getattr(
                    model_provider, "draft_cache_identity", None
                ),
                "projection": [
                    type(projection).__name__,
                    getattr(projection, "bits", None),
                    getattr(projection, "group_size", None),
                ],
            },
            sort_keys=True,
        )
        runtime = runtime_context.runtime
        allow_full = resolve_full_context_draft_layers(
            supports=self.target_ops.capabilities_for(
                self.model
            ).supports_full_context_draft_layers,
            projected_ctx=len(prompt) + max_new_tokens,
            min_ctx=runtime.draft_full_context_min_ctx,
        )
        sink, window = _resolve_effective_trim_window(
            draft_model,
            len(prompt) + max_new_tokens,
            draft_sink_size=runtime.draft_sink_size,
            draft_window_size=runtime.draft_window_size,
            allow_full_attention_context=allow_full,
        )
        snapshot = None
        ready_from = 0
        found = self._lookup(prompt, reconstruct=False)
        context = None
        previous_tip = found[1] if found is not None else None
        if previous_tip not in self.prefix._store_tip_hashes:
            previous_tip = self.prefix._tip_lineage.get(previous_tip, previous_tip)
        if found is not None:
            n, tip, caches = found
            context = self.ssd.load_prefix_context(tip, signature)
            if context is not None:
                tensors, meta = context
                try:
                    spans = tuple(
                        tuple(int(v) for v in span)
                        for span in json.loads(meta["spans"])
                    )
                    chunks = tuple(tensors[f"hidden_{i}"] for i in range(len(spans)))
                    if any(
                        c.ndim != 3
                        or c.shape[0] != 1
                        or c.shape[-1] != chunks[0].shape[-1]
                        for c in chunks
                    ):
                        raise ValueError("context tensor layout mismatch")
                    TargetHiddenChunks(n, chunks, spans)  # validate tensor/span shapes
                    required = (
                        ((0, n),) if allow_full or window <= 0
                        else ((0, min(sink, n)), (max(0, n - window), n))
                    )
                    if any(
                        not spans_cover_prefix(
                            ((lo - start, hi - start) for lo, hi in spans if hi > start),
                            end - start,
                        )
                        for start, end in required
                    ):
                        raise ValueError("context spans do not cover the drafter window")
                    logits = tensors.get("logits")
                    if logits is not None and (
                        logits.ndim != 2 or logits.shape[0] != 1
                    ):
                        raise ValueError("context logits layout mismatch")
                    if int(meta["prefix_len"]) != n:
                        raise ValueError("context boundary mismatch")
                    snapshot = NativePrefixSnapshot(
                        tuple(prompt[:n]),
                        caches,
                        chunks,
                        spans,
                        n,
                        tensors.get("logits"),
                    )
                    if (allow_full and not snapshot_covers_prefix(snapshot, n)) or (
                        n == len(prompt) and snapshot.last_logits is None
                    ):
                        snapshot = None
                except (KeyError, TypeError, ValueError):
                    snapshot = None
        if snapshot is not None:
            restored = self._lookup(prompt[:snapshot.prefix_len])
            if restored is None or restored[0] != snapshot.prefix_len:
                snapshot = None
            else:
                snapshot.target_cache = restored[2]
        if snapshot is None and window > 0 and not allow_full:
            # A target-only entry can skip a prefix while recomputing a complete
            # drafter window and recovering real sink features separately.
            cutoff = len(prompt) - window
            recovered = self._lookup(prompt[:cutoff]) if cutoff > max(sink, 1) else None
            if recovered is not None:
                n, _, caches = recovered
                if n > max(sink, 1) and len(prompt) - n >= window:
                    snapshot = NativePrefixSnapshot(
                        tuple(prompt[:n]), caches, (), (), n
                    )
                    ready_from = n + window
                    if sink:
                        cold = self.target_ops.make_cache(
                            self.model,
                            enable_speculative_linear_cache=True,
                            quantize_kv_cache=False,
                            target_fa_window=0,
                        )
                        try:
                            _, captured = self.target_ops.forward_with_hidden_capture(
                                self.model,
                                input_ids=mx.array(prompt[:sink], dtype=mx.uint32)[
                                    None
                                ],
                                cache=cold,
                                capture_layer_ids={
                                    i + 1 for i in key.capture_layer_ids
                                },
                                logits_last_only=True,
                            )
                            features = self.target_ops.extract_context_feature(
                                captured, draft_model.target_layer_ids
                            )
                            features = draft_model.project_target_hidden(features)
                            mx.eval(features)
                            snapshot = NativePrefixSnapshot(
                                tuple(prompt[:n]),
                                caches,
                                (features,),
                                ((0, sink),),
                                n,
                                replayed_tokens=sink,
                            )
                        finally:
                            self.target_ops.cleanup_generation_caches(cold, [])
        hit_tokens = (
            snapshot.prefix_len - snapshot.replayed_tokens
            if snapshot is not None
            else 0
        )
        self._hits += int(hit_tokens > 0)
        self._misses += int(hit_tokens == 0)
        self._tokens_saved += hit_tokens
        service = NativeSnapshotService(
            self, draft_model, signature, runtime, ready_from,
            previous_tip=previous_tip,
            previous_context_tips=self._context_tips.get(
                (signature, previous_tip),
                (found[1],) if context is not None else (),
            ),
        )
        return PrefixCacheFlow(
            cache_manager=self,
            key=key,
            stable_prefix_len=len(prompt),
            snapshot=snapshot,
            snapshot_service=service,
            hit_tokens=hit_tokens,
            hit_kind="native" if hit_tokens else "miss",
            lookup_ms=(time.perf_counter() - started) * 1000,
        )

    def prune_context_tips(
        self, signature: str, tip: bytes, *, saved: bool,
        previous_tips: tuple[bytes, ...], replaced_tip: bytes | None,
    ) -> None:
        """Retain this turn and its predecessor, replacing same-turn checkpoints."""
        if saved:
            tips = (tip,) + tuple(t for t in previous_tips[:1] if t != tip)
            for stale in (*previous_tips[1:], replaced_tip):
                if stale is not None and stale not in tips:
                    self.ssd.forget_prefix_context(stale, signature)
        else:
            tips = (
                (replaced_tip,) + previous_tips[:1]
                if replaced_tip is not None else previous_tips
            )
        self._context_tips[signature, tip] = tips
        if len(self._context_tips) > _TIP_LINEAGE_MAX_ENTRIES:
            self._context_tips.clear()

    def clear(self, *, hot=False, ssd=False):
        report = {"hot_cleared": 0, "ssd_deleted": 0, "ranks": []}
        if ssd:
            self.ssd.close()
            report["ssd_deleted"] = self.ssd.clear()
            self.prefix.clear()
            self.ssd = PagedSSDCacheManager(**self._ssd_kwargs)
            self.ssd.set_expected_layer_signature(
                self.layer_types,
                cachelist_subtypes=cachelist_subtypes_from_cache_list(self.templates),
                numerics=numerics_revision_for_model(self.model),
            )
            self.paged.set_paged_ssd_cache_manager(self.ssd)
            self.prefix.paged_ssd_cache = self.ssd
        elif hot:
            report["hot_cleared"] = self.ssd.clear_hot_cache()
        self._context_tips.clear()
        self._hits = self._misses = self._tokens_saved = 0
        return report

    def memory_waterfall_bytes(self):
        stats = self.ssd.get_stats_dict()
        return {
            "l1_snapshot_bytes": stats["hot_cache_size_bytes"],
            "l2_disk_bytes": stats["total_size"],
            # Native-side durable drafter contexts map onto the runtime's
            # draft-context snapshot bucket. prefix_cache_memory_bytes()
            # whitelists names, so the raw counts ride the same bucket.
            "l1_snapshot_draft_context_bytes": stats["prefix_context_size_bytes"],
            "l1_snapshot_target_hidden_bytes": stats["prefix_context_size_bytes"],
            "prefix_context_files": stats["prefix_context_count"],
            "prefix_context_bytes": stats["prefix_context_size_bytes"],
        }

    def close(self):
        self.ssd.close()
        if self.boundary_store is not None:
            self.boundary_store.cleanup_all()
            self.boundary_store.shutdown()
        self.templates = []
        self.prefix.model = self.model = None


class NativeSnapshotService:
    def __init__(
        self, cache, draft_model, signature, runtime, ready_from=0, *,
        previous_tip=None, previous_context_tips=(),
    ):
        self.cache = cache
        self.draft_model = draft_model
        self.signature = signature
        self.runtime = runtime
        self.ready_from = ready_from
        self.previous_tip = previous_tip
        self.previous_context_tips = previous_context_tips
        self._published_tip = None
        self._published_context_tip = None
        self._prefill_request_id = None
        self.insert_ms = 0.0
        self.active = True

    def should_publish_frontier(self, prefix_len):
        # The target proxy already stores every boundary. Keep drafter context
        # at request checkpoints; intermediate hits can replay their window.
        return False

    def store_target(self, token_ids, target_cache):
        if not self.active:
            return None
        n = len(token_ids)
        offsets = [offset for c in target_cache if (offset := _offset(c)) is not None]
        if not n or (offsets and any(offset != n for offset in offsets)):
            return None
        states, model_config = extract_cache_states(target_cache, self.cache.model_name)
        if not states:
            return None
        started = time.perf_counter()
        request_id = self._prefill_request_id or uuid.uuid4().hex
        try:
            compact_pooling_cache_snapshot(states, n, self.cache.block_size)
            compact_deepseek_v41_snapshot(states, n, self.cache.block_size)
            if self.cache.paged.get_block_table(request_id) is None:
                self.cache.prefix.fetch_cache(request_id, token_ids)
            boundaries = {n: states}
            if self.cache.boundary_store is not None:
                from omlx.scheduler import _BoundarySnapshotProvider

                store = self.cache.boundary_store
                if not store.save(
                    request_id,
                    n,
                    states,
                    lambda value: (value, None),
                    block_size=self.cache.block_size,
                ):
                    return None
                boundaries = _BoundarySnapshotProvider(
                    store, request_id, [n], {n: states}, self.cache.ssd
                )
            table = self.cache.prefix.store_cache(
                request_id,
                token_ids,
                states,
                model_cache_config=model_config,
                boundary_snapshots=boundaries,
                _store_tail_terminal=True,
                _track_tip_lineage=False,
            )
            # generation may skip a block boundary. Native storage
            # then retains the earlier checkpoint; committed-boundary callbacks
            # in dflash-mlx would let composite caches retain the continuation.
            if table is None or table.num_tokens != n:
                return None
            return self.cache.paged.allocated_blocks[table.block_ids[-1]].block_hash
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning("DFlash native boundary store failed: %s", exc)
            return None
        finally:
            if self._prefill_request_id is None:
                self.cache.prefix.release_cache(request_id)
                self.cache.paged.delete_block_table(request_id)
            if self.cache.boundary_store is not None:
                self.cache.boundary_store.cleanup_request(request_id)
            self.insert_ms += (time.perf_counter() - started) * 1000

    def publish(
        self,
        *,
        token_ids,
        target_cache,
        target_hidden,
        last_logits,
        kind,
        snapshot_boundary,
        allow_full_attention_context,
        from_snapshot=False,
        snap_prefix_len=0,
        require_logits=False,
        **kwargs,
    ):
        if not self.active or target_hidden is None:
            return None
        if require_logits and last_logits is None:
            raise ValueError(f"{kind} snapshot requires last_logits")
        started = time.perf_counter()
        previous_insert_ms = self.insert_ms
        admitted = False
        context_saved = False
        try:
            tip = self.store_target(token_ids, target_cache)
            admitted = tip is not None
            if admitted and len(token_ids) >= self.ready_from:
                chunks, spans, _ = _build_target_hidden_chunks(
                    target_hidden,
                    draft_model=self.draft_model,
                    draft_sink_size=self.runtime.draft_sink_size,
                    draft_window_size=self.runtime.draft_window_size,
                    allow_full_attention_context=allow_full_attention_context,
                    clone=False,
                )
                tensors = {f"hidden_{i}": value for i, value in enumerate(chunks)}
                if last_logits is not None:
                    tensors["logits"] = last_logits
                context_saved = self.cache.ssd.save_prefix_context(
                    tip,
                    self.signature,
                    tensors,
                    {"prefix_len": str(len(token_ids)), "spans": json.dumps(spans)},
                    token_count=len(token_ids),
                )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning("DFlash native cache publication failed: %s", exc)
        if admitted:
            self.cache.prefix.record_tip(tip, self.previous_tip, self.cache.layer_types)
            if (
                self._published_tip is not None
                and self._published_tip not in (tip, self.previous_tip)
                and (context_saved or self._published_tip != self._published_context_tip)
            ):
                self.cache.prefix._retire_tip(self._published_tip, self.cache.layer_types)
            self.cache.prune_context_tips(
                self.signature, tip, saved=context_saved,
                previous_tips=self.previous_context_tips,
                replaced_tip=self._published_context_tip,
            )
            self._published_tip = tip
            if context_saved:
                self._published_context_tip = tip
            if self.previous_tip is None:
                # A cold request's prompt is the fallback for its generation snapshot.
                self.previous_tip = tip
                if context_saved:
                    self.previous_context_tips = (tip,)
        elapsed = (time.perf_counter() - started) * 1000
        self.insert_ms = previous_insert_ms + elapsed
        return SnapshotPublication(
            kind,
            snapshot_boundary,
            len(token_ids),
            elapsed,
            admitted,
            from_snapshot,
            snap_prefix_len,
        )
