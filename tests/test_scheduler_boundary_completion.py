# SPDX-License-Identifier: Apache-2.0
"""Exercise completed hybrid requests through durable prefix reuse."""

import json
import struct
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _cache_base_sizes


def _tiny_hybrid_model():
    return Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                full_attention_interval=2,
                vocab_size=32,
                linear_num_value_heads=2,
                linear_num_key_heads=2,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_conv_kernel_dim=4,
                partial_rotary_factor=1.0,
            ),
        )
    )


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("total_tokens", [3, 4, 5])
def test_completed_hybrid_boundary_reuses_final_response(
    mock_tokenizer, tmp_path, split, total_tokens
):
    mx.random.seed(7)
    model = _tiny_hybrid_model()
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4, gdn_ssd_split_enabled=split),
    )
    paged = PagedCacheManager(
        block_size=4, max_blocks=16, initial_blocks=16, model_name="tiny-qwen"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=16 * 1024**2,
        expected_model_name="tiny-qwen",
        expected_num_layers=2,
        expected_block_size=4,
        expected_layer_cache_types=["ArraysCache", "KVCache"],
        gdn_ssd_split_enabled=split,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=split,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler._boundary_snapshot_store = boundary
    scheduler.block_aware_cache = prefix
    generator = BatchGenerator(
        model,
        max_tokens=1,
        sampler=lambda logits: mx.zeros(logits.shape[:-1], dtype=mx.int32),
        stream=scheduler._stream,
    )
    scheduler.batch_generator = generator
    tokens = list(range(3, 3 + total_tokens - 1))
    request = Request(
        request_id="finished",
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add_request(request)
    scheduler.waiting.clear()
    scheduler.running[request.request_id] = request
    uid = generator.insert([tokens])[0]
    scheduler.request_id_to_uid[request.request_id] = uid
    scheduler.uid_to_request_id[uid] = request.request_id

    try:
        for _ in range(5):
            _, responses = generator.next()
            if responses:
                break
        assert responses[0].finish_reason == "length"
        assert responses[0].prompt_cache[-1].offset == total_tokens
        assert generator.extract_cache([uid]) == {}
        _, finished = scheduler._process_batch_responses(responses)
        assert finished == {request.request_id}
        scheduler._cleanup_finished(finished)
        for future in list(scheduler._inflight_store_futures.values()):
            future.result(timeout=10)
        scheduler._drain_pending_async_removes()

        extended = tokens + [0, 9, 10]
        table, remaining = prefix.fetch_cache("next", extended)
        # The generator prefilled directly, so no prefill tail was captured;
        # only an aligned completion boundary is reusable here. Unaligned
        # replies are not stored: the next turn renders them differently on
        # reasoning templates.
        if total_tokens % 4:
            assert table is None
            assert remaining == extended
        else:
            assert table is not None and table.num_tokens == total_tokens
            if split:
                sidecars = list(
                    (tmp_path / "cache" / "_gdn_sidecars").rglob("*.safetensors")
                )
                assert len(sidecars) == 1
                with sidecars[0].open("rb") as checkpoint:
                    header_size = struct.unpack("<Q", checkpoint.read(8))[0]
                    header = json.loads(checkpoint.read(header_size))
                # KV comes from the paged blocks; completion checkpoints
                # must retain only recurrent state, like prefill checkpoints.
                assert "layer_0_state_0" in header
                assert "layer_0_state_1" in header
                assert not any(key.startswith("layer_1_") for key in header)
            restored = prefix.reconstruct_cache(table)
            assert restored is not None
            assert remaining == [9, 10]
            with mx.stream(scheduler._stream):
                warm_logits = model(mx.array([remaining]), cache=restored)
                cold_logits = model(mx.array([extended]), cache=model.make_cache())
                mx.eval(warm_logits, cold_logits)
            assert mx.allclose(warm_logits[:, -1], cold_logits[:, -1], atol=1e-4)
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "offsets,expected",
    [([None], False), ([5], False), ([4, 5], False), ([4], True)],
)
def test_completion_boundary_requires_known_consistent_positions(
    mock_model, mock_tokenizer, offsets, expected
):
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler._boundary_snapshot_required = True
    scheduler._on_prefill_boundary_snapshot = MagicMock()
    request = Request(
        request_id="finished",
        prompt=[3, 4, 5],
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.prompt_token_ids = [3, 4, 5]
    request.num_prompt_tokens = 3
    request.output_token_ids = [6]
    cache = [
        SimpleNamespace(
            caches=[
                SimpleNamespace(offset=offset),
                SimpleNamespace(),
            ]
        )
        for offset in offsets
    ]

    scheduler._capture_finished_boundary_snapshot(request, cache)

    assert scheduler._on_prefill_boundary_snapshot.called is expected
    scheduler.shutdown()


@pytest.mark.parametrize("preserve_reasoning, expected", [(False, False), (True, True)])
def test_completion_boundary_counts_output_only_when_reasoning_is_preserved(
    mock_model, mock_tokenizer, preserve_reasoning, expected
):
    """With a think prefix the terminal boundary covers prompt + output only if the history keeps the reasoning."""
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler._boundary_snapshot_required = True
    scheduler._on_prefill_boundary_snapshot = MagicMock()
    request = Request(
        request_id="finished-think",
        prompt=[3, 4, 5],
        sampling_params=SamplingParams(max_tokens=1),
        preserve_reasoning=preserve_reasoning,
    )
    request.prompt_token_ids = [3, 4, 5]
    request.num_prompt_tokens = 3
    request.output_token_ids = [6]
    request.needs_think_prefix = True
    cache = [SimpleNamespace(caches=[SimpleNamespace(offset=4), SimpleNamespace()])]

    scheduler._capture_finished_boundary_snapshot(request, cache)

    assert scheduler._on_prefill_boundary_snapshot.called is expected
    scheduler.shutdown()


@pytest.mark.parametrize(
    "total_tokens,existing,expected",
    [
        (8, {}, True),
        (8, {3: object()}, True),
        (8, {8: object()}, False),
        (5, {}, False),
    ],
)
def test_completion_capture_ignores_older_tail_but_stays_aligned(
    mock_model, mock_tokenizer, total_tokens, existing, expected
):
    """A prefill tail below the end must not block the aligned completion capture."""
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler._boundary_snapshot_required = True
    scheduler._on_prefill_boundary_snapshot = MagicMock()
    request = Request(
        request_id="finished-tail",
        prompt=[3, 4, 5],
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = [3, 4, 5]
    request.num_prompt_tokens = 3
    request.output_token_ids = list(range(6, 6 + total_tokens - 3))
    scheduler._boundary_cache_snapshots[request.request_id] = dict(existing)
    cache = [
        SimpleNamespace(
            caches=[SimpleNamespace(offset=total_tokens), SimpleNamespace()]
        )
    ]

    scheduler._capture_finished_boundary_snapshot(request, cache)

    assert scheduler._on_prefill_boundary_snapshot.called is expected
    if expected:
        args = scheduler._on_prefill_boundary_snapshot.call_args.args
        kwargs = scheduler._on_prefill_boundary_snapshot.call_args.kwargs
        assert args[2] == total_tokens
        assert kwargs["source"] == "completion"
    scheduler.shutdown()


def _tiny_rotating_model():
    from mlx_lm.models.gemma3_text import Model, ModelArgs

    return Model(
        ModelArgs(
            model_type="gemma3_text",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            head_dim=16,
            vocab_size=32,
            num_key_value_heads=1,
            sliding_window=4,
            sliding_window_pattern=2,
            query_pre_attn_scalar=16,
        )
    )


def _run_through_scheduler(scheduler, tokens, marker, request_id, system_end=0):
    """Serve one request end to end: prefix lookup, prefill, decode, store."""
    request = Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=1, temperature=0),
    )
    scheduler.add_request(request)
    request.generation_prompt_start = marker
    request.system_prefix_end = system_end
    for _ in range(50):
        scheduler.step()
        if request_id not in scheduler.requests:
            break
    for future in list(scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    scheduler._drain_pending_async_removes()
    return request


def _assert_restored_matches_cold(model, scheduler, prefix, tokens, expected_cached):
    table, remaining = prefix.fetch_cache("probe", tokens)
    assert table is not None and table.num_tokens == expected_cached
    restored = prefix.reconstruct_cache(table)
    assert restored is not None
    assert remaining == tokens[expected_cached:]
    with mx.stream(scheduler._stream):
        cache = model.make_cache()
        if scheduler._turboquant_kv_bits is None:
            cold = model(mx.array([tokens]), cache=cache)[:, -1]
        else:
            # TurboQuant blocks hold the quantized prefix.
            model(mx.array([tokens[:expected_cached]]), cache=cache)
            scheduler._apply_turboquant_kv_convert(cache)
            cold = model(mx.array([remaining]), cache=cache)[:, -1]
        warm = model(mx.array([remaining]), cache=restored)[:, -1]
        mx.eval(cold, warm)
    assert mx.allclose(warm, cold, atol=1e-4)
    prefix.release_cache("probe")


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "kind,split",
    [("gdn", False), ("gdn", True), ("rotating", False)],
)
def test_prefill_tail_reuse_matches_cold_forward(
    mock_tokenizer, tmp_path, kind, split, chunked
):
    """Tail blocks stored by the scheduler restore the same state a cold forward
    produces, for GDN (embedded and sidecar) and rotating layouts on both
    prefill paths, across a repeat, an extended prompt, a second turn, and a
    request that shares only the system prefix."""
    mx.random.seed(7)
    if kind == "gdn":
        model = _tiny_hybrid_model()
        layer_types = ["ArraysCache", "KVCache"]
    else:
        model = _tiny_rotating_model()
        layer_types = ["RotatingKVCache", "KVCache"]
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(
            paged_cache_block_size=4,
            prefill_step_size=4,
            gdn_ssd_split_enabled=split,
            chunked_prefill=chunked,
        ),
    )
    paged = PagedCacheManager(
        block_size=4, max_blocks=64, initial_blocks=64, model_name="tiny"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=16 * 1024**2,
        expected_model_name="tiny",
        expected_num_layers=2,
        expected_block_size=4,
        expected_layer_cache_types=layer_types,
        gdn_ssd_split_enabled=split,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=split,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler._boundary_snapshot_store = boundary
    scheduler.block_aware_cache = prefix

    try:
        # Ten tokens, marker at 7: the tail [4:7) precedes the "generation
        # prompt", the prefill then crosses the boundary at 8, and the one
        # generated token leaves the sequence off the grid. Without the
        # persists flag the boundary at 8 must not outrank the tail.
        first = list(range(3, 13))
        _run_through_scheduler(scheduler, first, 7, "turn-1")
        _assert_restored_matches_cold(model, scheduler, prefix, first, 7)
        _assert_restored_matches_cold(model, scheduler, prefix, first + [1, 2, 5], 7)

        # Second turn reuses the tail, then stores its own tail at the
        # prefill end (its marker is the last prefilled token).
        second = first + [1, 2, 5, 6]
        request = _run_through_scheduler(scheduler, second, 13, "turn-2")
        assert request.cached_tokens == 7
        _assert_restored_matches_cold(model, scheduler, prefix, second + [7, 8], 13)

        # Fan-out on a system prefix shorter than a block (a root tail), then
        # on one whose user turn crosses a boundary. The shared tail survives
        # the other conversation's later turns.
        for system in ([20, 21, 22], [20, 21, 22, 23, 24, 25]):
            n = len(system)
            other = system + [26, 27, 28, 29]
            _run_through_scheduler(
                scheduler, other, len(other) - 1, f"a-{n}", system_end=n
            )
            turn = system + [30, 31, 0, 29]
            for i, extra in enumerate(([], [9, 10, 11, 29], [12, 13, 14, 29])):
                turn = turn + extra
                request = _run_through_scheduler(
                    scheduler, turn, len(turn) - 1, f"b-{n}-{i}", system_end=n
                )
                if i == 0:
                    assert request.cached_tokens == n
            probe = system + [15, 16, 29]
            _assert_restored_matches_cold(model, scheduler, prefix, probe, n)
    finally:
        scheduler.shutdown()


def _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind):
    """A chunked-prefill scheduler over a tiny model and a real paged SSD cache."""
    mx.random.seed(7)
    if kind == "gdn":
        model = _tiny_hybrid_model()
        layer_types = ["ArraysCache", "KVCache"]
    elif kind == "deepseek_v41":
        from test_deepseek_v41 import load_reference_weights, tiny

        from omlx.patches.deepseek_v41 import apply_patch
        from omlx.patches.deepseek_v41.language import LanguageModel

        apply_patch()
        model = LanguageModel(tiny())
        load_reference_weights(model)
        layer_types = ["DeepseekV41Cache"] * len(model.layers)
    else:
        model = _tiny_rotating_model()
        layer_types = ["RotatingKVCache", "KVCache"]
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(
            paged_cache_block_size=4, prefill_step_size=4, chunked_prefill=True
        ),
    )
    paged = PagedCacheManager(
        block_size=4, max_blocks=128, initial_blocks=128, model_name="tiny"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=16 * 1024**2,
        expected_model_name="tiny",
        expected_num_layers=len(layer_types),
        expected_block_size=4,
        expected_layer_cache_types=layer_types,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
    prefix = BlockAwarePrefixCache(
        model=model, paged_cache_manager=paged, paged_ssd_cache_manager=ssd
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler._boundary_snapshot_store = boundary
    scheduler.block_aware_cache = prefix
    # The rig attaches the cache after construction, so add the async writer a
    # configured paged SSD cache would have.
    scheduler._store_cache_executor = ThreadPoolExecutor(max_workers=1)
    scheduler._get_stop_tokens = lambda: set()
    # Non-NAX hosts with 64 GB+ widen Qwen3.5 prefill chunks past these prompts.
    scheduler._qwen35_prefill_floor = 0
    # Fairness gates chunks on wall-clock decode time; a slow host would let the
    # decoding rows finish before the chunked prefill reaches its probe point.
    scheduler._decode_fairness = False
    return model, scheduler, prefix


@pytest.mark.parametrize("kind", ["gdn", "rotating", "deepseek_v41"])
@pytest.mark.parametrize("finish", ["cancel", "parser_stop"])
def test_requeued_request_stores_each_output_token_once(
    mock_tokenizer, tmp_path, kind, finish
):
    from omlx.request import RequestStatus

    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind)
    requests = [
        Request(
            request_id=str(i),
            prompt=list(range(3 + i, 13 + i)),
            sampling_params=SamplingParams(max_tokens=32, temperature=0),
        )
        for i in range(2)
    ]
    request = requests[0]
    original_prompt = list(request.prompt)
    try:
        for row in requests:
            scheduler.add_request(row)
        for _ in range(100):
            scheduler.step()
            if min(row.num_output_tokens for row in requests) >= 3:
                break
        assert scheduler.request_memory_requeue() == 2
        for _ in range(100):
            scheduler.step()
            if request.num_output_tokens >= 14:
                break
        assert request.num_output_tokens == 14
        tokens = original_prompt + request.output_token_ids

        if finish == "cancel":
            scheduler.abort_request(request.request_id, keep_cache=True)
            scheduler.step()
        else:
            request.set_finished(RequestStatus.FINISHED_STOPPED)
            scheduler._cleanup_finished({request.request_id})
        for future in list(scheduler._inflight_store_futures.values()):
            future.result(timeout=30)
        scheduler._drain_pending_async_removes()
        assert request.request_id not in scheduler.requests
        _assert_restored_matches_cold(
            model, scheduler, prefix, tokens + [7, 8], len(tokens)
        )
    finally:
        scheduler.shutdown()
        scheduler.batch_generator = None


def test_memory_requeue_preserves_thinking_budget(mock_tokenizer, tmp_path):
    from omlx.api.thinking import ThinkingBudgetProcessor

    _, scheduler, _ = _chunked_scheduler_rig(mock_tokenizer, tmp_path, "gdn")

    def fixed_logits(tokens, logits):
        result = mx.full(logits.shape, float("-inf"))
        result[..., 7] = 0.0
        return result

    scheduler._build_sampler_and_processors = lambda params, request: (
        lambda logits: mx.argmax(logits, axis=-1),
        [fixed_logits, ThinkingBudgetProcessor([31], 9)],
    )
    requests = [
        Request(
            request_id=str(i),
            prompt=list(range(3 + i, 13 + i)),
            sampling_params=SamplingParams(max_tokens=24, temperature=0),
        )
        for i in range(2)
    ]
    streamed = {request.request_id: [] for request in requests}

    def step():
        for output in scheduler.step().outputs:
            streamed[output.request_id].extend(output.new_token_ids)

    try:
        for request in requests:
            scheduler.add_request(request)
        for _ in range(100):
            step()
            if min(request.num_output_tokens for request in requests) >= 5:
                break
        assert scheduler.request_memory_requeue() == 2
        for _ in range(200):
            step()
            if all(request.is_finished() for request in requests):
                break
        for request in requests:
            assert request.memory_requeues == 1
            assert streamed[request.request_id] == [7] * 8 + [31] + [7] * 15
            assert request.output_token_ids == streamed[request.request_id]
    finally:
        scheduler.shutdown()
        scheduler.batch_generator = None


@pytest.mark.parametrize("kind", ["gdn", "rotating"])
def test_memory_requeue_resumes_decode_from_stored_prefix(
    mock_tokenizer, tmp_path, kind
):
    """A memory requeue stores two decoding rows and a chunked prefill, resumes
    them from the paged SSD cache, and streams every token once with the
    original usage."""
    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind)

    prompts = {
        "a": list(range(3, 13)),
        "b": list(range(4, 12)),
        "c": [5 + i % 20 for i in range(30)],
    }
    requests = {
        rid: Request(
            request_id=rid,
            prompt=list(tokens),
            sampling_params=SamplingParams(max_tokens=12, temperature=0),
        )
        for rid, tokens in prompts.items()
    }
    streamed = {rid: [] for rid in requests}
    finals = {}

    def step():
        for output in scheduler.step().outputs:
            streamed[output.request_id].extend(output.new_token_ids)
            if output.finished:
                finals[output.request_id] = output

    try:
        for rid in ("a", "b"):
            scheduler.add_request(requests[rid])
        for _ in range(100):
            step()
            if all(len(requests[rid].output_token_ids) >= 5 for rid in ("a", "b")):
                break
        scheduler.add_request(requests["c"])
        for _ in range(100):
            step()
            state = scheduler._prefill_states.get("c")
            if state is not None and _cache_base_sizes(state.cache) >= 8:
                break
        prefilled = _cache_base_sizes(scheduler._prefill_states["c"].cache)
        sent = {rid: len(r.output_token_ids) for rid, r in requests.items()}
        assert sent["c"] == 0

        assert scheduler.request_memory_requeue() == 3
        step()
        for rid, request in requests.items():
            assert request.memory_requeues == 1
            assert request.requeued_output_tokens == sent[rid]
            assert len(request.prompt_token_ids) == len(prompts[rid]) + sent[rid]
            resumed = request.prompt_token_ids
            # A decoding row keeps every full block but its last token; a
            # prefill keeps every block it finished.
            stored = (len(resumed) - 1) // 4 * 4 if sent[rid] else prefilled // 4 * 4
            _assert_restored_matches_cold(model, scheduler, prefix, resumed, stored)

        for _ in range(200):
            step()
            if len(finals) == len(requests):
                break
        for rid, request in requests.items():
            assert streamed[rid] == request.output_token_ids
            assert len(streamed[rid]) == 12
            assert finals[rid].prompt_tokens == len(prompts[rid])
            assert finals[rid].completion_tokens == 12
            assert finals[rid].cached_tokens == 0
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("kind", ["gdn", "rotating"])
def test_client_cancel_keeps_computed_prefix(mock_tokenizer, tmp_path, kind):
    """A client cancel stores a decoding row's prompt and a chunked prefill's
    progress for a resend; a cancel under memory pressure stores nothing."""
    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind)
    prompts = {
        "decode": list(range(3, 13)),
        "pressure": list(range(14, 24)),
        "prefill": [5 + i % 9 for i in range(30)],
    }
    requests = {
        rid: Request(
            request_id=rid,
            prompt=list(tokens),
            sampling_params=SamplingParams(max_tokens=50, temperature=0),
        )
        for rid, tokens in prompts.items()
    }
    try:
        for rid in ("decode", "pressure"):
            scheduler.add_request(requests[rid])
        for _ in range(100):
            scheduler.step()
            if all(
                len(requests[r].output_token_ids) >= 3 for r in ("decode", "pressure")
            ):
                break
        scheduler.add_request(requests["prefill"])
        for _ in range(100):
            scheduler.step()
            state = scheduler._prefill_states.get("prefill")
            if state is not None and _cache_base_sizes(state.cache) >= 8:
                break
        prefilled = _cache_base_sizes(scheduler._prefill_states["prefill"].cache)

        scheduler.abort_request("decode", keep_cache=True)
        scheduler.abort_request("prefill", keep_cache=True)
        scheduler.step()
        scheduler._admission_paused = True
        scheduler.abort_request("pressure", keep_cache=True)
        scheduler.step()
        for future in list(scheduler._inflight_store_futures.values()):
            future.result(timeout=30)
        scheduler._drain_pending_async_removes()
        assert not scheduler.requests

        table, _ = prefix.fetch_cache("probe-decode", prompts["decode"])
        assert table is not None and table.num_tokens >= 8
        prefix.release_cache("probe-decode")
        _assert_restored_matches_cold(
            model, scheduler, prefix, prompts["decode"], table.num_tokens
        )
        _assert_restored_matches_cold(
            model, scheduler, prefix, prompts["prefill"], prefilled // 4 * 4
        )
        table, _ = prefix.fetch_cache("probe-pressure", prompts["pressure"])
        assert table is None or table.num_tokens == 0
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "kind,turboquant", [("gdn", False), ("rotating", False), ("gdn", True)]
)
def test_client_cancel_keeps_external_prefill_progress(
    mock_tokenizer, tmp_path, kind, turboquant
):
    """A cancel that lands inside the external prefill loop stores the blocks
    that loop finished, so a resend restores them. Under TurboQuant they are
    stored in the quantized decode layout."""
    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind)
    scheduler.config.chunked_prefill = False
    if turboquant:
        apply_turboquant_attention_patch()
        scheduler._turboquant_kv_bits = 4.0
        assert scheduler.refresh_ssd_layer_signature() == [
            "ArraysCache",
            "TurboQuantKVCache",
        ]
    prompt = [5 + i % 9 for i in range(30)]
    scheduler.add_request(
        Request(
            request_id="external",
            prompt=list(prompt),
            sampling_params=SamplingParams(max_tokens=8, temperature=0),
        )
    )
    check_aborts = scheduler._check_pending_aborts_for_uids
    chunks = []

    def cancel_after_two_chunks(uids):
        chunks.append(len(uids))
        if len(chunks) == 2:
            scheduler.abort_request("external", keep_cache=True)
        return check_aborts(uids)

    scheduler._check_pending_aborts_for_uids = cancel_after_two_chunks
    try:
        for _ in range(10):
            scheduler.step()
            if "external" not in scheduler.requests:
                break
        for future in list(scheduler._inflight_store_futures.values()):
            future.result(timeout=30)
        scheduler._drain_pending_async_removes()
        assert len(chunks) == 2
        assert not scheduler.requests

        table, _ = prefix.fetch_cache("probe-external", prompt)
        assert table is not None and table.num_tokens >= 4
        prefix.release_cache("probe-external")
        _assert_restored_matches_cold(
            model, scheduler, prefix, prompt, table.num_tokens
        )
    finally:
        scheduler.shutdown()
