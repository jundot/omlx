"""Regression tests for SpecPrefill parameter forwarding in VLM engine."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.vlm import VLMBatchedEngine


@pytest.mark.asyncio
async def test_vlm_chat_forwards_specprefill_threshold_and_keep_pct():
    """VLM chat must pass both SpecPrefill overrides through to add_request()."""
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._vlm_model = MagicMock()
    engine._vlm_model.config.model_type = "test"
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template.return_value = "<prompt>"
    engine._tokenizer.encode.side_effect = lambda text, **kwargs: list(range(max(1, len(text.split()))))
    engine._engine = MagicMock()
    engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
    engine._engine.add_request = AsyncMock(return_value="req-1")
    engine._engine.abort_request = AsyncMock(return_value=True)

    async def _one_output_stream(_request_id):
        yield MagicMock(
            output_text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine.stream_outputs = _one_output_stream

    # Mock _process_chat_messages to skip mlx-vlm template processing
    def _mock_process(messages, tools, kwargs):
        return "<prompt>", None, {}, None, None, []

    with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
        async for _ in engine.stream_chat(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            specprefill=True,
            specprefill_keep_pct=0.2,
            specprefill_threshold=1024,
        ):
            pass

    try:
        _, kwargs = engine._engine.add_request.call_args
        assert kwargs["specprefill"] is True
        assert kwargs["specprefill_keep_pct"] == 0.2
        assert kwargs["specprefill_threshold"] == 1024
    finally:
        engine._engine._mlx_executor.shutdown(wait=False)


class TestVLMEngineSpecPrefillForwarding:
    """Non-streaming path must forward SpecPrefill overrides (issue #2274/#2281 parity).

    ``generate()``/``chat()`` previously dropped SpecPrefill kwargs on the VLM
    engine, so a configured keep_pct silently fell back to the engine default.
    """

    @staticmethod
    def _fake_output():
        return SimpleNamespace(
            output_text="hi",
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
            first_token_at=None,
        )

    def test_pop_specprefill_kwargs_extracts_and_pops(self):
        kwargs = {
            "specprefill_keep_pct": 0.25,
            "specprefill_threshold": 100,
            "specprefill_system_end": 12,
            "specprefill": True,
            "temperature": 0.7,
        }
        extracted = VLMBatchedEngine._pop_specprefill_kwargs(kwargs)

        assert extracted == {
            "specprefill_keep_pct": 0.25,
            "specprefill_threshold": 100,
            "specprefill_system_end": 12,
            "specprefill": True,
        }
        # Popped out of the original dict; unrelated kwargs are untouched.
        assert kwargs == {"temperature": 0.7}

    def test_pop_specprefill_kwargs_ignores_none_values(self):
        kwargs = {"specprefill_keep_pct": None, "specprefill": None}
        assert VLMBatchedEngine._pop_specprefill_kwargs(kwargs) == {}

    @pytest.mark.asyncio
    async def test_generate_forwards_specprefill_kwargs(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._engine = SimpleNamespace(
            generate=AsyncMock(return_value=self._fake_output())
        )

        await engine.generate(
            "a prompt",
            specprefill_keep_pct=0.25,
            specprefill_threshold=100,
        )

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert call_kwargs["specprefill_keep_pct"] == 0.25
        assert call_kwargs["specprefill_threshold"] == 100

    @pytest.mark.asyncio
    async def test_generate_omits_specprefill_when_absent(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._engine = SimpleNamespace(
            generate=AsyncMock(return_value=self._fake_output())
        )

        await engine.generate("a prompt")

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert "specprefill_keep_pct" not in call_kwargs
        assert "specprefill_threshold" not in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_injects_specprefill_system_end(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._model_settings = SimpleNamespace(specprefill_enabled=True)
        engine._engine = MagicMock()
        engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
        engine._engine.generate = AsyncMock(return_value=self._fake_output())

        # VLM prompts are pre-tokenized (list[int]) by _process_chat_messages;
        # full_tokens = len(prompt) = 10, non_system_tokens = 4, so
        # system_end = 10 - 4 = 6.
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "USER_ONLY"
        engine._tokenizer.encode.side_effect = lambda text, **kwargs: [0] * 4

        def _mock_process(messages, tools, kwargs):
            return list(range(10)), None, None, None, 0, []

        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        try:
            with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
                await engine.chat(messages)
        finally:
            engine._engine._mlx_executor.shutdown(wait=False)

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert call_kwargs["specprefill_system_end"] == 6

    @pytest.mark.asyncio
    async def test_chat_skips_system_end_when_specprefill_disabled(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._model_settings = SimpleNamespace(specprefill_enabled=False)
        engine._engine = MagicMock()
        engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
        engine._engine.generate = AsyncMock(return_value=self._fake_output())

        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "USER_ONLY"
        engine._tokenizer.encode.side_effect = lambda text, **kwargs: [0] * 4

        def _mock_process(messages, tools, kwargs):
            return list(range(8)), None, None, None, 0, []

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        try:
            with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
                await engine.chat(messages)
        finally:
            engine._engine._mlx_executor.shutdown(wait=False)

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert "specprefill_system_end" not in call_kwargs


class _FakeCacheLayer:
    """Scalar-offset KV cache layer, the shape sparse_prefill reads."""

    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        import mlx.core as mx

        return mx.zeros((1, 1))


class _RecordingQwenVLM:
    """An mlx_vlm-shaped target model.

    Attention holds ``rotary_emb`` and rotates from ``position_ids``; there is
    no ``.rope`` for sparse_prefill's position-mapped wrapper to replace.
    ``TestAdapterExplicitPositions`` covers the production position layout this
    stands in for; here the question is only which positions reach the forward.
    """

    def position_ids_for_absolute(self, positions):
        """Stand in for the adapter's layout seam."""
        return positions.reshape(1, -1)

    def __init__(self, cache):
        self.layers = [
            SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=object()))
        ]
        self._cache = cache
        self.seen_positions = []

    def __call__(self, input_ids, cache=None, position_ids=None):
        import mlx.core as mx

        length = input_ids.shape[1]
        if position_ids is None:
            # mlx_vlm's own fallback (qwen3_5/language.py): derive a
            # contiguous run from the cache offset. Reproduced here so a
            # forward that is handed no positions records what the real model
            # would have used, rather than a None the assertions cannot read.
            position_ids = mx.arange(
                cache[0].offset, cache[0].offset + length
            ).reshape(1, -1)
        self.seen_positions.append(position_ids)
        for layer in cache:
            layer.offset += length
        return mx.zeros((1, length, 8))


def _flatten_seen(seen):
    """Concatenate the per-chunk position rows into one position sequence."""
    import mlx.core as mx

    flat = []
    for position_ids in seen:
        # (3, 1, L) mRoPE planes are broadcast-identical; any plane is the row.
        row = position_ids.reshape(-1, position_ids.shape[-1])
        first = row[0]
        for other in range(1, row.shape[0]):
            assert mx.array_equal(row[other], first)
        flat.extend(int(v) for v in first.tolist())
    return flat


class TestSparsePrefillPositionsOnVLM:
    """Selected tokens must keep the positions they were selected from.

    SpecPrefill picks a sparse subset of the conversation and prefills only
    those tokens. Their RoPE positions are ``selected_indices +
    position_offset`` — the positions they hold in the real prompt — not
    ``0..N-1``. On a model whose attention exposes ``.rope`` the wrapper
    enforces that. mlx_vlm attention does not expose one, so before the
    explicit-position seam the forward derived positions from the cache offset
    and wrote the whole selection at dense consecutive positions.
    """

    @staticmethod
    def _run(selected, total_tokens, position_offset, step_size=4):
        import mlx.core as mx

        from omlx.patches.specprefill import sparse_prefill

        cache = [_FakeCacheLayer(offset=position_offset)]
        model = _RecordingQwenVLM(cache)
        sparse_prefill(
            model,
            mx.arange(total_tokens),
            mx.array(selected),
            cache,
            step_size=step_size,
            position_offset=position_offset,
        )
        return model

    def test_cold_cache_positions_are_the_selected_indices(self):
        selected = [0, 5, 6, 17, 40, 41, 99]
        model = self._run(selected, total_tokens=120, position_offset=0)
        assert _flatten_seen(model.seen_positions) == selected

    def test_cached_prefix_offsets_every_selected_position(self):
        selected = [0, 5, 6, 17, 40, 41, 99]
        offset = 12288
        model = self._run(selected, total_tokens=120, position_offset=offset)
        assert _flatten_seen(model.seen_positions) == [
            index + offset for index in selected
        ]

    def test_positions_are_not_dense(self):
        """The pre-fix behaviour, named so a regression cannot pass quietly."""
        selected = [0, 5, 6, 17, 40, 41, 99]
        offset = 12288
        model = self._run(selected, total_tokens=120, position_offset=offset)
        dense = list(range(offset, offset + len(selected)))
        assert _flatten_seen(model.seen_positions) != dense

    def test_decode_adjustment_is_recorded_for_a_ropeless_model(self):
        selected = [0, 5, 6, 17, 40, 41, 99]
        offset = 12288
        model = self._run(selected, total_tokens=120, position_offset=offset)
        # decode resumes at cache offset; the true next position is
        # position_offset + total_tokens.
        assert model._specprefill_decode_adjustment == 120 - len(selected)


class TestAdapterExplicitPositions:
    """The adapter must forward a caller-computed position timeline unchanged."""

    @staticmethod
    def _adapter():
        import mlx.core as mx

        from omlx.models.vlm import VLMModelAdapter

        language_model = MagicMock()
        language_model.return_value = SimpleNamespace(logits=mx.zeros((1, 3, 8)))
        vlm_model = MagicMock()
        vlm_model.language_model = language_model
        adapter = VLMModelAdapter(vlm_model)
        adapter._uses_mrope = True
        adapter._batch_rope_deltas = mx.array([0.0])
        return adapter, language_model

    def test_position_ids_for_absolute_preserves_every_position(self):
        import mlx.core as mx

        adapter, _ = self._adapter()
        positions = adapter.position_ids_for_absolute(mx.array([3, 9, 40]))
        assert positions.shape == (3, 1, 3)
        for plane in range(3):
            assert positions[plane, 0].tolist() == [3, 9, 40]

    def test_explicit_position_ids_win_over_cache_derived_ones(self):
        import mlx.core as mx

        adapter, language_model = self._adapter()
        supplied = adapter.position_ids_for_absolute(mx.array([3, 9, 40]))
        cache = [_FakeCacheLayer(offset=512)]

        adapter(mx.array([[1, 2, 3]]), cache=cache, position_ids=supplied)

        _, kwargs = language_model.call_args
        assert mx.array_equal(kwargs["position_ids"], supplied)

    def test_a_non_mrope_language_model_is_left_alone(self):
        """Nothing here establishes that every VLM takes ``position_ids``."""
        import mlx.core as mx

        adapter, _ = self._adapter()
        adapter._uses_mrope = False
        assert adapter.position_ids_for_absolute(mx.array([3, 9, 40])) is None


class TestSparsePrefillKeepsTheRopeWrapperPath:
    """A model with a wrappable ``.rope`` must be untouched by the seam."""

    def test_rope_models_still_use_the_position_mapped_wrapper(self):
        import mlx.core as mx

        from omlx.patches.specprefill import _PositionMappedRoPE, sparse_prefill

        class _Rope:
            dims = 8
            base = 10000.0
            scale = 1.0

            def __call__(self, x, offset=0):
                return x

        seen = []

        class _RopeModel:
            def __init__(self, cache):
                self.layers = [
                    SimpleNamespace(self_attn=SimpleNamespace(rope=_Rope()))
                ]
                self._cache = cache

            def __call__(self, input_ids, cache=None, **kwargs):
                seen.append(kwargs)
                assert isinstance(self.layers[0].self_attn.rope, _PositionMappedRoPE)
                length = input_ids.shape[1]
                for layer in cache:
                    layer.offset += length
                return mx.zeros((1, length, 8))

        cache = [_FakeCacheLayer(offset=0)]
        model = _RopeModel(cache)
        sparse_prefill(
            model,
            mx.arange(120),
            mx.array([0, 5, 6, 17, 40, 41, 99]),
            cache,
            step_size=4,
            position_offset=0,
        )
        assert seen and all(kwargs == {} for kwargs in seen)
        # The wrapper carries the decode adjustment; nothing is left behind
        # for a caller to apply a second time.
        assert model._specprefill_decode_adjustment is None


class TestDecodeAdjustmentLifetime:
    """The sparse decode delta must not outlive the sparse prefill.

    It is request-lifetime state written outside the scheduler's
    ``cleanup_rope`` path, so the ways it can be left behind are worth naming.
    """

    def test_cleanup_rope_clears_it(self):
        import mlx.core as mx

        from omlx.patches.specprefill import cleanup_rope, sparse_prefill

        cache = [_FakeCacheLayer(offset=12288)]
        model = _RecordingQwenVLM(cache)
        sparse_prefill(
            model,
            mx.arange(120),
            mx.array([0, 5, 6, 17, 40, 41, 99]),
            cache,
            step_size=4,
            position_offset=12288,
        )
        assert model._specprefill_decode_adjustment is not None
        cleanup_rope(model)
        assert model._specprefill_decode_adjustment is None

    def test_a_model_that_declines_positions_records_no_adjustment(self):
        """Weaker gates here than at the read would let the two drift apart."""
        import mlx.core as mx

        from omlx.patches.specprefill import sparse_prefill

        class _Declines(_RecordingQwenVLM):
            def position_ids_for_absolute(self, positions):
                return None

        cache = [_FakeCacheLayer(offset=12288)]
        model = _Declines(cache)
        sparse_prefill(
            model,
            mx.arange(120),
            mx.array([0, 5, 6, 17, 40, 41, 99]),
            cache,
            step_size=4,
            position_offset=12288,
        )
        assert model._specprefill_decode_adjustment is None

    def test_a_failed_chunk_leaves_nothing_the_scheduler_cannot_undo(self):
        """The scheduler's fallback undoes the rope wrapper; both carriers
        must end at the same call."""
        import mlx.core as mx
        import pytest

        from omlx.patches.specprefill import cleanup_rope, sparse_prefill

        class _FailsOnSecondChunk(_RecordingQwenVLM):
            def __call__(self, input_ids, cache=None, position_ids=None):
                if len(self.seen_positions) == 1:
                    raise RuntimeError("out of memory")
                return super().__call__(input_ids, cache=cache, position_ids=position_ids)

        cache = [_FakeCacheLayer(offset=12288)]
        model = _FailsOnSecondChunk(cache)
        with pytest.raises(RuntimeError):
            sparse_prefill(
                model,
                mx.arange(120),
                mx.array([0, 5, 6, 17, 40, 41, 99]),
                cache,
                step_size=2,
                position_offset=12288,
            )
        cleanup_rope(model)
        assert model._specprefill_decode_adjustment is None


class TestSparseDeltaDoesNotSurviveARequeue:
    """A requeued prefill re-prefills densely, so the sparse delta must go.

    The other position carrier already has a restore here (the
    ``_prefill_saved_rope_deltas`` stash); this one needs the same, or the
    retry decodes through an offset that describes a prefill that no longer
    exists.
    """

    def test_oom_requeue_clears_the_sparse_rope_delta(self):
        from omlx.scheduler import Scheduler, SchedulerConfig

        model = MagicMock()
        model.layers = []
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        scheduler = Scheduler(model=model, tokenizer=tokenizer, config=SchedulerConfig())

        request = MagicMock()
        request.request_id = "r1"
        request.prefill_oom_retries = 0
        request.rope_deltas = 12000.0
        request._prefill_saved_rope_deltas = None

        with patch.object(scheduler, "_reclaim_prefill_headroom"):
            requeued = Scheduler._requeue_or_fail_prefill(
                scheduler, request, RuntimeError("Memory limit exceeded")
            )

        assert requeued is True
        assert request.rope_deltas == 0.0
