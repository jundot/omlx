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
    def __init__(self, offset=0):
        self.offset = offset

    @property
    def state(self):
        import mlx.core as mx

        return mx.zeros((1, 1))


class _NoRopeVLM:
    """mlx_vlm Qwen3.5 shape: ``rotary_emb`` instead of ``.rope``."""

    def __init__(self):
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=object()))]
        self.seen = []

    def position_ids_for_absolute(self, positions):
        return positions.reshape(1, -1)

    def __call__(self, input_ids, cache=None, position_ids=None):
        import mlx.core as mx

        length = input_ids.shape[1]
        if position_ids is None:
            # mlx_vlm's fallback: a contiguous run from the cache offset.
            start = cache[0].offset
            position_ids = mx.arange(start, start + length).reshape(1, -1)
        self.seen.extend(position_ids.reshape(-1).tolist())
        for layer in cache:
            layer.offset += length
        return mx.zeros((1, length, 8))


_SELECTED = [0, 5, 6, 17, 40, 41, 99]


@pytest.mark.parametrize("offset", [0, 12288])
def test_sparse_prefill_keeps_selected_positions_without_rope(offset):
    import mlx.core as mx

    from omlx.patches.specprefill import sparse_prefill

    model = _NoRopeVLM()
    cache = [_FakeCacheLayer(offset)]
    sparse_prefill(
        model,
        mx.arange(120),
        mx.array(_SELECTED),
        cache,
        step_size=4,
        position_offset=offset,
    )

    assert model.seen == [index + offset for index in _SELECTED]
    assert model._specprefill_decode_adjustment == 120 - len(_SELECTED)


def test_adapter_passes_explicit_position_ids_through():
    import mlx.core as mx

    from omlx.models.vlm import VLMModelAdapter

    language_model = MagicMock()
    language_model.return_value = SimpleNamespace(logits=mx.zeros((1, 3, 8)))
    adapter = VLMModelAdapter(MagicMock(language_model=language_model))
    adapter._uses_mrope = True
    adapter._batch_rope_deltas = mx.array([0.0])

    supplied = adapter.position_ids_for_absolute(mx.array([3, 9, 40]))
    assert supplied.shape == (3, 1, 3)
    adapter(mx.array([[1, 2, 3]]), cache=[_FakeCacheLayer(512)], position_ids=supplied)

    assert mx.array_equal(language_model.call_args.kwargs["position_ids"], supplied)


def test_sparse_prefill_rope_models_keep_the_wrapper_path():
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
        def __init__(self):
            self.layers = [SimpleNamespace(self_attn=SimpleNamespace(rope=_Rope()))]

        def position_ids_for_absolute(self, positions):
            return positions.reshape(1, -1)

        def __call__(self, input_ids, cache=None, **kwargs):
            assert isinstance(self.layers[0].self_attn.rope, _PositionMappedRoPE)
            seen.append(kwargs)
            for layer in cache:
                layer.offset += input_ids.shape[1]
            return mx.zeros((1, input_ids.shape[1], 8))

    model = _RopeModel()
    sparse_prefill(
        model,
        mx.arange(120),
        mx.array(_SELECTED),
        [_FakeCacheLayer()],
        step_size=4,
        position_offset=0,
    )

    assert seen and all(kwargs == {} for kwargs in seen)
    assert model._specprefill_decode_adjustment is None
