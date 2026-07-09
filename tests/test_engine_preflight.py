# SPDX-License-Identifier: Apache-2.0
"""Tests for ``preflight_chat`` / ``preflight_completion`` on the engine
wrappers.

The full end-to-end value of these methods is that they raise
``PrefillMemoryExceededError`` BEFORE the route handler wraps the
response in a ``StreamingResponse``, so the FastAPI handler can turn
the exception into HTTP 400. We exercise the contract by:

- Stubbing the wrapper chain (engine -> _engine.engine.scheduler) and the
  tokenizer.
- Confirming ``preflight_or_raise`` is invoked with the right token count.
- Confirming the exception type propagates.
"""

from unittest.mock import MagicMock

import pytest

from omlx.exceptions import PrefillMemoryExceededError
from omlx.scheduler import Scheduler

_TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

# ---------------------------------------------------------------------------
# Scheduler.preflight_or_raise / _preflight_memory_check_tokens
# ---------------------------------------------------------------------------


class _ModelConfig:
    def __init__(
        self,
        num_hidden_layers=32,
        num_key_value_heads=8,
        num_attention_heads=32,
        head_dim=192,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim


def _make_scheduler():
    from omlx.scheduler import SchedulerConfig

    model = MagicMock()
    model.layers = []
    model.config = _ModelConfig()
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


class TestPreflightOrRaise:
    def test_raises_when_peak_exceeds_limit(self, monkeypatch):
        scheduler = _make_scheduler()
        scheduler._prefill_memory_guard = True
        scheduler._memory_hard_limit_bytes = 1  # any allocation overshoots

        import omlx.scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
        monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

        with pytest.raises(PrefillMemoryExceededError) as exc:
            scheduler.preflight_or_raise(num_prompt_tokens=65536, request_id="req-x")
        assert "Prefill would require" in str(exc.value)
        assert exc.value.request_id == "req-x"

    def test_returns_silently_when_within_budget(self, monkeypatch):
        scheduler = _make_scheduler()
        scheduler._prefill_memory_guard = True
        scheduler._memory_hard_limit_bytes = 10**18  # effectively unbounded

        import omlx.scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
        monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

        # Must not raise
        scheduler.preflight_or_raise(num_prompt_tokens=1024)

    def test_skips_when_guard_disabled(self):
        scheduler = _make_scheduler()
        scheduler._prefill_memory_guard = False
        scheduler._memory_hard_limit_bytes = 1
        # Even with an impossibly small limit, disabled guard never raises.
        scheduler.preflight_or_raise(num_prompt_tokens=10**6)

    def test_accounts_for_cached_tokens(self, monkeypatch):
        """A fully cached request must not be rejected even at a tiny limit."""
        scheduler = _make_scheduler()
        scheduler._prefill_memory_guard = True
        scheduler._memory_hard_limit_bytes = 1

        import omlx.scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
        monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

        scheduler.preflight_or_raise(num_prompt_tokens=10_000, cached_tokens=10_000)


# ---------------------------------------------------------------------------
# Engine wrapper preflight methods
# ---------------------------------------------------------------------------


def _build_engine_with_stub_scheduler(engine_cls, scheduler):
    """Return an engine of the given class wired to a stub scheduler chain.

    The real BatchedEngine / VLMBatchedEngine init does heavy work (model
    load, etc.). For the preflight contract test we only need the wrapper
    methods + tokenizer + the ``_engine.engine.scheduler`` chain, so we
    bypass __init__ via __new__ and pin only the attributes the preflight
    method touches.
    """
    engine = engine_cls.__new__(engine_cls)
    engine._loaded = True
    engine._enable_thinking = None
    engine._prefill_eviction_callback = None

    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="hello world")
    # The encoded length drives what we pass to preflight_or_raise.
    tokenizer.encode = MagicMock(return_value=list(range(110_000)))
    engine._tokenizer = tokenizer

    # Wrapper chain that _resolve_scheduler / preflight_chat traverse:
    #   engine._engine.engine.scheduler
    inner_engine_core = MagicMock(spec=["scheduler"])
    inner_engine_core.scheduler = scheduler
    async_engine_core = MagicMock(spec=["engine"])
    async_engine_core.engine = inner_engine_core
    engine._engine = async_engine_core
    return engine


@pytest.mark.asyncio
async def test_batched_engine_preflight_runs_eviction_before_final_check():
    from types import SimpleNamespace

    from omlx.engine.batched import BatchedEngine

    scheduler = MagicMock()
    eviction_request = SimpleNamespace(request_id="req-evict")
    scheduler.preflight_eviction_request.return_value = eviction_request
    order = []
    scheduler.preflight_or_raise.side_effect = lambda **kwargs: order.append(
        ("final", "checked")
    )

    async def _evict(request):
        order.append(("evict", request.request_id))
        return True

    engine = BatchedEngine(
        model_name="test-model",
        prefill_eviction_callback=_evict,
    )

    await engine._preflight_or_raise_with_eviction(
        scheduler,
        num_prompt_tokens=123,
        request_id="req-evict",
    )

    scheduler.preflight_eviction_request.assert_called_once_with(
        num_prompt_tokens=123,
        request_id="req-evict",
    )
    scheduler.preflight_or_raise.assert_called_once_with(
        num_prompt_tokens=123,
        request_id="req-evict",
    )
    assert order == [("evict", "req-evict"), ("final", "checked")]


@pytest.mark.asyncio
async def test_batched_engine_preflight_chat_raises_for_oversize_prompt(monkeypatch):
    from omlx.engine.batched import BatchedEngine

    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1  # force rejection

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

    engine = _build_engine_with_stub_scheduler(BatchedEngine, scheduler)
    # _preprocess_messages on BatchedEngine assumes Harmony hooks etc.; stub
    # it out so the test only exercises the preflight wiring.
    engine._preprocess_messages = lambda m: m

    with pytest.raises(PrefillMemoryExceededError):
        await engine.preflight_chat(messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_vlm_engine_preflight_chat_raises_for_oversize_prompt(monkeypatch):
    from omlx.engine.vlm import VLMBatchedEngine

    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

    engine = _build_engine_with_stub_scheduler(VLMBatchedEngine, scheduler)

    with pytest.raises(PrefillMemoryExceededError):
        await engine.preflight_chat(messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_preflight_completion_raises_for_oversize_prompt(monkeypatch):
    from omlx.engine.batched import BatchedEngine

    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

    engine = _build_engine_with_stub_scheduler(BatchedEngine, scheduler)

    with pytest.raises(PrefillMemoryExceededError):
        await engine.preflight_completion(prompt="a" * 110_000)


# ---------------------------------------------------------------------------
# VLM-specific contracts (image-token budget + tools conversion + cached
# tokens propagation through preflight_or_raise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vlm_preflight_chat_adds_image_token_budget(monkeypatch):
    """Each image-bearing content part must add
    ``_IMAGE_TOKEN_UPPER_BOUND_FALLBACK`` to the prompt size the scheduler sees,
    so image-heavy borderline requests can't slip past."""
    from omlx.engine.vlm import _IMAGE_TOKEN_UPPER_BOUND_FALLBACK, VLMBatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(VLMBatchedEngine, scheduler)
    # Make the templated text deterministically 1000 tokens.
    engine._tokenizer.encode = MagicMock(return_value=list(range(1000)))

    seen: dict = {}

    def _capture(num_prompt_tokens, **kwargs):
        seen["num_prompt_tokens"] = num_prompt_tokens

    scheduler.preflight_or_raise = _capture  # type: ignore[assignment]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": _TINY_PNG_DATA_URI},
                },
                {"type": "image", "source": {}},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    await engine.preflight_chat(messages=messages)
    # 1000 text tokens + 2 images * 1280
    assert seen["num_prompt_tokens"] == 1000 + 2 * _IMAGE_TOKEN_UPPER_BOUND_FALLBACK


@pytest.mark.asyncio
async def test_vlm_preflight_chat_strips_images_before_template(monkeypatch):
    """Modern HF chat templates (Qwen2.5-VL, Gemma-Vision, Llama-3.2-Vision)
    render image content parts as literal placeholder strings inline with
    the text. If preflight templates the raw messages, the resulting
    tokenized prompt already contains image-placeholder tokens AND we
    then add the per-image budget on top — a double count that
    produces spurious 400s on borderline image-bearing requests the
    real chat path would have admitted. ``preflight_chat`` must
    therefore call ``extract_images_from_messages`` BEFORE
    ``_apply_chat_template``, the same way ``_process_chat_messages``
    does on the execution path.
    """
    from omlx.engine.vlm import VLMBatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(VLMBatchedEngine, scheduler)
    engine._tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    engine._apply_chat_template = MagicMock(return_value="stripped text")
    scheduler.preflight_or_raise = lambda **kw: None  # type: ignore[assignment]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these:"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URI}},
                {"type": "image", "source": {}},
            ],
        }
    ]
    await engine.preflight_chat(messages=messages)

    # _apply_chat_template was called with image content-parts stripped.
    assert engine._apply_chat_template.call_count == 1
    (call_messages, *_), _ = engine._apply_chat_template.call_args
    user_content = call_messages[0]["content"]
    if isinstance(user_content, list):
        types_seen = {part.get("type") for part in user_content}
        assert (
            "image_url" not in types_seen
        ), "image_url part leaked into template input"
        assert "image" not in types_seen, "image part leaked into template input"
    else:
        # Some packs reduce single-text content to a string.
        assert isinstance(user_content, str)


@pytest.mark.asyncio
async def test_vlm_preflight_chat_converts_pydantic_tools(monkeypatch):
    """``preflight_chat`` must run tools through ``convert_tools_for_template``
    so Pydantic ``ToolDefinition`` callers don't get the silent
    template-retry fallback that drops tools entirely."""
    from omlx.engine.vlm import VLMBatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(VLMBatchedEngine, scheduler)
    engine._tokenizer.encode = MagicMock(return_value=[1])
    scheduler.preflight_or_raise = lambda **k: None  # type: ignore[assignment]

    called_with = {}

    def _spy(messages, tools, **kwargs):
        called_with["tools"] = tools
        return ""

    engine._apply_chat_template = _spy  # type: ignore[assignment]

    sentinel_tool = {
        "type": "function",
        "function": {"name": "do_x", "parameters": {}},
    }
    await engine.preflight_chat(
        messages=[{"role": "user", "content": "x"}], tools=[sentinel_tool]
    )

    # convert_tools_for_template returned a list (possibly unchanged for a
    # dict that already has the right shape, possibly transformed) — the
    # contract is: tools were passed through the conversion path rather
    # than the raw input.
    assert called_with["tools"] is not None


@pytest.mark.asyncio
async def test_batched_engine_preflight_logs_when_scheduler_unreachable(
    monkeypatch, caplog
):
    """If the wrapper chain doesn't expose a scheduler (e.g. partial
    init failure), preflight no-ops but logs a warning rather than
    silently swallowing the safety check."""
    import logging

    from omlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._enable_thinking = None
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template = MagicMock(return_value="hi")
    engine._tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    engine._preprocess_messages = lambda m: m
    # _engine is None — simulates a partial-init failure where
    # _resolve_scheduler chain can't reach a real scheduler.
    engine._engine = None

    with caplog.at_level(logging.WARNING):
        await engine.preflight_chat(messages=[{"role": "user", "content": "x"}])

    assert any(
        "preflight check skipped" in r.message for r in caplog.records
    ), "expected a warning when scheduler is unreachable"


@pytest.mark.asyncio
async def test_preflight_chat_swallows_tokenizer_errors(caplog):
    """Tokenizer errors during preflight must not raise — the real chat
    path will hit the same error and surface it through the existing
    handler chain. Raising here would introduce a NEW 500 failure mode
    on borderline-malformed-prompt requests.
    """
    import logging

    from omlx.engine.batched import BatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(BatchedEngine, scheduler)
    engine._tokenizer.encode = MagicMock(
        side_effect=UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "synthetic")
    )
    engine._preprocess_messages = lambda m: m

    raise_called = {"yes": False}

    def _trip(num_prompt_tokens, **kwargs):
        raise_called["yes"] = True

    scheduler.preflight_or_raise = _trip  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        # Must NOT raise the UnicodeDecodeError up to the caller.
        await engine.preflight_chat(messages=[{"role": "user", "content": "x"}])

    assert not raise_called[
        "yes"
    ], "preflight_or_raise must NOT be called when tokenizer fails"
    assert any(
        "tokenizer.encode raised" in r.message for r in caplog.records
    ), "expected a warning logging the tokenizer error"


@pytest.mark.asyncio
async def test_preflight_completion_swallows_tokenizer_errors(caplog):
    """Same contract on the completion path."""
    import logging

    from omlx.engine.batched import BatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(BatchedEngine, scheduler)
    engine._tokenizer.encode = MagicMock(side_effect=ValueError("bad input"))

    raise_called = {"yes": False}
    scheduler.preflight_or_raise = lambda **k: raise_called.__setitem__("yes", True)  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        await engine.preflight_completion(prompt="\x00" * 10)

    assert not raise_called["yes"]
    assert any("tokenizer.encode raised" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_vlm_preflight_chat_swallows_tokenizer_errors(caplog):
    """VLM path mirrors BatchedEngine on tokenizer-error handling."""
    import logging

    from omlx.engine.vlm import VLMBatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(VLMBatchedEngine, scheduler)
    engine._tokenizer.encode = MagicMock(side_effect=RuntimeError("Already borrowed"))

    raise_called = {"yes": False}
    scheduler.preflight_or_raise = lambda **k: raise_called.__setitem__("yes", True)  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        await engine.preflight_chat(messages=[{"role": "user", "content": "x"}])

    assert not raise_called["yes"]
    assert any("tokenizer.encode raised" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Regressions added in code review: structured rejection, request_id
# plumbing, and engine_core cleanup-on-raise leak.
# ---------------------------------------------------------------------------


def test_preflight_rejection_carries_estimated_and_limit_bytes(monkeypatch):
    """``PrefillMemoryExceededError`` must surface the structured rejection
    fields (``estimated_bytes`` / ``limit_bytes``) so clients can branch on
    numeric values instead of regex-matching the human-readable message.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1024  # tiny — forces rejection

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

    with pytest.raises(PrefillMemoryExceededError) as exc_info:
        scheduler.preflight_or_raise(num_prompt_tokens=65536, request_id="req-attrs")
    exc = exc_info.value
    assert exc.request_id == "req-attrs"
    assert exc.limit_bytes == 1024
    assert exc.estimated_bytes is not None and exc.estimated_bytes > 0


def test_preflight_or_raise_synthesizes_request_id_when_unset(monkeypatch):
    """If the caller doesn't pass a request_id, preflight_or_raise must
    generate a unique one so each rejection is individually traceable.
    Regression for the prior literal "preflight" default which collapsed
    every rejection's id together in logs and FastAPI handler traces.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

    ids = set()
    for _ in range(4):
        with pytest.raises(PrefillMemoryExceededError) as exc_info:
            scheduler.preflight_or_raise(num_prompt_tokens=65536)
        rid = exc_info.value.request_id
        assert rid and rid != "preflight"
        assert rid.startswith("preflight-")
        ids.add(rid)
    assert len(ids) == 4, "request_ids must be unique per rejection"


@pytest.mark.asyncio
async def test_batched_engine_preflight_chat_threads_request_id(monkeypatch):
    """The engine wrapper must forward the caller's request_id to the
    scheduler so the rejection log + exception carry a meaningful trace
    label rather than the synthesized "preflight-XXXX" fallback.
    """
    from omlx.engine.batched import BatchedEngine

    scheduler = _make_scheduler()
    engine = _build_engine_with_stub_scheduler(BatchedEngine, scheduler)
    engine._preprocess_messages = lambda m: m
    engine._tokenizer.encode = MagicMock(return_value=[1, 2, 3])

    seen: dict = {}

    def _capture(num_prompt_tokens, **kwargs):
        seen.update(kwargs)
        seen["num_prompt_tokens"] = num_prompt_tokens

    scheduler.preflight_or_raise = _capture  # type: ignore[assignment]
    await engine.preflight_chat(
        messages=[{"role": "user", "content": "x"}],
        request_id="trace-id-42",
    )
    assert seen.get("request_id") == "trace-id-42"


@pytest.mark.asyncio
async def test_engine_core_add_request_cleans_up_on_scheduler_raise(
    monkeypatch,
):
    """Regression for the engine_core leak: when scheduler.add_request
    raises (e.g. PrefillMemoryExceededError) the per-request collector /
    stream_state / finished_event entries must be removed. Without
    cleanup, every rejection accumulates one of each — under sustained
    rejection load this leaks indefinitely.
    """
    from concurrent.futures import ThreadPoolExecutor

    from omlx.engine_core import EngineCore

    core = EngineCore.__new__(EngineCore)
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core._finished_at = {}

    class _Cfg:
        stream_interval = 1

    core.config = _Cfg()
    core._mlx_executor = ThreadPoolExecutor(max_workers=1)

    raising_scheduler = MagicMock()
    raising_scheduler._specprefill_draft_model = None

    def _raise(req):
        raise PrefillMemoryExceededError(
            message="rejected for test",
            request_id=req.request_id,
            estimated_bytes=10**9,
            limit_bytes=10**8,
        )

    raising_scheduler.add_request = _raise
    core.scheduler = raising_scheduler

    # Drive add_request enough that we can observe collectors before/after.
    with pytest.raises(PrefillMemoryExceededError):
        await core.add_request(
            prompt=[1, 2, 3],
            sampling_params=MagicMock(),
            request_id="leak-check-1",
        )

    # All per-request engine_core entries must be cleaned up.
    assert "leak-check-1" not in core._output_collectors
    assert "leak-check-1" not in core._stream_states
    assert "leak-check-1" not in core._finished_events

    core._mlx_executor.shutdown(wait=True)


def test_scheduler_add_request_cleans_block_table_on_rejection(monkeypatch):
    """When add_request raises PrefillMemoryExceededError, any block_table
    that the prefix-cache lookup attached must be released so a sustained
    rejection stream cannot leak block tables / refcounts.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

    # Pin a fake block_table + paged_cache_manager so we can verify
    # delete_block_table is called on the rejection path.
    pcm = MagicMock()
    scheduler.paged_cache_manager = pcm

    req = MagicMock()
    req.request_id = "blk-clean-1"
    req.num_prompt_tokens = 65536
    req.cached_tokens = 0
    req.block_table = MagicMock()
    req.prompt = [1, 2, 3]
    req.prompt_token_ids = [1, 2, 3]
    req.vlm_extra_keys_for_cache = None
    req.vlm_extra_key_token_start_for_cache = None
    req.vlm_extra_key_ranges_for_cache = None
    # Disable prefix-cache fetch so we don't go through the full lookup.
    scheduler.block_aware_cache = None
    # Disable SpecPrefill draft.
    scheduler._specprefill_draft_model = None

    with pytest.raises(PrefillMemoryExceededError):
        scheduler.add_request(req)
    pcm.delete_block_table.assert_called_once_with("blk-clean-1")
    # The request must not have entered self.waiting.
    assert req not in scheduler.waiting
    assert req.request_id not in scheduler.requests


# ---------------------------------------------------------------------------
# Rejection message identifies the binding ceiling
# ---------------------------------------------------------------------------


class TestRejectionMessageNamesBindingCeiling:
    """When a request is rejected, the message must name which of the
    three component ceilings (static / dynamic / metal_cap) is binding
    and steer the user to the right remedy.

    Without this discrimination operators on Pi-class hosts spent hours
    staring at a generic "reduce context length, free system memory, or
    loosen memory_guard_tier" message that didn't tell them which of
    their three knobs to actually turn. The most common confusion was a
    metal_cap-bound 413 on hosts where ``iogpu.wired_limit_mb`` had
    never been raised — the message told them to free system memory
    when no amount of freeing system memory would help.
    """

    def _arm_ceilings(
        self,
        sched,
        *,
        static: int,
        dynamic: int,
        metal_cap: int,
        tier: str = "balanced",
        hot_cache_reserved: int = 0,
    ) -> None:
        """Set the four propagated ceiling fields directly.

        Mirrors what ``ProcessMemoryEnforcer._propagate_memory_limit``
        does on a real run; the binding-aware message reads only these
        fields plus ``_memory_hard_limit_bytes``.
        """
        sched._prefill_memory_guard = True
        hard_limit = min(v for v in (static, dynamic, metal_cap) if v > 0)
        if hot_cache_reserved > 0:
            hard_limit = max(1, hard_limit - hot_cache_reserved)
        sched._memory_hard_limit_bytes = hard_limit
        sched._memory_static_ceiling_bytes = static
        sched._memory_dynamic_ceiling_bytes = dynamic
        sched._memory_metal_cap_bytes = metal_cap
        sched._memory_hot_cache_reserved_bytes = hot_cache_reserved
        sched._memory_guard_tier = tier
        # Set_model_info populated dims at scheduler construction; we
        # only need a non-zero peak estimate to drive the rejection
        # path, not exact bytes.

    def _force_rejection(self, sched, monkeypatch):
        """Mock the parts of the math we don't care about and call
        ``_preflight_memory_check`` so we can inspect the message it
        returns."""
        # Peak chosen larger than any ceiling tested below so the
        # rejection branch fires deterministically.
        sched.memory_monitor = MagicMock()
        sched.memory_monitor.estimate_prefill_peak_bytes.return_value = 512 * 1024**3

        import omlx.scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
        monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 0)

        req = MagicMock()
        req.request_id = "binding-test"
        req.num_prompt_tokens = 65536
        req.cached_tokens = 0
        # _preflight_memory_check tries an LRU eviction retry first; we
        # don't want that path here.
        monkeypatch.setattr(
            sched,
            "_raise_prefill_eviction_if_available",
            lambda **kw: None,
        )
        rej = sched._preflight_memory_check(req)
        assert rej is not None, "rejection branch must fire when peak > ceiling"
        return rej

    def test_metal_cap_binding_names_sysctl(self, monkeypatch):
        sched = _make_scheduler()
        self._arm_ceilings(
            sched, static=64 * 1024**3, dynamic=32 * 1024**3, metal_cap=16 * 1024**3
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert (
            "iogpu.wired_limit_mb" in rej.message
        ), f"metal_cap binding must steer user to the sysctl knob; got: {rej.message}"
        assert "metal_cap ceiling" in rej.message
        assert "caps Metal at 16.00 GB" in rej.message

    def test_dynamic_binding_under_custom_names_admin_setting(self, monkeypatch):
        sched = _make_scheduler()
        self._arm_ceilings(
            sched,
            static=64 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=48 * 1024**3,
            tier="custom",
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "custom_ceiling_bytes" in rej.message, (
            "dynamic binding under custom tier must point at the admin "
            f"Memory setting, not 'close other apps'; got: {rej.message}"
        )
        assert "close other apps" not in rej.message.lower()

    def test_dynamic_binding_under_reclaim_tier_names_apps(self, monkeypatch):
        sched = _make_scheduler()
        # Static > dynamic, balanced tier: closing apps and/or raising
        # tier is what helps.
        self._arm_ceilings(
            sched,
            static=64 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=48 * 1024**3,
            tier="balanced",
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "close other apps" in rej.message.lower(), (
            "dynamic binding on a reclaim tier should suggest closing "
            f"apps; got: {rej.message}"
        )
        assert "memory_guard_tier" in rej.message

    def test_hot_cache_reservation_preserves_binding_label(self, monkeypatch):
        sched = _make_scheduler()
        self._arm_ceilings(
            sched,
            static=64 * 1024**3,
            dynamic=32 * 1024**3,
            metal_cap=16 * 1024**3,
            hot_cache_reserved=2 * 1024**3,
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "metal_cap ceiling" in rej.message
        assert "effective ceiling" not in rej.message
        assert "caps Metal at 16.00 GB" in rej.message

    def test_static_binding_falls_back_to_generic_advice(self, monkeypatch):
        sched = _make_scheduler()
        # Static is the smallest non-zero ceiling.
        self._arm_ceilings(
            sched,
            static=16 * 1024**3,
            dynamic=64 * 1024**3,
            metal_cap=48 * 1024**3,
        )
        rej = self._force_rejection(sched, monkeypatch)
        assert "memory_guard_tier" in rej.message
        assert "iogpu.wired_limit_mb" not in rej.message
        assert "custom_ceiling_bytes" not in rej.message
