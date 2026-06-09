# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests that the prefill memory guard is wired up.

Until 2026-05-15 the guard was dead code: ``Scheduler.memory_monitor`` was
left as ``None`` and ``_set_model_info_for_monitor`` had zero callers, so
``_preflight_memory_check`` short-circuited at the ``memory_monitor is None``
gate even when ``_prefill_memory_guard`` was flipped on by the enforcer.

These tests pin the wiring so a future refactor cannot silently revert it.
"""

from unittest.mock import MagicMock, patch

from omlx.memory_monitor import MemoryMonitor
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


class _ModelConfig:
    """Minimal config object exposing the fields the estimator reads."""

    def __init__(
        self,
        num_hidden_layers: int | None = 32,
        num_key_value_heads: int = 8,
        num_attention_heads: int = 32,
        head_dim: int = 192,  # > 128 → SDPA fallback path (the panic-prone one)
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim


def _make_scheduler() -> Scheduler:
    model = MagicMock()
    model.layers = []
    model.config = _ModelConfig()
    # Strip make_cache so the KVCache-counting branch in
    # _set_model_info_for_monitor doesn't try to iterate a MagicMock.
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


def _make_request(prompt_tokens: int = 65536) -> Request:
    req = Request(
        request_id="req-large",
        prompt=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    req.prompt_token_ids = list(range(prompt_tokens))
    req.num_prompt_tokens = prompt_tokens
    return req


def test_scheduler_init_instantiates_memory_monitor():
    scheduler = _make_scheduler()
    assert isinstance(scheduler.memory_monitor, MemoryMonitor)


def test_scheduler_init_populates_estimator_dims():
    scheduler = _make_scheduler()
    monitor = scheduler.memory_monitor
    assert monitor is not None
    assert monitor._num_attention_heads == 32
    assert monitor._head_dim == 192
    assert monitor._num_layers == 32
    assert monitor._num_kv_heads == 8


def test_estimator_produces_nonzero_peak_after_init():
    scheduler = _make_scheduler()
    assert scheduler.memory_monitor is not None
    peak = scheduler.memory_monitor.estimate_prefill_peak_bytes(65536, 2048)
    assert peak > 0


def test_preflight_positive_control_passes_normal_request():
    """Positive-control: a normal prompt under a generous limit must NOT
    be rejected. Defends against an accidental sign-flip on the
    threshold comparison in _preflight_memory_check.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    # Huge limit — even a multi-GB peak fits comfortably.
    scheduler._memory_hard_limit_bytes = 10**18
    with patch("omlx.scheduler.mx.get_active_memory", return_value=0), patch(
        "omlx.scheduler.get_phys_footprint", return_value=0
    ):
        assert scheduler._preflight_memory_check(_make_request(32768)) is None


def test_preflight_rejects_when_estimated_peak_exceeds_hard_limit():
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1  # any allocation exceeds

    with patch("omlx.scheduler.mx.get_active_memory", return_value=0), patch(
        "omlx.scheduler.get_phys_footprint", return_value=0
    ):
        rejection = scheduler._preflight_memory_check(_make_request(65536))

    assert rejection is not None
    assert "Prefill would require" in rejection.message
    assert "KV+SDPA" in rejection.message
    assert rejection.estimated_bytes > 0
    assert rejection.limit_bytes == 1


def test_preflight_returns_none_when_guard_disabled():
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = False
    scheduler._memory_hard_limit_bytes = 1
    assert scheduler._preflight_memory_check(_make_request(65536)) is None


def test_preflight_returns_none_when_request_fully_cached():
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1
    req = _make_request(1000)
    req.cached_tokens = 1000
    # Fully cached: no new tokens to prefill, no peak to estimate.
    assert scheduler._preflight_memory_check(req) is None


def test_preflight_rejects_heavily_cached_long_context():
    """Regression for M3: a request whose suffix is small but whose
    *full* prompt is long must still trip the guard, because the SDPA
    scores tensor spans the full prompt (cached + new), not just the
    new tokens. Previously the estimator passed only new_tokens to the
    scores formula and the heavily-cached path slipped through.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    # Tight limit so even a partial prefill against a 100k KV trips it.
    scheduler._memory_hard_limit_bytes = 100 * 1024**2  # 100 MB
    req = _make_request(100_000)
    req.cached_tokens = 99_000  # only 1k new tokens but kv_len = 100k
    with patch("omlx.scheduler.mx.get_active_memory", return_value=0), patch(
        "omlx.scheduler.get_phys_footprint", return_value=0
    ):
        error = scheduler._preflight_memory_check(req)
    assert error is not None, (
        "guard must trip on heavily-cached long-context: SDPA scores "
        "still span the full prompt"
    )


def test_preflight_rejects_uncached_long_context():
    """Symmetric to test_preflight_rejects_heavily_cached_long_context:
    a request with mostly NEW tokens (no cache) at a 100k prompt must
    also trip the guard. This locks in the SDPA-fallback K-dim formula
    in both directions; if a future refactor regressed the cached path
    OR the uncached path, only one of these two tests would fail.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 100 * 1024**2  # 100 MB
    req = _make_request(100_000)
    req.cached_tokens = 1_000  # almost everything is new
    with patch("omlx.scheduler.mx.get_active_memory", return_value=0), patch(
        "omlx.scheduler.get_phys_footprint", return_value=0
    ):
        error = scheduler._preflight_memory_check(req)
    assert error is not None, (
        "guard must trip on uncached long-context too"
    )


class _VLMConfig:
    """Top-level VLM config whose LM dims live under text_config (Qwen3.6-VL,
    Gemma-4 layout). The top-level surface deliberately has no num_hidden_layers,
    so this exercises the nested-config descent path."""

    def __init__(self):
        self.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        self.model_type = "qwen3_5_moe"
        self.text_config = _ModelConfig(
            num_hidden_layers=40,
            num_key_value_heads=2,
            num_attention_heads=16,
            head_dim=256,  # > 128 → SDPA fallback (the panic-prone path)
        )


def _make_vlm_scheduler() -> Scheduler:
    model = MagicMock()
    model.layers = []
    model.config = _VLMConfig()
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


def test_vlm_nested_config_populates_estimator_dims():
    """Regression: VLM models nest LM dims under config.text_config — the
    estimator must follow the sub-config or it stays silently dead at
    runtime (no Model info set log, peak == 0, guard short-circuits)."""
    scheduler = _make_vlm_scheduler()
    monitor = scheduler.memory_monitor
    assert monitor is not None
    assert monitor._num_layers == 40
    assert monitor._num_kv_heads == 2
    assert monitor._num_attention_heads == 16
    assert monitor._head_dim == 256


def test_vlm_estimator_produces_nonzero_peak():
    scheduler = _make_vlm_scheduler()
    assert scheduler.memory_monitor is not None
    # 90k tokens at head_dim=256 / n_q=16 should yield a multi-GiB peak via
    # the SDPA-fallback branch.
    peak = scheduler.memory_monitor.estimate_prefill_peak_bytes(90000, 2048)
    assert peak > 10 * 1024 * 1024 * 1024  # > 10 GiB


def test_rejection_releases_block_aware_cache_when_present():
    """Regression for the prefix-cache leak found in review: a request
    rejected by the prefill memory guard had its ref counts on every
    prefix-matched paged block (and its ``request_tables`` entry)
    incremented by ``add_request → fetch_cache``. Without releasing
    them on the rejection path, those refs pin the paged cache and
    compound the very memory pressure that triggered the rejection.
    """
    scheduler = _make_scheduler()
    block_aware_cache = MagicMock()
    paged_cache_manager = MagicMock()
    scheduler.block_aware_cache = block_aware_cache
    scheduler.paged_cache_manager = paged_cache_manager

    scheduler._release_paged_cache_for_request("req-leak")

    # When block_aware_cache is present it owns the cleanup chain
    # (release_cache → paged_cache_manager.delete_block_table).
    block_aware_cache.release_cache.assert_called_once_with("req-leak")
    paged_cache_manager.delete_block_table.assert_not_called()


def test_rejection_releases_paged_cache_when_no_prefix_cache():
    """When block_aware_cache is absent but a paged_cache_manager is
    wired up, the rejection path must call ``delete_block_table``
    directly — otherwise the request's ``request_tables`` entry and
    every block ref it holds leaks for the process lifetime.
    """
    scheduler = _make_scheduler()
    scheduler.block_aware_cache = None
    paged_cache_manager = MagicMock()
    scheduler.paged_cache_manager = paged_cache_manager

    scheduler._release_paged_cache_for_request("req-leak")

    paged_cache_manager.delete_block_table.assert_called_once_with("req-leak")


def test_rejection_releases_draft_prefix_cache_for_specprefill_requests():
    """SpecPrefill primes an independent ``_draft_prefix_cache`` in
    ``_try_specprefill_scoring`` (via its own ``fetch_cache``).
    The rejection path must release that draft cache too, symmetric
    to the target cache — otherwise a rejected SpecPrefill request
    leaks every draft-block ref and orphans its ``_request_tables``
    entry exactly like the target-cache bug this commit fixes."""
    scheduler = _make_scheduler()
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = MagicMock()
    draft_cache = MagicMock()
    scheduler._draft_prefix_cache = draft_cache

    scheduler._release_paged_cache_for_request("req-spec-leak")

    draft_cache.release_cache.assert_called_once_with("req-spec-leak")


def test_rejection_helper_noop_without_caches():
    """No caches wired up → helper must not raise. Embedded test
    schedulers (this file's ``_make_scheduler``) build without paged
    caches; the helper must be safe to call unconditionally on the
    rejection path."""
    scheduler = _make_scheduler()
    scheduler.block_aware_cache = None
    scheduler.paged_cache_manager = None
    # Must not raise.
    scheduler._release_paged_cache_for_request("req-leak")


def test_preflight_rejection_path_invokes_release_helper():
    """End-to-end wiring: the preflight rejection in ``_schedule_waiting``
    must invoke the cache-release helper before popping
    ``self.requests``. Pins the call-site fix for the leak — without
    this hook the helper could exist but never be called from the hot
    path.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1  # forces rejection

    req = _make_request(65536)
    scheduler.requests[req.request_id] = req
    scheduler.waiting.append(req)

    # Make the rejection branch take effect even before
    # _ensure_batch_generator runs — patch the preflight check to
    # short-circuit on entry and keep this test independent of the
    # batch-generator construction path.
    from omlx.scheduler import _PreflightRejection

    def _force_reject(_request):
        return _PreflightRejection(
            message="forced rejection for test",
            estimated_bytes=1,
            limit_bytes=1,
        )

    with patch.object(
        scheduler, "_release_paged_cache_for_request"
    ) as release_spy, patch.object(
        scheduler, "_preflight_memory_check", side_effect=_force_reject
    ), patch.object(
        scheduler, "_ensure_batch_generator", return_value=None
    ):
        # Pretend a batch_generator exists so the loop continues past
        # the ``if self.batch_generator is None: break`` guard.
        scheduler.batch_generator = MagicMock()
        scheduler._schedule_waiting()

    release_spy.assert_any_call(req.request_id)
    assert req.request_id not in scheduler.requests


def test_vlm_preflight_rejects_oversize_request():
    scheduler = _make_vlm_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 40 * 1024 * 1024 * 1024  # 40 GiB hard limit

    with patch("omlx.scheduler.mx.get_active_memory", return_value=28 * 1024 ** 3), patch(
        "omlx.scheduler.get_phys_footprint", return_value=28 * 1024 ** 3
    ):
        # 100k tokens at head_dim=256 should push (28 GiB baseline + KV+SDPA
        # peak) past the 40 GiB limit.
        rejection = scheduler._preflight_memory_check(_make_request(100000))

    assert rejection is not None
    assert "KV+SDPA" in rejection.message


# ---------------------------------------------------------------------------
# Config-descent edge cases (M3 in the upstream review of this commit)
# ---------------------------------------------------------------------------


class _VLMTopLevelVisionConfig:
    """Top-level config has num_hidden_layers that refers to the *vision*
    encoder. The estimator must descend into text_config rather than
    accept the top-level value, otherwise it miscalibrates the SDPA peak.
    """

    def __init__(self):
        self.architectures = ["FakeVisionLM"]
        self.model_type = "fake_vlm"
        # Vision encoder block count surfaces at top-level on some
        # HF auto-wrapped packs — accepting this would silently use
        # 27 layers / wrong heads for the LM math.
        self.num_hidden_layers = 27
        self.num_attention_heads = 16  # vision attn heads
        self.head_dim = 80  # vision head_dim (< 128, different SDPA path)
        self.text_config = _ModelConfig(
            num_hidden_layers=40,
            num_key_value_heads=2,
            num_attention_heads=16,
            head_dim=256,  # LM head_dim → SDPA-fallback path
        )


def test_vlm_descent_prefers_text_config_over_top_level_vision_field():
    """Regression: top-level num_hidden_layers can refer to the vision
    encoder; the estimator must prefer text_config when present."""
    model = MagicMock()
    model.layers = []
    model.config = _VLMTopLevelVisionConfig()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    cfg = SchedulerConfig(
        max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0,
    )
    sched = Scheduler(model=model, tokenizer=tokenizer, config=cfg)

    monitor = sched.memory_monitor
    assert monitor is not None
    # Must be the LM dims from text_config, NOT vision (27 / 80).
    assert monitor._num_layers == 40
    assert monitor._head_dim == 256


class _AltSubConfigContainer:
    """Some packs name the LM sub-config ``language_config`` (or
    ``llm_config``) instead of ``text_config``."""

    def __init__(self, sub_attr_name: str):
        self.architectures = ["AltSubConfigVLM"]
        sub = _ModelConfig(
            num_hidden_layers=24,
            num_key_value_heads=4,
            num_attention_heads=24,
            head_dim=192,
        )
        setattr(self, sub_attr_name, sub)


def test_vlm_descent_handles_language_config_alias():
    model = MagicMock()
    model.layers = []
    model.config = _AltSubConfigContainer("language_config")
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0,
        ),
    )
    assert sched.memory_monitor._num_layers == 24
    assert sched.memory_monitor._head_dim == 192


def test_vlm_descent_handles_llm_config_alias():
    model = MagicMock()
    model.layers = []
    model.config = _AltSubConfigContainer("llm_config")
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0,
        ),
    )
    assert sched.memory_monitor._num_layers == 24


class _LegacyLMConfig:
    """GPT-style legacy config exposing ``n_layer`` / ``n_head`` / ``n_embd``
    instead of HuggingFace's ``num_hidden_layers`` etc."""

    def __init__(self):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768  # head_dim derived as n_embd / n_head = 64


def test_legacy_n_layer_fallback_path():
    """The extractor falls back to ``n_layer`` / ``n_head`` / ``n_embd`` for
    GPT-style configs and derives head_dim when not directly present."""
    model = MagicMock()
    model.layers = []
    model.config = _LegacyLMConfig()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0,
        ),
    )
    monitor = sched.memory_monitor
    assert monitor is not None
    assert monitor._num_layers == 12
    assert monitor._num_kv_heads == 12  # falls back to n_head
    assert monitor._head_dim == 64  # n_embd / n_head


class _BrokenConfig:
    """A config whose attribute access raises — exercises the outer
    try/except wrap in _set_model_info_for_monitor."""

    @property
    def num_hidden_layers(self):
        raise RuntimeError("synthetic boom")


class _VLMWithNestedLegacyLayer:
    """Hypothetical VLM whose LM sub-config exposes only the legacy
    GPT-style ``n_layer`` (no ``num_hidden_layers``). The descent rule
    must accept this so the LM dims aren't shadowed by the top-level
    vision-encoder dims.
    """

    def __init__(self):
        self.architectures = ["LegacyNestedVLM"]
        # Top-level matches vision encoder dims that should be ignored.
        self.num_hidden_layers = 27
        self.num_key_value_heads = 16
        self.num_attention_heads = 16
        self.head_dim = 80
        self.text_config = _ModelConfig(
            num_hidden_layers=None,
            num_key_value_heads=8,
            num_attention_heads=32,
            head_dim=128,
        )
        # Force the sub-config to surface only n_layer, not
        # num_hidden_layers.
        self.text_config.num_hidden_layers = None
        self.text_config.n_layer = 36


def test_vlm_descent_prefers_text_config_via_legacy_n_layer():
    """Regression: the sub-config preference rule must accept legacy
    ``n_layer`` in addition to ``num_hidden_layers`` so the descent
    isn't silently skipped when only the legacy alias is present —
    otherwise the top-level (vision) dims leak into the SDPA-peak
    calculation.
    """
    model = MagicMock()
    model.layers = []
    model.config = _VLMWithNestedLegacyLayer()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0,
        ),
    )
    monitor = sched.memory_monitor
    assert monitor is not None
    # Must be the LM dims (n_layer=36, head_dim=128), NOT vision (27/80).
    assert monitor._num_layers == 36
    assert monitor._head_dim == 128


def test_exception_during_descent_is_swallowed():
    """The whole _set_model_info_for_monitor body is wrapped in
    try/except so a malformed config can't break Scheduler init."""
    model = MagicMock()
    model.layers = []
    model.config = _BrokenConfig()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    # Must not raise.
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0,
        ),
    )
    # Monitor exists but dims stayed None — estimator returns 0 / guard skips.
    assert sched.memory_monitor is not None
    assert sched.memory_monitor._num_layers is None
