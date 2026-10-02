# SPDX-License-Identifier: Apache-2.0
"""Tests for admin cache probe endpoint (POST /admin/api/cache/probe)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — triggers set_admin_getters
from omlx.cache.paged_cache import compute_block_hash
from omlx.model_settings import ModelSettings, merge_chat_template_kwargs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 4  # Small block size for readable tests
MODEL_ID = "test-model"
MODEL_NAME = "/models/test-model"


def _make_request(model_id=MODEL_ID, messages=None):
    """Build a CacheProbeRequest."""
    return admin_routes.CacheProbeRequest(
        model_id=model_id,
        messages=messages or [{"role": "user", "content": "hello"}],
    )


def _make_tokenizer(token_ids):
    """Return a minimal tokenizer mock that produces *token_ids*."""
    tok = MagicMock(spec=[])
    tok.apply_chat_template = MagicMock(return_value="rendered prompt")
    tok.encode = MagicMock(return_value=token_ids)
    return tok


def _compute_hashes(token_ids, block_size=BLOCK_SIZE, model_name=MODEL_NAME):
    """Compute the chain-hashed block sequence for *token_ids*."""
    hashes = []
    parent = b""
    for start in range(0, len(token_ids), block_size):
        block_tokens = token_ids[start : start + block_size]
        h = compute_block_hash(parent, block_tokens, model_name=model_name)
        hashes.append(h)
        parent = h
    return hashes


def _make_ssd_index(known_hashes):
    """Return a mock SSD index whose contains() recognises *known_hashes*."""
    idx = MagicMock(spec=[])
    idx.contains = MagicMock(side_effect=lambda h: h in known_hashes)
    return idx


def _make_engine_entry(
    tokenizer,
    scheduler,
    has_apply_chat_template=True,
):
    """Build the engine_pool._entries[model_id] namespace chain."""
    engine_ns = SimpleNamespace(
        _tokenizer=tokenizer,
        _engine=SimpleNamespace(
            engine=SimpleNamespace(scheduler=scheduler),
        ),
    )
    if has_apply_chat_template:
        engine_ns._apply_chat_template = lambda msgs, tools, **kw: "rendered prompt"
    return SimpleNamespace(engine=engine_ns)


def _make_scheduler(
    ssd_hot=None,
    ssd_index=None,
    model_name=MODEL_NAME,
    block_size=BLOCK_SIZE,
):
    """Build a scheduler SimpleNamespace with the attributes probe_cache reads."""
    return SimpleNamespace(
        block_aware_cache=SimpleNamespace(block_size=block_size),
        paged_ssd_cache_manager=SimpleNamespace(
            _hot_cache=ssd_hot or {},
            _index=ssd_index or _make_ssd_index(set()),
        ),
        paged_cache_manager=SimpleNamespace(model_name=model_name),
        config=SimpleNamespace(paged_cache_block_size=block_size),
    )


def _pool_with(entries):
    """Return a mock engine pool wrapping *entries*."""
    pool = MagicMock(spec=[])
    pool._entries = entries
    return pool


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


class TestCacheProbeToolCallNormalization:
    """Verify cache probing mirrors chat-path tool argument normalization."""

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ('{"city":"Seoul"}', {"city": "Seoul"}),
            ("   ", {}),
        ],
    )
    def test_normalizes_arguments_before_rendering(self, arguments, expected):
        messages = [
            {"role": "user", "content": "Check the weather."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        request = _make_request(messages=messages)
        tokenizer = _make_tokenizer([1, 2, 3, 4])
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        rendered_messages = None

        def render(normalized, tools, **kwargs):
            nonlocal rendered_messages
            rendered_messages = normalized
            return "rendered prompt"

        entry.engine._apply_chat_template = render
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            asyncio.run(admin_routes.probe_cache(request, is_admin=True))

        normalized_arguments = rendered_messages[1]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert normalized_arguments == expected
        assert (
            request.messages[1]["tool_calls"][0]["function"]["arguments"] == arguments
        )


class TestCacheProbeErrors:
    """Guard-clause and error-path coverage."""

    def test_engine_pool_not_initialized(self):
        with patch.object(admin_routes, "_get_engine_pool", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 503

    def test_model_not_found(self):
        pool = _pool_with({})
        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 404

    def test_model_not_loaded(self):
        pool = _pool_with({MODEL_ID: SimpleNamespace(engine=None)})
        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )
        assert result["model_loaded"] is False
        assert "reason" in result

    def test_no_tokenizer(self):
        entry = SimpleNamespace(
            engine=SimpleNamespace(
                _tokenizer=None,
                _engine=SimpleNamespace(
                    engine=SimpleNamespace(scheduler=_make_scheduler()),
                ),
            )
        )
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 400

    def test_scheduler_unavailable(self):
        tokenizer = _make_tokenizer([1, 2, 3])
        entry = SimpleNamespace(
            engine=SimpleNamespace(
                _tokenizer=tokenizer,
                _engine=None,
            )
        )
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 500

    def test_block_size_unavailable(self):
        tokenizer = _make_tokenizer([1, 2, 3])
        scheduler = SimpleNamespace(
            block_aware_cache=None,
            paged_ssd_cache_manager=None,
            paged_cache_manager=None,
            config=SimpleNamespace(paged_cache_block_size=0),
        )
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.probe_cache(_make_request(), is_admin=True))
            assert exc_info.value.status_code == 500

    def test_empty_tokens(self):
        tokenizer = _make_tokenizer([])
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})
        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )
        assert result["model_loaded"] is True
        assert result["total_tokens"] == 0
        assert result["total_blocks"] == 0


# ---------------------------------------------------------------------------
# Cache classification tests
# ---------------------------------------------------------------------------


class TestCacheProbeClassification:
    """Verify hot / disk / cold block classification."""

    def test_all_blocks_cold(self):
        """No SSD cache entries at all → everything cold."""
        token_ids = list(range(12))  # 3 blocks of 4
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["total_tokens"] == 12
        assert result["total_blocks"] == 3
        assert result["blocks_ssd_hot"] == 0
        assert result["blocks_ssd_disk"] == 0
        assert result["blocks_cold"] == 3
        assert result["ssd_hit_tokens"] == 0
        assert result["cold_tokens"] == 12

    def test_all_blocks_ssd_disk(self):
        """Every block found in SSD disk index."""
        token_ids = list(range(8))  # 2 blocks of 4
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index(set(hashes))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_hot"] == 0
        assert result["blocks_ssd_disk"] == 2
        assert result["blocks_cold"] == 0
        assert result["ssd_hit_tokens"] == 8
        assert result["cold_tokens"] == 0

    def test_all_blocks_ssd_hot(self):
        """Every block found in SSD hot cache (RAM copy)."""
        token_ids = list(range(8))  # 2 blocks of 4
        hashes = _compute_hashes(token_ids)
        ssd_hot = {h: {} for h in hashes}
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_hot=ssd_hot)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_hot"] == 2
        assert result["blocks_ssd_disk"] == 0
        assert result["blocks_cold"] == 0

    def test_mixed_hot_and_disk(self):
        """First block hot, second block disk-only."""
        token_ids = list(range(8))  # 2 blocks of 4
        hashes = _compute_hashes(token_ids)
        ssd_hot = {hashes[0]: {}}
        ssd_index = _make_ssd_index({hashes[1]})
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_hot=ssd_hot, ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_hot"] == 1
        assert result["blocks_ssd_disk"] == 1
        assert result["blocks_cold"] == 0

    def test_partial_prefix_hit(self):
        """First 2 blocks cached, third block miss → third is cold."""
        token_ids = list(range(12))  # 3 blocks of 4
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index({hashes[0], hashes[1]})
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_disk"] == 2
        assert result["blocks_cold"] == 1
        assert result["ssd_hit_tokens"] == 8
        assert result["cold_tokens"] == 4

    def test_gap_in_prefix_makes_rest_cold(self):
        """Block 0 cached, block 1 missing, block 2 cached → blocks 1+2 cold.

        The walk stops at the first miss because of the contiguous prefix
        assumption, so block 2 is never checked even though it exists in the
        index.
        """
        token_ids = list(range(12))  # 3 blocks of 4
        hashes = _compute_hashes(token_ids)
        # Only block 0 and block 2 in index (gap at block 1)
        ssd_index = _make_ssd_index({hashes[0], hashes[2]})
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["blocks_ssd_disk"] == 1  # Only block 0
        assert result["blocks_cold"] == 2  # Blocks 1 and 2

    def test_partial_last_block(self):
        """Token count not a multiple of block_size → last block is smaller."""
        token_ids = list(range(10))  # 2 full blocks + 1 partial (2 tokens)
        hashes = _compute_hashes(token_ids)
        ssd_index = _make_ssd_index(set(hashes))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler(ssd_index=ssd_index)
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert result["total_tokens"] == 10
        assert result["total_blocks"] == 3
        assert result["blocks_ssd_disk"] == 3
        assert result["blocks_cold"] == 0
        assert result["ssd_hit_tokens"] == 10
        assert result["cold_tokens"] == 0


# ---------------------------------------------------------------------------
# Response shape test
# ---------------------------------------------------------------------------


class TestCacheProbeResponseShape:
    """Verify the response contains exactly the expected fields."""

    EXPECTED_FIELDS = {
        "model_id",
        "model_loaded",
        "total_tokens",
        "block_size",
        "total_blocks",
        "blocks_ssd_hot",
        "blocks_ssd_disk",
        "blocks_cold",
        "ssd_hit_tokens",
        "cold_tokens",
    }

    def test_loaded_response_fields(self):
        token_ids = list(range(4))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert set(result.keys()) == self.EXPECTED_FIELDS

    def test_no_dead_fields(self):
        """Ensure blocks_ram, ram_hit_tokens, prefix_index_hits are gone."""
        token_ids = list(range(4))
        tokenizer = _make_tokenizer(token_ids)
        scheduler = _make_scheduler()
        entry = _make_engine_entry(tokenizer, scheduler)
        pool = _pool_with({MODEL_ID: entry})

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            result = asyncio.run(
                admin_routes.probe_cache(_make_request(), is_admin=True)
            )

        assert "blocks_ram" not in result
        assert "ram_hit_tokens" not in result
        assert "prefix_index_hits" not in result


class TestCacheProbeChatTemplateKwargs:
    """The probe must render the prompt the scheduler would actually prefill.

    Regression: the probe rendered with the caller's kwargs alone and never
    consulted the model's own settings. For any model with enable_thinking
    set, it hashed a prompt no real turn produces — and because the block
    walk stops at the first miss, *every* block came back cold and the cache
    looked empty even while it was being written.
    """

    @staticmethod
    def _rendered_kwargs(
        settings,
        request_kwargs=None,
        lookup_raises=False,
        thinking_budget=None,
        preserve_thinking_default=None,
    ):
        """Run probe_cache and return the kwargs handed to the template."""
        request = admin_routes.CacheProbeRequest(
            model_id=MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs=request_kwargs,
            thinking_budget=thinking_budget,
        )
        entry = _make_engine_entry(_make_tokenizer([1, 2, 3, 4]), _make_scheduler())
        entry.preserve_thinking_default = preserve_thinking_default
        seen = {}

        def render(messages, tools, **kwargs):
            seen["ct"] = kwargs.get("chat_template_kwargs")
            return "rendered prompt"

        entry.engine._apply_chat_template = render

        manager = MagicMock(spec=[])
        if lookup_raises:
            manager.get_settings_for_request = MagicMock(
                side_effect=RuntimeError("settings store unavailable")
            )
        else:
            manager.get_settings_for_request = MagicMock(return_value=settings)

        with (
            patch.object(
                admin_routes,
                "_get_engine_pool",
                return_value=_pool_with({MODEL_ID: entry}),
            ),
            patch.object(admin_routes, "_get_settings_manager", return_value=manager),
        ):
            asyncio.run(admin_routes.probe_cache(request, is_admin=True))
        return seen["ct"]

    def test_model_thinking_toggle_reaches_the_template(self):
        """The bug: this toggle was dropped, so the probe hashed the wrong prompt."""
        settings = ModelSettings()
        settings.enable_thinking = False
        assert self._rendered_kwargs(settings) == {"enable_thinking": False}

    def test_persisted_template_kwargs_reach_the_template(self):
        settings = ModelSettings()
        settings.chat_template_kwargs = {"custom": "value"}
        assert self._rendered_kwargs(settings) == {"custom": "value"}

    def test_preserve_thinking_toggle_reaches_the_template(self):
        settings = ModelSettings()
        settings.preserve_thinking = True
        assert self._rendered_kwargs(settings) == {"preserve_thinking": True}

    def test_model_thinking_budget_enables_thinking(self):
        settings = ModelSettings(
            thinking_budget_enabled=True,
            thinking_budget_tokens=1024,
        )
        assert self._rendered_kwargs(settings) == {"enable_thinking": True}

    def test_request_thinking_budget_enables_thinking(self):
        assert self._rendered_kwargs(None, thinking_budget=1024) == {
            "enable_thinking": True
        }

    def test_explicit_thinking_toggle_wins_over_budget(self):
        settings = ModelSettings(
            enable_thinking=False,
            thinking_budget_enabled=True,
            thinking_budget_tokens=1024,
        )
        assert self._rendered_kwargs(settings) == {"enable_thinking": False}

    def test_model_preserve_thinking_default_reaches_the_template(self):
        assert self._rendered_kwargs(None, preserve_thinking_default=True) == {
            "preserve_thinking": True
        }

    def test_explicit_preserve_thinking_wins_over_model_default(self):
        rendered = self._rendered_kwargs(
            None,
            {"preserve_thinking": False},
            preserve_thinking_default=True,
        )
        assert rendered == {"preserve_thinking": False}

    def test_request_kwargs_override_model_settings(self):
        settings = ModelSettings()
        settings.enable_thinking = False
        rendered = self._rendered_kwargs(settings, {"enable_thinking": True})
        assert rendered == {"enable_thinking": True}

    def test_forced_keys_win_over_request_kwargs(self):
        settings = ModelSettings()
        settings.enable_thinking = False
        settings.forced_ct_kwargs = ["enable_thinking"]
        rendered = self._rendered_kwargs(settings, {"enable_thinking": True})
        assert rendered == {"enable_thinking": False}

    def test_settings_lookup_failure_falls_back_to_request_kwargs(self):
        """A settings failure must degrade, not break probing outright."""
        rendered = self._rendered_kwargs(
            None, {"enable_thinking": True}, lookup_raises=True
        )
        assert rendered == {"enable_thinking": True}

    def test_no_settings_and_no_kwargs_renders_none(self):
        assert self._rendered_kwargs(None) is None

    def test_probe_and_chat_path_resolve_identical_kwargs(self):
        """Anti-drift: both paths must agree, since disagreement is the bug."""
        settings = ModelSettings()
        settings.enable_thinking = False
        settings.chat_template_kwargs = {"custom": "value"}
        request_kwargs = {"custom": "override"}
        assert self._rendered_kwargs(settings, request_kwargs) == (
            merge_chat_template_kwargs(settings, request_kwargs)
        )
