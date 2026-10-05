# SPDX-License-Identifier: Apache-2.0
"""Anthropic ``thinking.type: "adaptive"`` must reach the chat template.

Issue #4242: the Anthropic request schema accepts the three-state
``thinking.type`` (``enabled`` / ``disabled`` / ``adaptive``), but
``create_anthropic_message`` collapsed ``adaptive`` into
``enable_thinking=True``.  A three-state model such as MiniMax-M3 therefore
never saw ``thinking_mode: "adaptive"`` — it received ``"enabled"`` instead.

The three tests below pin the behaviour that must hold:

1. ``TestAnthropicThinkingType`` drives the real ``/v1/messages`` route and
   captures the ``chat_template_kwargs`` the engine is called with, for every
   value ``thinking.type`` can take (including absent / ``None``).
2. ``TestMinimaxThinkingKwargsTable`` pins the MiniMax-M3 translator's
   boolean -> ``thinking_mode`` mapping.
3. ``TestDeepseekV41ThinkingMode`` pins the counter-example that rules out
   injecting ``thinking_mode`` for *every* model: DeepSeek V4.1 asserts
   ``thinking_mode in ("chat", "thinking")``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import omlx.server as srv
from omlx.model_settings import ModelSettings

# ---------------------------------------------------------------------------
# 1. The /v1/messages route: what chat_template_kwargs does thinking.type make?
# ---------------------------------------------------------------------------


def _capture_anthropic_ct_kwargs(
    monkeypatch,
    model_type,
    body_extra,
    settings=None,
    *,
    pool_config_model_type=None,
):
    """POST /v1/messages and return the chat_template_kwargs the engine saw.

    ``preflight_chat`` raises so the request stops at the generation boundary;
    everything upstream (kwarg merge, thinking mapping, budget injection,
    grammar derivation) has already run by then.

    ``pool_config_model_type`` decouples the pool entry's ``config_model_type``
    from ``engine.model_type``; it defaults to ``model_type``.
    """
    engine = MagicMock()
    engine.model_type = model_type
    engine.is_diffusion_model = False
    engine.tokenizer = None
    engine.preflight_chat = AsyncMock(
        side_effect=HTTPException(status_code=418, detail="captured")
    )
    engine.start = AsyncMock()
    engine.count_chat_tokens.return_value = 128

    pool = MagicMock()
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.get_entry.return_value = SimpleNamespace(
        config_model_type=(
            model_type if pool_config_model_type is None else pool_config_model_type
        ),
        preserve_thinking_default=None,
    )

    monkeypatch.setattr(srv._server_state, "engine_pool", pool)
    monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
    monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
    monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
    monkeypatch.setattr(
        srv,
        "get_model_settings_for_request",
        lambda name: settings if settings is not None else ModelSettings(),
    )
    monkeypatch.setitem(
        srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
    )

    body = {
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(body_extra)
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        response = client.post("/v1/messages", json=body)

    assert response.status_code == 418, response.text
    return engine.preflight_chat.call_args.kwargs.get("chat_template_kwargs") or {}


class TestAnthropicThinkingType:
    """``thinking.type`` -> chat_template_kwargs, per model family."""

    @pytest.mark.parametrize(
        "thinking, expected",
        [
            # Explicit three states.
            ({"type": "enabled"}, {"enable_thinking": True}),
            ({"type": "disabled"}, {"enable_thinking": False}),
            (
                {"type": "adaptive"},
                # MiniMax-M3's template reads the three-state kwarg, so the
                # request must reach it as "adaptive" (#4242).  ``enable_thinking``
                # stays because it is the portable signal every other template
                # understands — the M3 translator drops it (model-native wins).
                {"thinking_mode": "adaptive", "enable_thinking": True},
            ),
            # Setting absent -> nothing injected (default path must not move).
            (None, {}),
        ],
        ids=["enabled", "disabled", "adaptive", "absent"],
    )
    def test_minimax_m3_thinking_type(self, monkeypatch, thinking, expected):
        ct_kwargs = _capture_anthropic_ct_kwargs(
            monkeypatch,
            "minimax_m3",
            {} if thinking is None else {"thinking": thinking},
        )
        assert ct_kwargs == expected

    @pytest.mark.parametrize(
        "thinking, expected",
        [
            ({"type": "enabled"}, {"enable_thinking": True}),
            ({"type": "disabled"}, {"enable_thinking": False}),
            # MiniMax-M3 is the only family whose template takes three states.
            # Every other model keeps today's exact behaviour: adaptive is
            # "enabled".  Injecting ``thinking_mode`` globally would reach
            # DeepSeek V4.1, which asserts thinking_mode in ("chat","thinking").
            ({"type": "adaptive"}, {"enable_thinking": True}),
            (None, {}),
        ],
        ids=["enabled", "disabled", "adaptive", "absent"],
    )
    def test_non_minimax_models_keep_two_state_behaviour(
        self, monkeypatch, thinking, expected
    ):
        for model_type in ("qwen3_5", "deepseek_v41", "gemma4"):
            ct_kwargs = _capture_anthropic_ct_kwargs(
                monkeypatch,
                model_type,
                {} if thinking is None else {"thinking": thinking},
            )
            assert ct_kwargs == expected, model_type

    def test_explicit_null_thinking_is_ignored(self, monkeypatch):
        """``thinking: null`` must behave exactly like the field being absent."""
        absent = _capture_anthropic_ct_kwargs(monkeypatch, "minimax_m3", {})
        explicit_null = _capture_anthropic_ct_kwargs(
            monkeypatch, "minimax_m3", {"thinking": None}
        )
        assert explicit_null == absent == {}

    def test_request_state_still_overrides_a_model_setting(self, monkeypatch):
        """Precedence is unchanged: a per-request state beats a model default.

        ``merge_chat_template_request_kwargs`` lets the request win over
        ``ModelSettings.enable_thinking`` unless the key is listed in
        ``forced_ct_kwargs``.  That is existing behaviour for "enabled" and
        "disabled"; "adaptive" must not be treated differently.
        """
        engine = MagicMock()
        engine.model_type = "minimax_m3"
        engine.is_diffusion_model = False
        engine.tokenizer = None
        engine.preflight_chat = AsyncMock(
            side_effect=HTTPException(status_code=418, detail="captured")
        )
        engine.start = AsyncMock()
        engine.count_chat_tokens.return_value = 128

        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            config_model_type="minimax_m3", preserve_thinking_default=None
        )
        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
        monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
        monkeypatch.setattr(
            srv,
            "get_model_settings_for_request",
            lambda name: ModelSettings(enable_thinking=False),
        )
        monkeypatch.setitem(
            srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
        )
        with TestClient(srv.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "adaptive"},
                },
            )
        assert response.status_code == 418, response.text
        ct_kwargs = engine.preflight_chat.call_args.kwargs["chat_template_kwargs"]
        assert ct_kwargs == {"thinking_mode": "adaptive", "enable_thinking": True}

    def test_forced_ct_kwargs_suppresses_adaptive_entirely(self, monkeypatch):
        """``forced_ct_kwargs`` pins the operator's value against the request."""
        ct_kwargs = _capture_anthropic_ct_kwargs(
            monkeypatch,
            "minimax_m3",
            {"thinking": {"type": "adaptive"}, "chat_template_kwargs": {}},
            settings=ModelSettings(forced_ct_kwargs=["enable_thinking"]),
        )
        # "enable_thinking" is forced, so no per-request state is injected at
        # all -- including the model-native thinking_mode.
        assert ct_kwargs == {}

    def test_adaptive_does_not_fight_a_positive_thinking_budget(self, monkeypatch):
        """Budget auto-enable keys off ``enable_thinking``.

        With ``enable_thinking=True`` already present the budget branch must not
        add anything on top, so ``adaptive`` survives a budgeted request.
        """
        ct_kwargs = _capture_anthropic_ct_kwargs(
            monkeypatch,
            "minimax_m3",
            {"thinking": {"type": "adaptive"}, "thinking_budget": 1024},
        )
        assert ct_kwargs == {"thinking_mode": "adaptive", "enable_thinking": True}

    def test_client_supplied_thinking_mode_wins_over_adaptive(self, monkeypatch):
        """A client's ``thinking_mode`` must not be clobbered by the injection.

        This is the case that regressed against main: assigning the key
        unconditionally overwrote the client's value.
        """
        ct_kwargs = _capture_anthropic_ct_kwargs(
            monkeypatch,
            "minimax_m3",
            {
                "thinking": {"type": "adaptive"},
                "chat_template_kwargs": {"thinking_mode": "disabled"},
            },
        )
        assert ct_kwargs == {"thinking_mode": "disabled", "enable_thinking": True}

    def test_forced_thinking_mode_setting_wins_over_adaptive(self, monkeypatch):
        """``forced_ct_kwargs=["thinking_mode"]`` pins the operator's value."""
        ct_kwargs = _capture_anthropic_ct_kwargs(
            monkeypatch,
            "minimax_m3",
            {"thinking": {"type": "adaptive"}},
            settings=ModelSettings(
                chat_template_kwargs={"thinking_mode": "disabled"},
                forced_ct_kwargs=["thinking_mode"],
            ),
        )
        assert ct_kwargs == {"thinking_mode": "disabled", "enable_thinking": True}

    def test_pool_config_model_type_enables_injection_when_engine_type_is_none(
        self, monkeypatch
    ):
        """The pool entry's ``config_model_type`` alone must enable injection.

        ``engine.model_type`` can be ``None`` and the served name need not look
        like MiniMax; the pool's configured type still identifies the template.
        """
        ct_kwargs = _capture_anthropic_ct_kwargs(
            monkeypatch,
            None,
            {"thinking": {"type": "adaptive"}},
            pool_config_model_type="minimax_m3",
        )
        assert ct_kwargs == {"thinking_mode": "adaptive", "enable_thinking": True}


# ---------------------------------------------------------------------------
# 2. The MiniMax-M3 translator: boolean -> thinking_mode
# ---------------------------------------------------------------------------


class TestMinimaxThinkingKwargsTable:
    def _translate(self, kwargs):
        from omlx.patches.mlx_vlm_minimax_m3_compat import (
            _apply_minimax_thinking_kwargs,
        )

        out = dict(kwargs)
        _apply_minimax_thinking_kwargs(out)
        return out

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"enable_thinking": True}, {"thinking_mode": "enabled"}),
            ({"enable_thinking": False}, {"thinking_mode": "disabled"}),
            # No boolean, nothing to derive.
            ({}, {}),
            # The model-native keyword wins and enable_thinking is dropped
            # without a message (pre-existing documented-by-test behaviour).
            (
                {"enable_thinking": True, "thinking_mode": "adaptive"},
                {"thinking_mode": "adaptive"},
            ),
            (
                {"enable_thinking": False, "thinking_mode": "disabled"},
                {"thinking_mode": "disabled"},
            ),
        ],
        ids=["true", "false", "absent", "both-adaptive", "both-disabled"],
    )
    def test_table(self, kwargs, expected):
        assert self._translate(kwargs) == expected

    def test_translator_emits_adaptive_when_the_boolean_carries_it(self):
        """A string-valued ``enable_thinking`` must not be silently dropped.

        ``thinking_mode`` is the three-state kwarg; ``enable_thinking`` is a
        two-state one.  If a caller ever puts the three-state value on the
        boolean key, dropping it would lose the state — the translator has to
        recognise it.
        """
        assert self._translate({"enable_thinking": "adaptive"}) == {
            "thinking_mode": "adaptive"
        }


class TestVlmMinimaxThinkingMode:
    """``VLMBatchedEngine`` calls its own translator, not the patch-module one.

    Both functions must map every input identically; the VL translator used to
    drop a string ``enable_thinking`` on the floor.
    """

    @staticmethod
    def _translate(model_type, kwargs):
        from omlx.engine.vlm import _apply_minimax_m3_thinking_mode

        out = dict(kwargs)
        _apply_minimax_m3_thinking_mode(model_type, out)
        return out

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"enable_thinking": True}, {"thinking_mode": "enabled"}),
            ({"enable_thinking": False}, {"thinking_mode": "disabled"}),
            # No boolean, nothing to derive.
            ({}, {}),
            ({"enable_thinking": "adaptive"}, {"thinking_mode": "adaptive"}),
            # The model-native keyword wins and enable_thinking is dropped.
            (
                {"enable_thinking": True, "thinking_mode": "adaptive"},
                {"thinking_mode": "adaptive"},
            ),
            (
                {"enable_thinking": False, "thinking_mode": "disabled"},
                {"thinking_mode": "disabled"},
            ),
        ],
        ids=[
            "true",
            "false",
            "absent",
            "adaptive",
            "both-adaptive",
            "both-disabled",
        ],
    )
    def test_vl_translator_table(self, kwargs, expected):
        assert self._translate("minimax_m3_vl", kwargs) == expected

    def test_vl_translator_emits_adaptive(self):
        assert self._translate("minimax_m3_vl", {"enable_thinking": "adaptive"}) == {
            "thinking_mode": "adaptive"
        }

    def test_non_minimax_type_is_a_noop(self):
        kwargs = {"enable_thinking": "adaptive"}
        assert self._translate("qwen3_5", kwargs) == kwargs

    def test_both_translators_agree_on_every_input(self):
        from omlx.patches.mlx_vlm_minimax_m3_compat import (
            _apply_minimax_thinking_kwargs,
        )

        cases = [
            {},
            {"enable_thinking": True},
            {"enable_thinking": False},
            {"enable_thinking": None},
            {"enable_thinking": "adaptive"},
            {"thinking_mode": "adaptive"},
            {"enable_thinking": True, "thinking_mode": "disabled"},
            {"enable_thinking": "adaptive", "thinking_mode": "enabled"},
        ]
        for kwargs in cases:
            patch_out = dict(kwargs)
            _apply_minimax_thinking_kwargs(patch_out)
            assert self._translate("minimax_m3_vl", kwargs) == patch_out, kwargs


# ---------------------------------------------------------------------------
# 2b. The grammar site: reasoning must agree with the rendered template
# ---------------------------------------------------------------------------


def _reasoning_for_ct_kwargs(ct_kwargs, monkeypatch):
    """Return the ``reasoning`` flag handed to xgrammar's structural tag.

    ``xgrammar`` carries a native extension that is absent in this checkout, so
    a stub module stands in — the derivation under test is pure and runs before
    anything is handed to the library.
    """
    import sys

    from omlx.server import _compile_with_structural_tag

    compiled_grammar_cls = type("CompiledGrammar", (), {})
    compiled_grammar = compiled_grammar_cls()

    fake_xgrammar = MagicMock()
    fake_xgrammar.CompiledGrammar = compiled_grammar_cls
    fake_tag = MagicMock()
    fake_tag.model_dump.return_value = {
        "type": "structural_tag",
        "format": {"type": "any_text", "excludes": []},
    }
    fake_xgrammar.get_builtin_structural_tag.return_value = fake_tag
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar)

    compiler = MagicMock()
    compiler.compile_structural_tag.return_value = compiled_grammar
    _compile_with_structural_tag(
        compiler,
        {"type": "json_schema", "json_schema": {"type": "object"}},
        "minimax",
        ct_kwargs,
    )
    # mark_grammar_thinking_phase records the same flag on the grammar.
    return (
        fake_xgrammar.get_builtin_structural_tag.call_args.kwargs["reasoning"],
        compiled_grammar._omlx_has_thinking_phase,
    )


class TestGrammarReasoningDerivation:
    """A grammar must never wait for a thinking phase the template disables."""

    @pytest.mark.parametrize(
        "ct_kwargs, expected",
        [
            (None, True),
            ({}, True),
            ({"enable_thinking": True}, True),
            ({"enable_thinking": False}, False),
            # Model-native keyword: the template renders thinking off, so the
            # grammar must not expect a thinking phase (#4242 step 3).
            ({"thinking_mode": "disabled"}, False),
            ({"thinking_mode": "enabled"}, True),
            # Precedence matches the template translator: thinking_mode wins.
            (
                {"thinking_mode": "disabled", "enable_thinking": True},
                False,
            ),
            (
                {"thinking_mode": "enabled", "enable_thinking": False},
                True,
            ),
            # A budget may inject enable_thinking=True next to a client-sent
            # thinking_mode; the model-native keyword still decides.
            (
                {"thinking_mode": "adaptive", "enable_thinking": True},
                True,
            ),
            # DeepSeek V4.x owns thinking_mode with values ("chat","thinking").
            # Neither is "disabled", so the grammar keeps its thinking phase.
            ({"thinking_mode": "chat"}, True),
            ({"thinking_mode": "thinking"}, True),
        ],
        ids=[
            "none",
            "empty",
            "bool-true",
            "bool-false",
            "mode-disabled",
            "mode-enabled",
            "mode-wins-over-bool-true",
            "mode-wins-over-bool-false",
            "adaptive-with-budget-true",
            "deepseek-chat",
            "deepseek-thinking",
        ],
    )
    def test_reasoning_flag(self, monkeypatch, ct_kwargs, expected):
        tag_reasoning, marked = _reasoning_for_ct_kwargs(ct_kwargs, monkeypatch)
        assert tag_reasoning is expected
        assert marked is expected


# ---------------------------------------------------------------------------
# 3. Counter-example: why thinking_mode is not injected for every model
# ---------------------------------------------------------------------------


class TestDeepseekV41ThinkingMode:
    """DeepSeek V4.1 owns ``thinking_mode`` with a *different* value space."""

    @staticmethod
    def _render(**kwargs):
        from omlx.patches.deepseek_v41.processing import Processor

        proc = SimpleNamespace(tokenizer=MagicMock())
        return Processor.apply_chat_template(
            proc, [{"role": "user", "content": "hi"}], **kwargs
        )

    def test_adaptive_is_rejected_by_the_v41_processor(self):
        # ``render_message`` asserts thinking_mode in ("chat", "thinking").
        # Sending "adaptive" raises — which is what makes a blanket injection
        # of ``thinking_mode="adaptive"`` in server.py unacceptable.
        with pytest.raises(AssertionError, match="Invalid thinking_mode"):
            self._render(thinking_mode="adaptive")

    def test_legacy_boolean_still_renders(self):
        """The portable signal the server actually sends today still works."""
        assert "<｜Assistant｜>" in self._render(enable_thinking=True)
        assert "<｜Assistant｜>" in self._render(enable_thinking=False)
