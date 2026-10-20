# SPDX-License-Identifier: Apache-2.0
"""
Tests for Claude Code auto-mode safety-classifier detection.

Claude Code sends an internal Bash safety-classifier request that carries no
``thinking`` field at all. On a reasoning model oMLX therefore falls through to
the model default (thinking on), the model reasons past Claude Code's ~35-45s
deadline, and the request is cancelled — surfacing to the user as "model is
temporarily unavailable".

The envelopes below are transcribed from a live capture against Claude Code
2.1.246, not invented.
"""

import pytest

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.anthropic_utils import is_auto_mode_classifier_request

CLASSIFIER_SYSTEM = (
    "You are a security monitor for autonomous AI coding agents. "
    "Decide whether the proposed action is safe."
)


def _classifier_payload(**overrides):
    """The real captured classifier envelope, with optional field overrides."""
    payload = {
        "model": "Qwen3.6-35B-A3B-OptiQ-4bit",
        "max_tokens": 2112,
        "stream": False,
        "stop_sequences": ["</block>"],
        "system": [{"type": "text", "text": CLASSIFIER_SYSTEM}],
        "messages": [
            {
                "role": "user",
                "content": "<transcript>rm /tmp/foo.txt</transcript> Is this safe?",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _normal_turn_payload(**overrides):
    """A normal Claude Code conversation turn, which must never match."""
    payload = {
        "model": "Qwen3.6-35B-A3B-OptiQ-4bit",
        "max_tokens": 32000,
        "stream": True,
        "thinking": {"type": "adaptive"},
        "system": [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }
        ],
        "messages": [{"role": "user", "content": "what does this function do?"}],
        "tools": [
            {
                "name": "Bash",
                "description": "run a command",
                "input_schema": {"type": "object"},
            }
        ],
    }
    payload.update(overrides)
    return payload


def _request(payload) -> MessagesRequest:
    return MessagesRequest.model_validate(payload)


class TestIsAutoModeClassifierRequest:
    """The predicate must match the classifier and nothing else."""

    def test_matches_real_captured_envelope(self):
        assert is_auto_mode_classifier_request(_request(_classifier_payload())) is True

    def test_matches_when_system_is_a_bare_string(self):
        req = _request(_classifier_payload(system=CLASSIFIER_SYSTEM))
        assert is_auto_mode_classifier_request(req) is True

    def test_marker_match_is_case_insensitive(self):
        req = _request(
            _classifier_payload(
                system=[{"type": "text", "text": CLASSIFIER_SYSTEM.upper()}]
            )
        )
        assert is_auto_mode_classifier_request(req) is True

    def test_marker_matches_when_not_the_first_system_block(self):
        """Substring match, not prefix — Claude Code may prepend a block.

        A prefix check would stop matching with no signal at all, which is the
        silent regression this change exists to prevent.
        """
        req = _request(
            _classifier_payload(
                system=[
                    {"type": "text", "text": "## Session Context"},
                    {"type": "text", "text": CLASSIFIER_SYSTEM},
                ]
            )
        )
        assert is_auto_mode_classifier_request(req) is True

    def test_rejects_normal_turn(self):
        assert (
            is_auto_mode_classifier_request(_request(_normal_turn_payload())) is False
        )

    def test_rejects_streaming_request(self):
        """A streaming request is never the classifier — it is non-streaming."""
        req = _request(_classifier_payload(stream=True))
        assert is_auto_mode_classifier_request(req) is False

    def test_rejects_request_carrying_tools(self):
        """A real conversation turn discussing the classifier still carries tools."""
        req = _request(
            _classifier_payload(
                tools=[
                    {
                        "name": "Bash",
                        "description": "run a command",
                        "input_schema": {"type": "object"},
                    }
                ]
            )
        )
        assert is_auto_mode_classifier_request(req) is False

    def test_empty_tools_list_still_matches(self):
        """``tools: []`` is absence of tools, not presence."""
        req = _request(_classifier_payload(tools=[]))
        assert is_auto_mode_classifier_request(req) is True

    def test_rejects_missing_marker(self):
        req = _request(
            _classifier_payload(
                system=[{"type": "text", "text": "You are a helpful assistant."}]
            )
        )
        assert is_auto_mode_classifier_request(req) is False

    def test_rejects_billing_header_alone(self):
        """Regression guard for issue #1.

        The legacy ``x-anthropic-billing-header:`` marker appears on ordinary
        Agent SDK turns, not just the classifier. Matching on it alone would
        disable thinking on every request.
        """
        req = _request(
            _normal_turn_payload(
                system=[
                    {
                        "type": "text",
                        "text": "x-anthropic-billing-header: cc_version=2.1.246",
                    },
                    {"type": "text", "text": "You are a Claude agent."},
                ],
                stream=False,
                tools=[],
            )
        )
        assert is_auto_mode_classifier_request(req) is False

    def test_handles_absent_system(self):
        req = _request(_classifier_payload(system=None))
        assert is_auto_mode_classifier_request(req) is False


class TestApplyClassifierThinking:
    """Thinking is turned off only for a classifier request, and an explicit
    choice (a model setting or the client's own template kwargs) still wins."""

    def _apply(self, request, merged=None, forced=None, is_classifier=True):
        from omlx.server import _apply_auto_mode_classifier_thinking

        merged = {} if merged is None else merged
        _apply_auto_mode_classifier_thinking(
            request, merged, set() if forced is None else forced, is_classifier
        )
        return merged

    def test_applies_on_the_captured_envelope(self):
        merged = self._apply(_request(_classifier_payload()))
        assert merged["enable_thinking"] is False

    def test_forced_key_still_wins(self):
        merged = self._apply(
            _request(_classifier_payload()), forced={"enable_thinking"}
        )
        assert "enable_thinking" not in merged

    def test_client_chat_template_kwargs_still_win(self):
        request = _request(
            _classifier_payload(chat_template_kwargs={"enable_thinking": True})
        )
        merged = self._apply(request, merged={"enable_thinking": True})
        assert merged["enable_thinking"] is True

    def test_normal_turn_is_untouched(self):
        merged = self._apply(
            _request(_normal_turn_payload(stream=False, tools=[])),
            is_classifier=False,
        )
        assert merged == {}


class TestIsAutoModeClassifierWrapper:
    """A bug in detection must never fail an unrelated request.

    Detection runs on every /v1/messages call. The wrapper turns an error into
    "not the classifier", which is the behavior before the workaround existed.
    """

    def test_detection_error_means_not_a_classifier(self, monkeypatch, caplog):
        import omlx.server as server

        def boom(_request):
            raise RuntimeError("simulated future schema change")

        monkeypatch.setattr(server, "is_auto_mode_classifier_request", boom)
        with caplog.at_level("WARNING", logger="omlx.server"):
            assert (
                server._is_auto_mode_classifier(_request(_classifier_payload()))
                is False
            )
        assert any("detection failed" in r.getMessage() for r in caplog.records)

    def test_matches_the_captured_envelope(self):
        import omlx.server as server

        assert server._is_auto_mode_classifier(_request(_classifier_payload())) is True


class TestSteeredClassifierModel:
    """Steering is off by default and falls back to the requested model when it
    is on but cannot name a model."""

    def _call(self, is_classifier=True):
        from omlx.server import _steered_classifier_model

        return _steered_classifier_model(is_classifier)

    def _server_state_with(self, monkeypatch, *, steer, tier="haiku", **tier_models):
        import omlx.server as server
        from omlx.settings import ClaudeCodeSettings, GlobalSettings

        state = server.ServerState()
        state.global_settings = GlobalSettings()
        state.global_settings.claude_code = ClaudeCodeSettings(
            steer_classifier_requests=steer,
            classifier_model_tier=tier,
            **tier_models,
        )
        monkeypatch.setattr(server, "_server_state", state)
        monkeypatch.setattr(server, "_steering_warned", set())
        return server

    def test_disabled_by_default_returns_none(self, monkeypatch):
        self._server_state_with(monkeypatch, steer=False, haiku_model="m")
        assert self._call() is None

    def test_enabled_returns_configured_tier_model(self, monkeypatch):
        self._server_state_with(
            monkeypatch, steer=True, tier="haiku", haiku_model="local-haiku-4bit"
        )
        assert self._call() == "local-haiku-4bit"

    def test_enabled_but_not_a_classifier_request_returns_none(self, monkeypatch):
        self._server_state_with(monkeypatch, steer=True, haiku_model="local-haiku-4bit")
        assert self._call(is_classifier=False) is None

    def test_no_global_settings_returns_none(self, monkeypatch):
        import omlx.server as server

        state = server.ServerState()
        state.global_settings = None
        monkeypatch.setattr(server, "_server_state", state)
        assert self._call() is None

    @pytest.mark.parametrize(
        ("tier", "expected"), [("opus", "o"), ("sonnet", "s"), ("haiku", "h")]
    )
    def test_each_tier_selects_its_own_model(self, monkeypatch, tier, expected):
        self._server_state_with(
            monkeypatch,
            steer=True,
            tier=tier,
            haiku_model="h",
            sonnet_model="s",
            opus_model="o",
        )
        assert self._call() == expected

    def test_unset_tier_model_warns_once_and_returns_none(self, monkeypatch, caplog):
        self._server_state_with(monkeypatch, steer=True, tier="opus")
        with caplog.at_level("WARNING", logger="omlx.server"):
            assert self._call() is None
            assert self._call() is None
        warnings = [r for r in caplog.records if "no model" in r.getMessage()]
        assert len(warnings) == 1
        assert "opus" in warnings[0].getMessage()


@pytest.mark.parametrize("marker_present", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("has_tools", [True, False])
def test_is_auto_mode_classifier_request_requires_all_three_conditions(
    marker_present, stream, has_tools
):
    system_text = CLASSIFIER_SYSTEM if marker_present else "you are a helpful assistant"
    tools = (
        [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]
        if has_tools
        else None
    )
    request = _request(
        _classifier_payload(
            stream=stream,
            system=[{"type": "text", "text": system_text}],
            tools=tools,
        )
    )
    expected = marker_present and not stream and not has_tools
    assert is_auto_mode_classifier_request(request) == expected
