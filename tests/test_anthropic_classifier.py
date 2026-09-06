# SPDX-License-Identifier: Apache-2.0
"""
Tests for Claude Code auto-mode safety-classifier detection.

Claude Code sends an internal Bash safety-classifier request that carries no
``thinking`` field at all. On a reasoning model oMLX therefore falls through to
the model default (thinking on), the model reasons past Claude Code's ~35-45s
deadline, and the request is cancelled — surfacing to the user as "model is
temporarily unavailable" (see issue #1).

The envelopes below are transcribed from a live capture against Claude Code
2.1.246, not invented.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.anthropic_utils import (
    classifier_envelope_drift,
    is_auto_mode_classifier_request,
)

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
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}
        ],
        "messages": [{"role": "user", "content": "what does this function do?"}],
        "tools": [
            {"name": "Bash", "description": "run a command", "input_schema": {"type": "object"}}
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
        assert is_auto_mode_classifier_request(_request(_normal_turn_payload())) is False

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
            _classifier_payload(system=[{"type": "text", "text": "You are a helpful assistant."}])
        )
        assert is_auto_mode_classifier_request(req) is False

    def test_rejects_total_envelope_drift(self):
        """The marker alone is not enough: a request sharing only the marker
        sentence with zero other classifier envelope traits (oversized
        max_tokens, no matching stop_sequence, no transcript tags) is not
        corroborated as the classifier."""
        req = _request(
            _classifier_payload(
                max_tokens=99999,
                stop_sequences=[],
                messages=[{"role": "user", "content": "no tags here"}],
            )
        )
        assert is_auto_mode_classifier_request(req) is False

    def test_matches_on_partial_envelope_drift(self):
        """A single drifted trait (Claude Code changed one envelope detail)
        must still match — only total drift disqualifies the marker match."""
        req = _request(_classifier_payload(max_tokens=99999))
        assert is_auto_mode_classifier_request(req) is True

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


class TestClassifierEnvelopeDrift:
    """Drift reporting is the early warning that Anthropic changed the format."""

    def test_captured_envelope_reports_no_drift(self):
        assert classifier_envelope_drift(_request(_classifier_payload())) == []

    def test_reports_oversized_max_tokens(self):
        drift = classifier_envelope_drift(_request(_classifier_payload(max_tokens=32000)))
        assert any("max_tokens" in reason for reason in drift)

    def test_reports_missing_stop_sequence(self):
        drift = classifier_envelope_drift(_request(_classifier_payload(stop_sequences=None)))
        assert any("stop_sequences" in reason for reason in drift)

    def test_reports_missing_transcript_tags(self):
        req = _request(
            _classifier_payload(
                messages=[{"role": "user", "content": "is rm /tmp/foo.txt safe?"}]
            )
        )
        drift = classifier_envelope_drift(req)
        assert any("transcript" in reason for reason in drift)

    def test_finds_transcript_in_structured_content_blocks(self):
        req = _request(
            _classifier_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<transcript>ls</transcript>"},
                        ],
                    }
                ]
            )
        )
        assert classifier_envelope_drift(req) == []

    def test_accumulates_multiple_drift_reasons(self):
        req = _request(
            _classifier_payload(
                max_tokens=32000,
                stop_sequences=None,
                messages=[{"role": "user", "content": "no tags here"}],
            )
        )
        assert len(classifier_envelope_drift(req)) == 3


class TestDetectionIsBestEffort:
    """A bug in the heuristic must never break an unrelated request.

    The branch sits on the hot path of every /v1/messages call, inside a try
    whose only handler releases the lease and re-raises. Without isolation a
    detection bug would 500 requests that aren't even classifier requests.
    """

    def _apply(self, request, merged=None, forced=None):
        from omlx.server import _apply_auto_mode_classifier_thinking

        merged = {} if merged is None else merged
        _apply_auto_mode_classifier_thinking(
            request, merged, set() if forced is None else forced
        )
        return merged

    def test_predicate_raising_does_not_propagate(self, monkeypatch):
        import omlx.server as server

        def boom(_request):
            raise RuntimeError("simulated future schema change")

        monkeypatch.setattr(server, "is_auto_mode_classifier_request", boom)
        merged = self._apply(_request(_classifier_payload()))
        assert merged == {}, "must fall through, leaving thinking at the default"

    def test_drift_check_raising_does_not_propagate(self, monkeypatch):
        import omlx.server as server

        def boom(_request):
            raise RuntimeError("simulated future schema change")

        monkeypatch.setattr(server, "classifier_envelope_drift", boom)
        merged = self._apply(_request(_classifier_payload()))
        assert merged["enable_thinking"] is False, (
            "the fix must still apply — only the diagnostic failed"
        )

    def test_applies_on_the_captured_envelope(self):
        merged = self._apply(_request(_classifier_payload()))
        assert merged["enable_thinking"] is False

    def test_forced_key_still_wins(self):
        merged = self._apply(
            _request(_classifier_payload()), forced={"enable_thinking"}
        )
        assert "enable_thinking" not in merged

    def test_normal_turn_is_untouched(self):
        merged = self._apply(_request(_normal_turn_payload(stream=False, tools=[])))
        assert merged == {}


class TestDriftBoundaries:
    """The three drift checks are independent, so cover them independently."""

    @pytest.mark.parametrize(
        "max_tokens,expect_drift",
        [(4095, False), (4096, False), (4097, True), (32000, True)],
    )
    def test_max_tokens_ceiling_boundary(self, max_tokens, expect_drift):
        req = _request(_classifier_payload(max_tokens=max_tokens))
        drifted = any("max_tokens" in r for r in classifier_envelope_drift(req))
        assert drifted is expect_drift

    @pytest.mark.parametrize("over_ceiling", [False, True])
    @pytest.mark.parametrize("drop_stop_sequence", [False, True])
    @pytest.mark.parametrize("drop_transcript", [False, True])
    def test_every_combination_reports_exactly_its_own_reasons(
        self, over_ceiling, drop_stop_sequence, drop_transcript
    ):
        """All 8 pass/fail combinations, not just all-pass and all-fail."""
        payload = _classifier_payload(
            max_tokens=32000 if over_ceiling else 2112,
            stop_sequences=None if drop_stop_sequence else ["</block>"],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "no tags here"
                        if drop_transcript
                        else "<transcript>ls</transcript>"
                    ),
                }
            ],
        )
        reasons = classifier_envelope_drift(_request(payload))
        assert any("max_tokens" in r for r in reasons) is over_ceiling
        assert any("stop_sequences" in r for r in reasons) is drop_stop_sequence
        assert any("transcript" in r for r in reasons) is drop_transcript
        assert len(reasons) == sum([over_ceiling, drop_stop_sequence, drop_transcript])

    def test_bool_max_tokens_is_coerced_by_pydantic_before_the_check(self):
        """``isinstance(True, int)`` is True in Python, but it never reaches us.

        Pydantic coerces ``max_tokens: True`` to ``1`` during validation, so the
        predicate only ever sees a real int.
        """
        req = _request(_classifier_payload(max_tokens=True))
        assert req.max_tokens == 1
        assert not any("max_tokens" in r for r in classifier_envelope_drift(req))


class TestUserMessageContentShapes:
    """User message content arrives as str, pydantic blocks, or raw dicts."""

    def test_ignores_non_text_blocks(self):
        """A tool_use block carries no text and must not satisfy the check."""
        req = _request(
            _classifier_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu_1",
                                "name": "Bash",
                                "input": {"command": "<transcript>fake</transcript>"},
                            }
                        ],
                    }
                ]
            )
        )
        assert any("transcript" in r for r in classifier_envelope_drift(req))

    def test_ignores_assistant_messages(self):
        req = _request(
            _classifier_payload(
                messages=[
                    {"role": "user", "content": "no tags"},
                    {"role": "assistant", "content": "<transcript>ls</transcript>"},
                ]
            )
        )
        assert any("transcript" in r for r in classifier_envelope_drift(req))

    def test_empty_messages_do_not_raise(self):
        req = _request(_classifier_payload(messages=[]))
        assert any("transcript" in r for r in classifier_envelope_drift(req))

    _ascii_text = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=20)
    _text_block = st.builds(lambda t: {"type": "text", "text": t}, _ascii_text)
    _non_text_block = st.sampled_from(
        [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
            {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {}},
        ]
    )
    _block = st.one_of(_text_block, _non_text_block)
    _content = st.one_of(_ascii_text, st.lists(_block, max_size=5))
    _message = st.fixed_dictionaries(
        {"role": st.sampled_from(["user", "assistant"]), "content": _content}
    )

    @given(messages=st.lists(_message, max_size=5))
    def test_user_message_text_property(self, messages):
        """_user_message_text must equal the concatenation of str content and
        text-block text from user messages only, in message order, regardless
        of how many non-text blocks or assistant messages are interleaved."""
        from omlx.api.anthropic_utils import _user_message_text

        req = _request(_classifier_payload(messages=messages))
        result = _user_message_text(req.messages)

        expected_parts = []
        for m in messages:
            if m["role"] != "user":
                continue
            content = m["content"]
            if isinstance(content, str):
                expected_parts.append(content)
            else:
                for block in content:
                    if block.get("type") == "text":
                        expected_parts.append(block["text"])
        assert result == "".join(expected_parts)


class TestNoContentLeakage:
    """Drift reasons are logged, so they must never carry prompt content."""

    @pytest.mark.parametrize(
        "secret",
        [
            "hunter2-do-not-log-this",
            # Secrets shaped like the reason strings themselves, to probe for
            # accidental f-string interpolation of user content.
            "max_tokens=99999 exceeds expected ceiling 4096",
            "stop_sequences missing '</block>'",
            "messages missing <transcript> tags",
            "",
            "\n\nWARNING: forged log line",
        ],
    )
    def test_drift_reasons_are_independent_of_message_content(self, secret):
        """Reasons must be a function of envelope structure, never of content.

        Comparing against a benign control is stronger than substring-checking
        the secret: it still catches interpolation when the secret happens to
        be shaped like one of the fixed reason templates, where a substring
        assertion would fire on the template itself.
        """

        def reasons_for(content):
            return classifier_envelope_drift(
                _request(
                    _classifier_payload(
                        max_tokens=32000,
                        stop_sequences=None,
                        messages=[{"role": "user", "content": content}],
                    )
                )
            )

        control = reasons_for("benign filler")
        assert control, "expected drift so there is something to compare"
        assert reasons_for(secret) == control

    def test_drift_reasons_are_independent_of_system_prompt(self):
        def reasons_for(system_text):
            return classifier_envelope_drift(
                _request(
                    _classifier_payload(
                        max_tokens=32000,
                        system=[{"type": "text", "text": system_text}],
                    )
                )
            )

        control = reasons_for(CLASSIFIER_SYSTEM)
        assert control
        assert reasons_for(f"{CLASSIFIER_SYSTEM} system-secret-do-not-log") == control


class TestSteeredClassifierModel:
    """_steered_classifier_model must fail safe exactly like
    _apply_auto_mode_classifier_thinking (#2) — a bug in detection or a
    missing/unset tier model must never propagate, only skip steering."""

    def _call(self, request, server):
        from omlx.server import _steered_classifier_model

        return _steered_classifier_model(request)

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
        return server

    def test_disabled_by_default_returns_none(self, monkeypatch):
        server = self._server_state_with(monkeypatch, steer=False, haiku_model="m")
        result = self._call(_request(_classifier_payload()), server)
        assert result is None

    def test_enabled_returns_configured_tier_model(self, monkeypatch):
        server = self._server_state_with(
            monkeypatch, steer=True, tier="haiku", haiku_model="local-haiku-4bit"
        )
        result = self._call(_request(_classifier_payload()), server)
        assert result == "local-haiku-4bit"

    def test_enabled_logs_steering_substitution(self, monkeypatch, caplog):
        server = self._server_state_with(
            monkeypatch, steer=True, tier="haiku", haiku_model="local-haiku-4bit"
        )
        with caplog.at_level("WARNING", logger="omlx.server"):
            self._call(_request(_classifier_payload()), server)
        assert any(
            "haiku" in record.message and "local-haiku-4bit" in record.message
            for record in caplog.records
        )

    def test_enabled_but_tier_model_unset_returns_none(self, monkeypatch):
        server = self._server_state_with(monkeypatch, steer=True, tier="opus")
        result = self._call(_request(_classifier_payload()), server)
        assert result is None

    def test_enabled_but_tier_model_unset_logs_warning(self, monkeypatch, caplog):
        server = self._server_state_with(monkeypatch, steer=True, tier="opus")
        with caplog.at_level("WARNING", logger="omlx.server"):
            self._call(_request(_classifier_payload()), server)
        assert any(
            "opus" in record.message and "no model" in record.message
            for record in caplog.records
        )

    def test_enabled_but_not_a_classifier_request_returns_none(self, monkeypatch):
        server = self._server_state_with(
            monkeypatch, steer=True, haiku_model="local-haiku-4bit"
        )
        result = self._call(
            _request(_normal_turn_payload(stream=False, tools=[])), server
        )
        assert result is None

    def test_no_global_settings_returns_none(self, monkeypatch):
        import omlx.server as server

        state = server.ServerState()
        state.global_settings = None
        monkeypatch.setattr(server, "_server_state", state)
        result = self._call(_request(_classifier_payload()), server)
        assert result is None

    def test_predicate_raising_does_not_propagate(self, monkeypatch):
        server = self._server_state_with(
            monkeypatch, steer=True, haiku_model="local-haiku-4bit"
        )

        def boom(_request):
            raise RuntimeError("simulated future schema change")

        monkeypatch.setattr(server, "is_auto_mode_classifier_request", boom)
        result = self._call(_request(_classifier_payload()), server)
        assert result is None

    def test_sonnet_tier_selects_sonnet_model(self, monkeypatch):
        server = self._server_state_with(
            monkeypatch,
            steer=True,
            tier="sonnet",
            haiku_model="h",
            sonnet_model="s",
            opus_model="o",
        )
        result = self._call(_request(_classifier_payload()), server)
        assert result == "s"


# -----------------------------------------------------------------------------
# Generative coverage — property-test-gap-finder's original suggestions:
# randomized max_tokens / stop_sequences / transcript content, and randomized
# combinations of the three match conditions. The example-based tests above
# stay as regression fixtures for the captured Claude Code 2.1.246 envelope
# and are not replaced.
# -----------------------------------------------------------------------------

_DRIFT_REASON_STATIC_TEXT = (
    "max_tokens= exceeds expected ceiling 4096"
    "stop_sequences missing '</block>'"
    "messages missing <transcript> tags"
)


@given(
    max_tokens=st.integers(min_value=-(2**31), max_value=2**31),
    stop_sequences=st.lists(st.text(), max_size=5),
    transcript_secret=st.text(),
)
def test_classifier_envelope_drift_never_leaks_transcript_content(
    max_tokens, stop_sequences, transcript_secret
):
    """Drift reasons name only which field drifted, never message content."""
    # A short generated secret can coincidentally be a substring of the
    # reasons' own fixed wording (e.g. "b" inside "</block>") with no
    # relation to actual leakage — exclude that false-positive case.
    assume(not transcript_secret or transcript_secret not in _DRIFT_REASON_STATIC_TEXT)
    request = _request(
        _classifier_payload(
            max_tokens=max_tokens if max_tokens > 0 else 1,
            stop_sequences=stop_sequences,
            messages=[
                {
                    "role": "user",
                    "content": f"<transcript>{transcript_secret}</transcript> is this safe?",
                }
            ],
        )
    )
    reasons = classifier_envelope_drift(request)
    for reason in reasons:
        assert transcript_secret not in reason or transcript_secret == ""


@given(
    system_marker_present=st.booleans(),
    stream=st.booleans(),
    has_tools=st.booleans(),
)
def test_is_auto_mode_classifier_request_requires_all_three_conditions(
    system_marker_present, stream, has_tools
):
    system_text = CLASSIFIER_SYSTEM if system_marker_present else "you are a helpful assistant"
    request = _request(
        _classifier_payload(
            stream=stream,
            system=[{"type": "text", "text": system_text}],
            tools=(
                [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]
                if has_tools
                else None
            ),
        )
    )
    expected = system_marker_present and not stream and not has_tools
    assert is_auto_mode_classifier_request(request) == expected
