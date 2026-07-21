from omlx.request_attribution import (
    REQUEST_ATTRIBUTION_MAX_VALUE_LENGTH,
    request_attribution_from_headers,
)


def test_extracts_explicit_request_attribution_headers():
    metadata = request_attribution_from_headers(
        {
            "x-omlx-source": "openclaw",
            "x-omlx-agent": "main",
            "x-omlx-channel": "telegram",
            "x-omlx-session-name": "Support chat",
            "x-omlx-session-id": "session-123",
            "x-omlx-conversation-id": "chat-456",
            "x-request-id": "caller-789",
        }
    )

    assert metadata == {
        "source": "openclaw",
        "agent": "main",
        "channel": "telegram",
        "session_name": "Support chat",
        "session_id": "session-123",
        "conversation_id": "chat-456",
        "external_request_id": "caller-789",
    }


def test_request_attribution_is_bounded_and_control_characters_are_collapsed():
    metadata = request_attribution_from_headers(
        {"x-omlx-session-name": "  hello\n\tworld  " + "x" * 300}
    )

    assert metadata["session_name"].startswith("hello world")
    assert len(metadata["session_name"]) == REQUEST_ATTRIBUTION_MAX_VALUE_LENGTH


def test_known_user_agent_is_only_a_best_effort_source_fallback():
    assert request_attribution_from_headers({"user-agent": "Hermes-Agent/1.2"}) == {
        "source": "hermes"
    }
    assert request_attribution_from_headers({"user-agent": "python-httpx/0.28"}) == {}


def test_explicit_source_wins_over_user_agent():
    assert (
        request_attribution_from_headers(
            {"x-omlx-source": "openclaw", "user-agent": "Hermes-Agent/1.2"}
        )["source"]
        == "openclaw"
    )
