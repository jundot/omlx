"""Bounded, display-only attribution for inference requests."""

from collections.abc import Mapping

REQUEST_ATTRIBUTION_HEADERS = {
    "x-omlx-source": "source",
    "x-omlx-agent": "agent",
    "x-omlx-channel": "channel",
    "x-omlx-session-name": "session_name",
    "x-omlx-session-id": "session_id",
    "x-omlx-conversation-id": "conversation_id",
    "x-request-id": "external_request_id",
}
REQUEST_ATTRIBUTION_MAX_VALUE_LENGTH = 160


def request_attribution_from_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Extract bounded attribution labels from case-insensitive HTTP headers."""
    attribution: dict[str, str] = {}
    for header, key in REQUEST_ATTRIBUTION_HEADERS.items():
        value = headers.get(header)
        if not value:
            continue
        # Collapse whitespace/control characters before returning values to the
        # admin API. The template renders with x-text, but keep the API safe too.
        value = " ".join(value.split())[:REQUEST_ATTRIBUTION_MAX_VALUE_LENGTH]
        if value:
            attribution[key] = value

    # Best-effort fallback only; precise harness/session data must be explicit.
    if "source" not in attribution:
        user_agent = (headers.get("user-agent") or "").lower()
        for source in ("openclaw", "hermes", "pi"):
            if source in user_agent:
                attribution["source"] = source
                break
    return attribution
