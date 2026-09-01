# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.structured_logging — Issue #2770."""

import json
import logging
import uuid

import pytest

from omlx.structured_logging import (
    StructuredJSONFormatter,
    StructuredTextFormatter,
    _redact_secrets,
    _truncate_prompt,
    configure_structured_logging,
    get_request_id,
    get_service_name,
    get_trace_id,
    log_cache_event,
    log_node_health_change,
    log_request_completed,
    log_scheduler_routing,
    sanitize_for_log,
    set_request_id,
    set_service_name,
    set_trace_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_json_logs(capture_handler):
    """Parse all JSON log lines written to *capture_handler*."""
    output = capture_handler.stream.getvalue()
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def _setup_capture(level="DEBUG"):
    """Configure root logger with a StringIO handler using JSON formatter.

    Returns (handler, logger) — caller must clean up by restoring root handlers.
    """
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(StructuredJSONFormatter())

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    return handler, root, original_handlers, original_level


def _teardown_capture(handler, original_handlers, original_level):
    root = logging.getLogger()
    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.setLevel(original_level)
    handler.close()


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


class TestLogNoSecrets:
    """API keys and full prompts must never appear in log output."""

    def test_openai_key_masked(self):
        key = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
        result = _redact_secrets(f"Using key: {key}")
        assert key not in result
        assert "****" in result

    def test_xai_key_masked(self):
        key = "xai-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
        result = _redact_secrets(f"api_key={key}")
        assert key not in result
        assert "****" in result

    def test_bearer_token_masked(self):
        token = "Bearer abcdefghijklmnopqrstuvwxyz123456"
        result = _redact_secrets(f"Auth header: {token}")
        assert "123456" not in result
        assert "****" in result

    def test_key_equals_value_masked(self):
        result = _redact_secrets("api_key: sk-realkeyvalue1234567890abcdef")
        assert "realkeyvalue" not in result
        assert "****" in result

    def test_short_key_not_masked(self):
        result = _redact_secrets("api_key=short")
        assert "short" in result

    def test_prompt_hash_truncated(self):
        prompt = "This is a very long system prompt that should never appear in logs verbatim."
        result = _truncate_prompt(prompt)
        assert prompt not in result
        assert result.startswith("sha256:")
        assert "len=" in result

    def test_none_prompt_returns_none(self):
        assert _truncate_prompt(None) is None

    def test_empty_prompt_returns_none(self):
        assert _truncate_prompt("") is None

    def test_sanitize_dict_recursive(self):
        data = {
            "api_key": "sk-supersecretkey1234567890abcdef",
            "nested": {"token": "Bearer abcdefghijklmnopqrstuvwxyz123456"},
        }
        sanitized = sanitize_for_log(data)
        assert "supersecretkey" not in str(sanitized)
        assert "****" in str(sanitized)

    def test_full_prompt_not_in_json_output(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            secret_prompt = "You are a secret assistant with API key sk-topsecret1234567890abcdef"
            logger = logging.getLogger("omlx.test.secrets")
            logger.info(
                "Processing prompt: %s",
                secret_prompt,
                extra={"event": "test.secret", "prompt_text": secret_prompt},
            )
            records = _collect_json_logs(handler)
            assert len(records) == 1
            full_text = json.dumps(records[0])
            assert "topsecret1234567890abcdef" not in full_text
            assert "****" in full_text
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------


class TestRequestCompletedLogSchema:
    """Emitted JSON must contain all required fields per acceptance criteria."""

    def test_request_completed_has_required_fields(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            rid = str(uuid.uuid4())
            log_request_completed(
                request_id=rid,
                model="test-model",
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=10,
                prefill_duration=0.5,
                generation_duration=1.2,
                finish_reason="stop",
                stream=True,
                status_code=200,
            )
            records = _collect_json_logs(handler)
            assert len(records) == 1
            rec = records[0]

            # Required fields
            assert "timestamp" in rec
            assert "level" in rec
            assert "service" in rec
            assert "event" in rec
            assert "request_id" in rec

            # Event-specific fields
            assert rec["event"] == "request.completed"
            assert rec["model"] == "test-model"
            assert rec["prompt_tokens"] == 100
            assert rec["completion_tokens"] == 50
            assert rec["cached_tokens"] == 10
            assert rec["prefill_duration_s"] == 0.5
            assert rec["generation_duration_s"] == 1.2
            assert rec["total_duration_s"] == 1.7
            assert rec["finish_reason"] == "stop"
            assert rec["stream"] is True
            assert rec["status_code"] == 200

            # Timestamp is ISO 8601
            from datetime import datetime as dt
            dt.fromisoformat(rec["timestamp"].replace("Z", "+00:00"))
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_request_completed_with_trace_id(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            tid = "abc123def456"
            set_trace_id(tid)
            try:
                rid = str(uuid.uuid4())
                log_request_completed(
                    request_id=rid,
                    model="test-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                )
                records = _collect_json_logs(handler)
                assert records[0]["trace_id"] == tid
            finally:
                set_trace_id(None)
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_request_completed_service_default(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            log_request_completed(
                request_id="r1",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
            )
            records = _collect_json_logs(handler)
            assert records[0]["service"] == "omlx-inference"
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_request_completed_service_custom(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            set_service_name("omlx-cache")
            try:
                log_request_completed(
                    request_id="r1",
                    model="m",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
                records = _collect_json_logs(handler)
                assert records[0]["service"] == "omlx-cache"
            finally:
                set_service_name("omlx-inference")
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)


# ---------------------------------------------------------------------------
# Format toggle
# ---------------------------------------------------------------------------


class TestLogFormatToggle:
    """json/text format switching works."""

    def test_json_mode_produces_valid_json(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            logger = logging.getLogger("omlx.test.toggle")
            logger.info("Plain message")
            records = _collect_json_logs(handler)
            assert len(records) == 1
            assert isinstance(records[0], dict)
            assert records[0]["event"] == "log"
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_text_mode_produces_human_readable(self):
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredTextFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
        ))

        root = logging.getLogger()
        orig_handlers = root.handlers[:]
        orig_level = root.level
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)

        try:
            logger = logging.getLogger("omlx.test.text")
            logger.info("Human readable message")
            output = stream.getvalue()
            assert "Human readable message" in output
            # Should NOT be JSON
            assert not output.strip().startswith("{")
        finally:
            root.handlers.clear()
            root.handlers.extend(orig_handlers)
            root.setLevel(orig_level)
            handler.close()

    def test_text_mode_includes_event_prefix(self):
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredTextFormatter(
            "%(message)s"
        ))

        root = logging.getLogger()
        orig_handlers = root.handlers[:]
        orig_level = root.level
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)

        try:
            log_request_completed(
                request_id="r1",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
                logger_name="omlx.test.text.event",
            )
            output = stream.getvalue()
            assert "[request.completed]" in output
        finally:
            root.handlers.clear()
            root.handlers.extend(orig_handlers)
            root.setLevel(orig_level)
            handler.close()


# ---------------------------------------------------------------------------
# Event-specific fields
# ---------------------------------------------------------------------------


class TestEventSpecificFields:
    """Each event type includes its specific fields."""

    def test_cache_event_fields(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            log_cache_event(
                event="cache.eviction",
                model="test-model",
                cache_type="paged",
                entries=1000,
                evicted_entries=50,
                hit_rate=0.85,
                cache_size_bytes=1048576,
            )
            records = _collect_json_logs(handler)
            assert len(records) == 1
            rec = records[0]
            assert rec["event"] == "cache.eviction"
            assert rec["model"] == "test-model"
            assert rec["cache_type"] == "paged"
            assert rec["entries"] == 1000
            assert rec["evicted_entries"] == 50
            assert rec["hit_rate"] == 0.85
            assert rec["cache_size_bytes"] == 1048576
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_node_health_change_fields(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            log_node_health_change(
                node_id="node-1",
                previous_status="healthy",
                current_status="unhealthy",
                reason="heartbeat timeout",
                rank=0,
                deployment_id="dep-abc",
            )
            records = _collect_json_logs(handler)
            assert len(records) == 1
            rec = records[0]
            assert rec["event"] == "node.health_change"
            assert rec["node_id"] == "node-1"
            assert rec["previous_status"] == "healthy"
            assert rec["current_status"] == "unhealthy"
            assert rec["reason"] == "heartbeat timeout"
            assert rec["rank"] == 0
            assert rec["deployment_id"] == "dep-abc"
            assert rec["level"] == "WARNING"
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_node_health_change_healthy_is_info(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            log_node_health_change(
                node_id="node-1",
                previous_status="unhealthy",
                current_status="healthy",
                reason="recovered",
            )
            records = _collect_json_logs(handler)
            assert records[0]["level"] == "INFO"
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_scheduler_routing_fields(self):
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            log_scheduler_routing(
                request_id="req-42",
                model="llama-7b",
                queue_depth=3,
                priority=1,
                estimated_wait_ms=150.5,
            )
            records = _collect_json_logs(handler)
            assert len(records) == 1
            rec = records[0]
            assert rec["event"] == "scheduler.routing"
            assert rec["request_id"] == "req-42"
            assert rec["model"] == "llama-7b"
            assert rec["queue_depth"] == 3
            assert rec["priority"] == 1
            assert rec["estimated_wait_ms"] == 150.5
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_request_id_propagation(self):
        """request_id set via context variable is included in events."""
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            rid = str(uuid.uuid4())
            set_request_id(rid)
            try:
                log_cache_event(
                    event="cache.hit",
                    model="m",
                )
                records = _collect_json_logs(handler)
                assert records[0]["request_id"] == rid
            finally:
                set_request_id(None)
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_explicit_request_id_over_context(self):
        """Explicit request_id on emitter takes precedence over context."""
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            set_request_id("context-rid")
            try:
                log_request_completed(
                    request_id="explicit-rid",
                    model="m",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
                records = _collect_json_logs(handler)
                assert records[0]["request_id"] == "explicit-rid"
            finally:
                set_request_id(None)
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)

    def test_no_extra_fields_when_absent(self):
        """Fields not provided should not pollute the log record."""
        handler, root, orig_handlers, orig_level = _setup_capture()
        try:
            log_request_completed(
                request_id="r1",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
            )
            records = _collect_json_logs(handler)
            rec = records[0]
            assert "hit_rate" not in rec
            assert "evicted_entries" not in rec
            assert "node_id" not in rec
        finally:
            _teardown_capture(handler, orig_handlers, orig_level)


# ---------------------------------------------------------------------------
# Configure integration
# ---------------------------------------------------------------------------


class TestConfigureStructuredLogging:
    """Integration test for configure_structured_logging."""

    def test_configure_json_mode(self):
        import io

        stream = io.StringIO()
        # Patch sys.stderr to capture
        import sys
        original_stderr = sys.stderr
        sys.stderr = stream
        try:
            configure_structured_logging(level="INFO", format="json")
            logger = logging.getLogger("omlx.test.configure")
            logger.info("Configured test")
            # Should produce JSON on stderr (via root handler)
        finally:
            sys.stderr = original_stderr
            # Restore root
            root = logging.getLogger()
            root.handlers.clear()
            root.setLevel(logging.WARNING)

    def test_configure_text_mode(self):
        import sys

        original_stderr = sys.stderr
        try:
            configure_structured_logging(level="DEBUG", format="text")
            # Just ensure it doesn't crash
        finally:
            sys.stderr = original_stderr
            root = logging.getLogger()
            root.handlers.clear()
            root.setLevel(logging.WARNING)
