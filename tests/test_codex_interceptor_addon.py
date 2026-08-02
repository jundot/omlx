"""Cover the mitmproxy addon glue that no other suite can reach.

mitmproxy ships its own interpreter, so the addon is never imported by the
deterministic suite and its wiring went unverified. A minimal stub is enough to
exercise the response paths where a mistyped hand-off silently failed every
compaction turn.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
ADDON_PATH = REPO_ROOT / "omlx" / "codex_interceptor" / "addon.py"


class FakeHeaders(dict):
    def get(self, key, default=""):
        for name, value in self.items():
            if name.lower() == str(key).lower():
                return value
        return default

    def pop(self, key, default=None):
        for name in list(self):
            if name.lower() == str(key).lower():
                return super().pop(name, default)
        return default


class FakeResponse:
    def __init__(self, status_code, content, headers):
        self.status_code = status_code
        self.reason = ""
        self.headers = FakeHeaders(headers)
        self._content = content

    @staticmethod
    def make(status_code=200, content=b"", headers=None):
        return FakeResponse(status_code, content, headers or {})

    def get_content(self, strict=True):
        del strict
        return self._content

    def set_content(self, content):
        self._content = content

    def decode(self, strict=True):
        del strict


class FakeRequest:
    def __init__(self, content=b""):
        self.headers = FakeHeaders({})
        self.method = "POST"
        self.host = "chatgpt.com"
        self.path = "/backend-api/codex/responses"
        self.url = "https://chatgpt.com/backend-api/codex/responses"
        self._content = content

    def get_content(self, strict=True):
        del strict
        return self._content

    def set_content(self, content):
        self._content = content

    def decode(self, strict=True):
        del strict


class FakeFlow:
    def __init__(self, response=None, request_content=b""):
        self.id = "flow-1"
        self.metadata: dict = {}
        self.request = FakeRequest(request_content)
        self.response = response
        self.error = None
        self.websocket = None


class RecordingCommands:
    def __init__(self):
        self.injected: list[bytes] = []

    def call(self, name, flow, from_client, payload):
        del name, flow, from_client
        self.injected.append(payload)


def load_addon(runtime: Path):
    stub = types.ModuleType("mitmproxy")
    http_module = types.ModuleType("mitmproxy.http")
    http_module.HTTPFlow = FakeFlow
    http_module.Response = FakeResponse
    ctx_module = types.ModuleType("mitmproxy.ctx")
    ctx_module.master = types.SimpleNamespace(commands=RecordingCommands())
    stub.http = http_module
    stub.ctx = ctx_module
    sys.modules["mitmproxy"] = stub
    sys.modules["mitmproxy.http"] = http_module
    sys.modules["mitmproxy.ctx"] = ctx_module
    os.environ.update(
        {
            "OMLX_CODEX_INTERCEPTOR_UPSTREAM_URL": "http://127.0.0.1:9999/v1/responses",
            "OMLX_CODEX_INTERCEPTOR_MODEL": "claude-opus-5",
            "OMLX_CODEX_INTERCEPTOR_STATUS_PATH": str(runtime / "status.jsonl"),
            "OMLX_CODEX_INTERCEPTOR_SESSION_ID": "test-session",
        }
    )
    spec = importlib.util.spec_from_file_location(
        "omlx.codex_interceptor.addon_under_test", ADDON_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, ctx_module.master.commands


class AddonTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        runtime = Path(self._tmp.name)
        self.module, self.commands = load_addon(runtime)
        self.interceptor = self.module.CodexLocalInterceptor()
        self.status_path = runtime / "status.jsonl"

    def events(self) -> list[dict]:
        return [json.loads(payload) for payload in self.commands.injected]

    def recorded(self) -> list[dict]:
        if not self.status_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.status_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def compaction_payload(self) -> dict:
        return {
            "model": "gpt-5.6-sol",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "keep routing local"}],
                }
            ],
            "context_management": {"compaction": {"enabled": True}},
        }

    def test_status_writer_rejects_content_and_raw_error_details(self):
        self.interceptor._record(
            "privacy_boundary",
            error="ValueError",
            input="secret prompt",
            output="secret answer",
            authorization="Bearer secret",
            error_detail="secret backend detail",
            upstream_error_message="secret upstream detail",
        )

        event = self.recorded()[-1]
        self.assertEqual(event["error"], "ValueError")
        self.assertFalse(
            {
                "input",
                "output",
                "authorization",
                "error_detail",
                "upstream_error_message",
            }
            & event.keys()
        )


class AddonCompactionTests(AddonTestCase):
    def test_a_failed_compaction_turn_is_answered_with_a_local_summary(self):
        failure = json.dumps(
            {
                "error": {
                    "type": "claude_cli_bridge_error",
                    "message": "this request already failed 2 times",
                }
            }
        ).encode("utf-8")
        flow = FakeFlow(
            FakeResponse(422, failure, {"Content-Type": "application/json"})
        )
        flow.metadata.update(
            {
                "harness_intercepted": True,
                "harness_compaction_request": True,
                "harness_compaction_payload": self.compaction_payload(),
                "harness_client_stream": True,
                "harness_requested_model": "gpt-5.6-sol",
            }
        )
        self.interceptor.response(flow)
        self.assertEqual(flow.response.status_code, 200)
        body = flow.response.get_content(strict=False).decode("utf-8")
        completed = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: {")
        ][-1]
        self.assertEqual(completed["type"], "response.completed")
        self.assertEqual(completed["response"]["output"][0]["type"], "compaction")
        self.assertIsNone(completed["response"]["error"])
        self.assertIn(
            "compaction_short_circuited",
            [event["event"] for event in self.recorded()],
        )

    def test_a_compaction_stream_is_reframed_for_the_websocket_bridge(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"The summary."}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n'
            b"data: [DONE]\n\n"
        )
        events = asyncio.run(self._collect_compaction_events(raw))
        self.assertEqual(events[-1]["type"], "response.completed")
        item = events[-1]["response"]["output"][0]
        self.assertEqual(item["type"], "compaction")
        self.assertEqual(item["encrypted_content"], "The summary.")

    def test_synthetic_websocket_compaction_is_remembered_for_the_next_turn(self):
        flow = FakeFlow()
        state = self.module.ResponsesWebSocketState()
        payload = {"type": "response.create", **self.compaction_payload()}
        payload["context_management"] = [{"type": "compaction"}]
        served = self.interceptor._serve_synthetic_compaction(
            flow,
            state,
            payload,
            started=0.0,
            requested_model="gpt-5.6-sol",
            reason="test",
            detail="forced fallback",
        )
        self.assertTrue(served)
        response_id = self.events()[-1]["response"]["id"]
        expanded = state.expand(
            {
                "type": "response.create",
                "model": "gpt-5.3-codex-spark",
                "previous_response_id": response_id,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "continue"}],
                    }
                ],
            }
        )
        self.assertTrue(
            any(item.get("type") == "compaction" for item in expanded["input"])
        )

    async def _collect_compaction_events(self, raw: bytes) -> list[dict]:
        """Drive the streaming reader with a canned upstream response."""
        collected: list[dict] = []
        queue: asyncio.Queue = asyncio.Queue()
        for line in raw.splitlines(keepends=True):
            queue.put_nowait(("data", line))
        queue.put_nowait(("done", None))

        async def skip_upstream_worker(func, *args, **kwargs):
            del func, args, kwargs

        original_to_thread = self.module.asyncio.to_thread
        original_queue = self.module.asyncio.Queue
        self.module.asyncio.to_thread = skip_upstream_worker
        self.module.asyncio.Queue = lambda: queue
        try:
            events, _, _, _, _ = await self.interceptor._local_sse_events(
                {"model": "claude-opus-5", "input": []},
                registered_tools=set(),
                namespace_tools={},
                emittable_tools=set(),
                client_tool_mappings={},
                compaction=True,
                on_event=collected.append,
                on_first_byte=lambda milliseconds: None,
            )
        finally:
            self.module.asyncio.to_thread = original_to_thread
            self.module.asyncio.Queue = original_queue
        self.assertEqual(events, collected)
        return events


class AddonReliabilityTests(AddonTestCase):
    """The response paths that carry telemetry and bound repeated failure."""

    def local_flow(self, response=None):
        flow = FakeFlow(response)
        flow.metadata.update(
            {
                "harness_intercepted": True,
                "harness_request_digest": "digest-1",
                "harness_registered_tools": [],
                "harness_namespace_tools": {},
                "harness_client_tool_mappings": {},
                "harness_request_started": 0.0,
            }
        )
        return flow

    def sse(self, usage=None, text="done"):
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    ],
                    **({"usage": usage} if usage else {}),
                },
            },
        ]
        body = (
            b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
            + b"data: [DONE]\n\n"
        )
        return FakeResponse(200, body, {"Content-Type": "text/event-stream"})

    def test_the_receipt_records_tokens_and_cache_hit_rate(self):
        usage = {
            "input_tokens": 23975,
            "output_tokens": 64,
            "input_tokens_details": {"cached_tokens": 22528},
        }
        flow = self.local_flow(self.sse(usage))
        self.interceptor.response(flow)
        completed = [e for e in self.recorded() if e["event"] == "inference_completed"]
        self.assertEqual(len(completed), 1)
        # Cache-hit rate is the signal that showed the compaction prefix miss.
        self.assertEqual(completed[0]["cached_tokens"], 22528)
        self.assertEqual(completed[0]["cache_hit_percent"], 94.0)
        self.assertEqual(completed[0]["input_tokens"], 23975)

    def test_an_empty_model_turn_is_recorded_instead_of_swallowed(self):
        flow = self.local_flow(self.sse(text="   "))
        self.interceptor.response(flow)
        completed = [e for e in self.recorded() if e["event"] == "inference_completed"]
        # One recorded session produced 18 of these and said nothing about them.
        self.assertTrue(completed[0].get("empty_model_output"))

    def test_a_request_that_failed_twice_is_refused_without_re_running(self):
        failure = FakeResponse(
            400,
            json.dumps({"error": {"message": "Chat template error"}}).encode(),
            {"Content-Type": "application/json"},
        )
        for expected in (1, 2):
            flow = self.local_flow(failure)
            self.interceptor.response(flow)
            errors = [e for e in self.recorded() if e["event"] == "inference_error"]
            self.assertEqual(errors[-1]["local_failure_count"], expected)
        self.assertTrue(self.interceptor._local_failures_exhausted("digest-1"))
        # A success clears it, so a transient failure does not poison the digest.
        self.interceptor.local_failures.clear()
        self.assertFalse(self.interceptor._local_failures_exhausted("digest-1"))

    def test_a_prompt_size_rejection_does_not_count_toward_the_failure_cap(self):
        # A 413 is refused before any inference runs, so retries are cheap and
        # the real error must keep surfacing instead of a 422 lockout.
        too_large = FakeResponse(
            413,
            json.dumps(
                {
                    "error": {
                        "message": "normalized Codex prompt exceeds the "
                        "Kimi CLI bridge limit"
                    }
                }
            ).encode(),
            {"Content-Type": "application/json"},
        )
        for _ in range(3):
            flow = self.local_flow(too_large)
            self.interceptor.response(flow)
        errors = [e for e in self.recorded() if e["event"] == "inference_error"]
        self.assertEqual(len(errors), 3)
        self.assertNotIn("local_failure_count", errors[-1])
        self.assertFalse(self.interceptor._local_failures_exhausted("digest-1"))

    def test_startup_records_the_fingerprint_of_the_code_that_is_running(self):
        ready = [e for e in self.recorded() if e["event"] == "interceptor_ready"]
        self.assertTrue(ready)
        self.assertRegex(ready[-1]["code_fingerprint"], r"^[0-9a-f]{12}$")
        # Two instances exist under test (the module's own addon plus this one),
        # and both must fingerprint the same source.
        self.assertEqual(
            {event["code_fingerprint"] for event in ready},
            {ready[-1]["code_fingerprint"]},
        )

    def test_native_tool_capability_switches_http_from_buffering_to_streaming(self):
        flow = self.local_flow(
            FakeResponse(
                200,
                b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n',
                {"Content-Type": "text/event-stream"},
            )
        )
        flow.metadata["harness_emittable_tools"] = {"exec_command"}
        with mock.patch.dict(
            os.environ,
            {"OMLX_CODEX_INTERCEPTOR_NATIVE_TOOL_STREAMING": "1"},
        ):
            self.interceptor.responseheaders(flow)

        self.assertNotIn("harness_buffer_tool_stream", flow.metadata)
        self.assertTrue(callable(flow.response.stream))
        installed = [
            event
            for event in self.recorded()
            if event["event"] == "sse_normalizer_installed"
        ]
        self.assertTrue(installed[-1]["native_tool_streaming"])

    def test_capability_file_enables_native_tool_streaming_dynamically(self):
        path = self.status_path.parent / "capability.json"
        path.write_text(
            json.dumps(
                {
                    "model": self.interceptor.model,
                    "capabilities": {"native_function_call": True},
                }
            )
        )
        self.interceptor.capabilities_path = path

        self.assertTrue(self.interceptor._native_tool_streaming_enabled())
        self.assertEqual(
            [
                event["event"]
                for event in self.recorded()
                if event["event"] == "native_tool_streaming_enabled"
            ],
            ["native_tool_streaming_enabled"],
        )


class AddonRequestPathTests(AddonTestCase):
    """The request path: validation, loop guard, and replaying a paid answer."""

    def local_request(self, items, tools=None):
        payload = {
            "model": "gpt-5.3-codex-spark",
            "instructions": "You are Codex.",
            "input": items,
            "tools": tools
            if tools is not None
            else [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {"type": "object"},
                }
            ],
            "stream": True,
        }
        flow = FakeFlow(request_content=json.dumps(payload).encode())
        self.interceptor.local_slot = "gpt-5.3-codex-spark"
        return flow

    def answered(
        self,
        flow,
        body=b'data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":10,"output_tokens":2,"input_tokens_details":{"cached_tokens":8}}}}\n\ndata: [DONE]\n\n',
    ):
        flow.response = FakeResponse(200, body, {"Content-Type": "text/event-stream"})
        self.interceptor.response(flow)

    def test_an_identical_retry_is_replayed_instead_of_re_run(self):
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the files"}],
            }
        ]
        first = self.local_request(items)
        self.interceptor.request(first)
        self.assertIsNone(first.response)  # went upstream
        self.answered(first)
        second = self.local_request(items)
        self.interceptor.request(second)
        # A cancelled stream makes Codex re-send a turn already paid for.
        self.assertIsNotNone(second.response)
        self.assertEqual(second.response.status_code, 200)
        self.assertIn(
            "local_response_replayed",
            [event["event"] for event in self.recorded()],
        )

    def test_a_churn_run_is_nudged_on_the_request_path(self):
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "find GenerationBatch"}],
            }
        ]
        for index in range(7):
            suffix = ".generate" if index % 2 else ""
            items.append(
                {
                    "type": "function_call",
                    "call_id": f"c{index}",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": f"python3 -c 'import mlx_lm{suffix}; print(dir(mlx_lm))'"
                        }
                    ),
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": f"c{index}",
                    "output": "batch_generate, convert, generate, load, models",
                }
            )
        flow = self.local_request(items)
        self.interceptor.request(flow)
        loops = [e for e in self.recorded() if e["event"] == "doom_loop_detected"]
        self.assertTrue(loops)
        # Varying arguments that learn nothing: the byte-identical rule misses it.
        self.assertEqual(loops[-1]["loop_kind"], "churn")
        self.assertEqual(loops[-1]["loop_guard_action"], "nudged")
        sent = json.loads(flow.request.get_content(strict=False))
        self.assertIn("LOOP GUARD", json.dumps(sent["input"][-1]))

    def test_an_orphan_tool_result_is_recovered_before_upstream_inference(self):
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "go"}],
            },
            {"type": "function_call_output", "call_id": "missing", "output": "done"},
        ]
        flow = self.local_request(items)
        self.interceptor.request(flow)
        repaired = [
            e for e in self.recorded() if e["event"] == "local_request_repaired"
        ]
        self.assertTrue(repaired)
        self.assertIn(
            "tool_result_without_a_call", repaired[-1]["local_request_rules_broken"]
        )
        sent = json.loads(flow.request.get_content(strict=False))
        self.assertNotIn(
            "function_call_output", [item.get("type") for item in sent["input"]]
        )
        self.assertIn("done", json.dumps(sent["input"]))

    def test_an_unrepairable_local_request_is_refused_before_upstream(self):
        flow = self.local_request([{"type": "message", "role": "user", "content": []}])
        self.interceptor.request(flow)
        self.assertEqual(flow.response.status_code, 422)
        invalid = [e for e in self.recorded() if e["event"] == "local_request_invalid"]
        self.assertIn(
            "message_content_is_empty", invalid[-1]["local_request_rules_broken"]
        )


class AddonWebSocketPathTests(AddonTestCase):
    """The bridge path, which carried 100% of a real session's traffic.

    The validator, both memos and the token telemetry were originally wired only
    into the HTTP path, so none of them applied to a live session at all.
    """

    def serve(self, raw: bytes, payload=None, state=None):
        flow = FakeFlow()
        flow.metadata["harness_responses_websocket"] = True
        state = state if state is not None else self.module.ResponsesWebSocketState()
        payload = payload or {
            "type": "response.create",
            "model": "gpt-5.3-codex-spark",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "list the files"}],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {"type": "object"},
                }
            ],
        }
        self.interceptor.local_slot = "gpt-5.3-codex-spark"
        queue: asyncio.Queue = asyncio.Queue()
        for line in raw.splitlines(keepends=True):
            queue.put_nowait(("data", line))
        queue.put_nowait(("done", None))

        async def skip_upstream_worker(function, *arguments, **keywords):
            del function, arguments, keywords

        async def drive():
            original_to_thread = self.module.asyncio.to_thread
            original_queue = self.module.asyncio.Queue
            self.module.asyncio.to_thread = skip_upstream_worker
            self.module.asyncio.Queue = lambda: queue
            try:
                await self.interceptor._serve_local_websocket_request(
                    flow, state, payload, 0.0
                )
            finally:
                self.module.asyncio.to_thread = original_to_thread
                self.module.asyncio.Queue = original_queue

        asyncio.run(drive())
        return flow, state, payload

    def stream(self, usage=True):
        completed = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        }
        if usage:
            completed["response"]["usage"] = {
                "input_tokens": 23975,
                "output_tokens": 64,
                "input_tokens_details": {"cached_tokens": 22528},
            }
        return (
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            + b"data: "
            + json.dumps(completed).encode()
            + b"\n\n"
            + b"data: [DONE]\n\n"
        )

    def test_the_bridge_records_tokens_and_cache_hit_rate(self):
        self.serve(self.stream())
        done = [e for e in self.recorded() if e["event"] == "inference_completed"]
        self.assertTrue(done)
        self.assertEqual(done[-1]["transport"], "websocket_bridge")
        self.assertEqual(done[-1]["cached_tokens"], 22528)
        self.assertEqual(done[-1]["cache_hit_percent"], 94.0)

    def test_native_tool_canary_path_streams_without_textual_buffer_repair(self):
        with (
            mock.patch.dict(
                os.environ,
                {"OMLX_CODEX_INTERCEPTOR_NATIVE_TOOL_STREAMING": "1"},
            ),
            mock.patch.object(
                self.module,
                "adapt_textual_tool_call_sse",
                side_effect=AssertionError("native streams must not be buffered"),
            ),
        ):
            self.serve(self.stream())

        routed = [e for e in self.recorded() if e["event"] == "inference_routed"]
        self.assertTrue(routed[-1]["native_tool_streaming"])
        visible = [
            e for e in self.recorded() if e["event"] == "inference_first_visible_event"
        ]
        self.assertTrue(visible)
        self.assertGreaterEqual(visible[-1]["first_visible_ms"], 1)

    def test_large_prewarm_prefix_is_backgrounded_and_deduplicated(self):
        payload = {
            "model": "gpt-5.3-codex-spark",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "x" * 9000}],
                }
            ],
            "tools": [],
            "stream": True,
        }

        async def drive():
            gate = asyncio.Event()
            calls = []

            async def fake_prefill(digest, body, *, requested_model):
                calls.append((digest, len(body), requested_model))
                await gate.wait()

            with mock.patch.object(
                self.interceptor,
                "_run_prefix_prefill",
                side_effect=fake_prefill,
            ):
                self.interceptor._schedule_prefix_prefill(
                    payload,
                    requested_model="gpt-5.3-codex-spark",
                )
                self.interceptor._schedule_prefix_prefill(
                    payload,
                    requested_model="gpt-5.3-codex-spark",
                )
                tasks = list(self.interceptor.websocket_tasks)
                gate.set()
                await asyncio.gather(*tasks)
            return calls

        calls = asyncio.run(drive())
        self.assertEqual(len(calls), 1)
        events = [event["event"] for event in self.recorded()]
        self.assertIn("prefix_prefill_started", events)
        self.assertIn("prefix_prefill_deduplicated", events)

    def test_prefix_prefill_can_be_disabled_for_paid_cli_bridges(self):
        self.interceptor.prefix_prefill_enabled = False
        self.interceptor._schedule_prefix_prefill(
            {
                "model": "gpt-5.3-codex-spark",
                "input": "x" * 9000,
            },
            requested_model="gpt-5.3-codex-spark",
        )
        self.assertFalse(self.interceptor.websocket_tasks)
        self.assertNotIn(
            "prefix_prefill_started",
            [event["event"] for event in self.recorded()],
        )

    def test_prefix_prefill_worker_consumes_one_token_stream(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                received["authorization"] = self.headers.get("Authorization")
                received["payload"] = json.loads(self.rfile.read(size))
                body = (
                    b'data: {"type":"response.created","response":{"id":"prefill"}}\n\n'
                    b'data: {"type":"response.completed","response":{"id":"prefill"}}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.interceptor.upstream_url = (
                f"http://127.0.0.1:{server.server_port}/v1/responses"
            )
            self.interceptor.api_key = "prefill-key"
            body = json.dumps(
                {
                    "model": self.interceptor.model,
                    "input": "stable prefix",
                    "stream": True,
                    "max_output_tokens": 1,
                }
            ).encode()
            asyncio.run(
                self.interceptor._run_prefix_prefill(
                    "digest",
                    body,
                    requested_model="gpt-5.3-codex-spark",
                )
            )
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(received["authorization"], "Bearer prefill-key")
        self.assertEqual(received["payload"]["max_output_tokens"], 1)
        completed = [
            event
            for event in self.recorded()
            if event["event"] == "prefix_prefill_completed"
        ]
        self.assertEqual(completed[-1]["status"], 200)
        self.assertGreaterEqual(completed[-1]["first_byte_ms"], 1)

    def test_private_upstream_connections_are_reused(self):
        connections = set()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                connections.add(id(self.connection))
                size = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(size)
                body = b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        pool = self.module.PersistentUpstreamPool(
            f"http://127.0.0.1:{server.server_port}/v1/responses"
        )
        try:
            metrics = []
            for _ in range(2):
                with pool.stream(
                    b"{}",
                    {"Content-Type": "application/json"},
                    timeout=2,
                ) as (response, detail):
                    response.read()
                    metrics.append(detail)
        finally:
            pool.close()
            server.shutdown()
            server.server_close()

        self.assertFalse(metrics[0]["connection_reused"])
        self.assertTrue(metrics[1]["connection_reused"])
        self.assertEqual(len(connections), 1)

    def test_prefix_fingerprint_survives_an_interceptor_restart(self):
        path = Path(self._tmp.name) / "prefix-cache.json"
        self.interceptor.prefix_cache_path = path
        self.interceptor.prefix_prefills["stable"] = (time.time() + 60, "ok")
        self.interceptor._save_prefix_prefills()

        with mock.patch.dict(
            os.environ,
            {"OMLX_CODEX_INTERCEPTOR_PREFIX_CACHE_PATH": str(path)},
        ):
            restarted = self.module.CodexLocalInterceptor()
        try:
            self.assertEqual(restarted.prefix_prefills["stable"][1], "ok")
        finally:
            restarted.done()

    def test_residency_keepalive_is_a_one_token_private_request(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                received["payload"] = json.loads(self.rfile.read(size))
                body = b'data: {"type":"response.completed","response":{"id":"r"}}\n\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.interceptor.upstream_url = (
                f"http://127.0.0.1:{server.server_port}/v1/responses"
            )
            asyncio.run(self.interceptor._run_residency_keepalive())
        finally:
            self.interceptor.upstream_pool.close()
            server.shutdown()
            server.server_close()

        self.assertEqual(received["payload"]["max_output_tokens"], 1)
        self.assertEqual(received["payload"]["model"], self.interceptor.model)
        self.assertIn(
            "residency_keepalive_completed",
            [event["event"] for event in self.recorded()],
        )

    def test_performance_warnings_are_specific_and_privacy_safe(self):
        self.interceptor._record_performance_warnings(
            request_bytes=256 * 1024,
            input_tokens=16000,
            cache_hit_percent=5,
            transform_ms=1,
            first_byte_ms=6000,
            first_visible_ms=8000,
            connect_ms=300,
            connection_reused=False,
        )
        warnings = {
            event["warning_code"]
            for event in self.recorded()
            if event["event"] == "performance_warning"
        }
        self.assertEqual(
            warnings,
            {
                "oversized_request",
                "slow_connection",
                "slow_prefill",
                "delayed_visible_output",
                "low_prefix_cache_hit",
            },
        )

    def test_the_bridge_replays_an_identical_turn(self):
        state = self.module.ResponsesWebSocketState()
        flow, state, payload = self.serve(self.stream(), state=state)
        first_events = len(self.events())
        self.serve(self.stream(), payload=payload, state=state)
        self.assertIn(
            "local_response_replayed",
            [event["event"] for event in self.recorded()],
        )
        # The stored answer is re-injected, so the client still sees a full stream.
        self.assertGreater(len(self.events()), first_events)

    def test_the_bridge_recovers_an_orphan_tool_result(self):
        payload = {
            "type": "response.create",
            "model": "gpt-5.3-codex-spark",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "missing",
                    "output": "done",
                },
            ],
            "tools": [],
        }
        self.serve(self.stream(), payload=payload)
        repaired = [
            e for e in self.recorded() if e["event"] == "local_request_repaired"
        ]
        self.assertTrue(repaired)
        self.assertEqual(repaired[-1]["transport"], "websocket_bridge")
        self.assertIn(
            "tool_result_without_a_call", repaired[-1]["local_request_rules_broken"]
        )

    def test_the_bridge_refuses_an_unrepairable_request_without_inference(self):
        payload = {
            "type": "response.create",
            "model": "gpt-5.3-codex-spark",
            "input": [{"type": "message", "role": "user", "content": []}],
            "tools": [],
        }
        self.serve(self.stream(), payload=payload)
        self.assertEqual(self.events()[-1]["type"], "error")
        self.assertEqual(self.events()[-1]["status"], 422)


if __name__ == "__main__":
    unittest.main()
