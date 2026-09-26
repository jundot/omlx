# SPDX-License-Identifier: Apache-2.0
"""Tests for the live Harbor agent transcript viewer."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import accuracy_benchmark
from omlx.admin import routes as admin_routes
from omlx.eval.agent_logs import MAX_TEXT_CHARS, AgentLogTailer, reduce_pi_line

ROOT = Path(__file__).resolve().parents[1]


def _line(obj: dict) -> str:
    return json.dumps(obj) + "\n"


ASSISTANT = {
    "type": "message_end",
    "message": {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan it"},
            {"type": "text", "text": "Editing now."},
            {"type": "text", "text": "  \n"},
            {
                "type": "toolCall", "id": "call_1", "name": "edit",
                "arguments": {"path": "/app/a.py", "edits": [{"oldText": "x = 1", "newText": "x = 2"}]},
            },
        ],
    },
}


class TestReducePiLine:
    def test_assistant_blocks_keep_order_and_skip_blank_text(self):
        events = reduce_pi_line(json.dumps(ASSISTANT))
        assert [e["kind"] for e in events] == ["thinking", "assistant", "tool_call"]
        assert events[0]["text"] == "plan it"
        call = events[2]
        assert call["name"] == "edit"
        assert call["args"]["edits"][0]["oldText"] == "x = 1"

    def test_tool_result_error_flag(self):
        events = reduce_pi_line(json.dumps({
            "type": "tool_execution_end", "toolCallId": "call_1", "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "boom"}]}, "isError": True,
        }))
        assert events == [{"kind": "tool_result", "id": "call_1", "name": "bash", "text": "boom", "is_error": True}]

    def test_noise_and_duplicates_are_dropped(self):
        assert reduce_pi_line(json.dumps({"type": "message_start", "message": {"role": "assistant"}})) == []
        # toolResult messages duplicate tool_execution_end; system prompts are huge.
        for role in ("toolResult", "system"):
            msg = {"type": "message_end", "message": {"role": role, "content": [{"type": "text", "text": "x"}]}}
            assert reduce_pi_line(json.dumps(msg)) == []

    def test_model_errors_and_retries_are_visible(self):
        failed = {"type": "message_end", "message": {
            "role": "assistant", "content": [], "stopReason": "error",
            "errorMessage": "507: model does not fit",
        }}
        assert reduce_pi_line(json.dumps(failed)) == [{"kind": "error", "text": "507: model does not fit"}]
        retry = {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000}
        assert reduce_pi_line(json.dumps(retry)) == [{"kind": "stderr", "text": "↻ retry 1/3 in 2s"}]

    def test_non_json_is_stderr(self):
        assert reduce_pi_line("npm WARN x\n") == [{"kind": "stderr", "text": "npm WARN x"}]

    def test_long_text_is_truncated(self):
        msg = {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "a" * (MAX_TEXT_CHARS + 7)}]}}
        (event,) = reduce_pi_line(json.dumps(msg))
        assert event["text"].startswith("a" * MAX_TEXT_CHARS)
        assert event["text"].endswith("[truncated 7 chars]")


def _trial(job: Path, name: str) -> Path:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text("{}")
    return trial


class TestAgentLogTailer:
    def test_partial_lines_wait_and_status_follows_result(self, tmp_path):
        job = tmp_path / "terminalbench_4-deadbeef"
        a = _trial(job, "alpha__AAAAAAA")
        _trial(job, "beta__BBBBBBB")
        (job / "config.json").write_text("{}")
        line = _line(ASSISTANT)
        pi = a / "agent" / "pi.txt"
        pi.write_text(line[:40])

        tailer = AgentLogTailer(job)
        first = tailer.poll()
        assert [(p["type"], p["trial"], p["status"]) for p in first] == [
            ("trial", "alpha__AAAAAAA", "running"),
            ("trial", "beta__BBBBBBB", "running"),
        ]
        assert first[0]["task"] == "alpha"

        with open(pi, "a") as f:
            f.write(line[40:])
        second = tailer.poll()
        assert len(second) == 1 and second[0]["type"] == "events"
        assert [e["kind"] for e in second[0]["events"]] == ["thinking", "assistant", "tool_call"]

        (a / "result.json").write_text(json.dumps({
            "finished_at": "2026-01-01T00:00:00",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }))
        assert tailer.poll() == [{"type": "trial", "trial": "alpha__AAAAAAA", "task": "alpha", "status": "pass"}]
        assert tailer.poll() == []

    def test_finish_flushes_tail_and_marks_unfinished(self, tmp_path):
        job = tmp_path / "terminalbench_4-deadbeef"
        a = _trial(job, "alpha__AAAAAAA")
        (a / "agent" / "pi.txt").write_text("killed mid-li")
        tailer = AgentLogTailer(job)
        tailer.poll()
        payloads = tailer.finish()
        assert payloads == [
            {"type": "events", "trial": "alpha__AAAAAAA", "events": [{"kind": "stderr", "text": "killed mid-li"}]},
            {"type": "trial", "trial": "alpha__AAAAAAA", "task": "alpha", "status": "incomplete"},
        ]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(accuracy_benchmark, "_agent_jobs_dir", lambda: tmp_path)

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


class TestAgentLogsRoute:
    @pytest.mark.parametrize("name", ["foo", "%2E%2E", "terminalbench_4-DEADBEEF", "mmlu-deadbeef"])
    def test_rejects_invalid_job_names(self, client, name):
        assert client.get(f"/admin/api/bench/accuracy/agent-logs/{name}/stream").status_code == 400

    def test_unknown_job_is_404(self, client):
        assert client.get("/admin/api/bench/accuracy/agent-logs/terminalbench_4-deadbeef/stream").status_code == 404

    def test_finished_job_replays_and_ends(self, client, tmp_path):
        trial = _trial(tmp_path / "terminalbench_4-deadbeef", "alpha__AAAAAAA")
        (trial / "agent" / "pi.txt").write_text(_line(ASSISTANT))
        with client.stream("GET", "/admin/api/bench/accuracy/agent-logs/terminalbench_4-deadbeef/stream") as resp:
            assert resp.status_code == 200
            frames = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
        assert [f["type"] for f in frames] == ["reset", "trial", "events", "trial", "end"]
        assert frames[-2]["status"] == "incomplete"


def test_dashboard_helpers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    result = subprocess.run(
        [node, "--test", str(ROOT / "tests/agent_logs_ui.test.cjs")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
