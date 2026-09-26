"""Live transcript tailing for Harbor ``pi`` agent trials.

Harbor's pi agent tees its ``--mode json`` event stream (minus
``message_update`` deltas, merged with stderr) to
``<job_dir>/<trial_dir>/agent/pi.txt``. This module reduces those lines into
compact display events for the admin transcript viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agentic import trial_outcome

MAX_TEXT_CHARS = 20000
_MAX_READ_BYTES = 4 * 1024 * 1024


def _clip(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return f"{text[:MAX_TEXT_CHARS]}\n… [truncated {len(text) - MAX_TEXT_CHARS} chars]"


def _clip_args(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, dict):
        return {k: _clip_args(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_args(v) for v in value]
    return value


def _joined_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def reduce_pi_line(line: str) -> list[dict]:
    """Map one raw ``pi.txt`` line to zero or more display events."""
    line = line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except ValueError:
        return [{"kind": "stderr", "text": _clip(line)}]
    if not isinstance(obj, dict):
        return [{"kind": "stderr", "text": _clip(line)}]

    kind = obj.get("type")
    if kind == "tool_execution_end":
        result = obj.get("result") or {}
        return [{
            "kind": "tool_result",
            "id": obj.get("toolCallId"),
            "name": obj.get("toolName"),
            "text": _clip(_joined_text(result.get("content") if isinstance(result, dict) else "")),
            "is_error": bool(obj.get("isError")),
        }]
    if kind == "auto_retry_start":
        delay = (obj.get("delayMs") or 0) / 1000
        return [{
            "kind": "stderr",
            "text": f"↻ retry {obj.get('attempt')}/{obj.get('maxAttempts')} in {delay:g}s",
        }]
    if kind != "message_end":
        return []

    message = obj.get("message") or {}
    role = message.get("role")
    content = message.get("content")
    if role == "user":
        text = _joined_text(content)
        return [{"kind": "user", "text": _clip(text)}] if text.strip() else []
    if role != "assistant" or not isinstance(content, list):
        return []

    events: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text") or ""
            if text.strip():
                events.append({"kind": "assistant", "text": _clip(text)})
        elif btype == "thinking":
            text = block.get("thinking") or ""
            if text.strip():
                events.append({"kind": "thinking", "text": _clip(text)})
        elif btype == "toolCall":
            args = block.get("arguments")
            events.append({
                "kind": "tool_call",
                "id": block.get("id"),
                "name": block.get("name"),
                "args": _clip_args(args if isinstance(args, dict) else {}),
            })
    # Failed model calls (HTTP errors, aborts) carry no content, only this.
    if message.get("errorMessage"):
        events.append({"kind": "error", "text": _clip(str(message["errorMessage"]))})
    return events


@dataclass
class TrialLog:
    dir: Path
    offset: int = 0
    partial: str = ""
    status: str = "running"


def _finished_status(trial_dir: Path) -> str | None:
    try:
        data = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("finished_at"):
        return None
    return trial_outcome(data)


class AgentLogTailer:
    """Incrementally reads every trial transcript under one Harbor job dir."""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.trials: dict[str, TrialLog] = {}

    def poll(self) -> list[dict]:
        payloads: list[dict] = []
        try:
            dirs = sorted(
                p for p in self.job_dir.iterdir()
                if p.is_dir() and (p / "config.json").is_file()
            )
        except OSError:
            return payloads
        for trial_dir in dirs:
            name = trial_dir.name
            task = name.rsplit("__", 1)[0]
            trial = self.trials.get(name)
            if trial is None:
                trial = self.trials[name] = TrialLog(trial_dir)
                payloads.append({"type": "trial", "trial": name, "task": task, "status": "running"})

            # Check completion before reading so the final flush sees every byte
            # written before result.json appeared.
            status = _finished_status(trial_dir) if trial.status == "running" else trial.status
            events = self._read_new(trial, final=status is not None)
            if events:
                payloads.append({"type": "events", "trial": name, "events": events})
            if status is not None and status != trial.status:
                trial.status = status
                payloads.append({"type": "trial", "trial": name, "task": task, "status": status})
        return payloads

    def finish(self) -> list[dict]:
        """Final poll after the job ended; trials without a result become ``incomplete``."""
        payloads = self.poll()
        for name, trial in self.trials.items():
            if trial.status != "running":
                continue
            events = self._read_new(trial, final=True)
            if events:
                payloads.append({"type": "events", "trial": name, "events": events})
            trial.status = "incomplete"
            payloads.append({
                "type": "trial", "trial": name,
                "task": name.rsplit("__", 1)[0], "status": "incomplete",
            })
        return payloads

    @staticmethod
    def _read_new(trial: TrialLog, final: bool) -> list[dict]:
        events: list[dict] = []
        try:
            with open(trial.dir / "agent" / "pi.txt", "rb") as f:
                f.seek(trial.offset)
                chunk = f.read(_MAX_READ_BYTES)
        except OSError:
            chunk = b""
        if chunk:
            # Advance only past complete lines so a half-written UTF-8 line stays
            # on disk; consume everything once the trial is finished or a single
            # line exceeds the read cap.
            cut = chunk.rfind(b"\n") + 1
            if final or (cut == 0 and len(chunk) == _MAX_READ_BYTES):
                cut = len(chunk)
            trial.offset += cut
            text = trial.partial + chunk[:cut].decode("utf-8", errors="replace")
            lines = text.split("\n")
            trial.partial = lines.pop()
            for line in lines:
                events.extend(reduce_pi_line(line))
        if final and trial.partial:
            events.extend(reduce_pi_line(trial.partial))
            trial.partial = ""
        return events
