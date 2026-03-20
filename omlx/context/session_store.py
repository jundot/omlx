# SPDX-License-Identifier: Apache-2.0
"""Persistent session log and rollover helpers for agent runtime."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..settings import AgentMemorySettings, GlobalSettings
from .memory import get_memory_root

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{4,}")


def _slugify(value: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in str(value).strip()
    ).strip("-")
    return cleaned or "default"


def _agent_key(metadata: dict[str, Any] | None) -> str:
    info = metadata or {}
    agent_id = info.get("agent_id")
    if isinstance(agent_id, str) and agent_id.strip():
        return _slugify(agent_id)
    context = info.get("omlx_context") or {}
    if isinstance(context, dict):
        if isinstance(context.get("workspace"), str) and context["workspace"].strip():
            return f"workspace-{_slugify(context['workspace'])}"
        if isinstance(context.get("profile"), str) and context["profile"].strip():
            return f"profile-{_slugify(context['profile'])}"
    return "default"


def get_sessions_root(
    settings: GlobalSettings,
    memory_settings: AgentMemorySettings | None = None,
) -> Path:
    """Resolve the root directory used for agent session logs."""
    return get_memory_root(settings, memory_settings) / "_sessions"


def _agent_sessions_dir(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    memory_settings: AgentMemorySettings | None = None,
) -> Path:
    return get_sessions_root(settings, memory_settings) / _agent_key(metadata)


def _session_dir(agent_dir: Path, session_id: str) -> Path:
    return agent_dir / session_id


def _session_log_path(session_dir: Path) -> Path:
    return session_dir / "session.jsonl"


def _session_meta_path(session_dir: Path) -> Path:
    return session_dir / "meta.json"


def _session_index_path(session_dir: Path) -> Path:
    return session_dir / "index.json"


def _session_handoff_path(session_dir: Path) -> Path:
    return session_dir / "HANDOFF.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_active_session(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    memory_settings: AgentMemorySettings | None = None,
) -> dict[str, Any]:
    """Ensure an active session exists for one agent and return its metadata."""
    memory = memory_settings or settings.agent_memory
    agent_dir = _agent_sessions_dir(settings, metadata, memory)
    agent_dir.mkdir(parents=True, exist_ok=True)
    current_path = agent_dir / "current.json"
    current = _read_json(current_path, {})
    session_id = current.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        session_id = _new_session_id()
        current = {
            "session_id": session_id,
            "created_at": _now_iso(),
            "agent_id": (metadata or {}).get("agent_id", "default"),
        }
        _write_json(current_path, current)

    session_dir = _session_dir(agent_dir, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_json(
        _session_meta_path(session_dir),
        {
            "session_id": session_id,
            "agent_id": (metadata or {}).get("agent_id", "default"),
            "created_at": current.get("created_at", _now_iso()),
            "last_response_id": None,
            "message_count": 0,
            "char_count": 0,
            "rolled_over": False,
        },
    )
    _write_json(_session_meta_path(session_dir), meta)
    return meta


def append_session_record(
    settings: GlobalSettings,
    public_response: dict[str, Any],
    input_messages: list[dict[str, Any]],
    output_messages: list[dict[str, Any]],
    memory_settings: AgentMemorySettings | None = None,
) -> dict[str, Any]:
    """Append one request/response exchange to the active session log."""
    memory = memory_settings or settings.agent_memory
    metadata = public_response.get("metadata") or {}
    session_meta = ensure_active_session(settings, metadata, memory)
    agent_dir = _agent_sessions_dir(settings, metadata, memory)
    session_dir = _session_dir(agent_dir, session_meta["session_id"])
    log_path = _session_log_path(session_dir)

    record = {
        "type": "exchange",
        "timestamp": _now_iso(),
        "response_id": public_response.get("id"),
        "previous_response_id": public_response.get("previous_response_id"),
        "input_messages": input_messages,
        "output_messages": output_messages,
        "metadata": metadata,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    delta_messages = len(input_messages) + len(output_messages)
    delta_chars = len(json.dumps(record, ensure_ascii=False))
    session_meta["last_response_id"] = public_response.get("id")
    session_meta["updated_at"] = _now_iso()
    session_meta["message_count"] = int(session_meta.get("message_count", 0)) + delta_messages
    session_meta["char_count"] = int(session_meta.get("char_count", 0)) + delta_chars
    _write_json(_session_meta_path(session_dir), session_meta)

    if (
        session_meta["message_count"] >= memory.session_rollover_messages
        or session_meta["char_count"] >= memory.session_rollover_chars
    ):
        rollover_session(settings, metadata, memory_settings=memory)

    return session_meta


def rollover_session(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    memory_settings: AgentMemorySettings | None = None,
) -> dict[str, Any]:
    """Finalize the current session with index/handoff and create a new one."""
    memory = memory_settings or settings.agent_memory
    agent_dir = _agent_sessions_dir(settings, metadata, memory)
    current_path = agent_dir / "current.json"
    current = _read_json(current_path, {})
    session_id = current.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return ensure_active_session(settings, metadata, memory)

    session_dir = _session_dir(agent_dir, session_id)
    meta_path = _session_meta_path(session_dir)
    meta = _read_json(meta_path, {"session_id": session_id})
    if not meta.get("rolled_over"):
        index = build_session_index(session_dir, meta)
        _write_json(_session_index_path(session_dir), index)
        _session_handoff_path(session_dir).write_text(render_session_handoff(index), encoding="utf-8")
        meta["rolled_over"] = True
        meta["rolled_over_at"] = _now_iso()
        _write_json(meta_path, meta)

    next_current = {
        "session_id": _new_session_id(),
        "created_at": _now_iso(),
        "agent_id": (metadata or {}).get("agent_id", meta.get("agent_id", "default")),
        "previous_session_id": session_id,
        "carry_index": str(_session_index_path(session_dir)),
    }
    _write_json(current_path, next_current)
    return ensure_active_session(settings, metadata, memory)


def build_session_index(session_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Build a lightweight index for a completed session segment."""
    log_path = _session_log_path(session_dir)
    previews: list[str] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines()[:12]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            input_messages = record.get("input_messages") or []
            if input_messages:
                preview = str(input_messages[-1].get("content", ""))[:120]
                if preview:
                    previews.append(preview)
    return {
        "session_id": meta.get("session_id"),
        "agent_id": meta.get("agent_id", "default"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "last_response_id": meta.get("last_response_id"),
        "message_count": meta.get("message_count", 0),
        "char_count": meta.get("char_count", 0),
        "previews": previews[:8],
    }


def render_session_handoff(index: dict[str, Any]) -> str:
    """Render a handoff file for a completed session segment."""
    lines = [
        "# Session Handoff",
        "",
        f"- Session ID: {index.get('session_id', '')}",
        f"- Agent ID: {index.get('agent_id', '')}",
        f"- Last Response ID: {index.get('last_response_id', '')}",
        f"- Message Count: {index.get('message_count', 0)}",
        f"- Character Count: {index.get('char_count', 0)}",
        "",
        "## Indexed Previews",
    ]
    for preview in index.get("previews", []):
        lines.append(f"- {preview}")
    return "\n".join(lines) + "\n"


def get_active_session_context(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    memory_settings: AgentMemorySettings | None = None,
) -> dict[str, Any]:
    """Return current session handoff/index pointers for one agent."""
    memory = memory_settings or settings.agent_memory
    agent_dir = _agent_sessions_dir(settings, metadata, memory)
    current = _read_json(agent_dir / "current.json", {})
    session_id = current.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        current = ensure_active_session(settings, metadata, memory)
        session_id = current.get("session_id")
    session_dir = _session_dir(agent_dir, session_id)
    return {
        "session_id": session_id,
        "previous_session_id": current.get("previous_session_id"),
        "carry_index": current.get("carry_index"),
        "handoff_text": (
            _session_handoff_path(session_dir).read_text(encoding="utf-8")
            if _session_handoff_path(session_dir).exists()
            else None
        ),
    }


def build_session_handoff_appendix(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    memory_settings: AgentMemorySettings | None = None,
) -> str | None:
    """Render the carried handoff/index of the previous session for prompt injection."""
    context = get_active_session_context(settings, metadata, memory_settings)
    carry_index = context.get("carry_index")
    if not isinstance(carry_index, str) or not carry_index:
        return None
    index_path = Path(carry_index)
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    lines = [
        "Previous session handoff and index. Use this as navigation into archived context.",
        f"Session ID: {index.get('session_id', '')}",
        f"Last Response ID: {index.get('last_response_id', '')}",
    ]
    for preview in index.get("previews", [])[:6]:
        lines.append(f"- {preview}")
    return "\n".join(lines)


def search_session_archives(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    query: str,
    limit: int = 10,
    include_current: bool = False,
    memory_settings: AgentMemorySettings | None = None,
) -> list[dict[str, Any]]:
    """Search archived session logs with simple substring matching."""
    q = query.strip().lower()
    if not q:
        return []
    tokens = [token.lower() for token in _QUERY_TOKEN_RE.findall(query)]
    if not tokens and q:
        tokens = [q]

    memory = memory_settings or settings.agent_memory
    agent_dir = _agent_sessions_dir(settings, metadata, memory)
    if not agent_dir.exists():
        return []
    current_id = _read_json(agent_dir / "current.json", {}).get("session_id")

    matches: list[dict[str, Any]] = []
    for session_dir in sorted([p for p in agent_dir.iterdir() if p.is_dir()], reverse=True):
        if not include_current and current_id == session_dir.name:
            continue
        log_path = _session_log_path(session_dir)
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            haystack = json.dumps(record, ensure_ascii=False).lower()
            score = sum(1 for token in tokens if token in haystack)
            if score <= 0 and q not in haystack:
                continue
            preview = ""
            input_messages = record.get("input_messages") or []
            if input_messages:
                preview = str(input_messages[-1].get("content", ""))[:160]
            matches.append(
                {
                    "session_id": session_dir.name,
                    "response_id": record.get("response_id"),
                    "timestamp": record.get("timestamp"),
                    "preview": preview,
                    "score": score,
                }
            )
            if len(matches) >= limit:
                break
    matches.sort(key=lambda item: (item.get("score", 0), item.get("timestamp", "")), reverse=True)
    return matches[:limit]


def build_session_recall_appendix(
    settings: GlobalSettings,
    metadata: dict[str, Any] | None,
    query: str | None,
    memory_settings: AgentMemorySettings | None = None,
) -> str | None:
    """Render lightweight archive recall hits for the current user query."""
    memory = memory_settings or settings.agent_memory
    if not memory.retrieval_enabled:
        return None
    if not isinstance(query, str) or len(query.strip()) < 12:
        return None

    matches = search_session_archives(
        settings,
        metadata=metadata,
        query=query,
        limit=memory.retrieval_max_matches,
        include_current=False,
        memory_settings=memory,
    )
    if not matches:
        return None

    lines = [
        "Relevant archived session recall based on the current user request.",
        f"Query: {query.strip()[:160]}",
    ]
    for match in matches:
        preview = str(match.get("preview", ""))[: memory.retrieval_preview_chars]
        lines.append(
            f"- {match.get('session_id', '')} / {match.get('response_id', '')}: {preview}"
        )
    return "\n".join(lines)
