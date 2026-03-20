# SPDX-License-Identifier: Apache-2.0
"""Heuristics for incrementally updating structured agent state."""

from __future__ import annotations

import re
from typing import Any

from .memory import AgentState, load_agent_state, save_agent_state
from ..settings import GlobalSettings

_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[。.!?])\s+")


def _normalize_line(value: str) -> str:
    text = " ".join(value.strip().split())
    return text.rstrip(" .")


def _append_unique_block(existing: str, additions: list[str], limit: int = 12) -> str:
    lines = [_normalize_line(line) for line in existing.splitlines() if _normalize_line(line)]
    known = {line.lower() for line in lines}
    for item in additions:
        normalized = _normalize_line(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in known:
            continue
        lines.append(normalized)
        known.add(key)
    return "\n".join(f"- {line}" for line in lines[-limit:])


def _extract_bullets(text: str) -> list[str]:
    return [_normalize_line(match) for match in _BULLET_RE.findall(text)]


def _extract_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.replace("\n", " "))
    return [_normalize_line(part) for part in parts if _normalize_line(part)]


def _pick_decision_candidates(text: str) -> list[str]:
    results = []
    for sentence in _extract_sentences(text):
        lowered = sentence.lower()
        if any(token in lowered for token in ("decide", "decision", "will use", "use ", "choose", "plan to")):
            results.append(sentence)
    return results[:6]


def _pick_todo_candidates(text: str) -> list[str]:
    candidates = []
    for bullet in _extract_bullets(text):
        lowered = bullet.lower()
        if any(token in lowered for token in ("todo", "next", "need", "should", "implement", "build", "wire", "add", "fix")):
            candidates.append(bullet)
    if not candidates:
        for sentence in _extract_sentences(text):
            lowered = sentence.lower()
            if any(token in lowered for token in ("next", "need to", "should", "implement", "build", "wire", "add", "fix")):
                candidates.append(sentence)
    return candidates[:8]


def _pick_fact_candidates(text: str) -> list[str]:
    candidates = []
    for sentence in _extract_sentences(text):
        lowered = sentence.lower()
        if any(token in lowered for token in ("is ", "are ", "means ", "supports", "uses", "stored", "writes", "reads")):
            candidates.append(sentence)
    return candidates[:8]


def _derive_mission(current: str, messages: list[dict[str, Any]]) -> str:
    if current.strip():
        return current.strip()
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = _normalize_line(str(msg.get("content", "")))
        if content:
            return content[:240]
    return ""


def update_agent_state_from_exchange(
    settings: GlobalSettings,
    selector: dict[str, str] | None,
    input_messages: list[dict[str, Any]],
    output_messages: list[dict[str, Any]],
) -> AgentState:
    """Update structured agent state heuristically from one exchange."""
    state = load_agent_state(settings, selector=selector)

    user_text = "\n".join(str(msg.get("content", "")) for msg in input_messages if msg.get("role") == "user")
    assistant_text = "\n".join(str(msg.get("content", "")) for msg in output_messages if msg.get("role") == "assistant")

    tool_lines = []
    for msg in output_messages:
        for call in msg.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if name:
                tool_lines.append(f"Used tool: {name}")

    state = AgentState(
        mission=_derive_mission(state.mission, input_messages),
        facts=_append_unique_block(state.facts, _pick_fact_candidates(user_text) + _pick_fact_candidates(assistant_text) + tool_lines),
        decisions=_append_unique_block(state.decisions, _pick_decision_candidates(assistant_text)),
        todo=_append_unique_block(state.todo, _pick_todo_candidates(user_text) + _pick_todo_candidates(assistant_text)),
    )
    return save_agent_state(settings, state, selector=selector)
