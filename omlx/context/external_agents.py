# SPDX-License-Identifier: Apache-2.0
"""External-agent registry and execution helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib import error, request

from ..settings import ExternalAgentDefinition, GlobalSettings

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^a-z0-9_]+")
_DEFAULT_TIMEOUT_SECONDS = 120.0


def normalize_external_agent_name(name: str) -> str:
    """Normalize a configured external-agent name into a safe tool suffix."""
    cleaned = _SAFE_NAME_RE.sub("_", name.strip().lower()).strip("_")
    return cleaned or "external"


def external_agent_tool_name(name: str) -> str:
    """Return the tool name exposed to the model for one external agent."""
    return f"agent__{normalize_external_agent_name(name)}"


def list_external_agents(settings: GlobalSettings | None) -> list[ExternalAgentDefinition]:
    """Return enabled external-agent definitions with an endpoint."""
    if settings is None:
        return []
    return [
        agent for agent in settings.integrations.external_agents
        if agent.enabled and agent.endpoint
    ]


def build_external_agent_tools(
    settings: GlobalSettings | None,
) -> list[dict[str, Any]]:
    """Expose configured external agents as OpenAI function tools."""
    tools: list[dict[str, Any]] = []
    for agent in list_external_agents(settings):
        tool_name = external_agent_tool_name(agent.name)
        description = agent.description.strip() or (
            f"Delegate a sub-task to external agent '{agent.display_name or agent.name}'."
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Task or question for the external agent.",
                            },
                            "profile": {
                                "type": "string",
                                "description": "Optional target profile override.",
                            },
                            "workspace": {
                                "type": "string",
                                "description": "Optional target workspace override.",
                            },
                            "skills": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional skills/persona packs to activate.",
                            },
                            "session_mode": {
                                "type": "string",
                                "enum": ["continue", "new"],
                                "description": "Continue the remote session or start a new one.",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional free-form metadata forwarded to the remote agent.",
                            },
                        },
                        "required": ["prompt"],
                    },
                },
            }
        )
    return tools


def get_external_agents_overview(
    settings: GlobalSettings | None,
) -> list[dict[str, Any]]:
    """Return summary rows for admin/runtime inspection."""
    overview: list[dict[str, Any]] = []
    for agent in list_external_agents(settings):
        overview.append(
            {
                "name": agent.name,
                "display_name": agent.display_name or agent.name,
                "tool_name": external_agent_tool_name(agent.name),
                "description": agent.description,
                "endpoint": agent.endpoint,
                "protocol": agent.protocol,
                "default_model": agent.model,
                "default_profile": agent.default_profile,
                "default_workspace": agent.default_workspace,
                "default_skills": list(agent.default_skills),
                "default_session_mode": agent.default_session_mode,
            }
        )
    return overview


def _resolve_external_agent(
    settings: GlobalSettings | None,
    tool_name: str,
) -> ExternalAgentDefinition | None:
    if settings is None:
        return None

    candidates = {
        tool_name.strip(),
        normalize_external_agent_name(tool_name),
        tool_name.removeprefix("agent__").strip(),
    }
    for agent in list_external_agents(settings):
        normalized = normalize_external_agent_name(agent.name)
        if (
            agent.name in candidates
            or normalized in candidates
            or external_agent_tool_name(agent.name) in candidates
        ):
            return agent
    return None


def _merge_string_list(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            continue
        for item in items:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in merged:
                    merged.append(cleaned)
    return merged


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc

    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError("Remote agent returned non-JSON response") from exc


def _extract_output_text(data: dict[str, Any]) -> str:
    for key in ("output_text", "text", "content", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if (
                        isinstance(content, dict)
                        and content.get("type") in {"output_text", "text"}
                        and isinstance(content.get("text"), str)
                    ):
                        parts.append(content["text"])
        if parts:
            return "\n".join(parts).strip()

    return json.dumps(data, ensure_ascii=False)


async def execute_external_agent_tool(
    settings: GlobalSettings | None,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    caller_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one configured external-agent tool over HTTP."""
    agent = _resolve_external_agent(settings, tool_name)
    if agent is None:
        raise ValueError(f"External agent '{tool_name}' is not configured")

    args = arguments or {}
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("External agent tool requires a non-empty 'prompt'")

    session_mode = args.get("session_mode")
    if session_mode not in {"continue", "new"}:
        session_mode = agent.default_session_mode or "continue"

    metadata = agent.default_metadata.copy()
    raw_metadata = args.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)
    if caller_metadata:
        metadata.setdefault("caller", caller_metadata)

    profile = args.get("profile") or agent.default_profile
    workspace = args.get("workspace") or agent.default_workspace
    skills = _merge_string_list(agent.default_skills, args.get("skills"))

    protocol = (agent.protocol or "agent_bridge").strip().lower()
    if protocol not in {"agent_bridge", "openai_responses"}:
        raise ValueError(f"Unsupported external agent protocol '{agent.protocol}'")

    headers = dict(agent.headers)
    if agent.api_key:
        headers.setdefault("Authorization", f"Bearer {agent.api_key}")

    if protocol == "openai_responses":
        context: dict[str, Any] = {}
        if profile:
            context["profile"] = profile
        if workspace:
            context["workspace"] = workspace
        if skills:
            context["skills"] = skills
        payload: dict[str, Any] = {
            "model": args.get("model") or agent.model or "default",
            "input": prompt,
            "store": True,
            "metadata": metadata,
        }
        if context:
            payload["omlx_context"] = context
            payload["metadata"] = {**metadata, "omlx_context": context}
        if session_mode == "new":
            payload.setdefault("metadata", {})
            payload["metadata"]["session_mode"] = "new"
    else:
        payload = {
            "prompt": prompt,
            "profile": profile,
            "workspace": workspace,
            "skills": skills,
            "session_mode": session_mode,
            "metadata": metadata,
            "model": args.get("model") or agent.model,
        }

    timeout = agent.timeout_seconds or _DEFAULT_TIMEOUT_SECONDS
    response_data = await asyncio.to_thread(
        _post_json,
        agent.endpoint,
        payload,
        headers,
        timeout,
    )
    return {
        "agent": agent.name,
        "display_name": agent.display_name or agent.name,
        "tool_name": external_agent_tool_name(agent.name),
        "protocol": protocol,
        "endpoint": agent.endpoint,
        "output_text": _extract_output_text(response_data),
        "response": response_data,
    }
