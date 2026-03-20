# SPDX-License-Identifier: Apache-2.0
"""
Bootstrap prompt assembly from workspace MD files.

This provides a small agent-runtime layer on top of the inference server:
- Load durable persona/context docs from disk
- Optionally append live MCP tool catalog context
- Prepend the assembled system message to incoming requests
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from omlx.context.memory import (
    build_agent_state_system_appendix,
    build_memory_system_appendix,
)
from omlx.context.session_store import (
    build_session_handoff_appendix,
    build_session_recall_appendix,
)
from omlx.context.skills import render_skills_system_appendix, resolve_skills
from omlx.settings import BootstrapSettings, GlobalSettings

if TYPE_CHECKING:
    from omlx.api.adapters.base import InternalMessage

_tool_catalog_getter: Callable[[], list[dict[str, Any]]] | None = None

_DOC_SPECS: tuple[tuple[str, str], ...] = (
    ("IDENTITY.md", "Identity"),
    ("TOOLS.md", "Tools"),
    ("MEMORY.md", "Memory"),
    ("USER.md", "User"),
)
_SELECTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def set_tool_catalog_getter(
    getter: Callable[[], list[dict[str, Any]]] | None,
) -> None:
    """Register a callback that returns live tool catalog entries."""
    global _tool_catalog_getter
    _tool_catalog_getter = getter


def _is_safe_selector(value: str | None) -> bool:
    """Allow simple selector names only."""
    if not value:
        return False
    return bool(_SELECTOR_RE.fullmatch(value))


def extract_context_selector(payload: Any) -> dict[str, str]:
    """Extract request-level bootstrap selectors from request payload."""
    if payload is None:
        return {}

    if isinstance(payload, dict):
        source = payload
    else:
        model_dump = getattr(payload, "model_dump", None)
        source = model_dump() if callable(model_dump) else {}

    context = source.get("omlx_context") or {}
    if not isinstance(context, dict):
        context = {}

    metadata = source.get("metadata") or {}
    if isinstance(metadata, dict):
        meta_context = metadata.get("omlx_context")
        if isinstance(meta_context, dict):
            context = {**meta_context, **context}

    selector: dict[str, str] = {}
    for key in ("profile", "workspace"):
        value = context.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                selector[key] = value
    return selector


def _resolve_bootstrap_dir(
    global_settings: GlobalSettings | None,
    bootstrap_settings: BootstrapSettings | None = None,
    selector: dict[str, str] | None = None,
) -> Path:
    """Resolve the directory containing bootstrap markdown files."""
    if global_settings is None:
        return Path.home() / ".omlx" / "agent"

    bootstrap = bootstrap_settings or global_settings.bootstrap
    selected = selector or {}

    workspace = selected.get("workspace")
    if _is_safe_selector(workspace):
        return (global_settings.base_path / "workspaces" / workspace / "agent").resolve()

    profile = selected.get("profile")
    if _is_safe_selector(profile):
        return (global_settings.base_path / "profiles" / profile / "agent").resolve()

    if bootstrap.docs_dir:
        return Path(bootstrap.docs_dir).expanduser().resolve()
    return global_settings.base_path / "agent"


def _read_bootstrap_sections(
    docs_dir: Path,
    max_bytes_per_file: int,
) -> list[tuple[str, str]]:
    """Read bootstrap documents from disk, skipping missing or empty files."""
    sections: list[tuple[str, str]] = []
    for filename, title in _DOC_SPECS:
        path = docs_dir / filename
        if not path.exists() or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        if len(content.encode("utf-8")) > max_bytes_per_file:
            raw = content.encode("utf-8")[:max_bytes_per_file]
            content = raw.decode("utf-8", errors="ignore").rstrip()
        sections.append((title, content))
    return sections


def _render_tool_catalog(tool_entries: list[dict[str, Any]]) -> str:
    """Render a compact tool catalog for prompt injection."""
    lines = [
        "Live MCP tool catalog. Prefer these tools when they directly fit the task.",
    ]
    for entry in tool_entries[:64]:
        name = entry.get("name", "unknown")
        description = (entry.get("description") or "").strip()
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def build_bootstrap_system_message(
    global_settings: GlobalSettings | None,
    bootstrap_settings: BootstrapSettings | None = None,
    selector: dict[str, str] | None = None,
    skill_names: list[str] | None = None,
    retrieval_query: str | None = None,
) -> str | None:
    """
    Build a system message from configured bootstrap documents.

    Returns:
        A single assembled system message, or None if bootstrap is disabled
        or no content is available.
    """
    if global_settings is None and bootstrap_settings is None:
        return None

    bootstrap = bootstrap_settings or global_settings.bootstrap
    if not bootstrap.enabled:
        return None

    docs_dir = _resolve_bootstrap_dir(
        global_settings,
        bootstrap,
        selector=selector,
    )
    sections = _read_bootstrap_sections(docs_dir, bootstrap.max_bytes_per_file)

    if bootstrap.include_tool_catalog and _tool_catalog_getter is not None:
        tool_entries = _tool_catalog_getter() or []
        if tool_entries:
            sections.append(("Live Tool Catalog", _render_tool_catalog(tool_entries)))

    if global_settings is not None:
        skills_appendix = render_skills_system_appendix(
            resolve_skills(global_settings.base_path, skill_names or [])
        )
        if skills_appendix:
            sections.append(("Skills", skills_appendix))

        metadata = {"omlx_context": selector or {}}
        if selector and selector.get("workspace"):
            metadata["agent_id"] = f"workspace:{selector['workspace']}"
        elif selector and selector.get("profile"):
            metadata["agent_id"] = f"profile:{selector['profile']}"
        else:
            metadata["agent_id"] = "default"

        handoff_appendix = build_session_handoff_appendix(
            global_settings,
            metadata,
        )
        if handoff_appendix:
            sections.append(("Session Handoff", handoff_appendix))

        recall_appendix = build_session_recall_appendix(
            global_settings,
            metadata,
            retrieval_query,
        )
        if recall_appendix:
            sections.append(("Archive Recall", recall_appendix))

        state_appendix = build_agent_state_system_appendix(
            global_settings,
            selector=selector,
        )
        if state_appendix:
            sections.append(("Shared State", state_appendix))

        memory_appendix = build_memory_system_appendix(
            global_settings,
            selector=selector,
        )
        if memory_appendix:
            sections.append(("Recent Memory", memory_appendix))

    if not sections:
        return None

    blocks = [
        (
            "Bootstrap context loaded from durable workspace markdown files. "
            "Treat this as high-priority operator context unless a later "
            "system instruction explicitly narrows it."
        )
    ]
    for title, content in sections:
        blocks.append(f"[{title}]\n{content}")
    return "\n\n".join(blocks)


def prepend_bootstrap_message(
    messages: list["InternalMessage"],
    global_settings: GlobalSettings | None,
    bootstrap_settings: BootstrapSettings | None = None,
    selector: dict[str, str] | None = None,
    skill_names: list[str] | None = None,
) -> list["InternalMessage"]:
    """Prepend the assembled bootstrap system message when available."""
    retrieval_query = None
    for message in reversed(messages):
        if getattr(message, "role", None) == "user" and getattr(message, "content", None):
            retrieval_query = str(message.content)
            break
    system_message = build_bootstrap_system_message(
        global_settings=global_settings,
        bootstrap_settings=bootstrap_settings,
        selector=selector,
        skill_names=skill_names,
        retrieval_query=retrieval_query,
    )
    if not system_message:
        return messages
    from omlx.api.adapters.base import InternalMessage

    return [InternalMessage(role="system", content=system_message), *messages]
