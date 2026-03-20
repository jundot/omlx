# SPDX-License-Identifier: Apache-2.0
"""Persistent agent memory helpers backed by local files or Obsidian."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..settings import AgentMemorySettings, GlobalSettings

_STATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("mission", "MISSION.md"),
    ("facts", "FACTS.md"),
    ("decisions", "DECISIONS.md"),
    ("todo", "TODO.md"),
)


@dataclass
class MemoryEntry:
    """A persisted agent memory note."""

    path: Path
    title: str
    content: str
    created_at: str
    tags: list[str]


@dataclass
class AgentState:
    """Structured shared task state for one agent."""

    mission: str = ""
    facts: str = ""
    decisions: str = ""
    todo: str = ""


def _slugify(value: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in value.strip()
    ).strip("-")
    return cleaned or "default"


def _resolve_agent_key(selector: dict[str, str] | None) -> str:
    selected = selector or {}
    if selected.get("workspace"):
        return f"workspace-{_slugify(selected['workspace'])}"
    if selected.get("profile"):
        return f"profile-{_slugify(selected['profile'])}"
    return "default"


def get_memory_root(
    settings: GlobalSettings,
    memory_settings: AgentMemorySettings | None = None,
) -> Path:
    """Resolve the root directory used for agent memory."""
    memory = memory_settings or settings.agent_memory
    if memory.backend == "obsidian" and memory.obsidian_vault_dir:
        root = Path(memory.obsidian_vault_dir).expanduser().resolve()
        if memory.obsidian_subdir:
            return root / memory.obsidian_subdir
        return root
    if memory.storage_dir:
        return Path(memory.storage_dir).expanduser().resolve()
    return settings.base_path / "memory"


def resolve_agent_memory_dir(
    settings: GlobalSettings,
    selector: dict[str, str] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> Path:
    """Resolve the directory containing memory notes for one agent."""
    return get_memory_root(settings, memory_settings) / _resolve_agent_key(selector)


def _resolve_agent_state_dir(
    settings: GlobalSettings,
    selector: dict[str, str] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> Path:
    """Resolve the directory containing structured state for one agent."""
    return resolve_agent_memory_dir(settings, selector, memory_settings) / "_state"


def append_memory_entry(
    settings: GlobalSettings,
    title: str,
    content: str,
    selector: dict[str, str] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> MemoryEntry:
    """Persist a memory note for the selected agent."""
    memory = memory_settings or settings.agent_memory
    target_dir = resolve_agent_memory_dir(settings, selector, memory)
    target_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = _slugify(title)[:48]
    filename = f"{created_at}-{slug}.md"
    path = target_dir / filename
    note_tags = [tag.strip() for tag in (tags or []) if tag.strip()]
    frontmatter = {
        "title": title.strip() or "Untitled Memory",
        "created_at": created_at,
        "tags": note_tags,
        "selector": selector or {},
        "metadata": metadata or {},
    }
    body = (
        "---\n"
        f"{json.dumps(frontmatter, ensure_ascii=False, indent=2)}\n"
        "---\n\n"
        f"# {frontmatter['title']}\n\n"
        f"{content.strip()}\n"
    )
    path.write_text(body, encoding="utf-8")
    return MemoryEntry(
        path=path,
        title=frontmatter["title"],
        content=content.strip(),
        created_at=created_at,
        tags=note_tags,
    )


def load_agent_state(
    settings: GlobalSettings,
    selector: dict[str, str] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> AgentState:
    """Load structured mission/facts/decisions/todo state for one agent."""
    state_dir = _resolve_agent_state_dir(settings, selector, memory_settings)
    data: dict[str, str] = {}
    for field_name, filename in _STATE_FIELDS:
        path = state_dir / filename
        if not path.exists():
            data[field_name] = ""
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("# "):
            _heading, _sep, body = text.partition("\n\n")
            text = body.strip()
        data[field_name] = text
    return AgentState(**data)


def save_agent_state(
    settings: GlobalSettings,
    state: AgentState,
    selector: dict[str, str] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> AgentState:
    """Persist structured mission/facts/decisions/todo state for one agent."""
    state_dir = _resolve_agent_state_dir(settings, selector, memory_settings)
    state_dir.mkdir(parents=True, exist_ok=True)
    for field_name, filename in _STATE_FIELDS:
        content = getattr(state, field_name).strip()
        title = field_name.replace("_", " ").title()
        path = state_dir / filename
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return load_agent_state(settings, selector=selector, memory_settings=memory_settings)


def list_memory_entries(
    settings: GlobalSettings,
    selector: dict[str, str] | None = None,
    limit: int | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> list[MemoryEntry]:
    """List recent memory notes for the selected agent."""
    target_dir = resolve_agent_memory_dir(settings, selector, memory_settings)
    if not target_dir.exists():
        return []

    entries: list[MemoryEntry] = []
    files = sorted(
        [path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"],
        reverse=True,
    )
    if limit is not None:
        files = files[:limit]

    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        title = path.stem
        content = text
        if "\n# " in text:
            _, _, after = text.partition("\n# ")
            heading, _, rest = after.partition("\n")
            title = heading.strip() or title
            content = rest.strip()
        entries.append(
            MemoryEntry(
                path=path,
                title=title,
                content=content,
                created_at=path.stem.split("-", 1)[0],
                tags=[],
            )
        )
    return entries


def build_memory_system_appendix(
    settings: GlobalSettings,
    selector: dict[str, str] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> str | None:
    """Render recent memory entries as request context."""
    memory = memory_settings or settings.agent_memory
    if not memory.enabled or not memory.inject_recent_memories:
        return None

    entries = list_memory_entries(
        settings,
        selector=selector,
        limit=memory.max_recent_memories,
        memory_settings=memory,
    )
    if not entries:
        return None

    blocks = ["Recent persistent memory notes. Use them as durable recalled context."]
    for entry in entries:
        content = entry.content.strip()
        if len(content) > memory.max_chars_per_memory:
            content = content[: memory.max_chars_per_memory].rstrip() + "..."
        blocks.append(f"[{entry.title}]\n{content}")
    return "\n\n".join(blocks)


def build_agent_state_system_appendix(
    settings: GlobalSettings,
    selector: dict[str, str] | None = None,
    memory_settings: AgentMemorySettings | None = None,
) -> str | None:
    """Render structured mission/facts/decisions/todo state as request context."""
    state = load_agent_state(settings, selector=selector, memory_settings=memory_settings)
    sections = []
    for field_name, _filename in _STATE_FIELDS:
        content = getattr(state, field_name).strip()
        if content:
            sections.append((field_name.replace("_", " ").title(), content))
    if not sections:
        return None

    blocks = ["Shared task state for this agent. Treat it as durable working state."]
    for title, content in sections:
        blocks.append(f"[{title}]\n{content}")
    return "\n\n".join(blocks)


def get_gallery_root(
    settings: GlobalSettings,
    memory_settings: AgentMemorySettings | None = None,
) -> Path:
    """Resolve the root directory used for agent image artifacts."""
    memory = memory_settings or settings.agent_memory
    if memory.gallery_dir:
        return Path(memory.gallery_dir).expanduser().resolve()
    return settings.base_path / "gallery"


def list_agent_gallery(
    settings: GlobalSettings,
    memory_settings: AgentMemorySettings | None = None,
) -> list[dict[str, Any]]:
    """List per-agent image artifacts for the admin gallery."""
    root = get_gallery_root(settings, memory_settings)
    if not root.exists():
        return []

    groups: list[dict[str, Any]] = []
    for agent_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        images = []
        for path in sorted(agent_dir.iterdir(), reverse=True):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                continue
            images.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "url": f"/admin/api/agent-gallery/file?path={path}",
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        if images:
            groups.append({"agent": agent_dir.name, "images": images})
    return groups
