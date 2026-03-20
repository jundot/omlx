# SPDX-License-Identifier: Apache-2.0
"""Skill bundle loading for the oMLX agent runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillBundle:
    """One loaded skill bundle."""

    name: str
    path: Path
    identity: str = ""
    rules: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    tool_mode: str | None = None
    workflow: dict[str, Any] = field(default_factory=dict)


def get_skills_root(base_path: Path) -> Path:
    """Return the root directory for local skill bundles."""
    return base_path / "skills"


def list_skills(base_path: Path) -> list[str]:
    """List available local skill names."""
    root = get_skills_root(base_path)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _read_optional_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_skill(base_path: Path, name: str) -> SkillBundle | None:
    """Load one local skill bundle by name."""
    skill_dir = get_skills_root(base_path) / name
    if not skill_dir.exists() or not skill_dir.is_dir():
        return None

    tools = _read_optional_json(skill_dir / "TOOLS.json")
    workflow = _read_optional_json(skill_dir / "WORKFLOW.json")
    return SkillBundle(
        name=name,
        path=skill_dir,
        identity=_read_optional_text(skill_dir / "IDENTITY.md"),
        rules=_read_optional_text(skill_dir / "RULES.md"),
        allowed_tools=[
            item for item in tools.get("allowed_tools", [])
            if isinstance(item, str) and item.strip()
        ],
        blocked_tools=[
            item for item in tools.get("blocked_tools", [])
            if isinstance(item, str) and item.strip()
        ],
        tool_mode=tools.get("tool_mode") if isinstance(tools.get("tool_mode"), str) else None,
        workflow=workflow if isinstance(workflow, dict) else {},
    )


def resolve_skills(base_path: Path, names: list[str]) -> list[SkillBundle]:
    """Load all existing skill bundles in order."""
    bundles: list[SkillBundle] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        bundle = load_skill(base_path, name)
        if bundle is not None:
            bundles.append(bundle)
    return bundles


def render_skills_system_appendix(skills: list[SkillBundle]) -> str | None:
    """Render loaded skills into bootstrap context."""
    if not skills:
        return None

    blocks = ["Active skill bundles loaded for this request."]
    for skill in skills:
        blocks.append(f"[Skill: {skill.name}]")
        if skill.identity:
            blocks.append(f"Identity:\n{skill.identity}")
        if skill.rules:
            blocks.append(f"Rules:\n{skill.rules}")
        if skill.workflow:
            workflow_lines = json.dumps(skill.workflow, ensure_ascii=False, indent=2)
            blocks.append(f"Workflow:\n{workflow_lines}")
    return "\n\n".join(blocks)
