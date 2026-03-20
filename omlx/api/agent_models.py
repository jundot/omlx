# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for agent runtime inspection APIs."""

from typing import Any

from pydantic import BaseModel, Field


class AgentRuntimeOverview(BaseModel):
    """Runtime overview for the local agent environment."""

    base_path: str
    default_agent_dir: str
    profiles: list[str] = Field(default_factory=list)
    workspaces: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mcp_tools: list[str] = Field(default_factory=list)
    external_agents: list[dict[str, Any]] = Field(default_factory=list)


class AgentRuntimeResolveRequest(BaseModel):
    """Request for resolving runtime state for a selector."""

    profile: str | None = None
    workspace: str | None = None
    tool_mode: str = "all"
    skills: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    prefer_parallel_tools: bool | None = None


class AgentRuntimeResolveResponse(BaseModel):
    """Resolved runtime state for a request selector."""

    selector: dict[str, str] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    bootstrap_preview: str | None = None
    effective_tools: list[str] = Field(default_factory=list)


class AgentToolExecuteRequest(BaseModel):
    """Execute one runtime tool on the server side."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None
    workspace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentToolExecuteResponse(BaseModel):
    """Normalized result for one runtime tool execution."""

    name: str
    kind: str
    ok: bool
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)


class AgentMemoryWriteRequest(BaseModel):
    """Append one persistent memory note."""

    profile: str | None = None
    workspace: str | None = None
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMemoryEntry(BaseModel):
    """One persisted memory note."""

    title: str
    content: str
    created_at: str
    path: str
    tags: list[str] = Field(default_factory=list)


class AgentMemoryListResponse(BaseModel):
    """Recent persistent memory notes."""

    selector: dict[str, str] = Field(default_factory=dict)
    entries: list[AgentMemoryEntry] = Field(default_factory=list)


class AgentStateRequest(BaseModel):
    """Structured shared task state write request."""

    profile: str | None = None
    workspace: str | None = None
    mission: str = ""
    facts: str = ""
    decisions: str = ""
    todo: str = ""


class AgentStateResponse(BaseModel):
    """Structured shared task state response."""

    selector: dict[str, str] = Field(default_factory=dict)
    mission: str = ""
    facts: str = ""
    decisions: str = ""
    todo: str = ""


class AgentSessionResponse(BaseModel):
    """Current or completed session metadata."""

    selector: dict[str, str] = Field(default_factory=dict)
    session_id: str
    agent_id: str
    created_at: str | None = None
    updated_at: str | None = None
    last_response_id: str | None = None
    message_count: int = 0
    char_count: int = 0
    rolled_over: bool = False
    carry_index: str | None = None
    previous_session_id: str | None = None


class AgentSessionSearchResponse(BaseModel):
    """Session archive search results."""

    selector: dict[str, str] = Field(default_factory=dict)
    query: str
    matches: list[dict[str, Any]] = Field(default_factory=list)
