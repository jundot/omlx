# SPDX-License-Identifier: Apache-2.0
"""Routes for inspecting and resolving agent runtime state."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .agent_models import (
    AgentMemoryEntry,
    AgentMemoryListResponse,
    AgentMemoryWriteRequest,
    AgentSessionResponse,
    AgentSessionSearchResponse,
    AgentStateRequest,
    AgentStateResponse,
    AgentToolExecuteRequest,
    AgentToolExecuteResponse,
    AgentRuntimeOverview,
    AgentRuntimeResolveRequest,
    AgentRuntimeResolveResponse,
)
from ..context.bootstrap import build_bootstrap_system_message
from ..context.external_agents import (
    build_external_agent_tools,
    execute_external_agent_tool,
)
from ..context.memory import (
    AgentState,
    append_memory_entry,
    list_memory_entries,
    load_agent_state,
    save_agent_state,
)
from ..context.session_store import (
    ensure_active_session,
    rollover_session,
    search_session_archives,
)
from ..context.runtime import (
    apply_skill_defaults,
    apply_tool_policy,
    get_runtime_overview,
    resolve_agent_runtime_request,
)

router = APIRouter(prefix="/v1/agent", tags=["agent"])

_get_settings = None
_get_mcp_manager = None


def set_agent_runtime_getters(settings_getter, mcp_manager_getter) -> None:
    """Register callbacks used by agent runtime routes."""
    global _get_settings, _get_mcp_manager
    _get_settings = settings_getter
    _get_mcp_manager = mcp_manager_getter


def _settings():
    if _get_settings is None:
        return None
    return _get_settings()


def _mcp_manager():
    if _get_mcp_manager is None:
        return None
    return _get_mcp_manager()


@router.get("/runtime")
async def get_agent_runtime() -> AgentRuntimeOverview:
    """Return available profiles, workspaces, and live MCP tool names."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    manager = _mcp_manager()
    tools = []
    if manager is not None:
        tools = [
            {"name": tool.full_name}
            for tool in manager.get_all_tools()
        ]

    return AgentRuntimeOverview(
        **get_runtime_overview(settings, tools),
    )


@router.post("/resolve")
async def resolve_agent_runtime(
    request: AgentRuntimeResolveRequest,
) -> AgentRuntimeResolveResponse:
    """Resolve bootstrap preview and effective tools for a runtime selector."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    runtime = resolve_agent_runtime_request({
        "omlx_context": request.model_dump(exclude_none=True),
    })
    runtime = apply_skill_defaults(settings, runtime)

    manager = _mcp_manager()
    runtime_tools = []
    if manager is not None:
        runtime_tools.extend(manager.get_all_tools_openai())
    runtime_tools.extend(build_external_agent_tools(settings))
    effective_tools = apply_tool_policy(None, runtime_tools, runtime) or []
    bootstrap_preview = build_bootstrap_system_message(
        settings,
        selector=runtime.selector,
        skill_names=runtime.skills,
    )

    return AgentRuntimeResolveResponse(
        selector=runtime.selector,
        skills=runtime.skills,
        tool_policy={
            "mode": runtime.tool_policy.mode,
            "allowed_tools": runtime.tool_policy.allowed_tools,
            "blocked_tools": runtime.tool_policy.blocked_tools,
            "prefer_parallel_tools": runtime.tool_policy.prefer_parallel_tools,
        },
        bootstrap_preview=bootstrap_preview,
        effective_tools=[
            tool.get("function", {}).get("name", "")
            for tool in effective_tools
            if tool.get("function", {}).get("name")
        ],
    )


@router.post("/tool/execute")
async def execute_agent_tool(
    request: AgentToolExecuteRequest,
) -> AgentToolExecuteResponse:
    """Execute one runtime tool, including MCP or configured external agents."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": request.profile,
            "workspace": request.workspace,
        }.items() if value
    }
    metadata = dict(request.metadata)
    if selector:
        metadata["omlx_context"] = selector
    if request.workspace:
        metadata.setdefault("agent_id", f"workspace:{request.workspace}")
    elif request.profile:
        metadata.setdefault("agent_id", f"profile:{request.profile}")
    else:
        metadata.setdefault("agent_id", "default")

    external_tool_names = {
        tool.get("function", {}).get("name", "")
        for tool in build_external_agent_tools(settings)
    }
    if request.name in external_tool_names:
        result = await execute_external_agent_tool(
            settings,
            request.name,
            request.arguments,
            caller_metadata=metadata,
        )
        return AgentToolExecuteResponse(
            name=request.name,
            kind="external_agent",
            ok=True,
            content=result.get("output_text", ""),
            raw=result,
        )

    manager = _mcp_manager()
    if manager is None:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' is not available")

    result = await manager.execute_tool(request.name, request.arguments)
    content = result.to_message("local")["content"]
    return AgentToolExecuteResponse(
        name=request.name,
        kind="mcp",
        ok=not result.is_error,
        content=content,
        raw={
            "tool_name": result.tool_name,
            "is_error": result.is_error,
            "error_message": result.error_message,
            "content": result.content,
        },
    )


@router.get("/memory")
async def list_agent_memory(
    profile: str | None = None,
    workspace: str | None = None,
    limit: int = 20,
) -> AgentMemoryListResponse:
    """List recent persistent memory notes for one agent."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": profile,
            "workspace": workspace,
        }.items() if value
    }
    entries = list_memory_entries(settings, selector=selector, limit=max(1, min(limit, 100)))
    return AgentMemoryListResponse(
        selector=selector,
        entries=[
            AgentMemoryEntry(
                title=entry.title,
                content=entry.content,
                created_at=entry.created_at,
                path=str(entry.path),
                tags=entry.tags,
            )
            for entry in entries
        ],
    )


@router.post("/memory")
async def write_agent_memory(
    request: AgentMemoryWriteRequest,
) -> AgentMemoryEntry:
    """Append a persistent memory note, optionally into an Obsidian vault."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": request.profile,
            "workspace": request.workspace,
        }.items() if value
    }
    entry = append_memory_entry(
        settings,
        title=request.title,
        content=request.content,
        selector=selector,
        tags=request.tags,
        metadata=request.metadata,
    )
    return AgentMemoryEntry(
        title=entry.title,
        content=entry.content,
        created_at=entry.created_at,
        path=str(entry.path),
        tags=entry.tags,
    )


@router.get("/state")
async def get_agent_state(
    profile: str | None = None,
    workspace: str | None = None,
) -> AgentStateResponse:
    """Return structured shared task state for one agent."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": profile,
            "workspace": workspace,
        }.items() if value
    }
    state = load_agent_state(settings, selector=selector)
    return AgentStateResponse(
        selector=selector,
        mission=state.mission,
        facts=state.facts,
        decisions=state.decisions,
        todo=state.todo,
    )


@router.put("/state")
async def put_agent_state(
    request: AgentStateRequest,
) -> AgentStateResponse:
    """Persist structured shared task state for one agent."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": request.profile,
            "workspace": request.workspace,
        }.items() if value
    }
    state = save_agent_state(
        settings,
        AgentState(
            mission=request.mission,
            facts=request.facts,
            decisions=request.decisions,
            todo=request.todo,
        ),
        selector=selector,
    )
    return AgentStateResponse(
        selector=selector,
        mission=state.mission,
        facts=state.facts,
        decisions=state.decisions,
        todo=state.todo,
    )


@router.get("/session")
async def get_agent_session(
    profile: str | None = None,
    workspace: str | None = None,
) -> AgentSessionResponse:
    """Return the active session metadata for one agent."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": profile,
            "workspace": workspace,
        }.items() if value
    }
    metadata = {"omlx_context": selector}
    if workspace:
        metadata["agent_id"] = f"workspace:{workspace}"
    elif profile:
        metadata["agent_id"] = f"profile:{profile}"
    else:
        metadata["agent_id"] = "default"
    session = ensure_active_session(settings, metadata)
    return AgentSessionResponse(selector=selector, **session)


@router.post("/session/new")
async def new_agent_session(
    profile: str | None = None,
    workspace: str | None = None,
) -> AgentSessionResponse:
    """Roll over to a new session and return the new active session metadata."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": profile,
            "workspace": workspace,
        }.items() if value
    }
    metadata = {"omlx_context": selector}
    if workspace:
        metadata["agent_id"] = f"workspace:{workspace}"
    elif profile:
        metadata["agent_id"] = f"profile:{profile}"
    else:
        metadata["agent_id"] = "default"
    session = rollover_session(settings, metadata)
    return AgentSessionResponse(selector=selector, **session)


@router.get("/session/search")
async def search_agent_sessions(
    query: str,
    profile: str | None = None,
    workspace: str | None = None,
    limit: int = 10,
) -> AgentSessionSearchResponse:
    """Search archived session logs for one agent."""
    settings = _settings()
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings are not initialized")

    selector = {
        key: value for key, value in {
            "profile": profile,
            "workspace": workspace,
        }.items() if value
    }
    metadata = {"omlx_context": selector}
    if workspace:
        metadata["agent_id"] = f"workspace:{workspace}"
    elif profile:
        metadata["agent_id"] = f"profile:{profile}"
    else:
        metadata["agent_id"] = "default"
    matches = search_session_archives(settings, metadata, query=query, limit=max(1, min(limit, 50)))
    return AgentSessionSearchResponse(selector=selector, query=query, matches=matches)
