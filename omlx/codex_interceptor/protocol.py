from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

INTERCEPT_HOSTS = frozenset({"chatgpt.com", "api.openai.com"})
RESPONSES_PATHS = frozenset(
    {
        "/backend-api/codex/responses",
        "/backend-api/codex/v1/responses",
        "/v1/responses",
    }
)
SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "openai-organization",
        "openai-project",
        "chatgpt-account-id",
        "x-openai-assistant-app-id",
        "x-openai-client-metadata",
    }
)
TOOL_NAME_SEPARATORS = re.compile(r"(?:__+|:+|\.+)")
PLUGIN_URI_PATTERN = re.compile(
    r"plugin://(?P<plugin>[A-Za-z0-9_-]+)@(?P<publisher>[A-Za-z0-9_-]+)"
)
LEGACY_LOCAL_MODEL_SLOTS = frozenset({"codex-local-harness"})
MAX_PLUGIN_SKILL_INSTRUCTIONS = 96 * 1024
DEFAULT_INTERCEPTOR_RUNTIME_DIR = Path.home() / ".omlx" / "codex-interceptor"
SAFE_EVENT_FIELDS = frozenset(
    {
        "time",
        "event",
        "session_id",
        "original_host",
        "original_path",
        "method",
        "duration_ms",
        "local_model",
        "local_server",
        "requested_model",
        "effective_model",
        "local_slot",
        "source_slot",
        "previous_source_slot",
        "repaired_compaction_id_count",
        "repaired_local_reasoning_item_count",
        "local_label",
        "transport",
        "inference_kind",
        "pro_mode",
        "first_byte_ms",
        "first_visible_ms",
        "client_to_bridge_ms",
        "bridge_overhead_ms",
        "route_to_first_byte_ms",
        "transform_ms",
        "dispatch_ms",
        "connect_ms",
        "connection_reused",
        "response_id",
        "previous_local_slot",
        "previous_local_model",
        "route_revision",
        "patched_model_count",
        "advertised_context_window",
        "opaque_compaction_item_count",
        "forced_local_compaction",
        "empty_model_output",
        "local_request_rules_broken",
        "local_failure_count",
        "local_response_replayed",
        "code_fingerprint",
        "output_tokens_per_second",
        "cache_hit_percent",
        "cached_tokens",
        "output_tokens",
        "input_tokens",
        "tool_count",
        "tool_names",
        "loop_tool_name",
        "loop_repeats",
        "loop_identical_outputs",
        "loop_arguments_digest",
        "loop_guard_action",
        "loop_kind",
        "advertised_tool_names",
        "advertised_tool_types",
        "request_bytes",
        "input_item_count",
        "input_roles",
        "input_type_counts",
        "coerced_input_item_count",
        "typeless_input_item_count",
        "non_array_content_item_count",
        "has_previous_response_id",
        "is_compaction_request",
        "store",
        "status",
        "content_type",
        "registered_tool_count",
        "tool_names_remapped",
        "native_tool_streaming",
        "retryable",
        "terminal_event",
        "prefix_prefill_status",
        "residency_status",
        "warning_code",
        "warning_value",
        "warning_threshold",
        "error",
    }
)
SAFE_SESSION_FIELDS = frozenset(
    {
        "schema_version",
        "session_id",
        "created_at",
        "updated_at",
        "phase",
        "command",
        "provider",
        "model",
        "local_slot",
        "local_label",
        "project",
        "listen_host",
        "listen_port",
        "proxy_pid",
        "app_pid",
        "app_path",
        "exit_code",
        "codex_config_path",
        "codex_config_sha256_before",
        "codex_config_sha256_after",
        "codex_config_unchanged",
        "effort",
        "preserves_provider_identity",
        "passes_through_non_inference_requests",
        "proxy_stopped_at",
        "desktop_attested_at",
    }
)
DESKTOP_ATTESTATION_FIELDS = frozenset(
    {
        "local_model_response_visible",
        "projects_sidebar_visible",
        "automations_visible",
        "agent_harness_used",
        "computer_use_used",
    }
)
FEATURE_PATH_PREFIXES = {
    "projects": (
        "/backend-api/projects",
        "/backend-api/codex/projects",
    ),
    "automations": (
        "/backend-api/automations",
        "/backend-api/scheduled-tasks",
        "/backend-api/tasks",
    ),
    "plugins": (
        "/backend-api/plugins",
        "/backend-api/ps/plugins",
    ),
}


@dataclass(frozen=True)
class InterceptorSelection:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    auth_header: bool

    @property
    def responses_url(self) -> str:
        return self.base_url.rstrip("/") + "/responses"


def list_pi_interceptor_choices(
    *, models_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return credential-free private Pi providers and their configured models."""
    path = _pi_models_path(models_path)
    raw = _read_pi_catalog(path)
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raise ValueError(f"Pi model catalog has no provider map: {path}")

    choices: list[dict[str, Any]] = []
    for provider, definition in providers.items():
        if (
            not isinstance(provider, str)
            or not provider
            or not isinstance(definition, dict)
        ):
            continue
        base_url = definition.get("baseUrl")
        try:
            _validate_private_base_url(base_url)
        except ValueError:
            # Transparent mode deliberately does not offer public providers.
            continue
        models = []
        for item in definition.get("models", []):
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            display_name = item.get("name")
            models.append(
                {
                    "id": model_id,
                    "name": (
                        display_name
                        if isinstance(display_name, str) and display_name
                        else model_id
                    ),
                }
            )
        if models:
            choices.append(
                {
                    "provider": provider,
                    "base_url": base_url.rstrip("/"),
                    "models": models,
                }
            )
    return choices


def list_opencode_interceptor_choices(
    *, config_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return credential-free private OpenCode providers and configured models."""
    path = _opencode_config_path(config_path)
    raw = _read_opencode_catalog(path)
    providers = raw.get("provider")
    if not isinstance(providers, dict):
        raise ValueError(f"OpenCode config has no provider map: {path}")

    choices: list[dict[str, Any]] = []
    for provider, definition in providers.items():
        if (
            not isinstance(provider, str)
            or not provider
            or not isinstance(definition, dict)
        ):
            continue
        options = definition.get("options")
        if not isinstance(options, dict):
            continue
        base_url = options.get("baseURL") or options.get("baseUrl")
        try:
            _validate_private_base_url(base_url)
        except ValueError:
            continue
        models = []
        configured_models = definition.get("models")
        if not isinstance(configured_models, dict):
            continue
        for model_id, item in configured_models.items():
            if not isinstance(model_id, str) or not model_id:
                continue
            display_name = item.get("name") if isinstance(item, dict) else None
            models.append(
                {
                    "id": model_id,
                    "name": (
                        display_name
                        if isinstance(display_name, str) and display_name
                        else model_id
                    ),
                }
            )
        if models:
            choices.append(
                {
                    "provider": provider,
                    "base_url": base_url.rstrip("/"),
                    "models": sorted(models, key=lambda item: item["name"].casefold()),
                }
            )
    return sorted(choices, key=lambda item: item["provider"].casefold())


def format_interceptor_event(event: dict[str, Any]) -> str | None:
    """Format one safe status event without reflecting arbitrary event fields."""
    if not isinstance(event, dict):
        return None
    event_name = event.get("event")
    if not isinstance(event_name, str):
        return None
    timestamp = event.get("time")
    clock = (
        time.strftime("%H:%M:%S", time.localtime(timestamp))
        if isinstance(timestamp, (int, float))
        else "--:--:--"
    )
    method = _display_token(event.get("method"), 12, fallback="REQUEST")
    host = _display_token(event.get("original_host"), 255, fallback="unknown-host")
    path = _safe_display_path(event.get("original_path"))
    status = event.get("status")
    status_text = str(status) if isinstance(status, int) else "?"
    duration = event.get("duration_ms")
    duration_text = f" {duration}ms" if isinstance(duration, int) else ""
    model = _display_token(event.get("local_model"), 255, fallback="local model")
    requested_model = _display_token(
        event.get("requested_model"), 255, fallback="remote model"
    )

    if event_name == "request_passthrough":
        return f"[{clock}] → {method} {host}{path} (pass-through)"
    if event_name == "response_passthrough":
        return f"[{clock}] ← {status_text} {method} {host}{path}{duration_text}"
    if event_name == "response_passthrough_error":
        return f"[{clock}] ! {method} {host}{path} failed{duration_text}"
    if event_name == "remote_inference_routed":
        return f"[{clock}] OPENAI → {requested_model} {method} {host}{path}"
    if event_name == "remote_inference_completed":
        return f"[{clock}] OPENAI ← {status_text} {requested_model}{duration_text}"
    if event_name == "remote_inference_error":
        return f"[{clock}] OPENAI ! {requested_model} failed{duration_text}"
    if event_name == "websocket_forced_to_http":
        return f"[{clock}] LOCAL ↻ WebSocket contained; using HTTP for {host}{path}"
    if event_name == "websocket_bridge_started":
        return f"[{clock}] LOCAL ✓ native WebSocket bridge connected"
    if event_name == "websocket_prewarm_completed":
        return f"[{clock}] LOCAL ✓ Codex WebSocket prewarm completed"
    if event_name == "prefix_prefill_started":
        return f"[{clock}] LOCAL … prefilling Codex instructions and tools"
    if event_name == "prefix_prefill_completed":
        first = event.get("first_byte_ms")
        first_text = f" · first byte {first}ms" if isinstance(first, int) else ""
        reused = (
            " · reused connection" if event.get("connection_reused") is True else ""
        )
        return (
            f"[{clock}] LOCAL ✓ Codex prefix cached{duration_text}{first_text}{reused}"
        )
    if event_name == "prefix_prefill_deduplicated":
        return f"[{clock}] LOCAL ✓ Codex prefix already warm"
    if event_name == "prefix_prefill_failed":
        return f"[{clock}] LOCAL ! prefix prefill unavailable{duration_text}"
    if event_name == "native_tool_streaming_enabled":
        return f"[{clock}] LOCAL ✓ native tools stream without full-response buffering"
    if event_name == "residency_keepalive_completed":
        return f"[{clock}] LOCAL ✓ model residency renewed{duration_text}"
    if event_name == "residency_keepalive_failed":
        return f"[{clock}] LOCAL ! model residency check failed{duration_text}"
    if event_name == "performance_warning":
        warning = _display_token(
            event.get("warning_code"), 48, fallback="performance warning"
        ).replace("_", " ")
        return f"[{clock}] PERF ! {warning}"
    if event_name == "inference_first_byte":
        first = event.get("first_byte_ms")
        first_text = f"{first}ms" if isinstance(first, int) else "ready"
        return f"[{clock}] LOCAL · first byte {first_text}"
    if event_name == "inference_upstream_connected":
        connect = event.get("connect_ms")
        connect_text = f"{connect}ms" if isinstance(connect, int) else "ready"
        reused = " · reused" if event.get("connection_reused") is True else ""
        return f"[{clock}] LOCAL · upstream connected {connect_text}{reused}"
    if event_name == "inference_first_visible_event":
        first = event.get("first_visible_ms")
        first_text = f"{first}ms" if isinstance(first, int) else "ready"
        return f"[{clock}] LOCAL · first visible output {first_text}"
    if event_name == "local_slot_recovered":
        slot = _display_token(event.get("local_slot"), 255, fallback="local slot")
        return f"[{clock}] LOCAL ↻ Codex slot recovered to {slot}"
    if event_name == "model_catalog_adapted":
        count = event.get("patched_model_count")
        count_text = str(count) if isinstance(count, int) else "0"
        return f"[{clock}] LOCAL ✓ native tool mode enabled for {count_text} model(s)"
    if event_name == "textual_tool_call_repaired":
        return f"[{clock}] LOCAL ✓ textual tool call converted to native function call"
    if event_name == "local_reasoning_item_repaired":
        count = event.get("repaired_local_reasoning_item_count")
        count_text = str(count) if isinstance(count, int) else "1"
        return f"[{clock}] LOCAL ✓ removed {count_text} incompatible reasoning item(s)"
    if event_name == "doom_loop_detected":
        name = _display_token(event.get("loop_tool_name"), 40, fallback="a tool")
        action = _display_token(event.get("loop_guard_action"), 12, fallback="observed")
        stuck = " · no progress" if event.get("loop_identical_outputs") else ""
        return f"[{clock}] LOOP {action} · {name} x{event.get('loop_repeats')}{stuck}"
    if event_name == "compaction_short_circuited":
        input_count = event.get("input_item_count")
        input_text = f" inputs={input_count}" if isinstance(input_count, int) else ""
        return f"[{clock}] LOCAL ✓ startup compaction satisfied without model call{input_text}"
    if event_name == "inference_routed":
        tool_count = event.get("tool_count")
        tools = f" tools={tool_count}" if isinstance(tool_count, int) else ""
        no_tools = " no-tools-advertised" if tool_count == 0 else ""
        request_bytes = event.get("request_bytes")
        byte_text = f" bytes={request_bytes}" if isinstance(request_bytes, int) else ""
        input_count = event.get("input_item_count")
        input_text = f" inputs={input_count}" if isinstance(input_count, int) else ""
        typeless_count = event.get("typeless_input_item_count")
        typeless_text = (
            f" typeless={typeless_count}"
            if isinstance(typeless_count, int) and typeless_count
            else ""
        )
        content_count = event.get("non_array_content_item_count")
        content_text = (
            f" badcontent={content_count}"
            if isinstance(content_count, int) and content_count
            else ""
        )
        coerced_count = event.get("coerced_input_item_count")
        coerced_text = (
            f" coerced={coerced_count}"
            if isinstance(coerced_count, int) and coerced_count
            else ""
        )
        compaction = (
            " compaction=1" if event.get("is_compaction_request") is True else ""
        )
        return f"[{clock}] LOCAL → {model} {method} {host}{path}{tools}{no_tools}{input_text}{typeless_text}{content_text}{coerced_text}{byte_text}{compaction}"
    if event_name in {"inference_completed", "inference_stream_closed"}:
        remapped = event.get("tool_names_remapped")
        repairs = f" repairs={remapped}" if isinstance(remapped, int) else ""
        first_visible = event.get("first_visible_ms")
        visible = (
            f" visible={first_visible}ms" if isinstance(first_visible, int) else ""
        )
        suffix = (
            " stream closed cleanly" if event_name == "inference_stream_closed" else ""
        )
        connection = (
            " reused=1"
            if event.get("connection_reused") is True
            else " reused=0"
            if event.get("connection_reused") is False
            else ""
        )
        return (
            f"[{clock}] LOCAL ← {status_text} {model}{duration_text}"
            f"{visible}{connection}{repairs}{suffix}"
        )
    if event_name == "sse_normalizer_installed":
        return f"[{clock}] LOCAL · streaming response active ({status_text})"
    if event_name == "request_rejected":
        return f"[{clock}] LOCAL ! inference request rejected"
    if event_name == "inference_error":
        detail = _display_token(event.get("error_detail"), 240, fallback="")
        upstream = _display_token(event.get("upstream_error_message"), 240, fallback="")
        suffix = f": {detail or upstream}" if detail or upstream else ""
        return f"[{clock}] LOCAL ! inference failed ({status_text}){suffix}"
    return None


def is_intercepted_request(method: str, host: str, path: str) -> bool:
    normalized_host = host.rsplit(":", 1)[0].lower()
    normalized_path = path.split("?", 1)[0]
    return (
        method.upper() == "POST"
        and normalized_host in INTERCEPT_HOSTS
        and normalized_path in RESPONSES_PATHS
    )


def is_intercepted_websocket(method: str, host: str, path: str) -> bool:
    normalized_host = host.rsplit(":", 1)[0].lower()
    normalized_path = path.split("?", 1)[0]
    return (
        method.upper() == "GET"
        and normalized_host in INTERCEPT_HOSTS
        and normalized_path in RESPONSES_PATHS
    )


def transform_responses_request(
    payload: dict[str, Any],
    model: str,
    *,
    inject_plugin_skills: bool = True,
) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Responses request body must be a JSON object")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("local model is required")
    transformed = json.loads(json.dumps(payload))
    transformed["model"] = model
    promoted_tool_names = deferred_tool_names_from_input(transformed.get("input"))
    transformed = promote_deferred_tools_for_local_model(transformed)
    transformed["input"] = _translate_compaction_items_for_local_model(
        transformed.get("input")
    )
    transformed["input"] = _translate_special_tool_items_for_local_model(
        transformed.get("input")
    )
    if _has_tool_search(transformed):
        transformed["input"] = _prepend_deferred_tool_hint(transformed.get("input"))
    if promoted_tool_names:
        transformed["input"] = _append_promoted_tool_hint(
            transformed.get("input"), promoted_tool_names
        )
    if inject_plugin_skills:
        transformed["input"] = append_explicit_plugin_skill_instructions(
            transformed.get("input")
        )
    if not isinstance(transformed.get("input"), str):
        transformed["input"] = _normalize_responses_input_for_local_model(
            transformed.get("input"), input_list_item=False
        )
        transformed["input"], _ = _coerce_responses_input_items_for_local_model(
            transformed.get("input")
        )
        top_level_instructions = transformed.pop("instructions", None)
        if isinstance(top_level_instructions, str) and top_level_instructions:
            transformed["input"].insert(
                0,
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": top_level_instructions,
                        }
                    ],
                },
            )
        transformed["input"] = _move_instruction_messages_to_front(
            transformed.get("input")
        )
    tool_names = {
        tool["name"]
        for tool in transformed.get("tools", [])
        if isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("name"), str)
        and tool["name"]
    }
    return transformed, tool_names


def _move_instruction_messages_to_front(value: Any) -> Any:
    """Merge local-server instruction messages into their required prefix.

    Codex histories can acquire developer messages while compatibility hints,
    plugin instructions, and deferred-tool results are translated. Some local
    OpenAI-compatible servers reject any system/developer message that appears
    after a conversation item; some accept only one such message. Preserve the
    instruction order and content while producing one leading developer item.

    Every instruction is merged, including the ones Codex adds mid-history. That
    rewrites the front of the prompt whenever a new one appears, which costs the
    local server its KV cache for the whole conversation. Leaving them in place
    as user notes preserves that cache but breaks two things that matter more: a
    converted item can land between a function call and its result, which a local
    chat template rejects outright ("A tool message must follow an assistant or
    tool message"), and the deferred-tool and plugin-skill hints stop being
    authoritative, so the model describes a tool call instead of emitting one.
    """
    if not isinstance(value, list):
        return value
    instruction_content: list[Any] = []
    conversation: list[Any] = []
    for item in value:
        if _is_instruction_message(item):
            instruction_content.extend(_instruction_content_items(item))
            continue
        conversation.append(item)
    if not instruction_content:
        return conversation
    return [
        {
            "type": "message",
            "role": "developer",
            "content": instruction_content,
        },
        *conversation,
    ]


def _is_instruction_message(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "message"
        and item.get("role") in {"system", "developer"}
    )


def _instruction_content_items(item: dict[str, Any]) -> list[Any]:
    content = item.get("content")
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    return []


def promote_deferred_tools_for_local_model(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Promote tools embedded in ``tool_search_output`` into ``tools``.

    Hosted Codex models understand that a client tool-search output extends the
    callable set. Local OpenAI-compatible servers only compile the request's
    top-level ``tools`` array, so without this promotion the model can read a
    discovered tool's name but cannot actually call it.
    """
    if not isinstance(payload, dict):
        return payload
    inputs = payload.get("input")
    if not isinstance(inputs, list):
        return payload
    discovered: list[dict[str, Any]] = []
    for item in inputs:
        if not (
            isinstance(item, dict)
            and item.get("type") == "tool_search_output"
            and isinstance(item.get("tools"), list)
        ):
            continue
        discovered.extend(tool for tool in item["tools"] if isinstance(tool, dict))
    if not discovered:
        return payload
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
    promoted = [copy.deepcopy(tool) for tool in tools]
    for tool in discovered:
        _merge_promoted_tool(promoted, tool)
    payload["tools"] = promoted
    return payload


def deferred_tool_names_from_input(value: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(value, list):
        return names
    for item in value:
        if not (isinstance(item, dict) and item.get("type") == "tool_search_output"):
            continue
        names.extend(_tool_definition_names(item.get("tools")))
    return list(dict.fromkeys(names))


def explicit_plugin_mentions(value: Any) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    # Plugin mentions come from the user; skip assistant and tool history.
    if isinstance(value, list):
        sources = [
            item
            for item in value
            if isinstance(item, dict) and item.get("role") == "user"
        ]
    else:
        sources = [value]
    for source in sources:
        for text in _iter_strings(source):
            for match in PLUGIN_URI_PATTERN.finditer(text):
                mention = (match.group("plugin"), match.group("publisher"))
                if mention not in mentions:
                    mentions.append(mention)
    return mentions


def append_explicit_plugin_skill_instructions(
    value: Any,
    *,
    plugin_cache_root: Path | None = None,
) -> Any:
    mentions = explicit_plugin_mentions(value)
    if not mentions:
        return value
    root = (
        plugin_cache_root
        if plugin_cache_root is not None
        else Path.home() / ".codex" / "plugins" / "cache"
    )
    items: list[dict[str, Any]] = []
    for plugin, publisher in mentions:
        loaded = _load_plugin_skill_bundle(root, publisher, plugin)
        if loaded is None:
            continue
        plugin_root, skills = loaded
        body = "\n\n".join(skills)
        if plugin == "chrome":
            body += "\n\n" + _chrome_skill_execution_guard(plugin_root)
        items.append(
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"INSTALLED `{plugin}` PLUGIN SKILL INSTRUCTIONS — "
                            "these are authoritative and must be followed exactly. "
                            "Do not improvise a different runtime or automation method.\n\n"
                            + body
                        ),
                    }
                ],
            }
        )
    if not items:
        return value
    if isinstance(value, list):
        return [*value, *items]
    if value is None:
        return items
    return [value, *items]


def _load_plugin_skill_bundle(
    cache_root: Path,
    publisher: str,
    plugin: str,
) -> tuple[Path, list[str]] | None:
    try:
        cache = cache_root.expanduser().resolve()
        plugin_base = (cache / publisher / plugin).resolve()
        if not plugin_base.is_relative_to(cache):
            return None
        latest = plugin_base / "latest"
        if latest.exists():
            plugin_root = latest.resolve()
        else:
            versions = sorted(
                (
                    path
                    for path in plugin_base.iterdir()
                    if path.is_dir() and path.name != "latest"
                ),
                key=lambda path: path.stat().st_mtime,
            )
            if not versions:
                return None
            plugin_root = versions[-1].resolve()
        if not plugin_root.is_relative_to(plugin_base):
            return None
        paths = sorted(plugin_root.glob("skills/*/SKILL.md"))
        skills: list[str] = []
        remaining = MAX_PLUGIN_SKILL_INSTRUCTIONS
        for path in paths:
            text = path.read_text(encoding="utf-8")
            size = len(text.encode("utf-8"))
            if not text or size > remaining:
                continue
            skills.append(text.replace("<plugin root>", str(plugin_root)))
            remaining -= size
        return (plugin_root, skills) if skills else None
    except (OSError, UnicodeError):
        return None


def _chrome_skill_execution_guard(plugin_root: Path) -> str:
    client = plugin_root / "scripts" / "browser-client.mjs"
    return (
        "LOCAL MODEL EXECUTION GUARD FOR AN EXPLICIT CHROME REQUEST:\n"
        "Your next action must be a native call to `mcp__node_repl__js`. "
        "Never use standalone Playwright, `chromium.launch`, an npm `chrome` "
        "package, `require`, child_process, shell commands, AppleScript, or MCP "
        "resource listing. First initialize the installed runtime using exactly "
        "this pattern (reuse existing globals when present):\n"
        "```js\n"
        "if (globalThis.agent?.browsers == null) {\n"
        f'  const {{ setupBrowserRuntime }} = await import("{client}");\n'
        "  await setupBrowserRuntime({ globals: globalThis });\n"
        "}\n"
        "if (globalThis.chrome == null) {\n"
        '  globalThis.chrome = await agent.browsers.get("extension");\n'
        "  nodeRepl.write(await chrome.documentation());\n"
        "}\n"
        "```\n"
        "After that call succeeds, follow the complete returned Chrome "
        "documentation to obtain a tab and navigate. Do not invent APIs."
    )


def _merge_promoted_tool(tools: list[Any], discovered: dict[str, Any]) -> None:
    tool_type = discovered.get("type")
    name = discovered.get("name")
    if tool_type == "namespace" and isinstance(name, str) and name:
        existing = next(
            (
                tool
                for tool in tools
                if isinstance(tool, dict)
                and tool.get("type") == "namespace"
                and tool.get("name") == name
            ),
            None,
        )
        if existing is None:
            tools.append(copy.deepcopy(discovered))
            return
        existing_nested = existing.get("tools")
        if not isinstance(existing_nested, list):
            existing_nested = []
            existing["tools"] = existing_nested
        known = {
            nested.get("name")
            for nested in existing_nested
            if isinstance(nested, dict) and isinstance(nested.get("name"), str)
        }
        for nested in discovered.get("tools", []):
            if (
                isinstance(nested, dict)
                and isinstance(nested.get("name"), str)
                and nested["name"] not in known
            ):
                existing_nested.append(copy.deepcopy(nested))
                known.add(nested["name"])
        return
    if isinstance(name, str) and any(
        isinstance(tool, dict)
        and tool.get("type") == tool_type
        and tool.get("name") == name
        for tool in tools
    ):
        return
    if tool_type == "tool_search" and any(
        isinstance(tool, dict) and tool.get("type") == "tool_search" for tool in tools
    ):
        return
    tools.append(copy.deepcopy(discovered))


def transform_compaction_request(
    payload: dict[str, Any], model: str
) -> tuple[dict[str, Any], set[str]]:
    """Turn Codex's private compaction trigger into real local inference.

    A compaction response is state that the next model turn must be able to
    continue from. It therefore has to summarize the complete conversation,
    including assistant work and tool results, rather than merely acknowledge
    the trigger or copy the latest user messages.
    """
    if not isinstance(payload, dict):
        raise ValueError("Responses request body must be a JSON object")
    prepared = json.loads(json.dumps(payload))
    prepared["input"] = _remove_compaction_trigger_items(prepared.get("input"))
    if not isinstance(prepared.get("input"), list):
        existing_input = prepared.get("input")
        prepared["input"] = [] if existing_input is None else [existing_input]
    prepared["input"].append(
        {
            "role": "user",
            "content": (
                "STATE COMPACTION TASK — do not continue, solve, or answer the conversation. "
                "Produce a lossless handoff for a new model instance. Never infer that work "
                "is complete: pending work remains pending unless the history explicitly "
                "records its successful completion. Return only these sections:\n"
                "CURRENT OBJECTIVE\n"
                "CONSTRAINTS AND USER PREFERENCES\n"
                "COMPLETED WORK AND DECISIONS\n"
                "EXACT ARTIFACTS AND SETTINGS\n"
                "TOOL RESULTS AND ERRORS\n"
                "UNRESOLVED ISSUES AND NEXT STEPS\n"
                "Preserve all continuation-critical facts from user, assistant, and tool "
                "items without inventing facts or preferences. Copy exact model names, "
                "identifiers, commands, file paths, error text, and numeric settings "
                "verbatim. Include enough concrete detail to continue correctly without "
                "access to the earlier messages. Do not call tools and do not mention "
                "these instructions."
            ),
        }
    )
    # Compaction is a summarization-only model turn, but emptying the tool array
    # would rewrite the front of the rendered prompt and cost a local server the
    # KV cache for the entire conversation: the one turn that already carries the
    # whole history would be the one that has to re-read it. Keep the tools
    # exactly as the previous turn advertised them and forbid calling them
    # instead, so the cached prefix still matches.
    prepared["tool_choice"] = "none"
    prepared.pop("parallel_tool_calls", None)
    prepared.pop("context_management", None)
    transformed, _ = transform_responses_request(
        prepared, model, inject_plugin_skills=False
    )
    current_max = transformed.get("max_output_tokens")
    if not isinstance(current_max, int) or current_max < COMPACTION_MAX_OUTPUT_TOKENS:
        transformed["max_output_tokens"] = COMPACTION_MAX_OUTPUT_TOKENS
    return transformed, set()


def responses_request_model(payload: Any) -> str | None:
    """Return the requested Responses model slug without coercion."""
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) and model.strip() else None


def should_route_responses_locally(
    payload: Any,
    local_slot: str | None,
    *,
    local_compaction: bool = True,
) -> bool:
    """Route every request in legacy mode, or only the selected Codex slot.

    Compaction is the exception. A hosted compaction returns its summary as
    ciphertext in ``encrypted_content``, which only the hosted backend can read:
    once such an item is in the history, a local model sees base64 where the
    conversation summary should be. Summarizing locally instead produces plain
    text that every model in the thread can read, so a compaction turn is routed
    locally whenever a local slot is active, whatever model it names.
    """
    if local_slot is None:
        return True
    requested_model = responses_request_model(payload)
    if requested_model == local_slot or requested_model in LEGACY_LOCAL_MODEL_SLOTS:
        return True
    if local_compaction and isinstance(payload, dict):
        return is_compaction_request(payload) or is_compaction_trigger_request(payload)
    return False


def select_lowest_visible_codex_model(payload: Any) -> dict[str, str] | None:
    """Select Codex's lowest-ranked user-visible model from a model catalogue."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return None
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, model in enumerate(payload["models"]):
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        visibility = model.get("visibility")
        if isinstance(visibility, str) and visibility not in {"list", "visible"}:
            continue
        if slug.startswith("codex-auto-"):
            continue
        priority = model.get("priority")
        rank = priority if isinstance(priority, int) else index
        candidates.append((rank, index, model))
    if not candidates:
        return None
    model = max(candidates, key=lambda item: (item[0], item[1]))[2]
    display = model.get("display_name") or model.get("title")
    return {
        "slug": model["slug"],
        "display_name": display
        if isinstance(display, str) and display
        else model["slug"],
    }


GPT_56_PRO_ALIAS = "gpt-5.6-sol-pro"
GPT_56_PRO_BASE_MODEL = "gpt-5.6-sol"


def add_gpt_56_pro_catalog_entry(payload: Any) -> tuple[Any, int]:
    """Expose GPT-5.6 Pro as a selectable wire-only alias for Sol."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return payload, 0
    if any(
        isinstance(item, dict) and item.get("slug") == GPT_56_PRO_ALIAS
        for item in payload["models"]
    ):
        return payload, 0
    source_index = next(
        (
            index
            for index, item in enumerate(payload["models"])
            if isinstance(item, dict) and item.get("slug") == GPT_56_PRO_BASE_MODEL
        ),
        None,
    )
    if source_index is None:
        return payload, 0
    patched = copy.deepcopy(payload)
    source = patched["models"][source_index]
    pro = copy.deepcopy(source)
    pro["slug"] = GPT_56_PRO_ALIAS
    pro["display_name"] = "GPT-5.6 Sol · Pro"
    pro["title"] = "GPT-5.6 Sol · Pro"
    pro["description"] = "GPT-5.6 Sol with Responses Pro reasoning mode"
    pro["visibility"] = "list"
    pro["default_reasoning_level"] = "medium"
    levels = pro.get("supported_reasoning_levels")
    if isinstance(levels, list):
        pro["supported_reasoning_levels"] = [
            level
            for level in levels
            if isinstance(level, dict) and level.get("effort") != "low"
        ]
    pro.pop("tool_mode", None)
    pro["use_responses_lite"] = False
    pro["include_skills_usage_instructions"] = True
    patched["models"].insert(source_index + 1, pro)
    return patched, 1


def adapt_gpt_56_pro_request(payload: Any) -> tuple[Any, bool]:
    """Rewrite the picker alias to the documented Responses Pro request shape."""
    if not isinstance(payload, dict) or payload.get("model") != GPT_56_PRO_ALIAS:
        return payload, False
    patched = copy.deepcopy(payload)
    patched["model"] = GPT_56_PRO_BASE_MODEL
    reasoning = patched.get("reasoning")
    reasoning = {} if not isinstance(reasoning, dict) else copy.deepcopy(reasoning)
    reasoning["mode"] = "pro"
    effort = reasoning.get("effort")
    if not isinstance(effort, str) or effort == "low":
        reasoning["effort"] = "medium"
    patched["reasoning"] = reasoning
    return patched, True


def adapt_model_catalog_for_local_tools(
    payload: Any,
    *,
    local_slot: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
    context_window: int | None = None,
) -> tuple[Any, int]:
    """Make a passing Codex model catalogue request ordinary native tools.

    This is deliberately a wire-response adaptation: it does not read or write
    Codex configuration or its on-disk model cache.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return payload, 0
    patched = copy.deepcopy(payload)
    count = 0
    for model in patched["models"]:
        if not isinstance(model, dict):
            continue
        if local_slot is not None and model.get("slug") != local_slot:
            continue
        tool_mode = model.get("tool_mode")
        responses_lite = model.get("use_responses_lite") is True
        changed = False
        if isinstance(tool_mode, str) and tool_mode.startswith("code_mode"):
            model.pop("tool_mode", None)
            changed = True
        if responses_lite:
            model["use_responses_lite"] = False
            changed = True
        if local_slot is not None:
            if display_name:
                # Older Codex builds render ``display_name``. Current desktop
                # builds validate and render ``title`` instead, falling back to
                # formatting the slug (for example, "Spark") when it is absent.
                # Set both wire fields so the selected private model remains
                # identifiable across app updates.
                for field in ("display_name", "title"):
                    if model.get(field) != display_name:
                        model[field] = display_name
                        changed = True
            if description and model.get("description") != description:
                model["description"] = description
                changed = True
            if model.get("include_skills_usage_instructions") is not True:
                model["include_skills_usage_instructions"] = True
                changed = True
            # Codex auto-compacts against the advertised window. Leaving the
            # hosted slot's figure in place makes it compact a local model long
            # before that model is full, and compaction is the most expensive
            # turn in a session.
            if isinstance(context_window, int) and context_window > 0:
                for field in ("context_window", "max_context_window"):
                    if model.get(field) != context_window:
                        model[field] = context_window
                        changed = True
        if changed:
            model["include_skills_usage_instructions"] = True
            count += 1
    return patched, count


# Doom-loop guard, calibrated against 56 recorded Codex sessions. Repeating one
# byte-identical call up to 11 times is ordinary work there (waiting on a process
# with write_stdin, polling wait_agent); stuck sessions repeat one call 19-43
# times. The nudge therefore starts above the observed legitimate ceiling. Note
# that identical results are NOT the signal they appear to be: every real loop in
# that data returned slightly different output each time, so progress is judged by
# repetition alone and matching results are only reported for context.
LOOP_WINDOW_ITEMS = 80
LOOP_NUDGE_REPEATS = 12
LOOP_BLOCK_REPEATS = 20
LOOP_GUARD_MODES = frozenset({"off", "observe", "guard"})


def detect_tool_call_loop(
    value: Any,
    *,
    window: int = LOOP_WINDOW_ITEMS,
    threshold: int = LOOP_NUDGE_REPEATS,
) -> dict[str, Any] | None:
    """Report a tool call the model keeps repeating verbatim.

    Only the tail is scanned, so the cost does not grow with the conversation.
    A repeat counts when the name and the arguments are byte-identical; whether
    the results were identical too is reported separately, because a poll that
    keeps returning the same thing is stuck while one whose output changes is
    making progress.
    """
    if not isinstance(value, list) or not value:
        return None
    tail = value[-window:] if window > 0 else value
    calls: dict[tuple[str, str], list[str | None]] = {}
    order: list[tuple[str, str]] = []
    outputs: dict[str, str] = {}
    for item in tail:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"function_call_output", "tool_call_output"}:
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                outputs[call_id] = _stringify_context_value(
                    item.get("output") if "output" in item else item.get("content")
                )[:2000]
            continue
        if item.get("type") not in {"function_call", "custom_tool_call"}:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = item.get("arguments") if "arguments" in item else item.get("input")
        signature = (name, _stringify_context_value(arguments)[:4000])
        if signature not in calls:
            calls[signature] = []
            order.append(signature)
        call_id = item.get("call_id")
        calls[signature].append(call_id if isinstance(call_id, str) else None)
    for signature in order:
        call_ids = calls[signature]
        if len(call_ids) < threshold:
            continue
        seen = [outputs.get(call_id) for call_id in call_ids if call_id in outputs]
        identical_outputs = len(seen) > 1 and len(set(seen)) == 1
        return {
            "name": signature[0],
            "repeats": len(call_ids),
            "identical_outputs": identical_outputs,
            "arguments_digest": hashlib.sha256(
                signature[1].encode("utf-8")
            ).hexdigest()[:16],
        }
    return None


CHURN_MIN_RUN = 6
CHURN_BLOCK_RUN = 10
CHURN_SIMILARITY = 0.8


def detect_churn_run(
    value: Any,
    *,
    window: int = LOOP_WINDOW_ITEMS,
    min_run: int = CHURN_MIN_RUN,
    similarity: float = CHURN_SIMILARITY,
) -> dict[str, Any] | None:
    """Report a run of near-identical calls that keep returning the same thing.

    The byte-identical guard misses a model that varies its arguments slightly
    while learning nothing - probing one import path after another, for example.
    Calibrated over 60 recorded sessions: consecutive same-tool calls whose
    arguments and results are both at least 80% similar run to 8, 15, 27 and 38
    in the four genuinely stuck sessions, and stop at 3 in every legitimate one,
    so the gap between 3 and 8 is empty and 6 sits safely inside it.
    """
    if not isinstance(value, list) or not value:
        return None
    tail = value[-window:] if window > 0 else value
    outputs: dict[str, str] = {}
    calls: list[tuple[str, str, str | None]] = []
    for item in tail:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"function_call_output", "tool_call_output"}:
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                outputs[call_id] = _stringify_context_value(
                    item.get("output") if "output" in item else item.get("content")
                )[:600]
            continue
        if item.get("type") not in {"function_call", "custom_tool_call"}:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = item.get("arguments") if "arguments" in item else item.get("input")
        call_id = item.get("call_id")
        calls.append(
            (
                name,
                _stringify_context_value(arguments)[:600],
                call_id if isinstance(call_id, str) else None,
            )
        )
    best_run = 1
    best_name = None
    run = 1
    for index in range(1, len(calls)):
        previous_name, previous_args, previous_id = calls[index - 1]
        name, arguments, call_id = calls[index]
        if name != previous_name or _similarity(previous_args, arguments) < similarity:
            run = 1
            continue
        previous_output = outputs.get(previous_id or "")
        output = outputs.get(call_id or "")
        if previous_output is None or output is None:
            run = 1
            continue
        if _similarity(previous_output, output) < similarity:
            run = 1
            continue
        run += 1
        if run > best_run:
            best_run, best_name = run, name
    if best_run < min_run or best_name is None:
        return None
    return {
        "name": best_name,
        "repeats": best_run,
        "identical_outputs": True,
        "kind": "churn",
        "arguments_digest": None,
    }


def _similarity(left: str, right: str, *, limit: int = 600) -> float:
    if left == right:
        return 1.0
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left[:limit], right[:limit]).ratio()


def apply_loop_guard(
    payload: dict[str, Any],
    detection: dict[str, Any] | None,
    *,
    block_repeats: int = LOOP_BLOCK_REPEATS,
) -> tuple[dict[str, Any], str | None]:
    """Break a detected loop without disturbing the rest of the request.

    The note is appended last, as a user item: a developer item would be merged
    into the leading instruction block, which both rewrites the prompt prefix the
    local server has cached and buries the correction a whole context away from
    where the model is generating. Nothing is inserted between a call and its
    result, and past the block threshold calling tools is refused for one turn,
    which forces the model to say something instead. ``tool_choice`` is a
    generation constraint rather than part of the rendered prompt, so refusing
    tools does not cost the cache.
    """
    if not detection:
        return payload, None
    items = payload.get("input")
    if not isinstance(items, list) or not items:
        return payload, None
    last = items[-1]
    if isinstance(last, dict) and last.get("type") in {
        "function_call",
        "custom_tool_call",
    }:
        # A call is still awaiting its result; appending here would separate them.
        return payload, None
    repeats = detection.get("repeats", 0)
    churn = detection.get("kind") == "churn"
    threshold = CHURN_BLOCK_RUN if churn else block_repeats
    blocking = isinstance(repeats, int) and repeats >= threshold
    outcome = "no progress" if detection.get("identical_outputs") else "the same way"
    shape = (
        "nearly identical arguments and it kept returning the same result"
        if churn
        else f"identical arguments and it resolved {outcome}"
    )
    note = (
        f"LOOP GUARD: you have already called `{detection.get('name')}` "
        f"{repeats} times with {shape}. "
        "Do not issue that call again with those arguments. Either take a "
        "materially different action, or stop and report what you have "
        "established and what is blocking you."
    )
    if blocking:
        note += (
            " Tool calls are refused for this turn: answer in text only, "
            "describing the state and the next distinct step."
        )
    guarded = dict(payload)
    guarded["input"] = [
        *items,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": note}],
        },
    ]
    if blocking:
        guarded["tool_choice"] = "none"
    return guarded, "blocked" if blocking else "nudged"


LOCAL_RESPONSE_INPUT_ITEM_TYPES = frozenset({"message"})
LOCAL_CONTEXT_TEXT_LIMIT = 20000
# Measured against a local model: the summary stops well short of this, so
# lowering the cap saved nothing and only risked truncating the handoff on a
# larger conversation, which is the one failure that loses the context outright.
COMPACTION_MAX_OUTPUT_TOKENS = 4096


def _normalize_responses_input_for_local_model(
    value: Any, *, input_list_item: bool
) -> Any:
    """Make implicit Codex Responses input items explicit for stricter servers.

    Codex can send message-like objects such as ``{"role": "user",
    "content": "hi"}`` without a top-level ``type``. Some local
    OpenAI-compatible servers reject those with errors like "Cannot determine
    type of 'item'". Function calls, tool outputs, reasoning, compaction
    triggers, and other explicitly typed items are left alone.
    """
    if isinstance(value, list):
        return [
            _normalize_responses_input_for_local_model(item, input_list_item=True)
            for item in value
        ]
    if input_list_item and isinstance(value, str):
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": value}],
        }
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _normalize_responses_input_for_local_model(item, input_list_item=False)
        for key, item in value.items()
    }
    role = normalized.get("role")
    if (
        "type" not in normalized
        and "content" in normalized
        and (
            input_list_item
            or (
                isinstance(role, str)
                and role in {"system", "developer", "user", "assistant"}
            )
        )
    ):
        normalized["type"] = "message"
        if "role" not in normalized:
            normalized["role"] = "user"
    if input_list_item and (
        "content" in normalized or normalized.get("type") == "message"
    ):
        normalized["content"] = _normalize_message_content(normalized.get("content"))
    return normalized


def _normalize_message_content(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "input_text", "text": value}]
    if isinstance(value, list):
        return [_normalize_message_content_item(item) for item in value]
    if isinstance(value, dict):
        return [_normalize_message_content_item(value)]
    return []


def _normalize_message_content_item(value: Any) -> Any:
    if isinstance(value, str):
        return {"type": "input_text", "text": value}
    if not isinstance(value, dict):
        return value
    if "type" not in value and isinstance(value.get("text"), str):
        return {**value, "type": "input_text"}
    return value


def _coerce_responses_input_items_for_local_model(value: Any) -> tuple[Any, int]:
    """Collapse Codex-only top-level input items into plain messages.

    Codex can include private context items such as ``reasoning`` alongside
    standard Responses items. A number of OpenAI-compatible local servers fail
    on the private items with generic schema errors like "Cannot determine type
    of 'item'". Preserve standard function calls and their outputs verbatim so
    the local model can complete a real call -> result -> answer cycle; collapse
    only unsupported/private items into developer context messages.
    """
    if isinstance(value, list):
        coerced: list[Any] = []
        count = 0
        for item in value:
            converted, item_count = _coerce_responses_input_item_for_local_model(item)
            coerced.append(converted)
            count += item_count
        return coerced, count
    converted, count = _coerce_responses_input_item_for_local_model(value)
    return converted, count


def _coerce_responses_input_item_for_local_model(item: Any) -> tuple[Any, int]:
    if isinstance(item, dict) and item.get("type") in LOCAL_RESPONSE_INPUT_ITEM_TYPES:
        normalized = dict(item)
        normalized["content"] = _normalize_message_content(normalized.get("content"))
        if not isinstance(normalized.get("role"), str) or not normalized.get("role"):
            normalized["role"] = "user"
        return normalized, 0
    if isinstance(item, dict) and item.get("type") in {
        "function_call",
        "function_call_output",
        "tool_call_output",
    }:
        if item.get("type") in {"function_call_output", "tool_call_output"}:
            return _compact_binary_tool_output(item), 0
        return item, 0
    if isinstance(item, dict) and item.get("type") == "compaction_trigger":
        return _context_item_to_developer_message(
            "compaction_trigger",
            "A startup compaction trigger was omitted for local model compatibility.",
        ), 1
    return _context_item_to_developer_message_from_value(item), 1


def _translate_special_tool_items_for_local_model(value: Any) -> Any:
    """Replay Codex-only tool items as ordinary Responses function items.

    OpenAI-compatible local servers generally understand ``function_call`` and
    ``function_call_output`` but not Codex's client-executed ``tool_search`` or
    free-form custom-tool history. Converting those history items keeps the
    local model's call -> result -> answer loop intact after Codex executes the
    tool.
    """
    if isinstance(value, list):
        return [_translate_special_tool_items_for_local_model(item) for item in value]
    if not isinstance(value, dict):
        return value
    item_type = value.get("type")
    if item_type == "tool_search_call":
        arguments = value.get("arguments")
        if not isinstance(arguments, dict):
            query = value.get("query")
            arguments = {"query": query} if isinstance(query, str) else {}
        query = arguments.get("query")
        query_text = query if isinstance(query, str) else "the requested capability"
        return _context_item_to_developer_message(
            "tool_search_call",
            f"Codex searched its deferred tool catalog for: {query_text}",
        )
    if item_type == "tool_search_output":
        loaded = _tool_definition_names(value.get("tools"))
        return _context_item_to_developer_message(
            "tool_search_output",
            (
                "Codex successfully loaded these deferred tools as ordinary "
                "callable functions for this turn:\n"
                + "\n".join(f"- {name}" for name in loaded)
            ),
        )
    if item_type == "custom_tool_call":
        return {
            "type": "function_call",
            "id": value.get("id"),
            "call_id": value.get("call_id") or value.get("id") or "call_custom_tool",
            "name": value.get("name") or "custom_tool",
            "arguments": json.dumps(
                {"input": value.get("input", "")},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    if item_type == "custom_tool_call_output":
        return {
            "type": "function_call_output",
            "id": value.get("id"),
            "call_id": value.get("call_id") or value.get("id") or "call_custom_tool",
            "output": value.get("output", ""),
        }
    return value


def _tool_definition_names(value: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(value, list):
        return names
    for tool in value:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if tool.get("type") == "namespace" and isinstance(name, str):
            for nested in tool.get("tools", []):
                if isinstance(nested, dict) and isinstance(nested.get("name"), str):
                    names.append(f"{name}__{nested['name']}")
        elif isinstance(name, str):
            names.append(name)
    return names


def _compact_binary_tool_output(item: dict[str, Any]) -> dict[str, Any]:
    """Remove inline binary images from tool history while keeping success text.

    Codex image tools return a data URL as an ``input_image`` item plus a small
    textual result. Forwarding that base64 payload to a local text model can
    consume hundreds of thousands of tokens. The desktop already owns and
    displays the image; the model only needs the textual result for its final
    response.
    """
    output = item.get("output")
    if not isinstance(output, list):
        return item
    compacted: list[Any] = []
    omitted_images = 0
    for part in output:
        if isinstance(part, dict) and (
            part.get("type") in {"input_image", "output_image", "image"}
            or "image_url" in part
        ):
            omitted_images += 1
            continue
        if isinstance(part, dict):
            updated = dict(part)
            for key in ("text", "content", "output"):
                value = updated.get(key)
                if isinstance(value, str) and len(value) > LOCAL_CONTEXT_TEXT_LIMIT:
                    updated[key] = (
                        value[:LOCAL_CONTEXT_TEXT_LIMIT]
                        + "\n[Large tool text truncated for local model compatibility.]"
                    )
            compacted.append(updated)
        else:
            compacted.append(part)
    if omitted_images:
        compacted.insert(
            0,
            {
                "type": "input_text",
                "text": (
                    f"[{omitted_images} generated image payload(s) omitted from local "
                    "model context; the image tool completed and Codex retains the images.]"
                ),
            },
        )
    updated_item = dict(item)
    updated_item["output"] = compacted
    return updated_item


def _context_item_to_developer_message_from_value(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        item_type = item.get("type")
        item_type_text = (
            item_type if isinstance(item_type, str) and item_type else "unknown"
        )
        if item_type_text == "reasoning":
            text = "Previous reasoning item omitted for local model compatibility."
        elif item_type_text == "function_call":
            name = item.get("name")
            arguments = item.get("arguments")
            parts = ["Previous function call."]
            if isinstance(name, str) and name:
                parts.append(f"Name: {name}")
            if arguments not in {None, ""}:
                parts.append(f"Arguments: {_stringify_context_value(arguments)}")
            text = "\n".join(parts)
        elif item_type_text in {"function_call_output", "tool_call_output"}:
            output = (
                item.get("output")
                if "output" in item
                else item.get("content")
                if "content" in item
                else item
            )
            text = "Previous tool output:\n" + _stringify_context_value(output)
        else:
            text = _stringify_context_value(item)
        return _context_item_to_developer_message(item_type_text, text)
    if isinstance(item, str):
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": item}],
        }
    return _context_item_to_developer_message("unknown", _stringify_context_value(item))


def _context_item_to_developer_message(item_type: str, text: str) -> dict[str, Any]:
    body = text.strip() or "[No textual content.]"
    if len(body) > LOCAL_CONTEXT_TEXT_LIMIT:
        body = (
            body[:LOCAL_CONTEXT_TEXT_LIMIT]
            + "\n[Truncated for local model compatibility.]"
        )
    return {
        "type": "message",
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Previous Codex context item converted for local model "
                    f"compatibility (type: {item_type}):\n\n{body}"
                ),
            }
        ],
    }


def _stringify_context_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


DEFERRED_TOOL_HINT = (
    "Some Codex tools are advertised through the deferred `tool_search` tool. "
    "If a skill, plugin, or user request says to use a tool that is not currently "
    "callable, call `tool_search` first to expose it; do not use shell commands, "
    "`which`, filesystem checks, `list_mcp_resources`, or `list_mcp_resource_templates` "
    "to decide whether a deferred tool exists. This is especially important for "
    "`node_repl`, `mcp__node_repl__js`, computer-use, Chrome/browser control, and "
    "other plugin tools. If a browser or computer-use skill says to use Node REPL "
    "and `mcp__node_repl__js` is not visible, call `tool_search` with the query "
    "`node_repl js` and no result limit; if needed, search again with limit 10. "
    "After tool search returns a matching namespace, call its exposed function "
    "directly; do not merely report that it is an MCP tool or question whether "
    "it is callable. "
    "Do not replace those tools with shell commands or AppleScript unless the user "
    "explicitly asks for that fallback."
)
DEFERRED_TOOL_FINAL_REMINDER = (
    "Deferred-tool reminder: when `node_repl`, `mcp__node_repl__js`, Chrome, "
    "browser control, or computer-use is needed but not currently visible, call "
    "`tool_search` for `node_repl js`. Do not probe with shell commands or MCP "
    "resource-listing tools. After discovery, call the exposed function directly."
)


def _has_tool_search(payload: dict[str, Any]) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("type") == "tool_search"
        for tool in payload.get("tools", [])
    )


def _prepend_deferred_tool_hint(value: Any) -> Any:
    hint_item = {
        "role": "developer",
        "content": [{"type": "input_text", "text": DEFERRED_TOOL_HINT}],
    }
    reminder_item = {
        "role": "developer",
        "content": [{"type": "input_text", "text": DEFERRED_TOOL_FINAL_REMINDER}],
    }
    if _contains_string(value, DEFERRED_TOOL_HINT):
        return value
    if isinstance(value, list):
        return [hint_item, *value, reminder_item]
    if value is None:
        return [hint_item, reminder_item]
    return [
        hint_item,
        {
            "role": "user",
            "content": [{"type": "input_text", "text": value}]
            if isinstance(value, str)
            else value,
        },
        reminder_item,
    ]


def _append_promoted_tool_hint(value: Any, names: Iterable[str]) -> Any:
    available = list(dict.fromkeys(name for name in names if isinstance(name, str)))
    if not available:
        return value
    displayed = available[:80]
    suffix = (
        f"\n- … and {len(available) - len(displayed)} more"
        if len(available) > len(displayed)
        else ""
    )
    hint = (
        "DEFERRED TOOL DISCOVERY SUCCEEDED. The following names are ordinary "
        "callable function tools in your current tool definitions:\n"
        + "\n".join(f"- {name}" for name in displayed)
        + suffix
        + "\nCall the required function directly using native function calling. "
        "An `mcp__` prefix does not make it non-callable. Do not search for any "
        "of these names again, do not inspect MCP resources, and do not claim "
        "that a listed function is absent. For Chrome, Browser, or Computer Use "
        "skills, call `mcp__node_repl__js` and run the skill's JavaScript runtime "
        "instructions."
    )
    item = {
        "role": "developer",
        "content": [{"type": "input_text", "text": hint}],
    }
    if isinstance(value, list):
        return [*value, item]
    if value is None:
        return [item]
    return [value, item]


def _contains_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, list):
        return any(_contains_string(item, needle) for item in value)
    if isinstance(value, dict):
        return any(_contains_string(item, needle) for item in value.values())
    return False


def summarize_responses_request(
    payload: dict[str, Any], raw_bytes: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "request_bytes": raw_bytes,
        "input_item_count": _input_item_count(payload.get("input")),
        "input_roles": _input_roles(payload.get("input")),
        "has_previous_response_id": isinstance(
            payload.get("previous_response_id"), str
        ),
        "is_compaction_request": is_compaction_request(payload),
    }
    if isinstance(payload.get("store"), bool):
        summary["store"] = payload["store"]
    return summary


def is_compaction_request(payload: dict[str, Any]) -> bool:
    """Best-effort detection for Codex's remote compaction task.

    Codex desktop currently sends compaction work through the same Responses
    route as ordinary inference. OpenAI returns a special output item of
    type ``compaction`` for that task; most local OpenAI-compatible servers
    ignore the distinction and emit assistant text instead. We only use this
    predicate to adapt the response shape, and we never log the text it scans.
    """
    if not isinstance(payload, dict):
        return False
    if _contains_compaction_trigger_item(payload.get("input")):
        return True
    # Once compaction succeeds, Codex sends ordinary follow-up requests that
    # include compaction items in the input context and the normal tool set. Do
    # not wrap those assistant responses as new compaction summaries.
    if payload.get("tools"):
        return False
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if (
                isinstance(key, str)
                and "compact" in key.lower()
                and value not in {False, None, ""}
            ):
                return True
            if isinstance(value, str) and _looks_like_compaction_marker(value):
                return True
    context_management = payload.get("context_management")
    if _contains_compaction_context_management(context_management) and not payload.get(
        "tools"
    ):
        return True
    return _contains_compaction_prompt_marker(payload.get("input"))


def is_compaction_trigger_request(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and _contains_compaction_trigger_item(
        payload.get("input")
    )


def synthetic_compaction_response(
    payload: dict[str, Any], model: str
) -> dict[str, Any]:
    summary = _synthetic_compaction_summary(payload)
    transformed, _ = adapt_compaction_response(
        {
            "id": _stable_response_id(summary),
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "status": "completed",
            "output_text": summary,
            "parallel_tool_calls": True,
            "tools": [],
        }
    )
    if isinstance(transformed, dict):
        return transformed
    raise ValueError("failed to build synthetic compaction response")


def normalize_legacy_compaction_item_ids(value: Any) -> tuple[Any, int]:
    """Repair compaction IDs emitted by older Harness versions.

    Codex accepts compaction item IDs with the ``cmp.`` prefix. Older Harness
    releases generated ``ci_local_`` IDs, which can remain in a task's replayed
    input and fail validation when that task is continued from another client.
    """

    def visit(node: Any) -> tuple[Any, int]:
        if isinstance(node, list):
            normalized: list[Any] = []
            count = 0
            for item in node:
                repaired, item_count = visit(item)
                normalized.append(repaired)
                count += item_count
            return normalized, count
        if not isinstance(node, dict):
            return node, 0
        normalized: dict[Any, Any] = {}
        count = 0
        for key, item in node.items():
            repaired, item_count = visit(item)
            normalized[key] = repaired
            count += item_count
        item_id = normalized.get("id")
        if (
            normalized.get("type") == "compaction"
            and isinstance(item_id, str)
            and item_id.startswith("ci_local_")
        ):
            normalized["id"] = "cmp." + item_id.removeprefix("ci_local_")
            count += 1
        return normalized, count

    return visit(value)


def normalize_invalid_local_reasoning_items(value: Any) -> tuple[Any, int]:
    """Remove persisted local chain-of-thought that hosted replay rejects.

    Valid hosted reasoning items have no visible ``content``. Local servers may
    emit non-empty reasoning text, which Codex can persist before an upgraded
    interceptor is running. Remove only that invalid shape so existing hybrid
    tasks recover without altering valid hosted reasoning state.
    """

    def visit(node: Any) -> tuple[Any, int, bool]:
        if isinstance(node, list):
            normalized: list[Any] = []
            count = 0
            for item in node:
                repaired, item_count, drop = visit(item)
                count += item_count
                if not drop:
                    normalized.append(repaired)
            return normalized, count, False
        if not isinstance(node, dict):
            return node, 0, False
        content = node.get("content")
        if (
            node.get("type") == "reasoning"
            and isinstance(content, list)
            and len(content) > 0
        ):
            return None, 1, True
        normalized: dict[Any, Any] = {}
        count = 0
        for key, item in node.items():
            repaired, item_count, drop = visit(item)
            count += item_count
            if not drop:
                normalized[key] = repaired
        return normalized, count, False

    normalized, count, _ = visit(value)
    return normalized, count


def adapt_compaction_response(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, dict):
        return value, False
    summary = extract_response_text(value).strip()
    if not summary:
        summary = "No prior visible conversation content required summarization."
    transformed = json.loads(json.dumps(value))
    item_id = _stable_compaction_item_id(summary)
    transformed["output"] = [
        {
            "id": item_id,
            "type": "compaction",
            "encrypted_content": summary,
            "created_by": "omlx-codex-interceptor",
        }
    ]
    transformed["status"] = "completed"
    transformed.pop("output_text", None)
    # The source can be an upstream failure body. A compaction reported as
    # completed must not also carry that failure, or Codex fails the task on the
    # error it finds instead of accepting the summary.
    transformed["error"] = None
    transformed["incomplete_details"] = None
    return transformed, True


def adapt_compaction_sse_response(raw: bytes) -> tuple[dict[str, Any], bool]:
    text_parts: list[str] = []
    fallback_text: str = ""
    template: dict[str, Any] | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            continue
        data = stripped.split(b":", 1)[1].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            decoded = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(decoded, dict):
            template = decoded
            event_type = decoded.get("type")
            delta = decoded.get("delta")
            if event_type == "response.output_text.delta" and isinstance(delta, str):
                text_parts.append(delta)
            elif event_type in {
                "response.output_text.done",
                "response.output_item.done",
                "response.completed",
            }:
                extracted = extract_response_text(decoded)
                if extracted:
                    fallback_text = extracted
    if template is None:
        template = {
            "id": "resp_compaction_local",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
        }
    base = {
        key: value
        for key, value in template.items()
        if key
        in {
            "id",
            "object",
            "created_at",
            "model",
            "usage",
            "service_tier",
        }
    }
    if not base.get("id"):
        base["id"] = "resp_compaction_local"
    base.setdefault("object", "response")
    base.setdefault("created_at", int(time.time()))
    base["status"] = "completed"
    summary = ("".join(text_parts) or fallback_text).strip()
    summary = re.sub(r"</?think>", "", summary, flags=re.IGNORECASE).strip()
    return adapt_compaction_response({**base, "output_text": summary})


def compaction_response_sse(value: Any) -> tuple[bytes, bool]:
    if _is_single_compaction_response(value):
        transformed = json.loads(json.dumps(value))
        item = transformed["output"][0]
        if not (isinstance(item.get("id"), str) and item["id"].startswith("cmp.")):
            item["id"] = _stable_compaction_item_id(item["encrypted_content"])
    else:
        transformed, adapted = adapt_compaction_response(value)
        if not adapted:
            return b"", False
    if not isinstance(transformed, dict):
        return b"", False
    output = transformed.get("output")
    if not isinstance(output, list) or len(output) != 1:
        return b"", False
    item = output[0]
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": transformed,
        },
    ]
    lines = [
        "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n"
        for event in events
    ]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8"), True


def _is_single_compaction_response(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    output = value.get("output")
    return (
        isinstance(output, list)
        and len(output) == 1
        and isinstance(output[0], dict)
        and output[0].get("type") == "compaction"
        and isinstance(output[0].get("encrypted_content"), str)
    )


def extract_response_text(value: Any) -> str:
    parts: list[str] = []

    def visit(node: Any, *, key: str | None = None) -> None:
        if isinstance(node, str):
            if key in {"text", "output_text", "content", "delta"}:
                parts.append(node)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        if isinstance(node_type, str) and node_type in {
            "reasoning",
            "function_call",
            "computer_call",
            "file_search_call",
            "web_search_call",
            "tool_search_call",
        }:
            return
        for child_key, child in node.items():
            visit(child, key=child_key if isinstance(child_key, str) else None)

    visit(value)
    return "".join(parts)


def _stable_compaction_item_id(summary: str) -> str:
    digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:24]
    return f"cmp.{digest}"


def _stable_response_id(summary: str) -> str:
    digest = hashlib.sha256(("response:" + summary).encode("utf-8")).hexdigest()[:24]
    return f"resp_local_{digest}"


def _input_item_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        return 1
    if value is None:
        return 0
    return 1


def typeless_input_item_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(
            1
            for item in value
            if not (isinstance(item, dict) and isinstance(item.get("type"), str))
        )
    if isinstance(value, dict):
        return 0 if isinstance(value.get("type"), str) else 1
    if value is None:
        return 0
    return 1


def local_incompatible_input_item_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(_is_local_incompatible_input_item(item) for item in value)
    if value is None:
        return 0
    return _is_local_incompatible_input_item(value)


def _is_local_incompatible_input_item(item: Any) -> int:
    if isinstance(item, str):
        return 0
    if not isinstance(item, dict):
        return 1
    item_type = item.get("type")
    if item_type == "message":
        return 0
    if item_type is None and "content" in item:
        return 0
    return 1


def input_type_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]
    for item in items:
        if isinstance(item, dict):
            item_type = item.get("type")
            if not isinstance(item_type, str) or not item_type:
                item_type = "implicit_message" if "content" in item else "typeless"
        elif isinstance(item, str):
            item_type = "string"
        else:
            item_type = type(item).__name__
        item_type = _display_token(item_type, 60, fallback="unknown")
        counts[item_type] = counts.get(item_type, 0) + 1
    return counts


def non_array_content_item_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(
            1
            for item in value
            if isinstance(item, dict)
            and "content" in item
            and not isinstance(item.get("content"), list)
        )
    if isinstance(value, dict):
        return (
            1
            if "content" in value and not isinstance(value.get("content"), list)
            else 0
        )
    return 0


def _input_roles(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = [item for item in value if isinstance(item, dict)]
    else:
        items = []
    for item in items:
        role = item.get("role")
        if not isinstance(role, str) or not role:
            role = item.get("type")
        if not isinstance(role, str) or not role:
            role = "unknown"
        role = _display_token(role, 40, fallback="unknown")
        counts[role] = counts.get(role, 0) + 1
    return counts


def is_readable_compaction_summary(value: Any) -> bool:
    """Whether a compaction item carries a summary this model can actually read.

    ``encrypted_content`` is literal: a hosted compaction stores ciphertext that
    only the hosted backend can decrypt. Handing that to a local model spends
    thousands of tokens on base64 and tells it nothing, which reads exactly like
    the model having forgotten the conversation. Locally produced summaries are
    plain text and are passed through.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    sample = value[:400]
    if sample.startswith("gAAAAA"):  # Fernet, as used by hosted compaction
        return False
    return not (
        len(sample) > 120 and not any(character.isspace() for character in sample)
    )


LOCAL_REQUEST_RULES = (
    "tool_result_must_follow_its_call",
    "tool_result_without_a_call",
    "instruction_after_conversation_started",
    "message_content_is_not_an_array",
    "message_content_is_empty",
    "item_type_cannot_be_determined",
)


def repair_local_request(payload: Any) -> tuple[Any, list[str]]:
    """Repair recoverable history shapes before a local chat template sees them.

    Responses streams are allowed to group calls and results, and older bridge
    versions could persist a result after losing its call. Local chat templates
    are generally stricter: each result must immediately follow its call. Pair
    known calls with their results and preserve an orphan result as an ordinary
    user-visible history note instead of either dropping it or sending a request
    the upstream has already told us it cannot parse.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), list):
        return payload, []
    original_rules = validate_local_request(payload)
    if not original_rules:
        return payload, []

    items = payload["input"]
    call_positions: dict[str, int] = {}
    outputs_by_call: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    output_indexes: set[int] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        if (
            item_type in {"function_call", "custom_tool_call"}
            and isinstance(call_id, str)
            and call_id
        ):
            call_positions.setdefault(call_id, index)
        elif (
            item_type
            in {
                "function_call_output",
                "tool_call_output",
                "custom_tool_call_output",
            }
            and isinstance(call_id, str)
            and call_id
        ):
            outputs_by_call.setdefault(call_id, []).append((index, item))
            output_indexes.add(index)

    repaired_items: list[Any] = []
    consumed_outputs: set[int] = set()
    for index, item in enumerate(items):
        if index in output_indexes:
            call_id = item.get("call_id") if isinstance(item, dict) else None
            if not isinstance(call_id, str) or call_id not in call_positions:
                repaired_items.append(
                    _recovered_history_message(
                        f"tool result {call_id or 'unknown'}",
                        _tool_output_value(item),
                    )
                )
                consumed_outputs.add(index)
            continue
        if not isinstance(item, dict):
            # The normalizer should already have handled these. Preserve their
            # value as readable context rather than forwarding a typeless item.
            repaired_items.append(_recovered_history_message("item", item))
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        repaired_items.append(copy.deepcopy(item))
        if (
            item_type in {"function_call", "custom_tool_call"}
            and isinstance(call_id, str)
            and call_id
        ):
            matches = outputs_by_call.get(call_id, [])
            if matches:
                result_index, result = matches[0]
                repaired_items.append(copy.deepcopy(result))
                consumed_outputs.add(result_index)
                # Duplicate results cannot all directly follow one call. Preserve
                # any extras as readable history rather than malformed tool items.
                for extra_index, extra in matches[1:]:
                    repaired_items.append(
                        _recovered_history_message(
                            f"duplicate tool result {call_id}",
                            _tool_output_value(extra),
                        )
                    )
                    consumed_outputs.add(extra_index)

    # Every item appended above is already a copy or a freshly built message,
    # so a shallow payload copy carrying the new input list is enough.
    repaired = dict(payload)
    repaired["input"] = repaired_items
    remaining = validate_local_request(repaired)
    return repaired, [rule for rule in original_rules if rule not in remaining]


def _tool_output_value(item: dict[str, Any]) -> Any:
    if "output" in item:
        return item.get("output")
    return item.get("content")


def _recovered_history_message(label: str, value: Any) -> dict[str, Any]:
    text = _stringify_context_value(value)
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"RECOVERED HISTORY — {label}: {text}",
            }
        ],
    }


def validate_local_request(payload: Any) -> list[str]:
    """Name the rules this request breaks before a local server rejects it.

    Every rule here was learned from an actual rejection: "Cannot determine type
    of 'item'", "item['content'] is not an array", "item['content'] is empty",
    "System/developer message must be at the beginning.", and "A tool message
    must follow an assistant or tool message." Each one costs a full round trip
    (60-200s on a large prompt) and is then retried, so naming the offending rule
    locally is worth far more than discovering it upstream.
    """
    if not isinstance(payload, dict):
        return []
    items = payload.get("input")
    if not isinstance(items, list):
        return []
    broken: list[str] = []
    pending_call_ids: set[str] = set()
    seen_conversation = False
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            broken.append("item_type_cannot_be_determined")
            continue
        item_type = item.get("type")
        role = item.get("role")
        if not isinstance(item_type, str) and not isinstance(role, str):
            broken.append("item_type_cannot_be_determined")
            continue
        if item_type in {"function_call", "custom_tool_call"}:
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                pending_call_ids.add(call_id)
            seen_conversation = True
            continue
        if item_type in {
            "function_call_output",
            "tool_call_output",
            "custom_tool_call_output",
        }:
            call_id = item.get("call_id")
            previous = items[index - 1] if index else None
            directly_follows = (
                isinstance(previous, dict)
                and previous.get("type") in {"function_call", "custom_tool_call"}
                and previous.get("call_id") == call_id
            )
            if not directly_follows:
                broken.append("tool_result_must_follow_its_call")
            if isinstance(call_id, str) and call_id not in pending_call_ids:
                broken.append("tool_result_without_a_call")
            seen_conversation = True
            continue
        if item_type == "message" or isinstance(role, str):
            if role in {"system", "developer"}:
                if seen_conversation:
                    broken.append("instruction_after_conversation_started")
            else:
                seen_conversation = True
            content = item.get("content")
            if content is not None and not isinstance(content, list):
                broken.append("message_content_is_not_an_array")
            elif isinstance(content, list) and not content:
                broken.append("message_content_is_empty")
            continue
        seen_conversation = True
    return list(dict.fromkeys(broken))


def canonical_digest(value: Any) -> str:
    """A stable digest of a request, for recognising an identical retry."""
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def source_fingerprint(paths: Iterable[Path]) -> str:
    """A short digest of the code actually loaded.

    Answering "is my fix running?" cost three debugging cycles in one day, and a
    90-minute session ran against code that had already been corrected on disk.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        try:
            digest.update(Path(path).read_bytes())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()[:12]


class SSEUsageObserver:
    """Capture the usage block a local server reports on its final event.

    Cache-hit rate is the single most useful local-health signal: a turn that
    re-reads its whole prompt costs orders of magnitude more than one that does
    not, and nothing in the receipt showed it before.
    """

    def __init__(self) -> None:
        self.buffer = b""
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return chunk
        self.buffer += chunk
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                break
            line = self.buffer[:newline]
            self.buffer = self.buffer[newline + 1 :]
            self._observe(line)
        return chunk

    def _observe(self, line: bytes) -> None:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return
        data = stripped.split(b":", 1)[1].strip()
        if not data or data == b"[DONE]":
            return
        # Usage only ever appears under a "usage" key; skip parsing plain deltas.
        if b'"usage"' not in data:
            return
        try:
            event = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(event, dict):
            return
        usage = usage_from_response_event(event)
        if usage is not None:
            self.usage = usage


def usage_from_response_event(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    if event.get("type") not in {None, "response.completed"}:
        return None
    response = event.get("response") if "response" in event else event
    usage = response.get("usage") if isinstance(response, dict) else None
    return usage if isinstance(usage, dict) else None


def usage_fields(usage: Any, *, duration_ms: int | None = None) -> dict[str, Any]:
    """Flatten a Responses usage block into receipt fields."""
    if not isinstance(usage, dict):
        return {}
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    fields: dict[str, Any] = {}
    if isinstance(input_tokens, int):
        fields["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        fields["output_tokens"] = output_tokens
    if isinstance(cached, int):
        fields["cached_tokens"] = cached
        if isinstance(input_tokens, int) and input_tokens > 0:
            fields["cache_hit_percent"] = round(cached / input_tokens * 100, 1)
    if (
        isinstance(output_tokens, int)
        and isinstance(duration_ms, int)
        and duration_ms > 0
    ):
        fields["output_tokens_per_second"] = round(
            output_tokens / (duration_ms / 1000), 1
        )
    return fields


def response_is_empty(value: Any) -> bool:
    """Whether a completed response said nothing and called nothing.

    One recorded session produced 18 empty assistant messages; the harness
    swallowed every one of them silently.
    """
    if not isinstance(value, dict):
        return False
    output = value.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"function_call", "custom_tool_call"}:
            return False
        if extract_response_text({"output": [item]}).strip():
            return False
    return True


def opaque_compaction_item_count(value: Any) -> int:
    """How many compaction items in this request are unreadable to the model."""
    if isinstance(value, list):
        return sum(opaque_compaction_item_count(item) for item in value)
    if not isinstance(value, dict):
        return 0
    if value.get("type") == "compaction":
        return (
            0 if is_readable_compaction_summary(value.get("encrypted_content")) else 1
        )
    return sum(opaque_compaction_item_count(item) for item in value.values())


def _translate_compaction_items_for_local_model(value: Any) -> Any:
    if isinstance(value, list):
        return [_translate_compaction_items_for_local_model(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "compaction" and isinstance(
        value.get("encrypted_content"), str
    ):
        summary = value["encrypted_content"]
        if not is_readable_compaction_summary(summary):
            return {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "An earlier part of this conversation was summarized "
                            "by the hosted service and that summary cannot be "
                            "read by this model. Treat the conversation as "
                            "starting from the messages that follow, and ask the "
                            "user to restate the goal if anything essential is "
                            "missing rather than guessing."
                        ),
                    }
                ],
            }
        return {
            "type": "message",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Previous conversation summary supplied by local "
                        f"compaction:\n\n{summary}"
                    ),
                }
            ],
        }
    return {
        key: _translate_compaction_items_for_local_model(item)
        for key, item in value.items()
    }


def _contains_compaction_context_management(value: Any) -> bool:
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "compaction":
            return True
    return False


def _contains_compaction_trigger_item(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_compaction_trigger_item(item) for item in value)
    if not isinstance(value, dict):
        return False
    item_type = value.get("type")
    if isinstance(item_type, str) and item_type == "compaction_trigger":
        return True
    return any(_contains_compaction_trigger_item(item) for item in value.values())


def _remove_compaction_trigger_items(value: Any) -> Any:
    if isinstance(value, list):
        return [
            _remove_compaction_trigger_items(item)
            for item in value
            if not (isinstance(item, dict) and item.get("type") == "compaction_trigger")
        ]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "compaction_trigger":
        return None
    return {key: _remove_compaction_trigger_items(item) for key, item in value.items()}


def _contains_compaction_prompt_marker(value: Any) -> bool:
    return any(_looks_like_compaction_marker(text) for text in _iter_strings(value))


def _synthetic_compaction_summary(payload: dict[str, Any]) -> str:
    user_messages = _recent_user_texts(payload.get("input"))
    lines = [
        "Automatic startup compaction completed locally by the oMLX.",
        "No remote/local model generation was required for this housekeeping step.",
    ]
    if user_messages:
        lines.append("Recent user message(s):")
        lines.extend(f"- {message}" for message in user_messages)
    return "\n".join(lines)


def _recent_user_texts(
    value: Any, *, limit: int = 5, max_chars: int = 500
) -> list[str]:
    messages: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("role") == "user":
            text = " ".join(_iter_strings(node.get("content"))).strip()
            if text:
                messages.append(_safe_summary_text(text, max_chars=max_chars))
            return
        for item in node.values():
            if isinstance(item, (dict, list)):
                visit(item)

    visit(value)
    return messages[-limit:]


def _safe_summary_text(value: str, *, max_chars: int) -> str:
    return " ".join(value.split())[:max_chars]


def _looks_like_compaction_marker(value: str) -> bool:
    lowered = value.lower()
    markers = (
        "remote compact task",
        "remote compaction",
        "compaction summary",
        "context compaction",
        "compact the conversation",
        "compact this conversation",
        "summarize the conversation so far",
    )
    return any(marker in lowered for marker in markers)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)


def extract_advertised_tools(payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    types: set[str] = set()
    for tool in payload.get("tools", []):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
        tool_type = tool.get("type")
        if isinstance(tool_type, str) and tool_type:
            types.add(tool_type)
    return names, types


def extract_namespace_tools(payload: dict[str, Any]) -> dict[str, set[str]]:
    namespaces: dict[str, set[str]] = {}
    for tool in payload.get("tools", []):
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "namespace"
            or not isinstance(tool.get("name"), str)
            or not tool["name"]
        ):
            continue
        names = {
            nested["name"]
            for nested in tool.get("tools", [])
            if isinstance(nested, dict)
            and nested.get("type") == "function"
            and isinstance(nested.get("name"), str)
            and nested["name"]
        }
        if names:
            namespaces[tool["name"]] = names
    return namespaces


def flatten_namespace_tools_for_local_model(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Expose Codex namespace members as ordinary local function tools.

    Hosted Codex models understand the Responses ``namespace`` tool type, but
    many OpenAI-compatible local servers only compile ``function`` entries into
    the model's tool template. Prefixing the member name keeps it unique. The
    response normalizer maps the flattened call back to ``namespace`` + member
    before Codex sees it.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload, set()
    flattened: list[Any] = []
    names: set[str] = set()
    for tool in tools:
        if not (
            isinstance(tool, dict)
            and tool.get("type") == "namespace"
            and isinstance(tool.get("name"), str)
            and tool["name"]
            and isinstance(tool.get("tools"), list)
        ):
            flattened.append(tool)
            continue
        namespace = tool["name"]
        for nested in tool["tools"]:
            if not (
                isinstance(nested, dict)
                and nested.get("type") == "function"
                and isinstance(nested.get("name"), str)
                and nested["name"]
            ):
                continue
            exposed = dict(nested)
            exposed_name = f"{namespace}__{nested['name']}"
            exposed["name"] = exposed_name
            flattened.append(exposed)
            names.add(exposed_name)
    payload["tools"] = flattened
    return payload, names


def expose_client_tools_for_local_model(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]], set[str]]:
    """Expose Codex client tools using local-server-compatible functions.

    ``tool_search`` and ``custom`` are first-class Responses tool types, but
    common local servers only compile ``function`` tools into the model chat
    template. The returned map lets the response bridge restore the exact
    Codex wire item after the local model calls the compatibility function.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload, {}, set()
    used = {
        tool["name"]
        for tool in tools
        if isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("name"), str)
        and tool["name"]
    }
    exposed: list[Any] = []
    mappings: dict[str, dict[str, str]] = {}
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            exposed.append(tool)
            continue
        tool_type = tool.get("type")
        if tool_type == "tool_search":
            name = _unique_tool_name("tool_search", used)
            used.add(name)
            mappings[name] = {"kind": "tool_search", "name": "tool_search"}
            names.add(name)
            exposed.append(
                {
                    "type": "function",
                    "name": name,
                    "description": (
                        "Search Codex's deferred tool catalog. Use this whenever a "
                        "required plugin, skill tool, node_repl, browser, Chrome, "
                        "computer-use, or other named tool is not currently visible. "
                        "Never search again for a tool already present in the current "
                        "function definitions; `mcp__` functions are directly callable."
                    ),
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Short search query naming the missing capability "
                                    "or tool, for example 'node_repl js'."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "description": "Optional maximum number of tool groups.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                }
            )
            continue
        if tool_type == "custom" and isinstance(tool.get("name"), str) and tool["name"]:
            original = tool["name"]
            name = _unique_tool_name(original, used)
            used.add(name)
            mappings[name] = {"kind": "custom", "name": original}
            names.add(name)
            description = tool.get("description")
            if not isinstance(description, str):
                description = f"Call the Codex {original} free-form tool."
            exposed.append(
                {
                    "type": "function",
                    "name": name,
                    "description": (
                        description
                        + "\nPass the complete free-form tool payload in `input`."
                    ),
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Complete free-form tool input.",
                            }
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                }
            )
            continue
        exposed.append(tool)
    payload["tools"] = exposed
    return payload, mappings, names


def _unique_tool_name(preferred: str, used: set[str]) -> str:
    if preferred not in used:
        return preferred
    base = f"codex_compat__{preferred}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def normalize_tool_name(emitted: str, registered_names: Iterable[str]) -> str:
    valid = {name for name in registered_names if isinstance(name, str) and name}
    if emitted in valid or not emitted or not valid:
        return emitted

    emitted_tokens = _tool_name_tokens(emitted)
    exact_shape = [name for name in valid if _tool_name_tokens(name) == emitted_tokens]
    if len(exact_shape) == 1:
        return exact_shape[0]

    # Some templates prepend a namespace to the exact registered name. Only
    # accept a unique, separator-aligned suffix; never use bare str.endswith.
    suffix_matches = []
    for name in valid:
        candidate_tokens = _tool_name_tokens(name)
        if (
            candidate_tokens
            and len(candidate_tokens) < len(emitted_tokens)
            and emitted_tokens[-len(candidate_tokens) :] == candidate_tokens
        ):
            suffix_matches.append(name)
    return suffix_matches[0] if len(suffix_matches) == 1 else emitted


def normalize_response_tool_names(
    value: Any,
    registered_names: Iterable[str],
    namespace_tools: dict[str, set[str]] | None = None,
) -> tuple[Any, list[dict[str, str]]]:
    registered = set(registered_names)
    namespaces = namespace_tools or {}
    remapped: list[dict[str, str]] = []

    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        updated = {key: visit(item) for key, item in node.items()}
        if updated.get("type") == "function_call" and isinstance(
            updated.get("name"), str
        ):
            original = updated["name"]
            namespace_match = None
            if not updated.get("namespace") and original not in registered:
                namespace_match = _normalize_namespace_call(original, namespaces)
            if namespace_match is not None:
                namespace, normalized = namespace_match
                updated["namespace"] = namespace
                updated["name"] = normalized
                remapped.append({"from": original, "to": f"{namespace}::{normalized}"})
            else:
                normalized = normalize_tool_name(original, registered)
            if namespace_match is None and normalized != original:
                updated["name"] = normalized
                remapped.append({"from": original, "to": normalized})
        return updated

    return visit(value), remapped


def normalize_client_tool_calls(
    value: Any,
    mappings: dict[str, dict[str, str]] | None,
) -> tuple[Any, int]:
    """Restore compatibility function calls to native Codex response items."""
    special = mappings or {}
    if not special:
        return value, 0
    adapted = 0

    def visit(node: Any) -> Any:
        nonlocal adapted
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        updated = {key: visit(item) for key, item in node.items()}
        if updated.get("type") != "function_call":
            return updated
        name = updated.get("name")
        mapping = special.get(name) if isinstance(name, str) else None
        if not isinstance(mapping, dict):
            return updated
        arguments = _decoded_function_arguments(updated.get("arguments"))
        call_id = updated.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            call_id = _stable_local_call_id(name or "tool", arguments)
        kind = mapping.get("kind")
        adapted += 1
        if kind == "tool_search":
            result = {
                "type": "tool_search_call",
                "call_id": call_id,
                "status": updated.get("status", "completed"),
                "execution": "client",
                "arguments": arguments,
            }
            if isinstance(updated.get("id"), str):
                result["id"] = _special_item_id("tsc_local", updated["id"])
            return result
        if kind == "custom":
            raw_input = arguments.get("input", "")
            if not isinstance(raw_input, str):
                raw_input = json.dumps(
                    raw_input, ensure_ascii=False, separators=(",", ":")
                )
            result = {
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": mapping.get("name") or name,
                "input": raw_input,
                "status": updated.get("status", "completed"),
            }
            if isinstance(updated.get("id"), str):
                result["id"] = _special_item_id("ctc_local", updated["id"])
            return result
        return updated

    return visit(value), adapted


def _decoded_function_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _parse_tool_arguments(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _stable_local_call_id(name: str, arguments: dict[str, Any]) -> str:
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256((name + "\0" + encoded).encode()).hexdigest()[:20]
    return f"call_local_{digest}"


def _special_item_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def sanitize_local_response_items(value: Any, *, copy: bool = True) -> tuple[Any, int]:
    """Remove local-only reasoning output before Codex persists the response.

    Some local Responses servers expose chain-of-thought as a ``reasoning``
    output item with non-empty ``content``. Codex may later replay the task to a
    hosted model, whose schema requires that private field to be empty. Keeping
    the item therefore poisons an otherwise valid hybrid task.

    The payload is deep-copied before mutation unless ``copy`` is false, which
    is only safe when the caller owns a freshly parsed value.
    """
    if not isinstance(value, dict):
        return value, 0
    transformed = json.loads(json.dumps(value)) if copy else value
    removed = 0

    def visit(node: Any) -> None:
        nonlocal removed
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        output = node.get("output")
        if isinstance(output, list):
            kept = []
            for item in output:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    removed += 1
                    continue
                kept.append(item)
                visit(item)
            node["output"] = kept
        for key, child in node.items():
            if key != "output":
                visit(child)

    visit(transformed)
    return transformed, removed


def adapt_textual_tool_call_sse(
    raw: bytes, registered_names: Iterable[str]
) -> tuple[bytes, bool]:
    """Convert a model's textual ``<tool_call>`` fallback into Responses SSE."""
    registered = {name for name in registered_names if isinstance(name, str) and name}
    if not registered or b"<tool_call>" not in raw:
        return raw, False
    events: list[dict[str, Any]] = []
    deltas: list[str] = []
    fallback_text: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            continue
        data = stripped.split(b":", 1)[1].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            event = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if event.get("type") == "response.output_text.delta" and isinstance(
            event.get("delta"), str
        ):
            deltas.append(event["delta"])
        if event.get("type") == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "message":
                fallback_text.append(extract_response_text(item))
    text = "".join(deltas) or "".join(fallback_text)
    parsed = _parse_textual_tool_call(text, registered)
    if parsed is None:
        return raw, False
    name, arguments = parsed
    arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256((name + "\0" + arguments_json).encode()).hexdigest()[:20]
    item = {
        "type": "function_call",
        "id": f"fc_local_{digest}",
        "call_id": f"call_local_{digest}",
        "name": name,
        "arguments": arguments_json,
    }
    response: dict[str, Any] = {}
    for event in events:
        candidate = event.get("response")
        if isinstance(candidate, dict):
            response = copy.deepcopy(candidate)
            break
    response.setdefault("id", f"resp_local_{digest}")
    response.setdefault("object", "response")
    response["status"] = "completed"
    response["output"] = [item]
    response.pop("output_text", None)
    synthetic = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": ""},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item["id"],
            "output_index": 0,
            "delta": arguments_json,
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": item["id"],
            "output_index": 0,
            "arguments": arguments_json,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    output = bytearray()
    for event in synthetic:
        output.extend(f"event: {event['type']}\n".encode())
        output.extend(b"data: ")
        output.extend(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
        )
        output.extend(b"\n\n")
    output.extend(b"data: [DONE]\n\n")
    return bytes(output), True


def adapt_client_tool_call_sse(
    raw: bytes,
    mappings: dict[str, dict[str, str]] | None,
) -> tuple[bytes, bool]:
    """Restore special Codex calls and remove incompatible function deltas.

    Local servers stream compatibility calls as ordinary function items.
    Codex only needs the added/done output items to dispatch a client tool; the
    intervening function-argument delta events refer to the old function item
    ID and are invalid once the item becomes ``tool_search_call`` or
    ``custom_tool_call``.
    """
    special = mappings or {}
    if not special:
        return raw, False
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            continue
        data = stripped.split(b":", 1)[1].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            event = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    special_item_ids: set[str] = set()
    contains_special = False

    def collect(node: Any) -> None:
        nonlocal contains_special
        if isinstance(node, list):
            for item in node:
                collect(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "function_call" and node.get("name") in special:
            contains_special = True
            if isinstance(node.get("id"), str):
                special_item_ids.add(node["id"])
        for item in node.values():
            collect(item)

    for event in events:
        collect(event)
    if not contains_special:
        return raw, False

    converted: list[dict[str, Any]] = []
    for event in events:
        if (
            event.get("type")
            in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }
            and event.get("item_id") in special_item_ids
        ):
            continue
        transformed, _ = normalize_client_tool_calls(event, special)
        converted.append(transformed)
    output = bytearray()
    for event in converted:
        event_type = event.get("type")
        if isinstance(event_type, str):
            output.extend(f"event: {event_type}\n".encode())
        output.extend(b"data: ")
        output.extend(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
        )
        output.extend(b"\n\n")
    output.extend(b"data: [DONE]\n\n")
    return bytes(output), True


def _parse_textual_tool_call(
    text: str, registered_names: set[str]
) -> tuple[str, dict[str, Any]] | None:
    marker = text.rfind("<tool_call>")
    if marker < 0:
        return None
    candidate = text[marker + len("<tool_call>") :].strip()
    candidate = candidate.split("</tool_call>", 1)[0].strip()
    while candidate.startswith("<tool_call>"):
        candidate = candidate[len("<tool_call>") :].strip()
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict) and isinstance(decoded.get("name"), str):
        name = decoded["name"]
        arguments = decoded.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _parse_tool_arguments(arguments)
        if name in registered_names and isinstance(arguments, dict):
            return name, arguments
    xml_call = _parse_xml_textual_tool_call(candidate, registered_names)
    if xml_call is not None:
        return xml_call
    match = re.search(r"([A-Za-z_][A-Za-z0-9_.:-]*)\s*\((\{.*\})\)", candidate, re.S)
    if not match or match.group(1) not in registered_names:
        return None
    arguments = _parse_tool_arguments(match.group(2))
    if not isinstance(arguments, dict):
        return None
    return match.group(1), arguments


def _parse_xml_textual_tool_call(
    candidate: str, registered_names: set[str]
) -> tuple[str, dict[str, Any]] | None:
    """Parse XML-like tool syntax emitted by some local chat templates.

    Models sometimes finish immediately after a raw parameter value, omitting
    ``</parameter>``, ``</function>``, and ``</tool_call>``.  The end of the
    response is therefore also treated as the end of the final parameter.
    """
    function_match = re.search(
        r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*>",
        candidate,
        re.I,
    )
    if function_match is None:
        function_match = re.search(
            r"<function\s+name\s*=\s*['\"]?([A-Za-z_][A-Za-z0-9_.:-]*)"
            r"['\"]?\s*>",
            candidate,
            re.I,
        )
    if function_match is None:
        return None
    name = function_match.group(1)
    if name not in registered_names:
        return None

    body = candidate[function_match.end() :]
    parameter_pattern = re.compile(
        r"<parameter(?:\s*=\s*|\s+name\s*=\s*['\"]?)"
        r"([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*>",
        re.I,
    )
    matches = list(parameter_pattern.finditer(body))
    if not matches:
        return name, {}

    arguments: dict[str, Any] = {}
    for index, match in enumerate(matches):
        parameter_name = match.group(1)
        if parameter_name in arguments:
            return None
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        raw_value = body[match.end() : end]
        raw_value = re.split(
            r"</parameter\s*>|</function\s*>|</tool_call\s*>",
            raw_value,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        arguments[parameter_name] = _parse_xml_parameter_value(raw_value)
    return name, arguments


def _parse_xml_parameter_value(value: str) -> Any:
    """Decode typed XML-like values while leaving ordinary prompt text intact."""
    if not value:
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            decoded = parser(value)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if decoded is None or isinstance(decoded, (str, int, float, bool, list, dict)):
            return decoded
    return value


def _parse_tool_arguments(value: str) -> dict[str, Any] | None:
    for candidate in (
        value,
        re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', value),
    ):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                decoded = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(decoded, dict):
            return decoded
    return None


class SSEToolNameNormalizer:
    def __init__(
        self,
        registered_names: Iterable[str],
        namespace_tools: dict[str, set[str]] | None = None,
        client_tool_mappings: dict[str, dict[str, str]] | None = None,
    ):
        self.registered_names = set(registered_names)
        self.namespace_tools = namespace_tools or {}
        self.client_tool_mappings = client_tool_mappings or {}
        self.buffer = b""
        self.remapped: list[dict[str, str]] = []
        self.client_tools_adapted = 0

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            trailing = self._transform_line(self.buffer) if self.buffer else b""
            self.buffer = b""
            return trailing
        self.buffer += chunk
        output: list[bytes] = []
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                break
            line = self.buffer[: newline + 1]
            self.buffer = self.buffer[newline + 1 :]
            output.append(self._transform_line(line))
        return b"".join(output)

    def _transform_line(self, line: bytes) -> bytes:
        newline = b""
        body = line
        if body.endswith(b"\n"):
            newline = b"\n"
            body = body[:-1]
            if body.endswith(b"\r"):
                body = body[:-1]
                newline = b"\r\n"
        if not body.startswith(b"data:"):
            return line
        prefix, raw = body.split(b":", 1)
        leading = raw[: len(raw) - len(raw.lstrip())]
        data = raw.strip()
        if not data or data == b"[DONE]":
            return line
        try:
            decoded = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return line
        transformed, remapped = normalize_response_tool_names(
            decoded, self.registered_names, self.namespace_tools
        )
        transformed, adapted = normalize_client_tool_calls(
            transformed, self.client_tool_mappings
        )
        self.client_tools_adapted += adapted
        self.remapped.extend(remapped)
        if not remapped and not adapted:
            return line
        encoded = json.dumps(
            transformed, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return prefix + b":" + leading + encoded + newline


class SSELocalOutputSanitizer:
    """Filter private local reasoning items and repair following output indexes."""

    def __init__(self) -> None:
        self.buffer = b""
        self.dropped_output_indexes: set[int] = set()
        self.removed = 0

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            trailing = self._transform_line(self.buffer) if self.buffer else b""
            self.buffer = b""
            return trailing
        self.buffer += chunk
        output: list[bytes] = []
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                break
            line = self.buffer[: newline + 1]
            self.buffer = self.buffer[newline + 1 :]
            output.append(self._transform_line(line))
        return b"".join(output)

    def _transform_line(self, line: bytes) -> bytes:
        newline = b""
        body = line
        if body.endswith(b"\n"):
            newline = b"\n"
            body = body[:-1]
            if body.endswith(b"\r"):
                body = body[:-1]
                newline = b"\r\n"
        if not body.startswith(b"data:"):
            return line
        prefix, raw = body.split(b":", 1)
        leading = raw[: len(raw) - len(raw.lstrip())]
        data = raw.strip()
        if not data or data == b"[DONE]":
            return line
        try:
            event = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return line
        if not isinstance(event, dict):
            return line

        output_index = event.get("output_index")
        item = event.get("item")
        event_type = event.get("type")
        if isinstance(item, dict) and item.get("type") == "reasoning":
            if isinstance(output_index, int):
                self.dropped_output_indexes.add(output_index)
            self.removed += 1
            return b""
        if (
            isinstance(output_index, int)
            and output_index in self.dropped_output_indexes
        ) or (
            isinstance(event_type, str) and event_type.startswith("response.reasoning")
        ):
            return b""
        adjusted = False
        if isinstance(output_index, int):
            offset = sum(
                1 for dropped in self.dropped_output_indexes if dropped < output_index
            )
            event["output_index"] = output_index - offset
            # A bool output_index collapses to 0/1 through the subtraction, so
            # the event must still be re-serialized when no index was dropped.
            adjusted = offset > 0 or isinstance(output_index, bool)
        event, nested_removed = sanitize_local_response_items(event, copy=False)
        self.removed += nested_removed
        if not adjusted and nested_removed == 0:
            return line
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return prefix + b":" + leading + encoded + newline


def load_pi_selection(
    provider: str,
    model: str,
    *,
    models_path: str | Path | None = None,
) -> InterceptorSelection:
    path = _pi_models_path(models_path)
    raw = _read_pi_catalog(path)
    providers = raw.get("providers")
    if not isinstance(providers, dict) or provider not in providers:
        raise ValueError(f"Pi provider {provider!r} is not configured")
    definition = providers[provider]
    if not isinstance(definition, dict):
        raise ValueError(f"Pi provider {provider!r} has an invalid definition")
    configured_models = {
        item.get("id")
        for item in definition.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model not in configured_models:
        raise ValueError(
            f"model {model!r} is not configured for Pi provider {provider!r}"
        )
    base_url = definition.get("baseUrl")
    _validate_private_base_url(base_url)
    api_key = definition.get("apiKey")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError(f"Pi provider {provider!r} has an invalid apiKey")
    return InterceptorSelection(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key or None,
        auth_header=bool(definition.get("authHeader", True)),
    )


def load_opencode_selection(
    provider: str,
    model: str,
    *,
    config_path: str | Path | None = None,
) -> InterceptorSelection:
    path = _opencode_config_path(config_path)
    raw = _read_opencode_catalog(path)
    providers = raw.get("provider")
    if not isinstance(providers, dict) or provider not in providers:
        raise ValueError(f"OpenCode provider {provider!r} is not configured")
    definition = providers[provider]
    if not isinstance(definition, dict):
        raise ValueError(f"OpenCode provider {provider!r} has an invalid definition")
    configured_models = definition.get("models")
    if not isinstance(configured_models, dict) or model not in configured_models:
        raise ValueError(
            f"model {model!r} is not configured for OpenCode provider {provider!r}"
        )
    options = definition.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"OpenCode provider {provider!r} has no options")
    base_url = options.get("baseURL") or options.get("baseUrl")
    _validate_private_base_url(base_url)
    api_key = options.get("apiKey")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError(f"OpenCode provider {provider!r} has an invalid apiKey")
    return InterceptorSelection(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key or None,
        auth_header=bool(options.get("authHeader", True)),
    )


def _opencode_config_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()
    configured = os.environ.get("OPENCODE_CONFIG")
    if configured:
        return Path(configured).expanduser()
    candidates = (
        Path.home() / ".config" / "opencode" / "opencode.json",
        Path.home() / ".opencode" / "opencode.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _read_opencode_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"OpenCode config not found: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(
            f"OpenCode config may contain credentials and must be mode 0600: {path}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"OpenCode config must contain a JSON object: {path}")
    return raw


def build_interceptor_plan(
    *,
    project: str,
    provider: str,
    model: str,
    listen_port: int = 8877,
    models_path: str | Path | None = None,
    plugin_root: str | Path | None = None,
) -> dict[str, Any]:
    project_path = Path(project).expanduser().resolve()
    if not project_path.is_dir():
        raise FileNotFoundError(f"project directory does not exist: {project_path}")
    if not isinstance(listen_port, int) or not 1024 <= listen_port <= 65535:
        raise ValueError("listen_port must be 1024..65535")
    selection = load_pi_selection(provider, model, models_path=models_path)
    root = (
        Path(plugin_root).resolve()
        if plugin_root is not None
        else Path(__file__).resolve().parents[2]
    )
    launcher = root / "scripts" / "codex_local_interceptor.py"
    if not launcher.is_file():
        raise FileNotFoundError(f"interceptor launcher is missing: {launcher}")
    config_path = (
        Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    )
    return {
        "mode": "transparent_process_proxy",
        "provider": selection.provider,
        "model": selection.model,
        "upstream_base_url": selection.base_url,
        "project": str(project_path),
        "listen_host": "127.0.0.1",
        "listen_port": listen_port,
        "codex_config_mutation": False,
        "codex_config_path": str(config_path),
        "codex_config_sha256": _sha256_file(config_path),
        "preserves_provider_identity": True,
        "passes_through_non_inference_requests": True,
        "intercepted_hosts": sorted(INTERCEPT_HOSTS),
        "intercepted_paths": sorted(RESPONSES_PATHS),
        "credentials_returned": False,
        "requirements": {
            "mitmdump": shutil.which("mitmdump"),
            "codex": shutil.which("codex"),
            "chatgpt_app": _find_chatgpt_app(),
        },
        "commands": {
            "doctor": [str(launcher), "doctor"],
            "status": [str(launcher), "status"],
            "cli_smoke": [
                str(launcher),
                "exec",
                "--provider",
                provider,
                "--model",
                model,
                "--listen-port",
                str(listen_port),
                "--project",
                str(project_path),
                "--",
                "Reply exactly LOCAL_INTERCEPTOR_OK",
            ],
            "app": [
                str(launcher),
                "app",
                "--provider",
                provider,
                "--model",
                model,
                "--listen-port",
                str(listen_port),
                "--project",
                str(project_path),
            ],
        },
        "app_restart_required": True,
        "warning": (
            "A running ChatGPT/Codex app cannot inherit a new process-scoped proxy. "
            "Quit it normally, then launch transparent mode; do not edit Codex config."
        ),
    }


def interceptor_status(*, runtime_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a credential-free receipt for the latest interceptor session."""
    runtime = (
        Path(runtime_dir).expanduser().resolve()
        if runtime_dir is not None
        else DEFAULT_INTERCEPTOR_RUNTIME_DIR
    )
    session_path = runtime / "session.json"
    status_path = runtime / "status.jsonl"
    if not session_path.is_file():
        return {
            "session_present": False,
            "runtime_dir": str(runtime),
            "message": "No interceptor session receipt exists yet.",
            "credentials_returned": False,
        }
    _require_private_receipt(runtime, session_path)
    if session_path.stat().st_size > 65_536:
        raise ValueError(
            f"interceptor session receipt is unexpectedly large: {session_path}"
        )
    raw = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("session_id"), str):
        raise ValueError(f"invalid interceptor session receipt: {session_path}")
    session_id = raw["session_id"]
    session = {key: raw[key] for key in SAFE_SESSION_FIELDS if key in raw}
    event_counts: dict[str, int] = {}
    recent_events: list[dict[str, Any]] = []
    registered_tools: set[str] = set()
    registered_tool_types: set[str] = set()
    feature_paths: dict[str, set[str]] = {
        feature: set() for feature in FEATURE_PATH_PREFIXES
    }
    if status_path.is_file():
        _require_private_receipt(runtime, status_path)
        if status_path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(
                f"interceptor status log is unexpectedly large: {status_path}"
            )
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(line) > 131_072:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("session_id") != session_id:
                    continue
                name = event.get("event")
                if isinstance(name, str):
                    event_counts[name] = event_counts.get(name, 0) + 1
                names = event.get("tool_names")
                advertised_names = event.get("advertised_tool_names")
                for candidate in (names, advertised_names):
                    if isinstance(candidate, list):
                        registered_tools.update(
                            item for item in candidate if isinstance(item, str) and item
                        )
                advertised_types = event.get("advertised_tool_types")
                if isinstance(advertised_types, list):
                    registered_tool_types.update(
                        item
                        for item in advertised_types
                        if isinstance(item, str) and item
                    )
                if (
                    name == "response_passthrough"
                    and isinstance(event.get("status"), int)
                    and 200 <= event["status"] < 400
                    and isinstance(event.get("original_path"), str)
                ):
                    path = event["original_path"].split("?", 1)[0]
                    for feature, prefixes in FEATURE_PATH_PREFIXES.items():
                        if any(
                            path == prefix or path.startswith(prefix + "/")
                            for prefix in prefixes
                        ):
                            feature_paths[feature].add(path)
                safe = {key: event[key] for key in SAFE_EVENT_FIELDS if key in event}
                if isinstance(safe.get("original_path"), str):
                    safe["original_path"] = safe["original_path"].split("?", 1)[0]
                recent_events.append(safe)
                if len(recent_events) > 20:
                    recent_events.pop(0)

    config_path = Path(
        raw.get(
            "codex_config_path",
            Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml",
        )
    ).expanduser()
    before = raw.get("codex_config_sha256_before")
    current = _sha256_file(config_path)
    config_unchanged = before == current if before is not None else None
    local_inference = event_counts.get("inference_routed", 0) > 0
    local_response = (
        event_counts.get("inference_completed", 0)
        + event_counts.get("inference_stream_closed", 0)
        > 0
    )
    websocket_fallback = event_counts.get("websocket_forced_to_http", 0) > 0
    websocket_bridge = event_counts.get("websocket_bridge_started", 0) > 0
    websocket_ready = websocket_bridge or websocket_fallback
    harness_advertised = any("harness" in name.lower() for name in registered_tools)
    computer_use_advertised = any(
        token in name.lower()
        for name in registered_tools
        for token in ("computer", "screenshot", "mouse", "keyboard")
    ) or any(
        token in tool_type.lower()
        for tool_type in registered_tool_types
        for token in ("computer", "computer_use")
    )
    raw_attestation = raw.get("desktop_attestation")
    attestation = {
        key: raw_attestation[key]
        for key in DESKTOP_ATTESTATION_FIELDS
        if isinstance(raw_attestation, dict)
        and isinstance(raw_attestation.get(key), bool)
    }
    projects_visible = attestation.get("projects_sidebar_visible") is True
    automations_visible = attestation.get("automations_visible") is True
    harness_used = attestation.get("agent_harness_used") is True
    computer_use_used = attestation.get("computer_use_used") is True
    local_response_visible = attestation.get("local_model_response_visible") is True
    automated_gates = {
        "local_inference_routed": local_inference,
        "local_inference_response_seen": local_response,
        # Retain the historical key for status consumers; the native bridge is
        # now the preferred successful containment mode.
        "websocket_contained_and_http_fallback_used": websocket_ready,
        "codex_config_unchanged": config_unchanged,
        "provider_identity_preserved_by_design": bool(
            raw.get("preserves_provider_identity")
        ),
        "harness_tool_advertised": harness_advertised,
        "computer_use_tool_advertised": computer_use_advertised,
    }
    complete = all(value is True for value in automated_gates.values()) and all(
        (
            projects_visible,
            automations_visible,
            harness_used,
            computer_use_used,
            local_response_visible,
        )
    )
    phase = raw.get("phase")
    return {
        "session_present": True,
        "runtime_dir": str(runtime),
        "session": session,
        "proxy_running": raw.get("proxy_stopped_at") is None
        and _pid_is_running(raw.get("proxy_pid")),
        "app_running": isinstance(phase, str)
        and phase.startswith("app_running")
        and _pid_is_running(raw.get("app_pid")),
        "codex_config_sha256_current": current,
        "codex_config_unchanged": config_unchanged,
        "event_counts": event_counts,
        "registered_tools": sorted(registered_tools),
        "registered_tool_types": sorted(registered_tool_types),
        "feature_backend_passthrough": {
            feature: {
                "seen": bool(paths),
                "successful_paths": sorted(paths),
            }
            for feature, paths in feature_paths.items()
        },
        "recent_events": recent_events,
        "desktop_acceptance": {
            **automated_gates,
            "websocket_transport": (
                "native_local_bridge"
                if websocket_bridge
                else "http_fallback"
                if websocket_fallback
                else "not_seen"
            ),
            "visual_attestation": attestation,
            "projects_sidebar": "verified_visible"
            if projects_visible
            else "pending_live_app_check",
            "automations": "verified_visible"
            if automations_visible
            else "pending_live_app_check",
            "agent_harness": "verified_used"
            if harness_used
            else "pending_live_app_check",
            "computer_use": "verified_used"
            if computer_use_used
            else "pending_live_app_check",
            "local_model_response": "verified_visible"
            if local_response_visible
            else "pending_live_app_check",
            "complete": complete,
        },
        "credentials_returned": False,
    }


def codex_thread_url(project: str | Path) -> str:
    path = str(Path(project).expanduser().resolve())
    return "codex://threads/new?path=" + quote(path, safe="")


def _tool_name_tokens(name: str) -> tuple[str, ...]:
    return tuple(part for part in TOOL_NAME_SEPARATORS.split(name) if part)


def _normalize_namespace_call(
    emitted: str, namespace_tools: dict[str, set[str]]
) -> tuple[str, str] | None:
    emitted_tokens = _tool_name_tokens(emitted)
    matches: set[tuple[str, str]] = set()
    for namespace, names in namespace_tools.items():
        for name in names:
            qualified_tokens = _tool_name_tokens(namespace) + _tool_name_tokens(name)
            if emitted_tokens == qualified_tokens or emitted == name:
                matches.add((namespace, name))
    return next(iter(matches)) if len(matches) == 1 else None


def pi_model_context_window(
    model: str,
    *,
    server: str | None = None,
    models_path: str | Path | None = None,
) -> int | None:
    """The context window Pi records for a private model, if it records one.

    Codex auto-compacts against the window advertised for the slot it believes
    it is using, so a local model with a larger window gets compacted long
    before it is full. Pi's catalogue is the only place that knows the real
    figure per server and model. An id can appear under several providers; an
    exact server match wins, and otherwise the smallest window is used because
    advertising more than the served model supports would overflow it.
    """
    if not isinstance(model, str) or not model:
        return None
    try:
        catalog = _read_pi_catalog(_pi_models_path(models_path))
    except (OSError, ValueError, PermissionError, json.JSONDecodeError):
        return None
    providers = catalog.get("providers")
    if not isinstance(providers, dict):
        return None
    matches: list[int] = []
    for provider, definition in providers.items():
        if not isinstance(definition, dict):
            continue
        for item in definition.get("models", []):
            if not isinstance(item, dict) or item.get("id") != model:
                continue
            window = item.get("contextWindow")
            if not isinstance(window, int) or window <= 0:
                continue
            if server is not None and provider == server:
                return window
            matches.append(window)
    return min(matches) if matches else None


def _pi_models_path(models_path: str | Path | None) -> Path:
    return (
        Path(models_path).expanduser()
        if models_path is not None
        else Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
        / "models.json"
    )


def _read_pi_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pi model catalog not found: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(
            f"Pi model catalog may contain credentials and must be mode 0600: {path}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Pi model catalog must contain a JSON object: {path}")
    return raw


def _display_token(value: Any, limit: int, *, fallback: str) -> str:
    if not isinstance(value, str) or not value:
        return fallback
    cleaned = "".join(
        character if character.isprintable() else "?" for character in value
    )
    return cleaned[:limit]


def _safe_display_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "/"
    without_query = value.split("?", 1)[0]
    cleaned = "".join(
        character if character.isprintable() else "?" for character in without_query
    )
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    return cleaned[:512]


def _validate_private_base_url(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("Pi provider baseUrl must be a string")
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Pi provider baseUrl must be a credential-free HTTP(S) URL")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        return
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve Pi provider host {host!r}") from exc
    if not addresses:
        raise ValueError(f"Pi provider host {host!r} resolved to no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not (ip.is_private or ip.is_loopback or ip.is_link_local):
            raise ValueError(
                f"transparent local mode rejects non-private upstream address {address}"
            )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pid_is_running(value: Any) -> bool:
    if not isinstance(value, int) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _require_private_receipt(runtime: Path, path: Path) -> None:
    if runtime.is_symlink() or path.is_symlink():
        raise PermissionError("interceptor receipts must not be symbolic links")
    for candidate in (runtime, path):
        stat = candidate.stat()
        if hasattr(os, "getuid") and stat.st_uid != os.getuid():
            raise PermissionError(
                f"interceptor receipt is owned by another user: {candidate}"
            )
        if stat.st_mode & 0o077:
            raise PermissionError(
                f"interceptor receipts must be owner-only: {candidate}"
            )


def _find_chatgpt_app() -> str | None:
    for candidate in (
        Path("/Applications/ChatGPT.app"),
        Path("/Applications/Codex.app"),
        Path.home() / "Applications" / "ChatGPT.app",
        Path.home() / "Applications" / "Codex.app",
    ):
        if candidate.is_dir():
            return str(candidate)
    return None
