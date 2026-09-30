# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import tomllib
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext, IntegrationModel
from omlx.utils.install import get_cli_command_prefix

CODEX_CONFIG_PATH = (
    Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
)

_CODEX_INSTRUCTIONS = (
    "You are a coding agent working in the user's local workspace. "
    "Follow the user's instructions and applicable AGENTS.md files. "
    "Inspect the workspace before making focused, maintainable changes. "
    "Preserve unrelated work, verify changes with relevant tests, and report limitations."
)


def _is_reasoning_model(model_id: str, reasoning: bool | None) -> bool:
    if reasoning is not None:
        return reasoning
    return bool(re.search(r"\b(thinking|o1|o3|r1)\b", model_id.lower()))


def codex_model_catalog(ctx: IntegrationContext) -> dict:
    """Build Codex's ModelInfo catalog, not an OpenAI /v1/models response.

    Keep capabilities conservative: thinking does not imply adjustable effort,
    reasoning-summary generation, verbosity control, or server-side search.
    """
    models = {
        model.id: model
        for model in ctx.models
        if model.model_type in (None, "llm", "vlm")
    }
    if ctx.model and ctx.model not in models and ctx.model_type in (None, "llm", "vlm"):
        models[ctx.model] = IntegrationModel(
            id=ctx.model,
            context_window=ctx.context_window,
            model_type=ctx.model_type,
            reasoning=ctx.reasoning,
            reasoning_effort_options=ctx.reasoning_effort_options,
            reasoning_effort_default=ctx.reasoning_effort_default,
        )
    ordered = sorted(models.values(), key=lambda model: model.id != ctx.model)
    entries = []
    for priority, model in enumerate(ordered):
        reasoning = _is_reasoning_model(model.id, model.reasoning)
        # Codex accepts a fixed effort enum. Do not invent levels from a
        # thinking boolean, or advertise custom template values it cannot send.
        efforts = (
            [
                effort
                for effort in (
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "ultra",
                )
                if effort in model.reasoning_effort_options
            ]
            if model.reasoning is not False
            else []
        )
        default_effort = (
            "medium" if "medium" in efforts else model.reasoning_effort_default
        )
        if efforts:
            if default_effort not in efforts:
                default_effort = efforts[0]
            reasoning_levels = [
                {"effort": effort, "description": f"Use {effort} reasoning effort"}
                for effort in efforts
            ]
        else:
            default_effort = "high" if reasoning else None
            reasoning_levels = (
                [{"effort": "high", "description": "Use model thinking"}]
                if reasoning
                else []
            )
        # Status can be unavailable for inference-only API keys. Do not let
        # Codex's large GPT context default overstate an unknown local limit.
        context_window = (
            model.context_window
            if model.context_window is not None and model.context_window > 0
            else 32768
        )
        entries.append(
            {
                "slug": model.id,
                "display_name": model.id,
                "description": "Local model served by oMLX",
                "visibility": "list",
                "supported_in_api": True,
                "priority": priority,
                "base_instructions": _CODEX_INSTRUCTIONS,
                "shell_type": "shell_command",
                "context_window": context_window,
                "input_modalities": (
                    ["text", "image"] if model.model_type == "vlm" else ["text"]
                ),
                "default_reasoning_level": default_effort,
                "supported_reasoning_levels": reasoning_levels,
                "supports_reasoning_summaries": False,
                "supports_reasoning_summary_parameter": False,
                "support_verbosity": False,
                "supports_parallel_tool_calls": False,
                "supports_search_tool": False,
                "prefer_websockets": False,
                "experimental_supported_tools": [],
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": min(10000, context_window // 4),
                },
            }
        )
    return {"models": entries}


def write_codex_model_catalog(config_path: Path, ctx: IntegrationContext) -> Path:
    """Refresh an oMLX-owned catalog without replacing the user's catalog.

    Separate endpoints get separate files; the catalog is read by Codex at
    startup, so subsequent refreshes do not mutate an active model selector.
    """
    endpoint_id = hashlib.sha256(ctx.openai_base_url.encode()).hexdigest()[:12]
    catalog_path = config_path.parent / f"omlx-models-{endpoint_id}.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    # Never expose a partially written catalog to simultaneous launches.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=catalog_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(
                json.dumps(codex_model_catalog(ctx), indent=2, ensure_ascii=False)
                + "\n"
            )
            temporary.close()
            temporary_path.replace(catalog_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return catalog_path


def write_codex_config(config_path: Path, ctx: IntegrationContext) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = ""
    if config_path.exists():
        # Create backup
        timestamp = time.time_ns()
        backup = config_path.with_suffix(f".{timestamp}.bak")
        shutil.copy2(config_path, backup)
        existing_content = config_path.read_text(encoding="utf-8")
        print(f"Backup: {backup}")

    # Refuse to rewrite invalid TOML, and back up before changing the catalog.
    tomllib.loads(existing_content)
    catalog_path = write_codex_model_catalog(config_path, ctx)

    # Parse existing config lines to preserve other settings
    lines = existing_content.splitlines()
    new_lines = []
    in_any_section = False
    in_omlx_section = False

    # Model-specific limits and capabilities belong in the catalog so switching
    # models does not retain the launch model's context or thinking settings.
    top_level_overrides = {
        "model": json.dumps(ctx.model or "select-a-model", ensure_ascii=False),
        "model_provider": '"omlx"',
        "model_catalog_json": json.dumps(str(catalog_path), ensure_ascii=False),
    }
    managed_keys = {
        "model_context_window",
        "model_auto_compact_token_limit",
        "model_reasoning_effort",
        "model_reasoning_summary",
        "model_supports_reasoning_summaries",
    }

    seen_keys = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_any_section = True
            in_omlx_section = bool(
                re.fullmatch(
                    r"\[model_providers\.omlx(?:\.[^\]]+)?\]\s*(?:#.*)?", stripped
                )
            )

        # Handle top-level keys
        if not in_any_section and "=" in stripped:
            key = stripped.split("=")[0].strip()
            if key in top_level_overrides:
                new_lines.append(f"{key} = {top_level_overrides[key]}")
                seen_keys.add(key)
                continue
            if key in managed_keys:
                continue

        # Skip old oMLX section
        if in_omlx_section:
            continue

        new_lines.append(line)

    # Add missing top-level keys
    for key, val in top_level_overrides.items():
        if key not in seen_keys:
            new_lines.insert(0, f"{key} = {val}")

    # Append new oMLX provider section
    new_lines.append("\n[model_providers.omlx]")
    new_lines.append('name = "oMLX"')
    new_lines.append(f"base_url = {json.dumps(ctx.openai_base_url)}")
    new_lines.append('env_key = "OMLX_API_KEY"')
    new_lines.append('wire_api = "responses"')
    new_lines.append("requires_openai_auth = false")
    new_lines.append("supports_websockets = false")

    new_content = "\n".join(new_lines) + "\n"
    tomllib.loads(new_content)
    config_path.write_text(new_content, encoding="utf-8")
    print(f"Config updated: {config_path}")


def codex_config_args(
    ctx: IntegrationContext, catalog_path: Path | None = None
) -> list[str]:
    """Build process-scoped Codex config overrides for an oMLX launch."""
    overrides: list[tuple[str, str]] = [
        ("model_provider", json.dumps("omlx")),
        ("model_providers.omlx.name", json.dumps("oMLX")),
        ("model_providers.omlx.base_url", json.dumps(ctx.openai_base_url)),
        ("model_providers.omlx.env_key", json.dumps("OMLX_API_KEY")),
        ("model_providers.omlx.wire_api", json.dumps("responses")),
        ("model_providers.omlx.requires_openai_auth", "false"),
        ("model_providers.omlx.supports_websockets", "false"),
    ]
    if catalog_path is not None:
        overrides.append(("model_catalog_json", json.dumps(str(catalog_path))))
    else:
        # Compatibility for callers configuring only a single model.
        if ctx.context_window is not None and ctx.context_window > 0:
            overrides.append(("model_context_window", str(ctx.context_window)))
        if _is_reasoning_model(ctx.model, ctx.reasoning):
            overrides.append(("model_reasoning_effort", json.dumps("high")))

    return [arg for key, value in overrides for arg in ("-c", f"{key}={value}")]


class CodexIntegration(Integration):
    """Codex integration using process-scoped configuration for oMLX."""

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="env_var",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        command = f"{get_cli_command_prefix()} launch codex"
        return f"{command} --model {ctx.model}" if ctx.model else command

    def configure(self, ctx: IntegrationContext) -> None:
        # Launch-time arguments carry the oMLX settings. Keeping this a no-op
        # ensures normal Codex sessions continue to use the user's config.
        return None

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        catalog_path = write_codex_model_catalog(CODEX_CONFIG_PATH, ctx)
        args = ["codex", *codex_config_args(ctx, catalog_path)]
        if ctx.model:
            args.extend(["-m", ctx.model])
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
