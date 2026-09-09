# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix

CODEX_PROFILE_NAME = "omlx"
CODEX_PROFILE_CONFIG_NAME = f"{CODEX_PROFILE_NAME}.config.toml"
CODEX_MODEL_CATALOG_NAME = "omlx-models.json"

_CODEX_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent. Follow the user's instructions carefully. "
    "Inspect relevant files before editing, make focused changes, preserve "
    "unrelated work, and verify changes with relevant tests."
)


def codex_home_path() -> Path:
    """Return the active Codex configuration directory."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home)
    return Path.home() / ".codex"


def codex_model_catalog_path() -> Path:
    """Return the dedicated oMLX catalog path in the active Codex home."""
    return codex_home_path() / CODEX_MODEL_CATALOG_NAME


def codex_profile_path() -> Path:
    """Return the standalone oMLX profile path in the active Codex home."""
    return codex_home_path() / CODEX_PROFILE_CONFIG_NAME


def _positive_int(*values: object, default: int) -> int:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def _codex_catalog_model(model: dict, priority: int, ctx: IntegrationContext) -> dict:
    model_id = str(model.get("id") or ctx.model)
    context_window = _positive_int(
        model.get("max_context_window"),
        model.get("model_context_length"),
        ctx.context_window if model_id == ctx.model else None,
        default=32_768,
    )
    modalities = ["text", "image"] if model.get("model_type") == "vlm" else ["text"]
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": "Local model served by oMLX",
        "default_reasoning_level": None,
        "supported_reasoning_levels": [],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": _CODEX_BASE_INSTRUCTIONS,
        "model_messages": None,
        # Keep both legacy and current names so catalogs work across recent
        # Codex releases. Unknown fields are ignored by a given release.
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": False,
        "supports_image_detail_original": False,
        "context_window": context_window,
        "max_context_window": context_window,
        "auto_compact_token_limit": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": modalities,
    }


def write_codex_model_catalog(catalog_path: Path, ctx: IntegrationContext) -> Path:
    """Write Codex metadata for every model exposed by the oMLX server."""
    models = list(ctx.available_models)
    if ctx.model and not any(model.get("id") == ctx.model for model in models):
        models.append(
            {
                "id": ctx.model,
                "model_type": ctx.model_type,
                "max_context_window": ctx.context_window,
            }
        )
    if not models:
        models.append({"id": ctx.model or "select-a-model"})

    catalog = {
        "models": [
            _codex_catalog_model(model, priority, ctx)
            for priority, model in enumerate(models)
        ]
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return catalog_path


def write_codex_config(
    config_path: Path,
    ctx: IntegrationContext,
    catalog_path: Path | None = None,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = ""
    if config_path.exists():
        # Create backup
        timestamp = int(time.time())
        backup = config_path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(config_path, backup)
            existing_content = config_path.read_text(encoding="utf-8")
            print(f"Backup: {backup}")
        except OSError as e:
            print(f"Warning: could not create backup or read config: {e}")

    # Parse existing config lines to preserve other settings
    lines = existing_content.splitlines()
    new_lines = []
    in_any_section = False
    in_omlx_section = False

    # Keys to override at the top level
    top_level_overrides = {
        "model": f'"{ctx.model or "select-a-model"}"',
        "model_provider": '"omlx"',
    }
    if catalog_path is not None:
        top_level_overrides["model_catalog_json"] = json.dumps(str(catalog_path))

    # If it is a reasoning model, add reasoning effort
    is_reasoning = (
        bool(ctx.reasoning)
        if ctx.reasoning is not None
        else bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))
    )
    if is_reasoning:
        top_level_overrides["model_reasoning_effort"] = '"high"'

    # Keys managed by oMLX that should be removed when not applicable
    managed_keys = {"model_reasoning_effort", "model_catalog_json"} - set(
        top_level_overrides.keys()
    )

    seen_keys = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_any_section = True
            in_omlx_section = stripped == "[model_providers.omlx]"

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
    new_lines.append(f'base_url = "{ctx.openai_base_url}"')
    new_lines.append('env_key = "OMLX_API_KEY"')

    config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
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
    ]
    if catalog_path is not None:
        overrides.append(("model_catalog_json", json.dumps(str(catalog_path))))
    if ctx.context_window is not None and ctx.context_window > 0:
        overrides.append(("model_context_window", str(ctx.context_window)))

    is_reasoning = (
        bool(ctx.reasoning)
        if ctx.reasoning is not None
        else bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))
    )
    if is_reasoning:
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
        return (
            f"{get_cli_command_prefix()} "
            f"launch codex --model {ctx.model or 'select-a-model'}"
        )

    def configure(self, ctx: IntegrationContext) -> None:
        # The dedicated catalog does not alter the user's normal Codex config;
        # launch-time arguments scope it to this oMLX process.
        write_codex_model_catalog(codex_model_catalog_path(), ctx)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        args = ["codex", *codex_config_args(ctx, codex_model_catalog_path())]
        if ctx.model:
            args.extend(["-m", ctx.model])
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
