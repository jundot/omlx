# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import tomllib
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix

CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def write_codex_config(config_path: Path, ctx: IntegrationContext) -> None:
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

    # Keys to override at the top level. String literals go through
    # json.dumps: its escaping matches TOML basic strings, so a model
    # alias containing quotes or newlines (server-side aliases are free
    # text) cannot inject additional TOML keys — e.g. a crafted alias
    # used to be able to write sandbox_mode/approval_policy here.
    top_level_overrides = {
        "model": json.dumps(ctx.model or "select-a-model"),
        "model_provider": json.dumps("omlx"),
    }

    # If it is a reasoning model, add reasoning effort
    is_reasoning = (
        bool(ctx.reasoning)
        if ctx.reasoning is not None
        else bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))
    )
    if is_reasoning:
        top_level_overrides["model_reasoning_effort"] = '"high"'

    # Keys managed by oMLX that should be removed when not applicable
    managed_keys = {"model_reasoning_effort"} - set(top_level_overrides.keys())

    seen_keys = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_any_section = True
            # TOML allows the key to be quoted: [model_providers."omlx"]
            # declares the same table; a bare comparison would leave the
            # old section in place and the re-run would emit a duplicate
            # table, corrupting the config.
            in_omlx_section = stripped in (
                "[model_providers.omlx]",
                '[model_providers."omlx"]',
                "[model_providers.'omlx']",
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
    new_lines.append(f"name = {json.dumps('oMLX')}")
    new_lines.append(f"base_url = {json.dumps(ctx.openai_base_url)}")
    new_lines.append(f"env_key = {json.dumps('OMLX_API_KEY')}")

    candidate = "\n".join(new_lines) + "\n"
    # Fail closed: if the assembled config no longer parses as TOML (or
    # the escaped values smuggled in NEW keys), keep the user's file
    # intact. Keys the user already had are of course preserved.

    try:
        parsed = tomllib.loads(candidate)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(
            f"refusing to write a config that does not parse as TOML: {e}"
        ) from e
    try:
        previous = tomllib.loads(existing_content) if existing_content else {}
    except tomllib.TOMLDecodeError:
        previous = {}
    injected_keys = set(parsed) - set(previous) - {
        "model",
        "model_provider",
        "model_reasoning_effort",
        "model_providers",
    }
    if injected_keys:
        raise ValueError(
            "refusing to write a config with unexpected top-level keys "
            f"(model alias may be malformed): {sorted(injected_keys)}"
        )

    config_path.write_text(candidate, encoding="utf-8")
    print(f"Config updated: {config_path}")


def codex_config_args(ctx: IntegrationContext) -> list[str]:
    """Build process-scoped Codex config overrides for an oMLX launch."""
    overrides: list[tuple[str, str]] = [
        ("model_provider", json.dumps("omlx")),
        ("model_providers.omlx.name", json.dumps("oMLX")),
        ("model_providers.omlx.base_url", json.dumps(ctx.openai_base_url)),
        ("model_providers.omlx.env_key", json.dumps("OMLX_API_KEY")),
    ]
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
        # Launch-time arguments carry the oMLX settings. Keeping this a no-op
        # ensures normal Codex sessions continue to use the user's config.
        return None

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        args = ["codex", *codex_config_args(ctx)]
        if ctx.model:
            args.extend(["-m", ctx.model])
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
