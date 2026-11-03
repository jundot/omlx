# SPDX-License-Identifier: Apache-2.0
"""GitHub Copilot CLI integration."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from omlx.integrations.base import (
    Integration,
    IntegrationContext,
    chat_models,
)
from omlx.utils.install import get_cli_command_prefix

PROVIDER_NAME = "omlx"


def _ignore_interrupt(signum, frame):
    pass


def copilot_providers_config_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the CLI providers file (``COPILOT_PROVIDERS_CONFIG`` or default)."""
    resolved: Mapping[str, str] = os.environ if env is None else env
    override = resolved.get("COPILOT_PROVIDERS_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".copilot" / "providers.json"


def _clobber_omlx_entries(entries: list, key: str, replacement: list) -> list:
    """Replace omlx-owned entries at their first slot, preserving foreign entries."""
    result: list = []
    inserted = False
    for entry in entries:
        if isinstance(entry, dict) and entry.get(key) == PROVIDER_NAME:
            if not inserted:
                result.extend(replacement)
                inserted = True
            continue
        result.append(entry)
    if not inserted:
        result.extend(replacement)
    return result


def copilot_provider_config(ctx: IntegrationContext) -> dict:
    """Build the omlx provider/model entries for the CLI providers registry."""
    selected_id = (
        ctx.models_status_map.get(ctx.model, {}).get("model_alias") or ctx.model
    )
    entries = []
    for model_id, status in chat_models(ctx).items():
        context_window = status.get("max_context_window")
        model_type = status.get("model_type")
        reasoning = status.get("enable_thinking")
        max_tokens = status.get("max_tokens")
        if model_id == selected_id:
            if ctx.context_window is not None:
                context_window = ctx.context_window
            model_type = model_type or ctx.model_type
            if ctx.reasoning is not None:
                reasoning = ctx.reasoning
            if ctx.max_tokens is not None:
                max_tokens = ctx.max_tokens
        if not isinstance(context_window, int) or not 0 < context_window <= 4294967295:
            context_window = 32768
        supports: dict[str, bool | list[str]] = {"vision": model_type == "vlm"}
        entry: dict[str, object] = {
            "id": model_id,
            "provider": PROVIDER_NAME,
            "maxPromptTokens": context_window,
            "maxContextWindowTokens": context_window,
            "capabilities": {"supports": supports},
        }
        if isinstance(max_tokens, int) and 0 < max_tokens <= 4294967295:
            entry["maxOutputTokens"] = max_tokens
        efforts = [
            effort
            for effort in status.get("reasoning_levels") or ()
            if effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        ]
        if efforts and reasoning is not False:
            supports["reasoning_effort"] = efforts
        entries.append(entry)

    return {
        "providers": [
            {
                "name": PROVIDER_NAME,
                "type": "openai",
                "baseUrl": ctx.openai_base_url,
                "wireApi": "responses",
                "bearerToken": ctx.auth_token,
            }
        ],
        "models": entries,
    }


class CopilotIntegration(Integration):
    """Copilot integration launching via the persistent provider registry."""

    def __init__(self):
        super().__init__(
            name="copilot",
            display_name="Copilot CLI",
            type="env_var",
            install_check="copilot",
            install_hint="npm install -g @github/copilot",
            requires_model_selection=False,
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        command = f"{get_cli_command_prefix()} launch copilot"
        return f"{command} --model {ctx.model}" if ctx.model else command

    def configure(self, ctx: IntegrationContext) -> None:
        self._write_registry(copilot_provider_config(ctx))

    def _write_registry(self, registry: dict) -> None:
        """Refresh omlx-owned entries, preserving other providers and settings."""
        path = copilot_providers_config_path().resolve()
        existing_text: str | None = None
        existing: object = None
        if path.is_file():
            path.chmod(0o600)
            try:
                existing_text = path.read_text(encoding="utf-8")
                existing = json.loads(existing_text)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                print(f"Warning: could not parse {path}: {e}")
        if existing is not None and not isinstance(existing, dict):
            print(f"Warning: {path} does not contain a JSON object; replacing it.")
        doc = existing if isinstance(existing, dict) else {}
        providers = doc.get("providers")
        models = doc.get("models")
        doc["providers"] = _clobber_omlx_entries(
            providers if isinstance(providers, list) else [],
            "name",
            registry["providers"],
        )
        doc["models"] = _clobber_omlx_entries(
            models if isinstance(models, list) else [],
            "provider",
            registry["models"],
        )
        payload = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        if existing_text != payload:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with (
                    path.open("rb") as source,
                    tempfile.NamedTemporaryFile(
                        prefix=f"{path.stem}.",
                        suffix=".bak",
                        dir=path.parent,
                        delete=False,
                    ) as backup,
                ):
                    shutil.copyfileobj(source, backup)
                print(f"Backup: {backup.name}")
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix=f"{path.stem}.",
                    suffix=".tmp",
                    dir=path.parent,
                    delete=False,
                ) as file:
                    temporary_path = Path(file.name)
                    os.fchmod(file.fileno(), 0o600)
                    file.write(payload)
                temporary_path.replace(path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            print(f"Copilot providers config: {path}")

    def launch(self, ctx: IntegrationContext) -> None:
        registry = copilot_provider_config(ctx)
        model_ids = [entry["id"] for entry in registry["models"]]
        if not model_ids:
            raise SystemExit("oMLX server reported no chat models. Load a model first.")
        model = ctx.models_status_map.get(ctx.model, {}).get("model_alias") or ctx.model
        if model and model not in model_ids:
            print(
                f"Warning: {ctx.model} is not a chat model on the server; ignoring it."
            )
        if model not in model_ids:
            default = (
                ctx.models_status_map.get(ctx.default_model, {}).get("model_alias")
                or ctx.default_model
            )
            model = default if default in model_ids else model_ids[0]
        self._write_registry(registry)
        env = self._scrubbed_env()
        config_path = copilot_providers_config_path(env)
        for key in list(env):
            if key.startswith("COPILOT_PROVIDER_") or key == "COPILOT_MODEL":
                del env[key]
        env["COPILOT_PROVIDERS_CONFIG"] = str(config_path)

        args = ["copilot", "--model", f"{PROVIDER_NAME}/{model}"]
        args.extend(ctx.extra_args)

        # A caught handler resets on exec, so Copilot receives SIGINT while
        # the launcher waits without treating Ctrl+C as its own interrupt.
        previous_handler = signal.signal(signal.SIGINT, _ignore_interrupt)
        try:
            result = subprocess.run(args, env=env, check=False)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        if result.returncode:
            raise SystemExit(
                128 - result.returncode if result.returncode < 0 else result.returncode
            )
