# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

from omlx.integrations.base import Integration, IntegrationContext, chat_model_ids
from omlx.utils.install import get_cli_command_prefix

CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"

# A slug Codex does not know, so it renders its built-in instructions.
_PROBE_MODEL = "omlx-instructions-probe"
_PROBE_TIMEOUT_SECONDS = 10.0
# Older Codex releases reject these catalog entries and do not start.
_MIN_CODEX_VERSION = (0, 148)

FALLBACK_INSTRUCTIONS = """\
You are a coding agent running in the Codex CLI on the user's machine. A local model served by oMLX powers you.

Keep working until the user's request is fully handled, using the tools available to you. Look at the relevant files before you change anything, and do not guess at facts you can check.

AGENTS.md files give instructions for the directory tree that contains them. Follow every AGENTS.md whose tree covers a file you change. A deeper AGENTS.md overrides a higher one, and direct instructions from the user override both.

To edit files, run apply_patch through the shell tool with the patch in a heredoc:

apply_patch <<'EOF'
*** Begin Patch
*** Update File: src/greeting.py
@@ def greet(name):
-    return "hi"
+    return f"Hello, {name}!"
*** End Patch
EOF

In an Update File section, each @@ line names the code the change sits in. Lines starting with a space are unchanged context, "-" removes a line, and "+" adds one. Use "*** Add File: <path>" with every line starting with "+" to create a file, and "*** Delete File: <path>" to remove one. One patch can hold several file sections.

Guidelines:
- Fix the root cause. Leave unrelated bugs and failing tests alone, and mention them instead.
- Keep changes small and in the style of the surrounding code.
- Do not commit, create branches, or add license headers unless the user asks.
- If the project has tests, run the most specific relevant ones first.
- Before a group of tool calls, tell the user in one short sentence what you are about to do.

When you finish, reply briefly (about 10 lines unless the task needs more detail). Refer to files as `path` or `path:line` so the user can open them. Use short "-" bullets only when they help, and do not print ANSI escape codes.
"""


def _provider_overrides(base_url: str) -> list[tuple[str, str]]:
    return [
        ("model_provider", json.dumps("omlx")),
        ("model_providers.omlx.name", json.dumps("oMLX")),
        ("model_providers.omlx.base_url", json.dumps(base_url)),
        ("model_providers.omlx.env_key", json.dumps("OMLX_API_KEY")),
    ]


def _as_cli_args(overrides: list[tuple[str, str]]) -> list[str]:
    return [arg for key, value in overrides for arg in ("-c", f"{key}={value}")]


def capture_codex_instructions(codex_bin: str, env: dict[str, str]) -> str | None:
    """Return the base instructions ``codex_bin`` sends, or None.

    Codex runs once with an empty CODEX_HOME against a loopback listener, so
    the probe needs no login, loads no model, and leaves no user session.
    """
    bodies: list[bytes] = []
    received = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            bodies.append(self.rfile.read(length))
            received.set()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "oMLX instructions probe"}}')

        def log_message(self, format: str, *args) -> None:
            pass

    # HTTPServer resolves the bound host's name, which a broken resolver holds for 35 s.
    server = TCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    overrides = _provider_overrides(f"http://127.0.0.1:{port}/v1") + [
        ("model_providers.omlx.request_max_retries", "0"),
        ("model_providers.omlx.stream_max_retries", "0"),
    ]
    try:
        with tempfile.TemporaryDirectory(
            prefix="omlx-codex-probe-", ignore_cleanup_errors=True
        ) as home:
            try:
                proc = subprocess.Popen(
                    [
                        codex_bin,
                        "exec",
                        "--skip-git-repo-check",
                        *_as_cli_args(overrides),
                        "-m",
                        _PROBE_MODEL,
                        "probe",
                    ],
                    cwd=home,
                    env={**env, "CODEX_HOME": home, "OMLX_API_KEY": "probe"},
                    # codex exec waits for more prompt text on an open stdin.
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                return None
            deadline = time.monotonic() + _PROBE_TIMEOUT_SECONDS
            while not received.wait(0.05):
                if proc.poll() is not None or time.monotonic() > deadline:
                    break
            with contextlib.suppress(subprocess.TimeoutExpired):
                # Codex exits on the error answer.
                proc.wait(timeout=5 if received.is_set() else 0)
            # The npm codex shim does not forward SIGKILL, so kill the whole
            # group before the temporary home is removed.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    finally:
        server.shutdown()
        server.server_close()

    if not bodies:
        return None
    try:
        instructions = json.loads(bodies[0]).get("instructions")
    except (ValueError, AttributeError):
        return None
    if isinstance(instructions, str) and instructions.strip():
        return instructions
    return None


def check_codex_version(codex_bin: str | None) -> None:
    """Exit with an update hint when ``codex_bin`` cannot read the catalog."""
    if codex_bin is None:
        return
    try:
        version = subprocess.run(
            [codex_bin, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return
    match = re.search(r"(\d+)\.(\d+)\.\d+", version)
    if match and tuple(map(int, match.groups())) < _MIN_CODEX_VERSION:
        print(
            f"{version.strip()} cannot read oMLX's model catalog. "
            "Update Codex to 0.148 or newer."
        )
        sys.exit(1)


def codex_instructions(codex_bin: str | None, env: dict[str, str]) -> str:
    instructions = capture_codex_instructions(codex_bin, env) if codex_bin else None
    if instructions is None:
        print(
            f"Warning: could not read the built-in instructions of "
            f"{codex_bin or 'codex'}; using oMLX's fallback coding instructions."
        )
        return FALLBACK_INSTRUCTIONS
    return instructions


def resolve_codex_models(ctx: IntegrationContext) -> tuple[list[str], str]:
    """Return the catalog model ids and the model Codex starts with."""
    model_ids = chat_model_ids(ctx)
    if not model_ids:
        print("oMLX server reported no chat models. Load a model first.")
        sys.exit(1)

    def served_id(model_id: str) -> str:
        return ctx.models_status_map.get(model_id, {}).get("model_alias") or model_id

    model = served_id(ctx.model)
    if model and model not in model_ids:
        print(f"Warning: {ctx.model} is not a chat model on the server; ignoring it.")
        model = ""
    if not model:
        default = served_id(ctx.default_model)
        model = default if default in model_ids else model_ids[0]
    return model_ids, model


def codex_model_catalog(
    ctx: IntegrationContext, model_ids: list[str], instructions: str
) -> dict:
    """Build ``model_catalog_json`` for the oMLX chat models.

    Fields not read from the server match Codex's metadata for unknown models.
    """
    entries = []
    for priority, model_id in enumerate(model_ids):
        info = ctx.models_status_map.get(model_id, {})
        # "off" and "on" toggle thinking; they are not effort levels.
        efforts = [
            level
            for level in info.get("reasoning_levels") or ()
            if level not in ("off", "on")
        ]
        entry = {
            "slug": model_id,
            "display_name": model_id,
            "description": "Local model served by oMLX.",
            "visibility": "list",
            "supported_in_api": True,
            "priority": priority,
            "base_instructions": instructions,
            "shell_type": "shell_command",
            # No default level: Codex sends no effort until the user picks one.
            "supported_reasoning_levels": [
                {"effort": effort, "description": f"{effort} reasoning effort"}
                for effort in efforts
            ],
            "support_verbosity": False,
            "apply_patch_tool_type": None,
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "experimental_supported_tools": [],
            "include_apps_usage_instructions": False,
        }
        context_window = info.get("max_context_window")
        if isinstance(context_window, int) and context_window > 0:
            entry["context_window"] = context_window
        model_type = info.get("model_type")
        if model_type in ("llm", "vlm"):
            entry["input_modalities"] = (
                ["text", "image"] if model_type == "vlm" else ["text"]
            )
        entries.append(entry)
    return {"models": entries}


def write_codex_model_catalog(
    codex_dir: Path,
    tool: str,
    ctx: IntegrationContext,
    model_ids: list[str],
    instructions: str,
) -> Path:
    # Per tool and endpoint, so a CLI launch or another server does not
    # rewrite the catalog a saved App config points to.
    endpoint_id = hashlib.sha256(ctx.openai_base_url.encode()).hexdigest()[:12]
    catalog_path = codex_dir / f"omlx-{tool}-models-{endpoint_id}.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = codex_model_catalog(ctx, model_ids, instructions)
    # Codex does not start on a partially written catalog.
    temporary = catalog_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, catalog_path)
    return catalog_path


def write_codex_config(
    config_path: Path, ctx: IntegrationContext, catalog_path: Path
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
        "model_catalog_json": json.dumps(str(catalog_path)),
    }

    # Left by older launches; the catalog now sets effort per model.
    managed_keys = {"model_reasoning_effort"}

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


def codex_config_args(ctx: IntegrationContext, catalog_path: Path) -> list[str]:
    """Build process-scoped Codex config overrides for an oMLX launch."""
    overrides = _provider_overrides(ctx.openai_base_url)
    # No global model_context_window: it would override every model's limit.
    overrides.append(("model_catalog_json", json.dumps(str(catalog_path))))
    return _as_cli_args(overrides)


class CodexIntegration(Integration):
    """Codex integration using process-scoped configuration for oMLX."""

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="env_var",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
            # Codex switches models with /model; --model picks the first one.
            requires_model_selection=False,
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        cmd = f"{get_cli_command_prefix()} launch codex"
        return f"{cmd} --model {ctx.model}" if ctx.model else cmd

    def configure(self, ctx: IntegrationContext) -> None:
        # Launch-time arguments carry the oMLX settings. Keeping this a no-op
        # ensures normal Codex sessions continue to use the user's config.
        return None

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        model_ids, model = resolve_codex_models(ctx)
        codex_bin = shutil.which("codex", path=env.get("PATH"))
        check_codex_version(codex_bin)
        catalog_path = write_codex_model_catalog(
            CODEX_CONFIG_PATH.parent,
            "codex",
            ctx,
            model_ids,
            codex_instructions(codex_bin, env),
        )
        env["OMLX_API_KEY"] = ctx.auth_token

        args = ["codex", *codex_config_args(ctx, catalog_path), "-m", model]
        args.extend(ctx.extra_args)
        print(f"Starting with {model}; /model switches among {len(model_ids)} models")

        os.execvpe("codex", args, env)
