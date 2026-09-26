# SPDX-License-Identifier: Apache-2.0
"""DeepSeek Harness (dsh) integration.

Registers the running oMLX server as an ``omlx`` provider route in the harness
profile patch layers — the desktop app's (``~/.dsh/profiles/desktop``) and the
web build's (``profiles/web``, which ``dsh web`` boots) — and opens the
desktop app. The ``llm-pi-ai`` entry gets ``config.providers.omlx``
(``api`` + ``baseURL`` + every model the server lists, with real capacity so
the harness never falls back to its 262k default context window); a named
``--model`` also sets ``agent-default-model``. The API key goes to the
credential store (``refs.OMLX_API_KEY`` in ``~/.dsh/.credentials.yaml``) and is
referenced via ``apiKeyEnv`` — secrets never enter the patch layer.

Both files are hand-edited and comment-carrying, so edits are surgical: only
managed blocks are rewritten, a timestamped backup is taken, and the result is
re-parsed before it replaces the original. The protocol defaults to
``openai-responses`` (``OMLX_DSH_API`` switches it).

Usage: ``omlx launch dsh [--model <id>]``
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix

DSH_APP_BUNDLE_ID = "com.deepseek.dsh"
DSH_APP_BUNDLE_NAME = "DeepSeek Harness.app"
DSH_APP_NAME = "DeepSeek Harness"
_APP_BUNDLE_ROOTS = (Path("/Applications"), Path.home() / "Applications")

# Provider route inside the llm-pi-ai patch entry.
PROVIDER_ROUTE = "omlx"
PROVIDER_DISPLAY_NAME = "oMLX"
API_KEY_REF = "OMLX_API_KEY"

# Protocol the route speaks to oMLX; OMLX_DSH_API switches it.
DEFAULT_API_PROTOCOL = "openai-responses"
API_PROTOCOLS = ("openai-responses", "openai-completions", "anthropic-messages")

# `open` forwards the environment to the app, and the app resolves its profile
# from these — a shell started *inside* a harness session exports them, which
# would bring the app up on the wrong home than the one just configured.
_DSH_LAUNCH_ENV_SCRUB = (
    "DSH_HOME",
    "DSH_SESSION_ID",
    "DSH_SESSION_JSONL",
    "DSH_SHELL",
)

_PATCH_ENTRY_HEADER = (
    "# Your patch layer for this dsh profile, applied after every bundle layer:\n"
    "# a top-level YAML array of loader patch entries (id-targeted config\n"
    "# overrides, disables, and insert lists; `!!js` expressions allowed).\n"
)
_LLM_PI_AI_ENTRY_COMMENT = "# oMLX provider route, written by `omlx launch dsh`.\n"
_AGENT_DEFAULT_ENTRY_COMMENT = (
    "# Default model for new sessions, written by `omlx launch dsh`.\n"
)

_LLM_PI_AI_ENTRY_ID = "llm-pi-ai"
_LLM_PI_AI_ENTRY_NAME = "@deepseek-ai/dsh-llm-pi-ai"
_AGENT_DEFAULT_ENTRY_ID = "agent-default-model"
_AGENT_DEFAULT_ENTRY_NAME = "@deepseek-ai/dsh-agent-default-model"


class DshConfigShapeError(RuntimeError):
    """The harness config file cannot be updated surgically."""


# ---------------------------------------------------------------------------
# Environment / path resolution
# ---------------------------------------------------------------------------


def dsh_home() -> Path:
    """Harness home holding profiles and credentials (``OMLX_DSH_HOME`` wins).

    The harness' own ``DSH_HOME`` is deliberately ignored: a shell inside a
    harness session exports it, and honouring it would write the provider
    route somewhere the desktop app never reads.
    """
    override = os.environ.get("OMLX_DSH_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".dsh"


def patch_file_path() -> Path:
    """Profile patch layer read by the DeepSeek Harness desktop app."""
    return dsh_home() / "profiles" / "desktop" / "cordis.patch.yml"


def web_patch_file_path() -> Path:
    """Web build's patch layer — `dsh web` boots profiles/web, not desktop."""
    return dsh_home() / "profiles" / "web" / "cordis.patch.yml"


def credentials_file_path() -> Path:
    """Harness credential store holding the ``OMLX_API_KEY`` ref."""
    return dsh_home() / ".credentials.yaml"


def api_protocol() -> str:
    """Resolve the protocol the provider route speaks (see OMLX_DSH_API)."""
    value = (os.environ.get("OMLX_DSH_API") or DEFAULT_API_PROTOCOL).strip()
    if value not in API_PROTOCOLS:
        raise DshConfigShapeError(
            f"OMLX_DSH_API must be one of: {', '.join(API_PROTOCOLS)} (got {value!r})"
        )
    return value


def find_dsh_app_bundle() -> Path | None:
    """App bundle path, or None. Matches on CFBundleIdentifier, not folder name."""
    for root in _APP_BUNDLE_ROOTS:
        bundle = root / DSH_APP_BUNDLE_NAME
        plist_path = bundle / "Contents" / "Info.plist"
        if not plist_path.is_file():
            continue
        try:
            with plist_path.open("rb") as f:
                info = plistlib.load(f)
        except (OSError, plistlib.InvalidFileException):
            continue
        if info.get("CFBundleIdentifier") == DSH_APP_BUNDLE_ID:
            return bundle
    return None


def _app_is_running() -> bool:
    """True when the desktop app is running (never launches it)."""
    try:
        proc = subprocess.run(
            ["osascript", "-e", f'application "{DSH_APP_NAME}" is running'],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip().lower() == "true"


# ---------------------------------------------------------------------------
# Surgical YAML block edits
#
# Both files are hand-edited and carry comments, so they are never
# re-serialized wholesale. These helpers locate a mapping key inside a line
# range and replace just its value block, keeping every other line as-is.
# ---------------------------------------------------------------------------


def _yaml_quote(value: str) -> str:
    """Value as a YAML-safe double-quoted scalar."""
    return json.dumps(value, ensure_ascii=False)


def _find_key_line(
    lines: list[str], key: str, start: int, end: int
) -> tuple[int, int] | None:
    """``(index, indent)`` of the shallowest ``key:`` line in ``lines[start:end]``.

    Shallowest wins so a deeper key of the same name cannot shadow the one the
    caller means; comment lines are skipped.
    """
    pattern = re.compile(rf"^(\s*){re.escape(key)}:(\s|$)")
    best: tuple[int, int] | None = None
    for i in range(start, min(end, len(lines))):
        line = lines[i]
        if line.lstrip().startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        if best is None or indent < best[1]:
            best = (i, indent)
            if indent == 0:
                break
    return best


def _inline_value(line: str) -> str:
    """Inline value following ``key:`` (empty for block style)."""
    _, sep, rest = line.partition(":")
    if not sep:
        return ""
    return rest.split(" #", 1)[0].strip()


def _block_content_end(lines: list[str], key_idx: int, key_indent: int) -> int:
    """Index just past a key's last content line.

    Trailing blanks/comments are excluded so they survive a replacement;
    comments before later content belong to the block and go with it.
    """
    j = key_idx + 1
    content_end = j
    while j < len(lines):
        line = lines[j]
        if line.strip() and (len(line) - len(line.lstrip())) <= key_indent:
            break
        j += 1
        if line.strip() and not line.lstrip().startswith("#"):
            content_end = j
    return content_end


def _entry_content_end(lines: list[str], start: int, end: int) -> int:
    """Index just past an entry's last content line (trailing comments kept)."""
    j = end
    while j > start + 1:
        line = lines[j - 1]
        if line.strip() and not line.lstrip().startswith("#"):
            break
        j -= 1
    return j


def _find_entry_bounds(
    lines: list[str], entry_id: str, entry_name: str
) -> tuple[int, int] | None:
    """Locate a top-level patch entry (``- `` at column 0) by id or module name.

    The entry's ``id``/``name`` may sit on the dash line or on later lines.
    """
    starts = [i for i, line in enumerate(lines) if line.startswith("- ")]
    for pos, entry_start in enumerate(starts):
        entry_end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        for line in lines[entry_start:entry_end]:
            m = re.match(r"^\s*(?:-\s+)?(id|name):\s*(.+?)\s*$", line)
            if not m:
                continue
            value = m.group(2).split(" #", 1)[0].strip().strip("'\"")
            if (m.group(1) == "id" and value == entry_id) or (
                m.group(1) == "name" and value == entry_name
            ):
                return entry_start, entry_end
    return None


def _upsert_block(
    lines: list[str],
    start: int,
    end: int,
    key: str,
    block: list[str],
    allow_inline: bool = False,
) -> list[str]:
    """Replace (or create) the block of ``key`` with ``block`` (key line first).

    Surrounding lines are untouched. A populated inline value
    (``config: {...}``) would be dropped by the rewrite and raises, unless the
    key is the managed target itself (``allow_inline``) or the value is an
    empty flow container (nothing to drop).
    """
    found = _find_key_line(lines, key, start, end)
    if found is None:
        return lines[:end] + block + lines[end:]
    key_idx, key_indent = found
    inline = _inline_value(lines[key_idx])
    if inline and inline not in ("{}", "[]") and not allow_inline:
        raise DshConfigShapeError(
            f"`{key}` uses inline flow style ({lines[key_idx].strip()}); "
            "convert it to block style first"
        )
    content_end = _block_content_end(lines, key_idx, key_indent)
    return lines[:key_idx] + block + lines[content_end:]


def _ensure_block(
    lines: list[str],
    start: int,
    end: int,
    key: str,
    indent: int,
    render_body,
) -> tuple[list[str], int, int, int]:
    """Ensure ``key`` exists as a block; returns ``(lines, idx, indent, end)``.

    ``render_body(indent)`` renders the nested content below the key line.
    Creates the key at ``end``, rewrites an empty flow value in place, and
    refuses a populated inline value (via ``_upsert_block``).
    """
    found = _find_key_line(lines, key, start, end)
    if found is None:
        block = [f"{' ' * indent}{key}:"] + render_body(indent)
        lines = lines[:end] + block + lines[end:]
        return lines, end, indent, end + len(block)
    key_idx, key_indent = found
    inline = _inline_value(lines[key_idx])
    if inline and inline not in ("{}", "[]"):
        raise DshConfigShapeError(
            f"`{key}` uses inline flow style ({lines[key_idx].strip()}); "
            "convert it to block style first"
        )
    if inline:  # empty flow container: rewrite in place
        block = [f"{' ' * key_indent}{key}:"] + render_body(key_indent)
        lines = lines[:key_idx] + block + lines[key_idx + 1 :]
        return lines, key_idx, key_indent, key_idx + len(block)
    return lines, key_idx, key_indent, _block_content_end(lines, key_idx, key_indent)


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------


def _provider_route_lines(
    base_url: str, models: list[dict], protocol: str, indent: int
) -> list[str]:
    """Render the ``omlx`` provider route block at the given indent."""
    pad = " " * indent
    out = [
        f"{pad}{PROVIDER_ROUTE}:",
        f"{pad}  displayName: {PROVIDER_DISPLAY_NAME}",
        f"{pad}  api: {protocol}",
        f"{pad}  baseURL: {_yaml_quote(base_url)}",
        f"{pad}  apiKeyEnv: {API_KEY_REF}",
        f"{pad}  models:",
    ]
    for model in models:
        out.append(f"{pad}    - id: {_yaml_quote(model['id'])}")
        out.append(f"{pad}      name: {_yaml_quote(model.get('name') or model['id'])}")
        if model.get("contextWindow"):
            out.append(f"{pad}      contextWindow: {int(model['contextWindow'])}")
        if model.get("maxTokens"):
            out.append(f"{pad}      maxTokens: {int(model['maxTokens'])}")
        modalities = model.get("input") or ["text"]
        out.append(f"{pad}      input:")
        for modality in modalities:
            out.append(f"{pad}        - {modality}")
    return out


def _llm_pi_ai_entry_lines(
    base_url: str, models: list[dict], protocol: str
) -> list[str]:
    """Render a complete ``llm-pi-ai`` patch entry carrying the route."""
    return [
        _LLM_PI_AI_ENTRY_COMMENT,
        f"- id: {_LLM_PI_AI_ENTRY_ID}",
        f'  name: "{_LLM_PI_AI_ENTRY_NAME}"',
        "  config:",
        "    providers:",
    ] + _provider_route_lines(base_url, models, protocol, indent=6)


def _agent_default_entry_lines(model: str) -> list[str]:
    """Render a complete ``agent-default-model`` patch entry."""
    return [
        _AGENT_DEFAULT_ENTRY_COMMENT,
        f"- id: {_AGENT_DEFAULT_ENTRY_ID}",
        f'  name: "{_AGENT_DEFAULT_ENTRY_NAME}"',
        "  config:",
        f"    provider: {PROVIDER_ROUTE}",
        f"    model: {_yaml_quote(model)}",
    ]


def _append_entry(lines: list[str], entry: list[str]) -> list[str]:
    if lines and lines[-1].strip():
        lines = lines + [""]
    return lines + entry


# ---------------------------------------------------------------------------
# Patch layer / credential store writers
# ---------------------------------------------------------------------------


class _PatchLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the ``!!js`` tags the patch layer allows."""


def _unknown_tag(loader, tag_suffix, node):  # type: ignore[no-untyped-def]
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_PatchLoader.add_multi_constructor("!", _unknown_tag)
_PatchLoader.add_multi_constructor("tag:yaml.org,2002:", _unknown_tag)


def _parse_yaml(text: str, what: str):
    try:
        return yaml.load(text, Loader=_PatchLoader)
    except yaml.YAMLError as e:
        raise DshConfigShapeError(f"{what} is not valid YAML: {e}") from e


def _upsert_llm_pi_ai(
    lines: list[str], base_url: str, models: list[dict], protocol: str
) -> list[str]:
    bounds = _find_entry_bounds(lines, _LLM_PI_AI_ENTRY_ID, _LLM_PI_AI_ENTRY_NAME)
    if bounds is None:
        return _append_entry(lines, _llm_pi_ai_entry_lines(base_url, models, protocol))
    start, end = bounds

    def config_body(indent: int) -> list[str]:
        return [" " * (indent + 2) + "providers:"] + _provider_route_lines(
            base_url, models, protocol, indent + 4
        )

    lines, cfg_idx, cfg_indent, cfg_end = _ensure_block(
        lines, start, _entry_content_end(lines, start, end), "config", 2, config_body
    )
    lines, prov_idx, prov_indent, prov_end = _ensure_block(
        lines,
        cfg_idx + 1,
        cfg_end,
        "providers",
        cfg_indent + 2,
        lambda indent: _provider_route_lines(base_url, models, protocol, indent + 2),
    )
    # Managed target: an inline value here is the old route, replaced whole.
    return _upsert_block(
        lines,
        prov_idx + 1,
        prov_end,
        PROVIDER_ROUTE,
        _provider_route_lines(base_url, models, protocol, prov_indent + 2),
        allow_inline=True,
    )


def _upsert_agent_default(lines: list[str], model: str) -> list[str]:
    bounds = _find_entry_bounds(
        lines, _AGENT_DEFAULT_ENTRY_ID, _AGENT_DEFAULT_ENTRY_NAME
    )
    if bounds is None:
        return _append_entry(lines, _agent_default_entry_lines(model))
    start, end = bounds

    def config_body(indent: int) -> list[str]:
        pad = " " * (indent + 2)
        return [f"{pad}provider: {PROVIDER_ROUTE}", f"{pad}model: {_yaml_quote(model)}"]

    lines, cfg_idx, cfg_indent, cfg_end = _ensure_block(
        lines, start, _entry_content_end(lines, start, end), "config", 2, config_body
    )
    pad = " " * (cfg_indent + 2)
    lines = _upsert_block(
        lines,
        cfg_idx + 1,
        cfg_end,
        "provider",
        [f"{pad}provider: {PROVIDER_ROUTE}"],
        allow_inline=True,
    )
    # The edit may have shifted lines; re-locate our config block (the
    # shallowest one after `start` is ours).
    found = _find_key_line(lines, "config", start, len(lines))
    if found is None:  # pragma: no cover - the block was there one line ago
        raise DshConfigShapeError("agent-default-model config block vanished")
    cfg_idx, cfg_indent = found
    cfg_end = _block_content_end(lines, cfg_idx, cfg_indent)
    pad = " " * (cfg_indent + 2)
    return _upsert_block(
        lines,
        cfg_idx + 1,
        cfg_end,
        "model",
        [f"{pad}model: {_yaml_quote(model)}"],
        allow_inline=True,
    )


def _validate_patch(
    text: str,
    base_url: str,
    models: list[dict],
    default_model: str | None,
    protocol: str,
) -> None:
    """Re-parse the generated patch and assert the managed state landed."""
    data = _parse_yaml(text, "the generated patch")

    def entry(entry_id: str, entry_name: str) -> dict:
        for item in data or []:
            if isinstance(item, dict) and (
                item.get("id") == entry_id or item.get("name") == entry_name
            ):
                return item
        raise ValueError(f"patch entry {entry_id!r} missing after write")

    route = (
        (entry(_LLM_PI_AI_ENTRY_ID, _LLM_PI_AI_ENTRY_NAME).get("config") or {}).get(
            "providers"
        )
        or {}
    ).get(PROVIDER_ROUTE)
    if not isinstance(route, dict):
        raise ValueError(f"provider route {PROVIDER_ROUTE!r} missing after write")
    for field, want in (("baseURL", base_url), ("api", protocol)):
        if route.get(field) != want:
            raise ValueError(f"{field} mismatch: {route.get(field)!r} != {want!r}")
    if [m.get("id") for m in route.get("models") or []] != [m["id"] for m in models]:
        raise ValueError("model list mismatch after write")
    if default_model:
        agent = (
            entry(_AGENT_DEFAULT_ENTRY_ID, _AGENT_DEFAULT_ENTRY_NAME).get("config")
            or {}
        )
        if (agent.get("provider"), agent.get("model")) != (
            PROVIDER_ROUTE,
            default_model,
        ):
            raise ValueError("agent-default-model was not rewritten")


def _backup_then_write(path: Path, text: str) -> None:
    if path.exists():
        timestamp = int(time.time())
        backup = path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(path, backup)
            print(f"Backup: {backup}")
        except OSError as e:
            print(f"Warning: could not create backup for {path}: {e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Config updated: {path}")


def write_dsh_patch(
    config_path: Path,
    base_url: str,
    models: list[dict],
    default_model: str | None = None,
) -> None:
    """Upsert the oMLX provider route (and optional default model) in place."""
    protocol = api_protocol()
    if not models:
        raise DshConfigShapeError("refusing to write a provider route with no models")

    existing = ""
    if config_path.exists():
        try:
            existing = config_path.read_text(encoding="utf-8")
        except OSError as e:
            raise DshConfigShapeError(f"cannot read {config_path}: {e}") from e

    lines = existing.splitlines() if existing else _PATCH_ENTRY_HEADER.splitlines()
    try:
        lines = _upsert_llm_pi_ai(lines, base_url, models, protocol)
        if default_model:
            lines = _upsert_agent_default(lines, default_model)
    except DshConfigShapeError as e:
        raise DshConfigShapeError(f"{config_path}: {e}") from e
    text = "\n".join(lines) + "\n"

    # Never land a file we cannot re-parse with the managed state in place.
    try:
        _validate_patch(text, base_url, models, default_model, protocol)
    except ValueError as e:
        raise DshConfigShapeError(
            f"generated config for {config_path} failed validation: {e}"
        ) from e

    _backup_then_write(config_path, text)


def write_credentials_ref(credentials_path: Path, ref_name: str, value: str) -> None:
    """Upsert ``refs.<ref_name>`` in the harness credential store."""
    if credentials_path.exists():
        try:
            lines = credentials_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise DshConfigShapeError(f"cannot read {credentials_path}: {e}") from e
        refs = _find_key_line(lines, "refs", 0, len(lines))
        if refs is not None and refs[1] == 0 and not _inline_value(lines[refs[0]]):
            # Block-style section: upsert just our key inside it.
            r_idx, r_indent = refs
            lines = _upsert_block(
                lines,
                r_idx + 1,
                _block_content_end(lines, r_idx, r_indent),
                ref_name,
                [f"{' ' * (r_indent + 2)}{ref_name}: {_yaml_quote(value)}"],
                allow_inline=True,
            )
        elif refs is not None and refs[1] == 0:
            # `refs: {}` rewritten as a block; a populated inline value
            # raises instead of dropping its entries.
            lines = _upsert_block(
                lines,
                refs[0],
                refs[0] + 1,
                "refs",
                ["refs:", f"  {ref_name}: {_yaml_quote(value)}"],
            )
        else:
            # No top-level section (or only a nested `refs:` key): create one.
            lines = lines + ["refs:", f"  {ref_name}: {_yaml_quote(value)}"]
    else:
        lines = [
            "version: 1",
            "records: {}",
            "refs:",
            f"  {ref_name}: {_yaml_quote(value)}",
        ]

    text = "\n".join(lines) + "\n"
    data = _parse_yaml(text, "the generated credential store")
    if not isinstance(data, dict) or (data.get("refs") or {}).get(ref_name) != value:
        raise DshConfigShapeError(
            f"generated credential store failed validation for {ref_name!r}"
        )

    _backup_then_write(credentials_path, text)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class DshIntegration(Integration):
    """DeepSeek Harness integration that registers oMLX as a provider route."""

    def __init__(self):
        super().__init__(
            name="dsh",
            display_name=DSH_APP_NAME,
            type="config_file",
            install_check="dsh",
            install_hint=(
                "Install the DeepSeek Harness desktop app "
                "(https://www.deepseek.com/harness/en/)"
            ),
            # The route carries the whole model catalog; --model only picks
            # the session default, so there is nothing to prompt for.
            requires_model_selection=False,
        )

    def is_installed(self) -> bool:
        # Desktop app only: dsh ships a web build and a preview desktop app,
        # but no CLI to launch.
        return find_dsh_app_bundle() is not None

    def get_command(self, ctx: IntegrationContext) -> str:
        cmd = f"{get_cli_command_prefix()} launch dsh"
        if ctx.model:
            cmd += f" --model {ctx.model}"
        return cmd

    def configure(self, ctx: IntegrationContext) -> None:
        models = self._fetch_models(ctx)
        if not models:
            print("oMLX server reported no models. Load a model first.")
            sys.exit(1)
        try:
            for target in (patch_file_path(), web_patch_file_path()):
                write_dsh_patch(
                    target,
                    ctx.openai_base_url,
                    models,
                    default_model=ctx.model or None,
                )
            write_credentials_ref(credentials_file_path(), API_KEY_REF, ctx.auth_token)
        except DshConfigShapeError as e:
            print(f"Cannot update the DeepSeek Harness config: {e}")
            sys.exit(1)
        default_note = f", default model {ctx.model}" if ctx.model else ""
        print(
            f"Registered oMLX provider with {len(models)} model(s){default_note} "
            f"in {DSH_APP_NAME}."
        )

    def _fetch_models(self, ctx: IntegrationContext) -> list[dict]:
        """List the server's models with capacity metadata for the route."""
        import requests

        headers = {"Authorization": f"Bearer {ctx.auth_token}"} if ctx.api_key else {}

        status_map: dict[str, dict] = {}
        try:
            resp = requests.get(
                f"{ctx.base_url}/v1/models/status", headers=headers, timeout=5
            )
            if resp.ok:
                for m in resp.json().get("models", []):
                    if m_id := m.get("id"):
                        status_map[m_id] = m
                    if alias := m.get("model_alias"):
                        status_map[alias] = m
        except Exception:
            pass

        ids: list[str] = []
        try:
            resp = requests.get(f"{ctx.base_url}/v1/models", headers=headers, timeout=5)
            resp.raise_for_status()
            ids = [
                m["id"]
                for m in resp.json().get("data", [])
                # Guard for servers that type the list; the status-driven
                # check below does the real filtering when this is unset.
                if m.get("id") and m.get("model_type") in ("llm", "vlm", None)
            ]
        except Exception:
            pass
        if not ids and ctx.model:
            ids = [ctx.model]

        models: list[dict] = []
        for m_id in ids:
            info = status_map.get(m_id, {})
            # Chat models only (matches the launcher's picker): embedding /
            # audio models on the server stay out of the route. Unknown types
            # — status fetch failed — stay in.
            model_type = info.get("model_type") or None
            if model_type and model_type not in ("llm", "vlm"):
                continue
            entry: dict = {"id": m_id, "name": m_id, "input": ["text"]}
            context_window = info.get("max_context_window")
            if isinstance(context_window, int) and context_window > 0:
                entry["contextWindow"] = context_window
            max_tokens = info.get("max_tokens")
            if isinstance(max_tokens, int) and max_tokens > 0:
                entry["maxTokens"] = max_tokens
            if str(info.get("model_type") or "").lower() == "vlm":
                entry["input"] = ["text", "image"]
            models.append(entry)
        return models

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        bundle = find_dsh_app_bundle()
        if bundle is None:
            # launch_command checks is_installed() first; this guards direct calls.
            print(f"{DSH_APP_NAME} is not installed.")
            print(f"Install: {self.install_hint}")
            sys.exit(1)

        env = self._scrubbed_env()
        was_running = _app_is_running()
        for key in _DSH_LAUNCH_ENV_SCRUB:
            env.pop(key, None)
        subprocess.run(["open", str(bundle)], env=env, check=False)
        if was_running:
            print(
                f"{DSH_APP_NAME} is already running — restart it to load the "
                "updated provider."
            )
        else:
            print(f"Opened {DSH_APP_NAME}.")
