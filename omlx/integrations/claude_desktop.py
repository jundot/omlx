# SPDX-License-Identifier: Apache-2.0
"""Auto-configure Claude Desktop (macOS) to use oMLX as a gateway.

When the ``Claude Desktop`` switch (``claude_code.desktop_enabled``)
is enabled, oMLX writes the Claude Desktop JSON configs so the app talks
**directly** to the oMLX server in gateway mode — no external reverse proxy
(ollama-switcher) needed. When disabled, the previous configuration is
restored.

File logic mirrors ``docs/task/references/ClaudeConfig.swift`` (validated
in the field), adapted to point at oMLX itself:

1. ``~/Library/Application Support/Claude/claude_desktop_config.json``
   → ``deploymentMode: "3p"``
2. ``~/Library/Application Support/Claude-3p/claude_desktop_config.json``
   → ``deploymentMode: "3p"``
3. ``~/Library/Application Support/Claude-3p/configLibrary/_meta.json``
   → entry ``{id: PROFILE_ID, name: "oMLX"}`` + ``appliedId: PROFILE_ID``
4. ``~/Library/Application Support/Claude-3p/configLibrary/<PROFILE_ID>.json``:
   - ``inferenceProvider: "gateway"``
   - ``inferenceGatewayBaseUrl: "http://127.0.0.1:<omlx port>"``
   - ``inferenceGatewayApiKey: <omlx api key, or "omlx" when open>``
   - ``inferenceGatewayAuthScheme: "bearer"``
   - ``disableDeploymentModeChooser: true``

Gateway format note verification:
Claude Desktop in gateway mode calls ``GET /v1/models`` and needs the
Anthropic family metadata (``display_name``, ``created_at``,
``anthropic_family_tier``, ``is_family_default``, ``max_tokens``) — see
``ollama-switcher` ModelMap.swift::catalog()`. With tier aliases exposed, oMLX
``/v1/models`` (``response_model_exclude_none=True``) already returns all
five fields on the tier slot entries, and bearer auth works on
``/v1/models`` and ``/v1/messages`` (``verify_api_key`` accepts the main /
sub keys plus the ``x-api-key`` fallback, and is open when no key is
configured). Field-tested by Peppe on 2026-09-07 against
``http://127.0.0.1:8000`` with a ``fable-5`` alias, so no ``/v1/models``
format change is needed here.

Only macOS is supported: on any other platform every public function is a
logged no-op returning ``False``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Stable oMLX gateway profile ID (UUID v4, hardcoded).
# Deliberately different from the ollama-switcher profile
# (``00000000-0000-4000-8000-000000000114``) so the two tools never share —
# and never clobber — each other's gateway profile. The 3rd group starts
# with ``4`` (v4) and the 4th with ``8`` (RFC 4122 variant).
PROFILE_ID = "00000000-0000-4000-8000-0000000000aa"
PROFILE_NAME = "oMLX"

# Gateway keys managed by this module inside the profile file. Restore
# removes exactly these (plus the legacy username/password/models keys the
# Swift reference also cleans up) and leaves every other key untouched.
_GATEWAY_KEYS = (
    "inferenceProvider",
    "inferenceGatewayBaseUrl",
    "inferenceGatewayAuthScheme",
    "inferenceGatewayApiKey",
    "inferenceGatewayUsername",
    "inferenceGatewayPassword",
    "inferenceModels",
)

# Private key inside the oMLX profile file recording the ``appliedId`` that
# was active in ``_meta.json`` before :func:`configure_omlx_gateway` took
# over, so :func:`restore` can hand the selection back to a third-party
# gateway profile (e.g. ollama-switcher) instead of dropping it.
_PREVIOUS_APPLIED_ID_KEY = "_omlxPreviousAppliedId"


def _is_macos() -> bool:
    """Return True only on macOS (patchable in tests)."""
    return sys.platform == "darwin"


def _home(root: Path | None) -> Path:
    """Resolve the home directory (overridable for tests)."""
    return root if root is not None else Path.home()


def _paths(home: Path) -> dict[str, Path]:
    """Resolve the four Claude Desktop config paths under *home*."""
    claude_dir = home / "Library" / "Application Support" / "Claude"
    claude3p_dir = home / "Library" / "Application Support" / "Claude-3p"
    library_dir = claude3p_dir / "configLibrary"
    return {
        "claude_config": claude_dir / "claude_desktop_config.json",
        "claude3p_config": claude3p_dir / "claude_desktop_config.json",
        "meta": library_dir / "_meta.json",
        "profile": library_dir / f"{PROFILE_ID}.json",
    }


def _read_json(path: Path) -> dict:
    """Read a JSON object; missing files (or invalid JSON) yield ``{}``.

    Corrupt files are backed up first so no user data is ever lost.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Claude Desktop config %s is not valid JSON: %s", path, exc)
        _backup(path)
        return {}
    if not isinstance(data, dict):
        logger.warning("Claude Desktop config %s is not a JSON object; ignoring", path)
        return {}
    return data


def _backup(path: Path) -> None:
    """Back up *path* to ``<path>.bak`` — but never overwrite an existing backup."""
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    except OSError as exc:
        logger.warning("Could not back up %s: %s", path, exc)


def _write_json(path: Path, data: dict) -> None:
    """Write *data* as pretty JSON, backing up the previous file first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_deployment_mode(path: Path, mode: str) -> None:
    """Read-modify-write ``deploymentMode`` on a desktop config file."""
    data = _read_json(path)
    if data.get("deploymentMode") == mode:
        return
    data["deploymentMode"] = mode
    _write_json(path, data)


def configure_omlx_gateway(
    port: int,
    api_key: str | None = None,
    home: Path | None = None,
) -> bool:
    """Point Claude Desktop at the oMLX gateway.

    Args:
        port: oMLX server port (``inferenceGatewayBaseUrl`` becomes
            ``http://127.0.0.1:<port>`` — oMLX directly, no proxy).
        api_key: oMLX API key, or ``"omlx"`` when the server is open
            (empty/None falls back to ``"omlx"``).
        home: fake home directory for tests (defaults to ``Path.home()``).

    Returns:
        True when the configuration was applied, False on non-macOS
        (logged no-op).
    """
    if not _is_macos():
        logger.info("Claude Desktop auto-config is macOS-only; skipping")
        return False
    resolved = _home(home)
    paths = _paths(resolved)
    key = (api_key or "").strip() or "omlx"

    _set_deployment_mode(paths["claude_config"], "3p")
    _set_deployment_mode(paths["claude3p_config"], "3p")

    meta = _read_json(paths["meta"])
    entries = [e for e in meta.get("entries", []) if e.get("id") != PROFILE_ID]
    entries.append({"id": PROFILE_ID, "name": PROFILE_NAME})
    meta["entries"] = entries
    previous_applied_id = meta.get("appliedId")
    meta["appliedId"] = PROFILE_ID
    _write_json(paths["meta"], meta)

    profile = _read_json(paths["profile"])
    # Remember who was applied before us (unless we were already applied, in
    # which case the previously stored marker stays valid).
    if previous_applied_id and previous_applied_id != PROFILE_ID:
        profile[_PREVIOUS_APPLIED_ID_KEY] = previous_applied_id
    profile["inferenceProvider"] = "gateway"
    profile["inferenceGatewayBaseUrl"] = f"http://127.0.0.1:{port}"
    profile["inferenceGatewayApiKey"] = key
    profile["inferenceGatewayAuthScheme"] = "bearer"
    profile["disableDeploymentModeChooser"] = True
    _write_json(paths["profile"], profile)

    logger.info(
        "Claude Desktop configured for oMLX gateway on 127.0.0.1:%s (profile %s)",
        port,
        PROFILE_ID,
    )
    return True


def restore(home: Path | None = None) -> bool:
    """Undo :func:`configure_omlx_gateway` (logged no-op off macOS).

    Removes the gateway keys from the oMLX profile, drops the ``_meta.json``
    entry (handing ``appliedId`` back to the profile that was applied before
    us, or clearing it when there was none), and reports
    ``deploymentMode`` back to ``"1p"``. Missing files are a safe no-op;
    unrelated keys are preserved; existing ``.bak`` backups are kept.
    """
    if not _is_macos():
        logger.info("Claude Desktop auto-config is macOS-only; skipping")
        return False
    resolved = _home(home)
    paths = _paths(resolved)

    _set_deployment_mode(paths["claude_config"], "1p")
    _set_deployment_mode(paths["claude3p_config"], "1p")

    meta = _read_json(paths["meta"])
    profile = _read_json(paths["profile"])
    previous_applied_id = profile.get(_PREVIOUS_APPLIED_ID_KEY)

    if meta:
        entries = [e for e in meta.get("entries", []) if e.get("id") != PROFILE_ID]
        if entries != meta.get("entries", []):
            meta["entries"] = entries
        if meta.get("appliedId") == PROFILE_ID:
            if previous_applied_id:
                # Hand the selection back to whoever held it before us.
                meta["appliedId"] = previous_applied_id
            else:
                meta.pop("appliedId", None)
        _write_json(paths["meta"], meta)

    if profile:
        changed = False
        for key in _GATEWAY_KEYS:
            if key in profile:
                del profile[key]
                changed = True
        if _PREVIOUS_APPLIED_ID_KEY in profile:
            del profile[_PREVIOUS_APPLIED_ID_KEY]
            changed = True
        if profile.get("disableDeploymentModeChooser") is not False:
            profile["disableDeploymentModeChooser"] = False
            changed = True
        if changed:
            _write_json(paths["profile"], profile)

    logger.info("Claude Desktop oMLX gateway configuration restored")
    return True


def is_configured(home: Path | None = None) -> bool:
    """Return True when the oMLX gateway profile is applied."""
    if not _is_macos():
        return False
    paths = _paths(_home(home))
    try:
        profile = _read_json(paths["profile"])
        if profile.get("inferenceProvider") != "gateway":
            return False
        base_url = str(profile.get("inferenceGatewayBaseUrl") or "")
        if not base_url.startswith("http://127.0.0.1:"):
            return False
        meta = _read_json(paths["meta"])
        return meta.get("appliedId") == PROFILE_ID
    except OSError:
        return False


def restart_claude_desktop() -> bool:
    """Quit and relaunch the Claude Desktop app (macOS only, best-effort)."""
    if not _is_macos():
        logger.info("Claude Desktop restart is macOS-only; skipping")
        return False
    try:
        subprocess.run(
            ["osascript", "-e", 'quit app "Claude"'],
            check=False,
            capture_output=True,
            timeout=15,
        )
        subprocess.run(
            ["open", "-a", "Claude"],
            check=False,
            capture_output=True,
            timeout=15,
        )
        logger.info("Claude Desktop restart requested")
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not restart Claude Desktop: %s", exc)
        return False


__all__ = [
    "PROFILE_ID",
    "PROFILE_NAME",
    "configure_omlx_gateway",
    "is_configured",
    "restore",
    "restart_claude_desktop",
]
