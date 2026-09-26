# SPDX-License-Identifier: Apache-2.0
"""Wire contract between the macOS app and ``/admin/api/global-settings``.

``GlobalSettingsPatch`` is hand-written in Swift against the flat
``GlobalSettingsRequest`` model, and both sides are free to drift: when a
field disappears server-side the app keeps sending it, Pydantic ignores the
unknown key, the PATCH still answers 200, and the control silently stops
persisting. That is what happened to the Claude Code
``context_scaling_enabled`` / ``target_context_size`` pair, dropped as a
BREAKING change in #2400 — the app kept a toggle that could never be saved
or read back.

These tests read the Swift declaration and the server model, so the next
removal fails here instead of shipping another dead toggle.
"""

import re
from pathlib import Path

from omlx.admin.routes import GlobalSettingsRequest

ROOT = Path(__file__).resolve().parents[1]
SWIFT_SOURCES = ROOT / "apps" / "omlx-mac" / "Sources"
PATCH_DTO = SWIFT_SOURCES / "Net" / "DTO" / "GlobalSettingsDTO.swift"

# `var fooBar: Type? = nil` inside GlobalSettingsPatch. The patch type encodes
# with `.convertToSnakeCase`, so the wire key is the snake-cased member name.
PATCH_MEMBER = re.compile(r"^\s+var ([A-Za-z][A-Za-z0-9]*): .* = nil$", re.M)

# Removed by #2400: auto-compact now follows the model's real context window,
# so neither the toggle nor its target has a server field any more.
REMOVED_CLAUDE_CODE_KEYS = (
    "claude_code_context_scaling_enabled",
    "claude_code_target_context_size",
    "claudeCodeContextScalingEnabled",
    "claudeCodeTargetContextSize",
    "contextScalingEnabled",
    "targetContextSize",
)


def _snake_case(member: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", member).lower()


def _patch_members() -> set[str]:
    body = PATCH_DTO.read_text().split("struct GlobalSettingsPatch", 1)[1]
    return set(PATCH_MEMBER.findall(body))


def test_patch_declaration_is_discovered():
    """Guards the parser itself: a renamed struct must not silently yield nothing."""
    members = _patch_members()
    assert len(members) > 30
    assert "maxAudioUploadSize" in members


def test_every_patch_field_is_accepted_by_the_server():
    accepted = set(GlobalSettingsRequest.model_fields)
    unknown = sorted(
        member for member in _patch_members() if _snake_case(member) not in accepted
    )
    assert not unknown, (
        "GlobalSettingsPatch sends fields the server does not accept: "
        f"{unknown}. Pydantic drops unknown keys, so the request still "
        "succeeds and the setting never persists."
    )


def test_app_does_not_target_the_removed_claude_code_scaling_settings():
    offenders = []
    for path in sorted(SWIFT_SOURCES.rglob("*.swift")):
        text = path.read_text()
        for key in REMOVED_CLAUDE_CODE_KEYS:
            if key in text:
                offenders.append(f"{path.relative_to(ROOT)}: {key}")
    assert not offenders, (
        "The app still targets the Claude Code context-scaling settings "
        "removed in #2400, which the server no longer accepts: " + ", ".join(offenders)
    )
